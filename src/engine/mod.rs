mod tdt;
mod unified;

use anyhow::{Result, bail};
use crossbeam_channel::{Receiver, Sender};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use crate::audio::RATE;

/// Every engine sees every frame; segment edges come from the shared VAD and an engine is free to
/// ignore them (the cache-aware model does, it commits as it goes). Text pushed to `out` is final
/// and appended verbatim, so engines include their own leading space.
pub trait Engine {
    fn frame(&mut self, frame: &[f32], out: &mut Vec<String>) -> Result<()>;
    fn speech_start(&mut self) {}
    fn speech_end(&mut self, out: &mut Vec<String>) -> Result<()>;
    /// End of input; the engine must be ready for a new stream afterwards.
    fn finish(&mut self, out: &mut Vec<String>) -> Result<()>;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    /// Offline Parakeet TDT (Phonon-2, parakeet-tdt-0.6b-v3), transcribed per VAD segment.
    Tdt,
    /// Cache-aware Parakeet (parakeet-unified-en), true streaming.
    Unified,
}

#[derive(Clone, Debug)]
pub struct Spec {
    pub name: String,
    pub kind: Kind,
    pub dir: PathBuf,
}

pub fn data_dir() -> PathBuf {
    std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".local/share"))
}

fn models_dir() -> PathBuf {
    data_dir().join("murmur/models")
}

impl std::str::FromStr for Spec {
    type Err = String;

    /// A preset name (`phonon2`, `unified`) or `name=kind:path`.
    fn from_str(s: &str) -> Result<Self, String> {
        let spec = |name: &str, kind, dir| Spec { name: name.into(), kind, dir };
        match s {
            "phonon2" => return Ok(spec("phonon2", Kind::Tdt, models_dir().join("phonon-2"))),
            "unified" => {
                let voxtype = data_dir().join("voxtype/models/parakeet-unified-en-0.6b-int8");
                let dir = if voxtype.is_dir() { voxtype } else { models_dir().join("parakeet-unified-en-0.6b-int8") };
                return Ok(spec("unified", Kind::Unified, dir));
            }
            _ => {}
        }
        let (name, rest) = s.split_once('=').ok_or(format!("unknown model `{s}`; use a preset or name=kind:path"))?;
        let (kind, path) = rest.split_once(':').ok_or(format!("`{s}`: expected name=kind:path"))?;
        let kind = match kind {
            "tdt" => Kind::Tdt,
            "unified" => Kind::Unified,
            other => return Err(format!("unknown kind `{other}` (tdt, unified)")),
        };
        Ok(spec(name, kind, PathBuf::from(path)))
    }
}

pub fn load(spec: &Spec, threads: usize) -> Result<Box<dyn Engine>> {
    if !spec.dir.is_dir() {
        bail!("{}: model dir {} missing (scripts/fetch-models)", spec.name, spec.dir.display());
    }
    Ok(match spec.kind {
        Kind::Tdt => Box::new(tdt::Tdt::load(&spec.dir, threads)?),
        Kind::Unified => Box::new(unified::Unified::load(&spec.dir, threads)?),
    })
}

static ORT: OnceLock<ort::environment::Environment> = OnceLock::new();

/// One ONNX Runtime environment for every session; ort loads the library from `ORT_DYLIB_PATH`.
pub fn init_ort() -> Result<()> {
    if ORT.get().is_none() {
        let env = ort::init().with_name("murmur").build().map_err(|e| anyhow::anyhow!("initialising onnxruntime: {e}"))?;
        let _ = ORT.set(env);
    }
    Ok(())
}

fn ort_env() -> ort::environment::Environment {
    ORT.get().expect("init_ort() not called").clone()
}

/// ONNX Runtime's worker threads spin between jobs by default. One model alone barely notices,
/// but several side by side burn every core waiting and starve each other.
fn no_spin(builder: ort::session::builder::SessionBuilder) -> ort::Result<ort::session::builder::SessionBuilder> {
    builder.with_intra_op_spinning(false).and_then(|b| b.with_inter_op_spinning(false)).map_err(Into::into)
}

fn exec_config(threads: usize) -> parakeet_rs::ExecutionConfig {
    parakeet_rs::ExecutionConfig::new()
        .with_intra_threads(threads)
        .with_environment(ort_env())
        .with_custom_configure(no_spin)
}

#[derive(Clone)]
pub enum Input {
    Frame(Arc<[f32]>),
    SpeechStart,
    SpeechEnd { voice_end: f64 },
    Finish,
}

pub enum Msg {
    Loaded { engine: usize, took: Duration },
    Failed { engine: usize, error: String },
    Text {
        engine: usize,
        text: String,
        /// Stream time the output answers to: the end of the frame that produced it, or the end
        /// of voice for a segment commit.
        stream_secs: f64,
        at: Instant,
    },
    Done { engine: usize },
    Audio { stream_secs: f64, speaking: bool },
    SourceDone,
}

pub fn spawn(engine: usize, spec: Spec, threads: usize, rx: Receiver<Input>, tx: Sender<Msg>) {
    std::thread::Builder::new()
        .name(spec.name.clone())
        .spawn(move || run(engine, &spec, threads, rx, tx))
        .expect("spawning engine thread");
}

fn run(engine: usize, spec: &Spec, threads: usize, rx: Receiver<Input>, tx: Sender<Msg>) {
    let started = Instant::now();
    let mut model = match load(spec, threads) {
        Ok(m) => m,
        Err(e) => {
            let _ = tx.send(Msg::Failed { engine, error: format!("{e:#}") });
            let _ = tx.send(Msg::Done { engine });
            return;
        }
    };
    let _ = tx.send(Msg::Loaded { engine, took: started.elapsed() });

    let mut samples = 0u64;
    let mut out = Vec::new();
    for input in rx {
                let (result, stream_secs, last) = match input {
            Input::Frame(frame) => {
                samples += frame.len() as u64;
                (model.frame(&frame, &mut out), samples as f64 / RATE as f64, false)
            }
            Input::SpeechStart => {
                model.speech_start();
                (Ok(()), samples as f64 / RATE as f64, false)
            }
            Input::SpeechEnd { voice_end } => (model.speech_end(&mut out), voice_end, false),
            Input::Finish => (model.finish(&mut out), samples as f64 / RATE as f64, true),
        };
        if let Err(e) = result {
            let _ = tx.send(Msg::Failed { engine, error: format!("{e:#}") });
        }
        for text in out.drain(..) {
            let _ = tx.send(Msg::Text { engine, text, stream_secs, at: Instant::now() });
        }
        if last {
            break;
        }
    }
    let _ = tx.send(Msg::Done { engine });
}

/// Silence either side of a segment: int8 Parakeet drops whole sub-2s utterances without it.
pub fn pad(samples: &[f32], secs: f32) -> Vec<f32> {
    let n = (secs * RATE as f32) as usize;
    let mut v = Vec::with_capacity(samples.len() + 2 * n);
    v.resize(n, 0.0);
    v.extend_from_slice(samples);
    v.resize(v.len() + n, 0.0);
    v
}
