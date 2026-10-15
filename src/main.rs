mod audio;
mod bench;
mod compare;
mod daemon;
mod engine;
mod stats;
mod text;
mod vad;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

use engine::Spec;

#[derive(Parser)]
#[command(about = "Local dictation, and side-by-side comparison of speech models")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Feed the microphone (or a recording) to several models at once and show their text side by side.
    Compare {
        #[command(flatten)]
        common: Common,
        /// Replay a WAV file as if it were the microphone, instead of recording.
        #[arg(long)]
        replay: Option<PathBuf>,
        /// With --replay: feed as fast as the models go instead of in real time.
        #[arg(long, requires = "replay")]
        fast: bool,
        /// Print each model's final text as lines instead of drawing columns.
        #[arg(long)]
        plain: bool,
    },
    /// Run the dictation daemon: types what you say into the focused window.
    Daemon {
        /// Model: a preset (phonon2, unified) or name=kind:path with kind tdt|unified.
        #[arg(short, long = "model", default_value = "unified")]
        model: Spec,
        /// ONNX Runtime threads.
        #[arg(long, default_value_t = 4)]
        threads: usize,
        /// Stay loaded between dictations instead of exiting after each one.
        #[arg(long)]
        resident: bool,
        /// Case-insensitive whole-word replacement, FROM=TO; repeatable.
        #[arg(long = "replace", value_name = "FROM=TO")]
        replacements: Vec<String>,
        /// Silence that closes a segment (segment models only).
        #[arg(long, default_value_t = 500)]
        silence_ms: u32,
        /// Longest segment before a forced cut (segment models only).
        #[arg(long = "max-segment", default_value_t = 25.0)]
        max_segment_secs: f32,
        /// Testing: use a recording, paced in real time, instead of the microphone.
        #[arg(long, hide = true)]
        replay: Option<PathBuf>,
        /// Testing: print what would be typed instead of typing it.
        #[arg(long, hide = true)]
        print: bool,
    },
    /// Start or stop dictating, starting the daemon if it isn't running.
    Toggle {
        #[arg(long, default_value = "murmur.service")]
        unit: String,
    },
    /// Print the daemon's state.
    Status,
    /// Score models on a manifest of `file.wav<TAB>reference text` lines.
    Bench {
        #[command(flatten)]
        common: Common,
        #[arg(long)]
        manifest: PathBuf,
        /// Print reference and hypothesis for every clip with errors.
        #[arg(long)]
        show: bool,
    },
}

#[derive(Args)]
struct Common {
    /// Model: a preset (phonon2, unified) or name=kind:path with kind tdt|unified.
    #[arg(short, long = "model", required = true)]
    models: Vec<Spec>,
    /// ONNX Runtime threads per model (default: cores split across models, at most 4).
    #[arg(long)]
    threads: Option<usize>,
    /// Silence that closes a segment.
    #[arg(long, default_value_t = 500)]
    silence_ms: u32,
    /// Longest segment before a forced cut.
    #[arg(long = "max-segment", default_value_t = 25.0)]
    max_segment_secs: f32,
}

impl Common {
    fn threads(&self) -> usize {
        self.threads.unwrap_or_else(|| {
            let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
            (cores / self.models.len()).clamp(1, 4)
        })
    }
}

fn timestamp() -> String {
    let now = unsafe { libc::time(std::ptr::null_mut()) };
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    unsafe { libc::localtime_r(&now, &mut tm) };
    format!(
        "{:04}{:02}{:02}-{:02}{:02}{:02}",
        tm.tm_year + 1900,
        tm.tm_mon + 1,
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec
    )
}

fn main() -> Result<()> {
    engine::init_ort()?;
    match Cli::parse().command {
        Command::Compare { common, replay, fast, plain } => compare::run(compare::Options {
            threads: common.threads(),
            specs: common.models,
            replay,
            fast,
            silence_ms: common.silence_ms,
            max_segment_secs: common.max_segment_secs,
            session_dir: engine::data_dir().join("murmur/sessions").join(timestamp()),
            plain,
        }),
        Command::Daemon { model, threads, resident, replacements, silence_ms, max_segment_secs, replay, print } => daemon::run(daemon::Options {
            spec: model,
            threads,
            resident,
            replacements: text::Replacements::parse(&replacements).map_err(anyhow::Error::msg)?,
            silence_ms,
            max_segment_secs,
            replay,
            print,
        }),
        Command::Toggle { unit } => {
            println!("{}", daemon::send("toggle", &unit)?);
            Ok(())
        }
        Command::Status => {
            println!("{}", daemon::send("status", "").unwrap_or_else(|_| "stopped".into()));
            Ok(())
        }
        Command::Bench { common, manifest, show } => bench::run(bench::Options {
            threads: common.threads(),
            specs: common.models,
            manifest,
            silence_ms: common.silence_ms,
            max_segment_secs: common.max_segment_secs,
            show,
        }),
    }
}
