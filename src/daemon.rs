//! Dictation daemon. Driven by line commands on `$XDG_RUNTIME_DIR/murmur/control.sock`
//! (`toggle`, `start`, `stop`, `status`), it types what you say into the focused window and
//! mirrors its state into `$XDG_RUNTIME_DIR/murmur/state` for the DMS overlay: `streaming` (the
//! mic is open, even while the model is still loading), `transcribing` (draining after a stop),
//! `idle`. `murmur toggle` writes `initializing` before the daemon is up. The file is removed
//! when the daemon exits.

use anyhow::{Context, Result, bail};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, unbounded};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::audio::Source;
use crate::engine::{self, Spec};
use crate::text::{Replacements, WordGate};
use crate::vad::{Edge, Vad};

/// A word held by the gate is let out once the model has been quiet this long: a fragment and
/// its continuation arrive within one 0.56s chunk.
const QUIET_FLUSH: Duration = Duration::from_millis(800);
/// Presses closer together than this are one press.
const DEBOUNCE: Duration = Duration::from_millis(300);

pub struct Options {
    pub spec: Spec,
    pub threads: usize,
    /// Stay loaded between dictations instead of exiting after each one.
    pub resident: bool,
    pub replacements: Replacements,
    pub silence_ms: u32,
    pub max_segment_secs: f32,
    /// Testing: a recording, paced in real time, instead of the microphone.
    pub replay: Option<PathBuf>,
    /// Testing: print what would be typed instead of typing it.
    pub print: bool,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum State {
    Idle,
    /// Only ever written by the client, for the gap before the daemon is up.
    Initializing,
    Streaming,
    Transcribing,
}

impl State {
    fn as_str(self) -> &'static str {
        match self {
            State::Idle => "idle",
            State::Initializing => "initializing",
            State::Streaming => "streaming",
            State::Transcribing => "transcribing",
        }
    }
}

enum Work {
    Frame(Vec<f32>),
    End,
}

struct Shared {
    state: State,
    capture_pid: Option<u32>,
    capture_stop: Arc<AtomicBool>,
    replay: Option<PathBuf>,
    last_toggle: Option<Instant>,
    paused_players: Vec<String>,
}

pub fn runtime_dir() -> PathBuf {
    std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("/run/user/{}", unsafe { libc::getuid() })))
        .join("murmur")
}

pub fn state_path() -> PathBuf {
    runtime_dir().join("state")
}

pub fn socket_path() -> PathBuf {
    runtime_dir().join("control.sock")
}

fn set_state(shared: &mut Shared, state: State) {
    shared.state = state;
    if let Err(e) = std::fs::write(state_path(), state.as_str()) {
        eprintln!("murmur: writing state file: {e}");
    }
    eprintln!("murmur: {}", state.as_str());
}

fn cleanup() {
    let _ = std::fs::remove_file(state_path());
    let _ = std::fs::remove_file(socket_path());
}

fn exit(code: i32) -> ! {
    cleanup();
    std::process::exit(code)
}

static STATE_PATH_C: OnceLock<std::ffi::CString> = OnceLock::new();

extern "C" fn on_term(_: libc::c_int) {
    // Only async-signal-safe calls here: a state file left behind would pin the overlay open.
    if let Some(p) = STATE_PATH_C.get() {
        unsafe { libc::unlink(p.as_ptr()) };
    }
    unsafe { libc::_exit(0) };
}

pub fn run(opts: Options) -> Result<()> {
    std::fs::create_dir_all(runtime_dir())?;
    if UnixStream::connect(socket_path()).is_ok() {
        bail!("another murmur daemon is already running");
    }
    let _ = std::fs::remove_file(socket_path());
    let listener = UnixListener::bind(socket_path()).context("binding control socket")?;

    let _ = STATE_PATH_C.set(std::ffi::CString::new(state_path().as_os_str().as_encoded_bytes())?);
    unsafe {
        let handler = on_term as extern "C" fn(libc::c_int) as libc::sighandler_t;
        libc::signal(libc::SIGTERM, handler);
        libc::signal(libc::SIGINT, handler);
    }
    if opts.resident {
        let _ = std::fs::remove_file(state_path());
    }

    let shared = Arc::new(Mutex::new(Shared {
        state: State::Idle,
        capture_pid: None,
        capture_stop: Arc::new(AtomicBool::new(false)),
        replay: opts.replay.clone(),
        last_toggle: None,
        paused_players: Vec::new(),
    }));
    let (work_tx, work_rx) = unbounded();
    let resident = opts.resident;
    {
        let shared = shared.clone();
        std::thread::Builder::new()
            .name("worker".into())
            .spawn(move || worker(opts, shared, work_rx))?;
    }

    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        if let Err(e) = handle(stream, &shared, &work_tx, resident) {
            eprintln!("murmur: {e:#}");
        }
    }
    Ok(())
}

fn handle(stream: UnixStream, shared: &Arc<Mutex<Shared>>, work_tx: &Sender<Work>, resident: bool) -> Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    let mut line = String::new();
    BufReader::new(&stream).read_line(&mut line)?;
    let mut s = shared.lock().unwrap();
    let cmd = line.trim();
    match cmd {
        "toggle" => {
            let bounce = s.last_toggle.is_some_and(|t| t.elapsed() < DEBOUNCE);
            s.last_toggle = Some(Instant::now());
            if bounce {
                eprintln!("murmur: toggle within {}ms of the last; ignored", DEBOUNCE.as_millis());
            } else {
                match s.state {
                    State::Idle | State::Initializing => start(&mut s, work_tx, resident)?,
                    State::Streaming => stop(&mut s),
                    State::Transcribing => eprintln!("murmur: toggle while transcribing; ignored"),
                }
            }
        }
        "start" if s.state == State::Idle => start(&mut s, work_tx, resident)?,
        "stop" if s.state == State::Streaming => stop(&mut s),
        "start" | "stop" | "status" => {}
        other => bail!("unknown command `{other}`"),
    }
    writeln!(&stream, "{}", s.state.as_str())?;
    Ok(())
}

fn start(s: &mut Shared, work_tx: &Sender<Work>, resident: bool) -> Result<()> {
    if s.replay.is_none() {
        s.paused_players = pause_media();
    }
    let source = match &s.replay {
        Some(path) => Source::wav(path, true),
        None => Source::mic(),
    };
    let mut source = match source {
        Ok(src) => src,
        Err(e) => {
            resume_media(&std::mem::take(&mut s.paused_players));
            if !resident {
                eprintln!("murmur: {e:#}");
                exit(1);
            }
            return Err(e);
        }
    };
    s.capture_pid = source.pid();
    s.capture_stop = Arc::new(AtomicBool::new(false));
    let stop = s.capture_stop.clone();
    // The mic is read to EOF after its kill, so audio already in the pipe isn't dropped; only a
    // replay, which never reaches EOF on its own, needs the flag.
    let stoppable = source.pid().is_none();
    let tx = work_tx.clone();
    std::thread::Builder::new().name("capture".into()).spawn(move || {
        while !(stoppable && stop.load(Ordering::Relaxed)) {
            match source.next_frame() {
                Ok(Some(frame)) => {
                    if tx.send(Work::Frame(frame)).is_err() {
                        return;
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    eprintln!("murmur: capture: {e:#}");
                    break;
                }
            }
        }
        let _ = tx.send(Work::End);
    })?;
    // Capture starts before the model is ready; that audio queues and is transcribed once it is,
    // so you can start talking at the keypress.
    set_state(s, State::Streaming);
    Ok(())
}

fn stop(s: &mut Shared) {
    set_state(s, State::Transcribing);
    s.capture_stop.store(true, Ordering::Relaxed);
    // Unblocks the capture thread's read; the audio already in the pipe is still delivered.
    if let Some(pid) = s.capture_pid.take() {
        unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
    }
}

fn worker(opts: Options, shared: Arc<Mutex<Shared>>, rx: Receiver<Work>) {
    let begin = Instant::now();
    let mut engine = match engine::load(&opts.spec, opts.threads) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("murmur: loading {}: {e:#}", opts.spec.name);
            exit(1);
        }
    };
    eprintln!("murmur: {} loaded in {:.2}s", opts.spec.name, begin.elapsed().as_secs_f64());

    let mut vad = Vad::new(opts.silence_ms, opts.max_segment_secs);
    let mut gate = WordGate::default();
    let mut typer = Typer::new(opts.replacements, opts.print);
    let mut out = Vec::new();
    let mut last_text = Instant::now();

    let take = |out: &mut Vec<String>, gate: &mut WordGate, typer: &mut Typer, last_text: &mut Instant| {
        for t in out.drain(..) {
            *last_text = Instant::now();
            if let Some(words) = gate.push(&t) {
                typer.type_text(&words);
            }
        }
    };

    loop {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(Work::Frame(frame)) => {
                let edges = vad.push(&frame);
                let mut result = engine.frame(&frame, &mut out);
                for edge in edges {
                    match edge {
                        Edge::Start => engine.speech_start(),
                        Edge::End { .. } => result = result.and(engine.speech_end(&mut out)),
                    }
                }
                if let Err(e) = result {
                    eprintln!("murmur: transcribing: {e:#}");
                }
                take(&mut out, &mut gate, &mut typer, &mut last_text);
            }
            Ok(Work::End) => {
                let mut result = Ok(());
                if vad.close().is_some() {
                    result = engine.speech_end(&mut out);
                }
                result = result.and(engine.finish(&mut out));
                if let Err(e) = result {
                    eprintln!("murmur: finishing: {e:#}");
                }
                take(&mut out, &mut gate, &mut typer, &mut last_text);
                if let Some(rest) = gate.flush() {
                    typer.type_text(&rest);
                }
                typer.finish();
                gate.reset();
                vad = Vad::new(opts.silence_ms, opts.max_segment_secs);

                let mut s = shared.lock().unwrap();
                resume_media(&std::mem::take(&mut s.paused_players));
                set_state(&mut s, State::Idle);
                if !opts.resident {
                    exit(0);
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return,
        }
        // Checked on every pass, not only on timeouts: while the mic is open a frame arrives every
        // 16ms, so the receive never times out.
        if last_text.elapsed() > QUIET_FLUSH
            && let Some(rest) = gate.flush()
        {
            typer.type_text(&rest);
        }
    }
}

/// Types through wtype; whatever fails to type is put on the clipboard at the end of the
/// dictation so nothing said is lost.
struct Typer {
    replacements: Replacements,
    untyped: String,
    print: Option<Instant>,
}

impl Typer {
    fn new(replacements: Replacements, print: bool) -> Self {
        Typer { replacements, untyped: String::new(), print: print.then(Instant::now) }
    }

    fn type_text(&mut self, text: &str) {
        let text = self.replacements.apply(text);
        if let Some(t0) = self.print {
            println!("{:>7.2}s type {text:?}", t0.elapsed().as_secs_f64());
            return;
        }
        // Through stdin, not argv: wtype would read a transcript starting with `-` as a flag.
        let typed = Command::new("wtype")
            .arg("-")
            .stdin(Stdio::piped())
            .spawn()
            .and_then(|mut child| {
                child.stdin.take().unwrap().write_all(text.as_bytes())?;
                child.wait()
            });
        match typed {
            Ok(status) if status.success() => {}
            other => {
                eprintln!("murmur: wtype failed ({other:?}); keeping {text:?} for the clipboard");
                self.untyped.push_str(&text);
            }
        }
    }

    fn finish(&mut self) {
        if self.untyped.is_empty() {
            return;
        }
        let text = std::mem::take(&mut self.untyped);
        let copied = Command::new("wl-copy")
            .arg("--")
            .arg(text.trim())
            .status()
            .is_ok_and(|s| s.success());
        let body = if copied { "Typing failed; the transcript is on the clipboard." } else { "Typing and the clipboard both failed." };
        let _ = Command::new("notify-send").args(["-a", "murmur", "Dictation", body]).status();
    }
}

fn pause_media() -> Vec<String> {
    let Ok(out) = Command::new("playerctl").arg("-l").stderr(Stdio::null()).output() else { return Vec::new() };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|p| {
            Command::new("playerctl")
                .args(["-p", p, "status"])
                .stderr(Stdio::null())
                .output()
                .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).trim() == "Playing")
        })
        .filter(|p| Command::new("playerctl").args(["-p", p, "pause"]).status().is_ok_and(|s| s.success()))
        .map(str::to_string)
        .collect()
}

fn resume_media(players: &[String]) {
    for p in players {
        let _ = Command::new("playerctl").args(["-p", p, "play"]).status();
    }
}

/// Sends one command, starting the daemon through systemd when it isn't running and the command
/// is one that should start it.
pub fn send(cmd: &str, unit: &str) -> Result<String> {
    if let Ok(reply) = request(cmd) {
        return Ok(reply);
    }
    if cmd != "toggle" && cmd != "start" {
        bail!("murmur daemon is not running");
    }
    // The overlay shows "Starting…" from the keypress on, not only once the daemon is up.
    std::fs::create_dir_all(runtime_dir())?;
    let _ = std::fs::write(state_path(), State::Initializing.as_str());
    let started = Command::new("systemctl").args(["--user", "start", unit]).status();
    if !started.is_ok_and(|s| s.success()) {
        let _ = std::fs::remove_file(state_path());
        bail!("systemctl --user start {unit} failed");
    }
    for _ in 0..400 {
        if let Ok(reply) = request(cmd) {
            return Ok(reply);
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let _ = std::fs::remove_file(state_path());
    bail!("{unit} never answered on {}", socket_path().display())
}

fn request(cmd: &str) -> Result<String> {
    let mut stream = UnixStream::connect(socket_path())?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    writeln!(stream, "{cmd}")?;
    let mut reply = String::new();
    BufReader::new(&stream).read_line(&mut reply)?;
    Ok(reply.trim().to_string())
}
