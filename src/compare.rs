use anyhow::Result;
use crossbeam_channel::{Sender, unbounded};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::Stylize;
use ratatui::text::{Line, Text};
use ratatui::widgets::{Block, Paragraph, Wrap};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::audio::{FRAME, RATE, Recorder, Source};
use crate::engine::{self, Input, Msg, Spec};
use crate::stats;
use crate::vad::{Edge, Vad};

pub struct Options {
    pub specs: Vec<Spec>,
    pub threads: usize,
    pub replay: Option<PathBuf>,
    pub fast: bool,
    pub silence_ms: u32,
    pub max_segment_secs: f32,
    pub session_dir: PathBuf,
    /// Print final text as lines instead of drawing columns; for replays without a terminal.
    pub plain: bool,
}

struct Column {
    name: String,
    loaded: Option<Duration>,
    error: Option<String>,
    committed: String,
    lags: Vec<f64>,
    done: bool,
}

pub fn run(opts: Options) -> Result<()> {
    let (tx, rx) = unbounded::<Msg>();
    let mut inputs = Vec::new();
    for (i, spec) in opts.specs.iter().enumerate() {
        let (itx, irx) = unbounded();
        engine::spawn(i, spec.clone(), opts.threads, irx, tx.clone());
        inputs.push(itx);
    }

    let mut columns: Vec<Column> = opts
        .specs
        .iter()
        .map(|s| Column {
            name: s.name.clone(),
            loaded: None,
            error: None,
            committed: String::new(),
            lags: Vec::new(),
            done: false,
        })
        .collect();

    let stop = Arc::new(AtomicBool::new(false));
    let mut started: Option<Instant> = None;
    let mut source_done = false;
    let mut stream_secs = 0.0;
    let mut speaking = false;
    let mut feeder = None;
    let mut terminal = (!opts.plain).then(ratatui::init);
    let mut last_stats = (Instant::now(), stats::Snapshot::read());

    let result = (|| -> Result<()> {
        loop {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    Msg::Loaded { engine, took } => columns[engine].loaded = Some(took),
                    Msg::Failed { engine, error } => {
                        if opts.plain {
                            eprintln!("{}: {error}", columns[engine].name);
                        }
                        columns[engine].error = Some(error);
                    }
                    Msg::Text { engine, text, stream_secs, at } => {
                        let col = &mut columns[engine];
                        col.committed.push_str(&text);
                        if let Some(t0) = started {
                            let lag = at.duration_since(t0).as_secs_f64() - stream_secs;
                            col.lags.push(lag);
                            if opts.plain {
                                println!("{:>6.2}s {:<10} +{lag:.2}s {}", stream_secs, col.name, text.trim());
                            }
                        }
                    }
                    Msg::Done { engine } => columns[engine].done = true,
                    Msg::Audio { stream_secs: s, speaking: sp } => {
                        stream_secs = s;
                        speaking = sp;
                    }
                    Msg::SourceDone => source_done = true,
                }
            }

            let ready = columns.iter().all(|c| c.loaded.is_some() || c.error.is_some() || c.done);
            if ready && feeder.is_none() {
                let source = match &opts.replay {
                    Some(path) => Source::wav(path, !opts.fast)?,
                    None => Source::mic()?,
                };
                let recorder = match opts.replay {
                    Some(_) => None,
                    None => {
                        std::fs::create_dir_all(&opts.session_dir)?;
                        Some(Recorder::create(&opts.session_dir.join("audio.wav"))?)
                    }
                };
                let vad = Vad::new(opts.silence_ms, opts.max_segment_secs);
                let (inputs, stop, tx) = (inputs.clone(), stop.clone(), tx.clone());
                started = Some(Instant::now());
                feeder = Some(std::thread::spawn(move || feed(source, vad, recorder, inputs, stop, tx)));
            }

            if last_stats.0.elapsed() > Duration::from_millis(500) {
                last_stats = (Instant::now(), stats::Snapshot::read());
            }
            let all_done = columns.iter().all(|c| c.done);
            let Some(terminal) = terminal.as_mut() else {
                if all_done {
                    return Ok(());
                }
                std::thread::sleep(Duration::from_millis(20));
                continue;
            };
            terminal.draw(|f| {
                let [body, footer] = Layout::vertical([Constraint::Min(3), Constraint::Length(1)]).areas(f.area());
                let areas = Layout::horizontal(vec![Constraint::Ratio(1, columns.len() as u32); columns.len()]).split(body);
                for (col, area) in columns.iter().zip(areas.iter()) {
                    let status = match (&col.error, col.loaded) {
                        (Some(_), _) => "error".to_string(),
                        (None, None) => "loading…".to_string(),
                        (None, Some(t)) => {
                            let mean = if col.lags.is_empty() {
                                "-".to_string()
                            } else {
                                format!("{:.2}s", col.lags.iter().sum::<f64>() / col.lags.len() as f64)
                            };
                            let last = col.lags.last().map_or("-".to_string(), |l| format!("{l:.2}s"));
                            format!("load {:.1}s · lag {last} (avg {mean})", t.as_secs_f64())
                        }
                    };
                    let mut text = Text::from(Line::from(col.committed.trim_start().to_string()));
                    if let Some(e) = &col.error {
                        text.push_line(Line::from(e.clone()).red());
                    }
                    let block = Block::bordered().title(format!(" {} ", col.name).bold()).title_bottom(format!(" {status} "));
                    let para = Paragraph::new(text).wrap(Wrap { trim: false });
                    let inner_width = area.width.saturating_sub(2);
                    let lines = para.line_count(inner_width) as u16;
                    let scroll = lines.saturating_sub(area.height.saturating_sub(2));
                    f.render_widget(para.scroll((scroll, 0)).block(block), *area);
                }
                let s = &last_stats.1;
                let state = if all_done {
                    "finished · q to quit".to_string()
                } else if feeder.is_none() {
                    "loading models…".to_string()
                } else if source_done || stop.load(Ordering::Relaxed) {
                    "finishing…".to_string()
                } else if speaking {
                    "● speech · q to stop".to_string()
                } else {
                    "○ silence · q to stop".to_string()
                };
                f.render_widget(
                    Line::from(format!(
                        " {state}   t={stream_secs:.1}s   rss {:.2}G   zswapped {:.2}G   majflt {}",
                        s.rss_gb, s.zswapped_gb, s.majflt
                    ))
                    .reversed(),
                    footer,
                );
            })?;

            if event::poll(Duration::from_millis(50))?
                && let Event::Key(k) = event::read()?
                && k.kind == KeyEventKind::Press
            {
                let quit = matches!(k.code, KeyCode::Char('q') | KeyCode::Esc)
                    || (k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL));
                if quit {
                    if all_done || feeder.is_none() {
                        return Ok(());
                    }
                    stop.store(true, Ordering::Relaxed);
                }
            }
        }
    })();

    if terminal.is_some() {
        ratatui::restore();
    }
    stop.store(true, Ordering::Relaxed);
    if let Some(h) = feeder {
        match h.join() {
            Ok(r) => r?,
            Err(_) => anyhow::bail!("audio thread panicked"),
        }
    }
    result?;

    let mut report = String::new();
    for col in &columns {
        let mean = if col.lags.is_empty() { 0.0 } else { col.lags.iter().sum::<f64>() / col.lags.len() as f64 };
        report.push_str(&format!("## {} (avg lag {mean:.2}s)\n{}\n\n", col.name, col.committed.trim()));
    }
    print!("{report}");
    if opts.replay.is_none() {
        std::fs::write(opts.session_dir.join("transcripts.md"), &report)?;
        println!("saved to {}", opts.session_dir.display());
    }
    Ok(())
}

fn feed(
    mut source: Source,
    mut vad: Vad,
    mut recorder: Option<Recorder>,
    inputs: Vec<Sender<Input>>,
    stop: Arc<AtomicBool>,
    tx: Sender<Msg>,
) -> Result<()> {
    let send = |input: Input| {
        for i in &inputs {
            let _ = i.send(input.clone());
        }
    };
    let mut frames = 0u64;
    while !stop.load(Ordering::Relaxed) {
        let Some(frame) = source.next_frame()? else { break };
        if let Some(r) = &mut recorder {
            r.write(&frame)?;
        }
        let edges = vad.push(&frame);
        send(Input::Frame(frame.into()));
        for edge in edges {
            send(match edge {
                Edge::Start => Input::SpeechStart,
                Edge::End { voice_end } => Input::SpeechEnd { voice_end },
            });
        }
        frames += 1;
        if frames % 4 == 0 {
            let _ = tx.send(Msg::Audio { stream_secs: (frames * FRAME as u64) as f64 / RATE as f64, speaking: vad.speaking() });
        }
    }
    drop(source);
    if let Some(Edge::End { voice_end }) = vad.close() {
        send(Input::SpeechEnd { voice_end });
    }
    send(Input::Finish);
    if let Some(r) = recorder {
        r.finish()?;
    }
    let _ = tx.send(Msg::SourceDone);
    Ok(())
}
