//! Runs clips through the same VAD + engine path as `compare`, without pacing, one engine per
//! child process so each one's peak memory is its own.

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::audio::{FRAME, RATE, read_wav};
use crate::engine::{self, Spec};
use crate::stats;
use crate::vad::{Edge, Vad};

const LEAD_SECS: f32 = 0.5;

pub struct Options {
    pub manifest: PathBuf,
    pub specs: Vec<Spec>,
    pub threads: usize,
    pub silence_ms: u32,
    pub max_segment_secs: f32,
    pub show: bool,
}

struct Clip {
    path: PathBuf,
    reference: String,
}

fn read_manifest(path: &Path) -> Result<Vec<Clip>> {
    let base = path.parent().unwrap_or(Path::new("."));
    std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let (file, reference) = l.split_once('\t').with_context(|| format!("expected `file<TAB>reference`: {l}"))?;
            Ok(Clip { path: base.join(file), reference: reference.to_string() })
        })
        .collect()
}

pub fn run(opts: Options) -> Result<()> {
    if opts.specs.len() == 1 {
        return run_one(&opts, &opts.specs[0]);
    }
    let exe = std::env::current_exe()?;
    println!("{:<12} {:>7} {:>7} {:>6} {:>8} {:>9} {:>9} {:>8}", "model", "WER", "short", "empty", "RTFx", "final p50", "final p95", "peak");
    for spec in &opts.specs {
        let arg = format!("{}={}:{}", spec.name, kind_name(spec), spec.dir.display());
        let status = std::process::Command::new(&exe)
            .args(["bench", "--manifest"])
            .arg(&opts.manifest)
            .args(["--model", &arg, "--threads", &opts.threads.to_string()])
            .args(["--silence-ms", &opts.silence_ms.to_string(), "--max-segment", &opts.max_segment_secs.to_string()])
            .args(opts.show.then_some("--show"))
            .env("MURMUR_BENCH_CHILD", "1")
            .status()?;
        if !status.success() {
            eprintln!("{}: bench failed ({status})", spec.name);
        }
    }
    Ok(())
}

fn kind_name(spec: &Spec) -> &'static str {
    match spec.kind {
        engine::Kind::Tdt => "tdt",
        engine::Kind::Unified => "unified",
    }
}

fn run_one(opts: &Options, spec: &Spec) -> Result<()> {
    let clips = read_manifest(&opts.manifest)?;
    if clips.is_empty() {
        bail!("{}: no clips", opts.manifest.display());
    }
    let mut model = engine::load(spec, opts.threads)?;

    let (mut edits, mut words) = (0usize, 0usize);
    let (mut short_edits, mut short_words) = (0usize, 0usize);
    let mut empty = 0;
    let mut audio_secs = 0.0;
    let mut compute = Duration::ZERO;
    let mut finals: Vec<Duration> = Vec::new();
    let mut out = Vec::new();

    for clip in &clips {
        // Dictation always has some silence around speech; corpus clips are cut tight to it.
        let samples = room_tone(&read_wav(&clip.path)?, LEAD_SECS);
        audio_secs += samples.len() as f64 / RATE as f64;
        let mut vad = Vad::new(opts.silence_ms, opts.max_segment_secs);
        let mut text = String::new();

        let begin = Instant::now();
        for chunk in samples.chunks(FRAME) {
            let mut frame = chunk.to_vec();
            frame.resize(FRAME, 0.0);
            let edges = vad.push(&frame);
            model.frame(&frame, &mut out)?;
            for edge in edges {
                match edge {
                    Edge::Start => model.speech_start(),
                    Edge::End { .. } => {
                        let t = Instant::now();
                        model.speech_end(&mut out)?;
                        finals.push(t.elapsed());
                    }
                }
            }
            text.extend(out.drain(..));
        }
        if vad.close().is_some() {
            let t = Instant::now();
            model.speech_end(&mut out)?;
            finals.push(t.elapsed());
        }
        model.finish(&mut out)?;
        text.extend(out.drain(..));
        compute += begin.elapsed();

        let reference = normalize(&clip.reference);
        let hypothesis = normalize(&text);
        let e = edit_distance(&reference, &hypothesis);
        edits += e;
        words += reference.len();
        if reference.len() <= 4 {
            short_edits += e;
            short_words += reference.len();
        }
        if hypothesis.is_empty() {
            empty += 1;
        }
        if opts.show && e > 0 {
            eprintln!("{} [{}]\n  ref: {}\n  hyp: {}", spec.name, clip.path.display(), reference.join(" "), hypothesis.join(" "));
        }
    }

    finals.sort();
    let pct = |p: f64| finals.get(((finals.len() as f64 - 1.0) * p).round() as usize).map_or(0.0, |d| d.as_secs_f64() * 1000.0);
    let rate = |e: usize, w: usize| if w == 0 { "-".to_string() } else { format!("{:.2}%", 100.0 * e as f64 / w as f64) };
    let line = format!(
        "{:<12} {:>7} {:>7} {:>6} {:>7.1}x {:>7.0}ms {:>7.0}ms {:>7.2}G",
        spec.name,
        rate(edits, words),
        rate(short_edits, short_words),
        format!("{empty}/{}", clips.len()),
        audio_secs / compute.as_secs_f64(),
        pct(0.5),
        pct(0.95),
        stats::Snapshot::read().peak_rss_gb,
    );
    if std::env::var_os("MURMUR_BENCH_CHILD").is_none() {
        println!("{:<12} {:>7} {:>7} {:>6} {:>8} {:>9} {:>9} {:>8}", "model", "WER", "short", "empty", "RTFx", "final p50", "final p95", "peak");
    }
    println!("{line}");
    Ok(())
}

/// Pads with faint noise rather than digital zeros, which no microphone produces.
fn room_tone(samples: &[f32], secs: f32) -> Vec<f32> {
    let n = (secs * RATE as f32) as usize;
    let mut seed = 0x2545_f491_u32;
    let mut noise = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32 - 0.5) * 2e-4
    };
    let mut v: Vec<f32> = (0..n).map(|_| noise()).collect();
    v.extend_from_slice(samples);
    v.extend((0..n).map(|_| noise()));
    v
}

/// Hesitations that references transcribe inconsistently; the Open ASR leaderboard drops them too.
const FILLERS: &[&str] = &["uh", "um", "uhm", "umm", "mm", "mmm", "mhm", "hmm", "hm", "ah", "er", "erm", "eh"];
/// Earnings-22 marks unintelligible spans as `<unk>` / `<inaudible>`; no model can match those.
const MARKERS: &[&str] = &["unk", "inaudible"];

/// Case- and punctuation-insensitive words; apostrophes kept so "don't" stays one word.
fn normalize(s: &str) -> Vec<String> {
    s.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '\'' { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .map(|w| w.trim_matches('\'').to_string())
        .filter(|w| !w.is_empty() && !FILLERS.contains(&w.as_str()) && !MARKERS.contains(&w.as_str()))
        .collect()
}

fn edit_distance(a: &[String], b: &[String]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.iter().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, y) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(x != y)).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wer_counts_word_edits() {
        let r = normalize("The cat, sat on the mat.");
        assert_eq!(edit_distance(&r, &normalize("the cat sat on mat")), 1);
        assert_eq!(edit_distance(&r, &normalize("the dog sat on the mat")), 1);
        assert_eq!(edit_distance(&r, &[]), 6);
        assert_eq!(normalize("Don't STOP"), vec!["don't", "stop"]);
        assert_eq!(normalize("ports where um, we uh can't"), vec!["ports", "where", "we", "can't"]);
    }
}
