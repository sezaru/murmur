use anyhow::{Context, Result, bail};
use std::io::Read;
use std::path::Path;
use std::process::{Child, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

pub const RATE: u32 = 16_000;
/// earshot's fixed frame size: 16ms at 16kHz.
pub const FRAME: usize = 256;

pub enum Source {
    Mic {
        child: Child,
        stdout: ChildStdout,
    },
    Wav {
        samples: Vec<f32>,
        pos: usize,
        paced: Option<Instant>,
    },
}

impl Source {
    pub fn mic() -> Result<Self> {
        let mut child = Command::new("pw-record")
            .args(["--raw", "--rate", "16000", "--channels", "1", "--format", "f32", "--latency", "16ms", "-"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("spawning pw-record")?;
        let stdout = child.stdout.take().unwrap();
        Ok(Source::Mic { child, stdout })
    }

    pub fn pid(&self) -> Option<u32> {
        match self {
            Source::Mic { child, .. } => Some(child.id()),
            Source::Wav { .. } => None,
        }
    }

    /// `paced` replays at real-time speed, as if the file were the microphone.
    pub fn wav(path: &Path, paced: bool) -> Result<Self> {
        Ok(Source::Wav {
            samples: read_wav(path)?,
            pos: 0,
            paced: paced.then(Instant::now),
        })
    }

    pub fn next_frame(&mut self) -> Result<Option<Vec<f32>>> {
        match self {
            Source::Mic { stdout, .. } => {
                let mut bytes = [0u8; FRAME * 4];
                match stdout.read_exact(&mut bytes) {
                    Ok(()) => Ok(Some(
                        bytes
                            .chunks_exact(4)
                            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                            .collect(),
                    )),
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
                    Err(e) => Err(e.into()),
                }
            }
            Source::Wav { samples, pos, paced } => {
                if *pos >= samples.len() {
                    return Ok(None);
                }
                if let Some(start) = paced {
                    let due = *start + Duration::from_secs_f64(*pos as f64 / RATE as f64);
                    if let Some(wait) = due.checked_duration_since(Instant::now()) {
                        std::thread::sleep(wait);
                    }
                }
                let end = (*pos + FRAME).min(samples.len());
                let mut frame = samples[*pos..end].to_vec();
                frame.resize(FRAME, 0.0);
                *pos = end;
                Ok(Some(frame))
            }
        }
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        if let Source::Mic { child, .. } = self {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

pub fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|s| s as f32 / scale))
                .collect::<Result<_, _>>()?
        }
    };
    let mono: Vec<f32> = interleaved
        .chunks_exact(channels)
        .map(|c| c.iter().sum::<f32>() / channels as f32)
        .collect();
    if spec.sample_rate == RATE {
        return Ok(mono);
    }
    if spec.sample_rate < 8_000 {
        bail!("{}: unsupported sample rate {}", path.display(), spec.sample_rate);
    }
    Ok(resample_linear(&mono, spec.sample_rate, RATE))
}

/// Good enough for speech going *down* to 16kHz; nothing here needs a polyphase filter.
fn resample_linear(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    let ratio = from as f64 / to as f64;
    let len = (input.len() as f64 / ratio) as usize;
    (0..len)
        .map(|i| {
            let x = i as f64 * ratio;
            let j = x as usize;
            let frac = (x - j as f64) as f32;
            let a = input[j];
            let b = *input.get(j + 1).unwrap_or(&a);
            a + (b - a) * frac
        })
        .collect()
}

pub struct Recorder(hound::WavWriter<std::io::BufWriter<std::fs::File>>);

impl Recorder {
    pub fn create(path: &Path) -> Result<Self> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        Ok(Recorder(hound::WavWriter::create(path, spec)?))
    }

    pub fn write(&mut self, frame: &[f32]) -> Result<()> {
        for s in frame {
            self.0.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
        }
        Ok(())
    }

    pub fn finish(self) -> Result<()> {
        Ok(self.0.finalize()?)
    }
}
