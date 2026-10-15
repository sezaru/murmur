use anyhow::{Result, anyhow};
use parakeet_rs::{ParakeetTDT, Transcriber};
use std::path::Path;

use super::{Engine, exec_config, pad};
use crate::audio::RATE;

const PREROLL_SECS: f32 = 0.4;
const PAD_SECS: f32 = 0.3;

pub struct Tdt {
    model: ParakeetTDT,
    buf: Vec<f32>,
    speaking: bool,
}

impl Tdt {
    pub fn load(dir: &Path, threads: usize) -> Result<Self> {
        let model = ParakeetTDT::from_pretrained(dir, Some(exec_config(threads))).map_err(|e| anyhow!("{e}"))?;
        Ok(Tdt { model, buf: Vec::new(), speaking: false })
    }

    fn transcribe(&mut self, out: &mut Vec<String>) -> Result<()> {
        let audio = pad(&self.buf, PAD_SECS);
        self.buf.clear();
        let text = self.model.transcribe_samples(audio, RATE, 1, None).map_err(|e| anyhow!("{e}"))?.text;
        let text = text.trim();
        if !text.is_empty() {
            out.push(format!(" {text}"));
        }
        Ok(())
    }
}

impl Engine for Tdt {
    fn frame(&mut self, frame: &[f32], _out: &mut Vec<String>) -> Result<()> {
        self.buf.extend_from_slice(frame);
        let keep = (PREROLL_SECS * RATE as f32) as usize;
        if !self.speaking && self.buf.len() > keep {
            self.buf.drain(..self.buf.len() - keep);
        }
        Ok(())
    }

    fn speech_start(&mut self) {
        self.speaking = true;
    }

    fn speech_end(&mut self, out: &mut Vec<String>) -> Result<()> {
        self.speaking = false;
        self.transcribe(out)
    }

    fn finish(&mut self, out: &mut Vec<String>) -> Result<()> {
        if self.speaking {
            self.speaking = false;
            self.transcribe(out)?;
        }
        Ok(())
    }
}
