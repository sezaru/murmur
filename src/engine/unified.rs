use anyhow::{Result, anyhow};
use parakeet_rs::{ParakeetUnified, UnifiedStreamingConfig};
use std::path::Path;

use super::{Engine, exec_config};

pub struct Unified {
    model: ParakeetUnified,
}

impl Unified {
    pub fn load(dir: &Path, threads: usize) -> Result<Self> {
        // The same windows voxtype runs: 0.56s chunks, 5.6s left / 0.56s right context.
        let model =
            ParakeetUnified::from_pretrained_with_streaming_config(dir, Some(exec_config(threads)), UnifiedStreamingConfig::default())
                .map_err(|e| anyhow!("{e}"))?;
        Ok(Unified { model })
    }
}

impl Engine for Unified {
    fn frame(&mut self, frame: &[f32], out: &mut Vec<String>) -> Result<()> {
        let delta = self.model.transcribe_chunk(frame).map_err(|e| anyhow!("{e}"))?;
        if !delta.is_empty() {
            out.push(delta);
        }
        Ok(())
    }

    fn speech_end(&mut self, _out: &mut Vec<String>) -> Result<()> {
        Ok(())
    }

    fn finish(&mut self, out: &mut Vec<String>) -> Result<()> {
        let delta = self.model.flush().map_err(|e| anyhow!("{e}"));
        self.model.reset();
        let delta = delta?;
        if !delta.is_empty() {
            out.push(delta);
        }
        Ok(())
    }
}
