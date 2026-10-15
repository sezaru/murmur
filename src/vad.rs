use crate::audio::{FRAME, RATE};

const THRESHOLD: f32 = 0.5;
/// Consecutive voiced frames before a segment opens (48ms), so a click doesn't start one.
const START_FRAMES: usize = 3;

pub enum Edge {
    Start,
    /// `voice_end` is the stream time of the last voiced frame, not of the frame that closed the
    /// segment, so latency is measured from when you actually stopped talking.
    End { voice_end: f64 },
}

pub struct Vad {
    detector: Box<earshot::Detector>,
    speaking: bool,
    voiced_run: usize,
    silent_run: usize,
    segment_frames: usize,
    silence_frames: usize,
    max_frames: usize,
    frames: u64,
    last_voiced: u64,
    pub score: f32,
}

impl Vad {
    pub fn new(silence_ms: u32, max_segment_secs: f32) -> Self {
        let frame_ms = FRAME as f32 * 1000.0 / RATE as f32;
        Vad {
            detector: earshot::Detector::default_boxed(),
            speaking: false,
            voiced_run: 0,
            silent_run: 0,
            segment_frames: 0,
            silence_frames: (silence_ms as f32 / frame_ms).ceil() as usize,
            max_frames: (max_segment_secs * 1000.0 / frame_ms) as usize,
            frames: 0,
            last_voiced: 0,
            score: 0.0,
        }
    }

    pub fn speaking(&self) -> bool {
        self.speaking
    }

    /// Returns the edges this frame causes: a forced cut at the segment cap yields End then Start.
    pub fn push(&mut self, frame: &[f32]) -> Vec<Edge> {
        self.frames += 1;
        self.score = self.detector.predict_f32(frame);
        let voiced = self.score >= THRESHOLD;
        if voiced {
            self.last_voiced = self.frames;
        }

        let mut edges = Vec::new();
        if !self.speaking {
            self.voiced_run = if voiced { self.voiced_run + 1 } else { 0 };
            if self.voiced_run >= START_FRAMES {
                self.speaking = true;
                self.silent_run = 0;
                self.segment_frames = self.voiced_run;
                edges.push(Edge::Start);
            }
            return edges;
        }

        self.segment_frames += 1;
        self.silent_run = if voiced { 0 } else { self.silent_run + 1 };
        if self.silent_run >= self.silence_frames {
            self.speaking = false;
            self.voiced_run = 0;
            edges.push(self.end());
        } else if self.segment_frames >= self.max_frames {
            edges.push(self.end());
            self.segment_frames = 0;
            edges.push(Edge::Start);
        }
        edges
    }

    /// Closes an open segment at end of input.
    pub fn close(&mut self) -> Option<Edge> {
        self.speaking.then(|| {
            self.speaking = false;
            self.end()
        })
    }

    fn end(&self) -> Edge {
        Edge::End {
            voice_end: (self.last_voiced * FRAME as u64) as f64 / RATE as f64,
        }
    }
}
