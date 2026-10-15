/// Holds back the word still being spoken. The streaming model emits fragments that can end
/// mid-word ("upg" then "uards"), and typed text can't be taken back, so only whole words go out:
/// everything up to the last whitespace, the rest once more arrives or the stream goes quiet.
#[derive(Default)]
pub struct WordGate {
    pending: String,
    started: bool,
}

impl WordGate {
    pub fn push(&mut self, delta: &str) -> Option<String> {
        self.pending.push_str(delta);
        let cut = self.pending.rfind(char::is_whitespace)? + 1;
        let ready: String = self.pending.drain(..cut).collect();
        self.emit(ready)
    }

    pub fn flush(&mut self) -> Option<String> {
        let rest = std::mem::take(&mut self.pending);
        self.emit(rest)
    }

    /// Next dictation starts fresh: no leading space before its first word.
    pub fn reset(&mut self) {
        self.pending.clear();
        self.started = false;
    }

    fn emit(&mut self, text: String) -> Option<String> {
        let text = if self.started { text } else { text.trim_start().to_string() };
        if text.is_empty() {
            return None;
        }
        self.started = true;
        Some(text)
    }
}

/// Case-insensitive, whole-word replacements, for words the model reliably mangles.
pub struct Replacements(Vec<(Vec<char>, String)>);

impl Replacements {
    /// Each rule is `FROM=TO`.
    pub fn parse(rules: &[String]) -> Result<Self, String> {
        rules
            .iter()
            .map(|r| {
                let (from, to) = r.split_once('=').ok_or(format!("replacement `{r}`: expected FROM=TO"))?;
                if from.is_empty() {
                    return Err(format!("replacement `{r}`: empty FROM"));
                }
                Ok((from.to_lowercase().chars().collect(), to.to_string()))
            })
            .collect::<Result<_, _>>()
            .map(Replacements)
    }

    pub fn apply(&self, text: &str) -> String {
        let mut out: Vec<char> = text.chars().collect();
        for (from, to) in &self.0 {
            let lower: Vec<char> = out.iter().flat_map(|c| c.to_lowercase()).collect();
            if lower.len() != out.len() {
                continue;
            }
            let mut result = Vec::with_capacity(out.len());
            let mut i = 0;
            while i < out.len() {
                let boundary_before = i == 0 || !out[i - 1].is_alphanumeric();
                let end = i + from.len();
                let matches = end <= out.len() && lower[i..end] == from[..];
                let boundary_after = end >= out.len() || !out[end].is_alphanumeric();
                if boundary_before && matches && boundary_after {
                    result.extend(to.chars());
                    i = end;
                } else {
                    result.push(out[i]);
                    i += 1;
                }
            }
            out = result;
        }
        out.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_types_whole_words_only() {
        let mut g = WordGate::default();
        assert_eq!(g.push(" upg"), None);
        assert_eq!(g.push("uards and at"), Some("upguards and ".into()));
        assert_eq!(g.push("om pain"), Some("atom ".into()));
        assert_eq!(g.flush(), Some("pain".into()));
        assert_eq!(g.flush(), None);
        g.reset();
        assert_eq!(g.push(" Next man"), Some("Next ".into()));
    }

    #[test]
    fn replacements_respect_word_boundaries() {
        let r = Replacements::parse(&["MX=Emacs".into(), "M X=Emacs".into()]).unwrap();
        assert_eq!(r.apply("open mx and m x, not mxl"), "open Emacs and Emacs, not mxl");
    }
}
