/// Process and system memory figures, so a run that spilled to the swapfile is visible next to
/// its latencies. Zswapped is system-wide: pages held compressed in RAM, not on disk.
#[derive(Default)]
pub struct Snapshot {
    pub rss_gb: f64,
    pub peak_rss_gb: f64,
    pub zswapped_gb: f64,
    pub majflt: u64,
}

fn kb_field(text: &str, key: &str) -> Option<f64> {
    text.lines()
        .find(|l| l.starts_with(key))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse::<f64>().ok())
}

impl Snapshot {
    pub fn read() -> Self {
        let gb = |kb: f64| kb / 1024.0 / 1024.0;
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        // Field 12 of /proc/self/stat, counted after the parenthesised command name.
        let majflt = std::fs::read_to_string("/proc/self/stat")
            .ok()
            .and_then(|s| s.rsplit_once(')').and_then(|(_, rest)| rest.split_whitespace().nth(9)?.parse().ok()))
            .unwrap_or(0);
        Snapshot {
            rss_gb: gb(kb_field(&status, "VmRSS:").unwrap_or(0.0)),
            peak_rss_gb: gb(kb_field(&status, "VmHWM:").unwrap_or(0.0)),
            zswapped_gb: gb(kb_field(&meminfo, "Zswapped:").unwrap_or(0.0)),
            majflt,
        }
    }
}
