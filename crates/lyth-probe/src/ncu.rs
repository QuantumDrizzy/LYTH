//! Measured DRAM traffic from Nsight Compute, against the hand-written accounting.
//!
//! WHY THIS EXISTS. `intensity.rs` checks that `declared == Σflops / Σbytes`, where both
//! numbers come from an accounting a human wrote after reading the kernel source. That
//! verifies the arithmetic is self-consistent. It cannot verify the accounting describes the
//! kernel: write 29 bytes for something that moves 40 and it still passes.
//!
//! That is the same shape as dividing instruction counts by an assumed 3 GHz — it looks like
//! verification and it is bookkeeping. This module closes it the way NIBBLE already did by
//! hand: measured DRAM traffic against the analytic count, matched to 0.5% on 13 of 14
//! shape/format pairs.
//!
//! THE RATIO IS THE RESULT, NOT A PASS MARK. The accounting is a LOWER BOUND: the bytes the
//! algorithm must move. Measurement can legitimately come out on either side of it.
//!
//!   measured > expected cache-line granularity (touching 4 bytes pulls a 32-byte sector),
//!                         uncoalesced access, partial-line read-modify-write, spilled
//!                         registers, or traffic the accounting simply forgot
//!
//!   measured < expected the data never reached DRAM. L2 absorbed it, which means the
//!                         accounting is describing the wrong level of the hierarchy
//!
//! So a ratio of 1.03 confirms the byte model. A ratio of 4.0 says you are moving four times
//! what you think, and names the shortlist above. Both are findings; only one of them is a
//! surprise.

use std::collections::BTreeMap;

use serde::Serialize;
use thiserror::Error;

/// Byte metrics, per level of the hierarchy the accounting can name.
///
/// `dram__bytes` counts what crosses the memory controller. `lts__t_bytes` counts what crosses
/// the L2 slices. Comparing an accounting to the wrong one of these is the single easiest way
/// to conclude that a correct byte model is wrong: a kernel whose working set is L2-resident
/// moves every byte the accounting claims and shows half of them at DRAM.
const LEVEL_METRICS: &[(&str, &[&str])] = &[
    ("dram", &["dram__bytes.sum"]),
    ("l2", &["lts__t_bytes.sum", "lts__t_bytes.sum.per_second"]),
];

/// Per-direction metrics, when the report carries them.
///
/// The spelling matters and is easy to get wrong: on sm_120 these are `dram__bytes_op_read`,
/// **not** `dram__bytes_read`. The wrong name does not error — ncu prints the row with a
/// value of `n/a` — so a fallback keyed on it silently never fires. Checked against
/// `ncu --query-metrics` on the device rather than recalled.
const DIR_METRICS: &[(&str, Direction, &[&str])] = &[
    (
        "dram",
        Direction::Read,
        &["dram__bytes_op_read.sum", "dram__bytes_read.sum"],
    ),
    (
        "dram",
        Direction::Write,
        &["dram__bytes_op_write.sum", "dram__bytes_write.sum"],
    ),
    ("l2", Direction::Read, &["lts__t_bytes_op_read.sum"]),
    ("l2", Direction::Write, &["lts__t_bytes_op_write.sum"]),
];

/// Which half of the traffic a comparison is about.
///
/// This is not a display option. Within one launch a write-back cache may retire every
/// store into L2 and evict none of them before the kernel ends, so the writes an accounting
/// correctly lists can contribute **zero** DRAM bytes. Comparing a read+write count against
/// a read-only measurement then reads as a failure of the accounting, which it is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Direction {
    All,
    Read,
    Write,
}

impl Direction {
    pub fn label(self) -> &'static str {
        match self {
            Direction::All => "all",
            Direction::Read => "read",
            Direction::Write => "write",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "all" | "rw" => Some(Direction::All),
            "read" | "r" => Some(Direction::Read),
            "write" | "w" => Some(Direction::Write),
            _ => None,
        }
    }

    /// The share of one move's bytes that belongs to this direction.
    ///
    /// A move marked `rw` is a read-modify-write of the same bytes, so its count is split
    /// evenly. That is an assumption, and it is the only one in this file: a move that
    /// reads and writes *different* amounts must be declared as two moves.
    pub fn share_of(self, move_dir: &str) -> f64 {
        let d = move_dir.trim().to_ascii_lowercase();
        match (self, d.as_str()) {
            (Direction::All, _) => 1.0,
            (Direction::Read, "r") | (Direction::Write, "w") => 1.0,
            (Direction::Read, "rw") | (Direction::Write, "rw") => 0.5,
            _ => 0.0,
        }
    }
}

/// Units that are counts rather than sizes: no scale, and never a fractional value.
const COUNT_UNITS: &[&str] = &[
    "sector", "sectors", "request", "requests", "inst", "warp", "thread", "cycle", "block",
];

/// Order from farthest-from-register to nearest. Matches `kernel.rs::deeper_level`.
pub const LEVEL_ORDER: &[&str] = &["dram", "l2", "smem", "reg"];

/// The deepest level any move in an accounting names — the one the byte count is about.
pub fn deepest_level<'a>(levels: impl IntoIterator<Item = &'a str>) -> String {
    let mut best = usize::MAX;
    for l in levels {
        if let Some(i) = LEVEL_ORDER.iter().position(|x| *x == l) {
            best = best.min(i);
        }
    }
    LEVEL_ORDER
        .get(if best == usize::MAX { 0 } else { best })
        .unwrap_or(&"dram")
        .to_string()
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NcuTraffic {
    pub kernel: String,
    /// The level this `bytes` figure is measured at (`dram` or `l2`).
    pub level: String,
    /// Bytes crossing that boundary, summed over the launches in the report.
    pub bytes: f64,
    pub launches: u32,
    /// Which metric the number came from, so a reader can check it themselves.
    pub metric: String,
    /// Every other level the same report also measured. Printed beside the verdict so a
    /// "wrong level" diagnosis is confirmed or killed from the same profiling run, rather
    /// than left as a hypothesis for the reader to go and test.
    pub others: Vec<(String, f64, String)>,
    /// Read and write halves at this level, when the report carries them. A zero here is
    /// a finding, not a missing value: it says the cache retired every store.
    pub read: Option<f64>,
    pub write: Option<f64>,
    /// Every metric the report carried for this kernel, so a counter that scales with the
    /// element can supply the element count. See `--elements-from`.
    pub metrics: BTreeMap<String, f64>,
}

impl NcuTraffic {
    /// Measured bytes for one direction, or `None` when the report did not split them.
    pub fn bytes_for(&self, dir: Direction) -> Option<f64> {
        match dir {
            Direction::All => Some(self.bytes),
            Direction::Read => self.read,
            Direction::Write => self.write,
        }
    }
}

#[derive(Debug, Error)]
pub enum NcuError {
    #[error("no header row in the ncu csv — expected a line containing \"Metric Name\"")]
    NoHeader,
    #[error("csv has no column named `{0}`")]
    NoColumn(String),
    #[error(
        "no byte metric found. profile with:          ncu --csv --metrics dram__bytes.sum,lts__t_bytes.sum"
    )]
    NoDramMetric,
    #[error(
        "the accounting is about `{0}` traffic but the report measures only {1}. Add the metric for that level (dram -> dram__bytes.sum, l2 -> lts__t_bytes.sum)."
    )]
    NoLevel(String, String),
    #[error(
        "the report does not split traffic by direction, so `{0}` cannot be isolated. Profile with: ncu --csv --metrics dram__bytes_op_read.sum,dram__bytes_op_write.sum"
    )]
    NoDirection(String),
    #[error("kernel `{0}` not in the report. kernels present: {1}")]
    NoSuchKernel(String, String),
    #[error("{0}")]
    Message(String),
}

/// One field of an ncu csv line. ncu quotes every field and escapes `"` by doubling it.
fn split_csv(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_q = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if in_q && chars.peek() == Some(&'"') => {
                cur.push('"');
                chars.next();
            }
            '"' => in_q = !in_q,
            ',' if !in_q => out.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    out.push(cur);
    out
}

/// Turn one ncu `Metric Value` cell into a number of bytes.
///
/// TWO TRAPS, both found by running it rather than by reading about it.
///
/// LOCALE. ncu formats through the host locale. On this machine it printed 2,508,800 bytes as
/// `2.508.800` — dots as thousands separators. Reading that as a decimal gives 2.5 bytes and a
/// ratio wrong by six orders of magnitude, which is exactly the kind of number that looks like
/// a result. So the separator is resolved rather than assumed:
///
///   both `.` and `,` present the LAST one is the decimal point, the other groups
///   one kind, appearing 2+     grouping
///   one kind, appearing once ambiguous on its own — resolved by the unit below
///
/// UNIT. `--csv` implies `--print-units base`, so a byte metric arrives as a whole number of
/// bytes and a lone separator there must be grouping: a fractional byte does not exist. Under
/// an explicit `--print-units auto` the unit is scaled (`Mbyte`) and a lone separator is a
/// decimal point. An unrecognised unit is refused rather than given a guessed scale.
fn parse_value(raw: &str, unit: &str) -> Option<f64> {
    let u = unit.trim().to_ascii_lowercase();
    let scale = match u.as_str() {
        "" | "byte" | "bytes" => 1.0,
        "kbyte" | "kb" => 1e3,
        "mbyte" | "mb" => 1e6,
        "gbyte" | "gb" => 1e9,
        "tbyte" | "tb" => 1e12,
        // Kibibyte spellings, which ncu also emits depending on version.
        "kibyte" => 1024.0,
        "mibyte" => 1024.0 * 1024.0,
        "gibyte" => 1024.0 * 1024.0 * 1024.0,
        // Counts, not sizes. A sector or an instruction count has no byte scale to apply,
        // and refusing them would throw away exactly the counters that can supply an
        // element count for a data-dependent kernel.
        u if COUNT_UNITS.contains(&u) => 1.0,
        other => {
            eprintln!("  [ncu] unrecognised unit `{other}` — refusing to guess a scale");
            return None;
        }
    };
    // An exact count has no fractional part, so a lone separator in it groups digits.
    let base_unit = scale == 1.0;

    let t: String = raw.chars().filter(|c| !c.is_whitespace()).collect();
    if t.is_empty() || t.eq_ignore_ascii_case("n/a") {
        return None;
    }
    let dots = t.matches('.').count();
    let commas = t.matches(',').count();
    let decimal: Option<char> = match (dots, commas) {
        (0, 0) => None,
        // Both kinds: whichever comes last is the decimal point.
        (d, c) if d > 0 && c > 0 => Some(if t.rfind('.') > t.rfind(',') {
            '.'
        } else {
            ','
        }),
        // Repeated: cannot be a decimal point, so it groups.
        (d, 0) if d > 1 => None,
        (0, c) if c > 1 => None,
        // Exactly one, of one kind. The unit decides.
        (1, 0) => {
            if base_unit {
                None
            } else {
                Some('.')
            }
        }
        (0, 1) => {
            if base_unit {
                None
            } else {
                Some(',')
            }
        }
        _ => None,
    };
    let normalised: String = match decimal {
        None => t.chars().filter(|c| *c != '.' && *c != ',').collect(),
        Some(dec) => t
            .chars()
            .filter(|c| *c == dec || !matches!(c, '.' | ','))
            .map(|c| if c == dec { '.' } else { c })
            .collect(),
    };
    let v: f64 = normalised.parse().ok()?;
    Some(v * scale)
}

/// Parse `ncu --csv` output and total the DRAM bytes, per kernel.
///
/// `kernel` selects one by substring; `None` totals every kernel in the report, which is
/// what you want when the report contains a single launch and wrong when it does not.
pub fn parse(csv: &str, kernel: Option<&str>, level: Option<&str>) -> Result<NcuTraffic, NcuError> {
    let mut lines = csv.lines();
    let header = lines
        .by_ref()
        .find(|l| l.contains("Metric Name"))
        .ok_or(NcuError::NoHeader)?;
    let cols = split_csv(header);
    let idx = |name: &str| -> Result<usize, NcuError> {
        cols.iter()
            .position(|c| c.trim() == name)
            .ok_or_else(|| NcuError::NoColumn(name.into()))
    };
    let (c_name, c_val) = (idx("Metric Name")?, idx("Metric Value")?);
    let c_unit = idx("Metric Unit").ok();
    let c_kernel = idx("Kernel Name").ok();

    // metric -> (bytes, launches), per kernel
    let mut per_kernel: BTreeMap<String, BTreeMap<String, (f64, u32)>> = BTreeMap::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let f = split_csv(line);
        if f.len() <= c_val.max(c_name) {
            continue;
        }
        let metric = f[c_name].trim().to_string();
        let unit = c_unit
            .and_then(|i| f.get(i))
            .map(String::as_str)
            .unwrap_or("");
        let Some(bytes) = parse_value(&f[c_val], unit) else {
            continue;
        };
        if !bytes.is_finite() {
            continue;
        }
        let kname = c_kernel
            .and_then(|i| f.get(i))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "<all>".into());
        let e = per_kernel
            .entry(kname)
            .or_default()
            .entry(metric)
            .or_insert((0.0, 0));
        e.0 += bytes;
        e.1 += 1;
    }

    let (kname, metrics) = match kernel {
        Some(want) => per_kernel
            .iter()
            .find(|(k, _)| k.contains(want))
            .map(|(k, v)| (k.clone(), v.clone()))
            .ok_or_else(|| {
                let present: Vec<&str> = per_kernel.keys().map(String::as_str).collect();
                NcuError::NoSuchKernel(want.into(), present.join(", "))
            })?,
        None => {
            // Fold every kernel together. Correct for a single-launch report, wrong for a
            // multi-kernel one -- so say which kernels were folded rather than hiding it.
            let names: Vec<&str> = per_kernel.keys().map(String::as_str).collect();
            let label = if names.len() == 1 {
                names[0].to_string()
            } else {
                names.join(" + ")
            };
            let mut folded: BTreeMap<String, (f64, u32)> = BTreeMap::new();
            for m in per_kernel.values() {
                for (metric, (b, n)) in m {
                    let e = folded.entry(metric.clone()).or_insert((0.0, 0));
                    e.0 += b;
                    e.1 += n;
                }
            }
            (label, folded)
        }
    };

    // Collect every level the report measured, then hand back the one asked for.
    let mut found: Vec<(String, f64, String, u32)> = Vec::new();
    for (level, names) in LEVEL_METRICS {
        if let Some((name, (bytes, n))) =
            names.iter().find_map(|n| metrics.get(*n).map(|v| (*n, v)))
        {
            found.push(((*level).to_string(), *bytes, name.to_string(), *n));
        }
    }
    // Older metric sets give only the two halves; their sum is the same quantity.
    if !found.iter().any(|(l, ..)| l == "dram") {
        let pick = |dir: Direction| {
            DIR_METRICS
                .iter()
                .filter(|(l, d, _)| *l == "dram" && *d == dir)
                .flat_map(|(_, _, names)| names.iter())
                .find_map(|n| metrics.get(*n))
        };
        if let (Some((rb, rn)), Some((wb, _))) = (pick(Direction::Read), pick(Direction::Write)) {
            found.push((
                "dram".into(),
                rb + wb,
                "dram__bytes_op_read.sum + dram__bytes_op_write.sum".into(),
                *rn,
            ));
        }
    }
    if found.is_empty() {
        return Err(NcuError::NoDramMetric);
    }

    let want = level.unwrap_or("dram");
    let pick = found.iter().position(|(l, ..)| l == want).ok_or_else(|| {
        let have: Vec<&str> = found.iter().map(|(l, ..)| l.as_str()).collect();
        NcuError::NoLevel(want.into(), have.join(", "))
    })?;
    let (lvl, bytes, metric, launches) = found[pick].clone();
    let half = |dir: Direction| -> Option<f64> {
        DIR_METRICS
            .iter()
            .filter(|(l, d, _)| *l == lvl && *d == dir)
            .flat_map(|(_, _, names)| names.iter())
            .find_map(|n| metrics.get(*n))
            .map(|(b, _)| *b)
    };
    let (read, write) = (half(Direction::Read), half(Direction::Write));
    let all_metrics: BTreeMap<String, f64> =
        metrics.iter().map(|(k, (v, _))| (k.clone(), *v)).collect();
    let others = found
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != pick)
        .map(|(_, (l, b, m, _))| (l.clone(), *b, m.clone()))
        .collect();
    Ok(NcuTraffic {
        kernel: kname,
        level: lvl,
        bytes,
        launches,
        metric,
        others,
        read,
        write,
        metrics: all_metrics,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum TrafficVerdict {
    /// The byte model is confirmed against silicon.
    Confirmed {
        expected: f64,
        measured: f64,
        ratio: f64,
        tol: f64,
    },
    /// More traffic than the accounting predicts.
    Inflated {
        expected: f64,
        measured: f64,
        ratio: f64,
        tol: f64,
        causes: Vec<String>,
    },
    /// Less traffic than predicted: it never reached DRAM.
    Absorbed {
        expected: f64,
        measured: f64,
        ratio: f64,
        tol: f64,
        causes: Vec<String>,
    },
}

/// Compare a per-element accounting, scaled by the problem size, against measurement.
///
/// `bytes_per_element` must already be restricted to `dir` — use [`Direction::share_of`]
/// over the accounting's moves — and `measured` must carry that direction's metric.
pub fn compare(
    bytes_per_element: f64,
    elements: f64,
    measured: &NcuTraffic,
    dir: Direction,
    tol: f64,
) -> Result<TrafficVerdict, NcuError> {
    if elements <= 0.0 {
        return Err(NcuError::Message("elements must be > 0".into()));
    }
    if bytes_per_element <= 0.0 {
        return Err(NcuError::Message(format!(
            "the accounting has no {} bytes to compare",
            dir.label()
        )));
    }
    if !(0.0..1.0).contains(&tol) {
        return Err(NcuError::Message(format!(
            "tol must be in [0,1), got {tol}"
        )));
    }
    let Some(measured_bytes) = measured.bytes_for(dir) else {
        return Err(NcuError::NoDirection(dir.label().into()));
    };
    let expected = bytes_per_element * elements;
    let ratio = measured_bytes / expected;

    if (ratio - 1.0).abs() <= tol {
        return Ok(TrafficVerdict::Confirmed {
            expected,
            measured: measured_bytes,
            ratio,
            tol,
        });
    }
    if ratio > 1.0 {
        Ok(TrafficVerdict::Inflated {
            expected,
            measured: measured_bytes,
            ratio,
            tol,
            causes: vec![
                "cache-line granularity: a 4-byte access pulls a 32-byte sector, so a strided or scattered pattern moves up to 8x the bytes the accounting counts"
                    .into(),
                "uncoalesced access within a warp".into(),
                "partial-line writes turning into read-modify-write".into(),
                "register spills to local memory, which is DRAM-backed".into(),
                "traffic the accounting does not list at all".into(),
            ],
        })
    } else {
        Ok(TrafficVerdict::Absorbed {
            expected,
            measured: measured_bytes,
            ratio,
            tol,
            causes: absorbed_causes(dir),
        })
    }
}

/// Why fewer bytes arrived than the accounting counts, most specific first.
fn absorbed_causes(dir: Direction) -> Vec<String> {
    let mut causes = Vec::new();
    if dir != Direction::Read {
        causes.push(
            "write-back absorption: a store retires into L2 as a dirty line and only reaches DRAM when that line is evicted. Within one launch a working set that fits in L2              evicts nothing, so correctly counted writes contribute ZERO measured bytes. Check the write half: if it is 0, the accounting is not wrong, the measurement is read-only. Re-run with --ncu-dir read."
                .into(),
        );
    }
    causes.push(
        "the accounting names the wrong level — the traffic crosses L2 but not the memory controller"
            .into(),
    );
    causes.push(
        "the working set fits in cache at this problem size, so the measurement does not generalise to the size the accounting was written for"
            .into(),
    );
    causes.push("fewer elements were processed than the element count claims".into());
    causes
}

fn human(bytes: f64) -> String {
    const U: [(&str, f64); 4] = [("GB", 1e9), ("MB", 1e6), ("kB", 1e3), ("B", 1.0)];
    for (name, scale) in U {
        if bytes >= scale {
            return format!("{:.3} {name}", bytes / scale);
        }
    }
    format!("{bytes:.0} B")
}

pub fn format_verdict(v: &TrafficVerdict, m: &NcuTraffic, elements: f64, dir: Direction) -> String {
    let (expected, measured, ratio, tol) = match v {
        TrafficVerdict::Confirmed {
            expected,
            measured,
            ratio,
            tol,
        }
        | TrafficVerdict::Inflated {
            expected,
            measured,
            ratio,
            tol,
            ..
        }
        | TrafficVerdict::Absorbed {
            expected,
            measured,
            ratio,
            tol,
            ..
        } => (*expected, *measured, *ratio, *tol),
    };
    let mut out = String::new();
    out.push_str(&format!(
        "\ntraffic-check ({} {}): accounting against measurement\n",
        m.level,
        dir.label()
    ));
    out.push_str(&format!("  kernel:   {}\n", short_kernel(&m.kernel)));
    out.push_str(&format!(
        "  metric:   {} ({} launch(es))\n",
        m.metric, m.launches
    ));
    out.push_str(&format!("  elements: {elements}\n"));
    out.push_str(&format!(
        "  analytic: {:>10}  ({:.4} bytes/element)\n",
        human(expected),
        expected / elements
    ));
    out.push_str(&format!(
        "  measured: {:>10}  ({:.4} bytes/element)\n",
        human(measured),
        measured / elements
    ));
    out.push_str(&format!(
        "  ratio:    {ratio:.4}x measured/analytic (tol ±{:.1}%)\n",
        tol * 100.0
    ));
    // The read/write split, when the report carries it. A write half of zero is the whole
    // diagnosis for a memory-bound kernel whose state fits in L2, so it is never hidden.
    if let (Some(r), Some(w)) = (m.read, m.write) {
        out.push_str(&format!(
            "  split:    read {} / write {}
",
            human(r),
            human(w)
        ));
        if w == 0.0 {
            out.push_str(
                "            write half is 0: every store retired into L2, none evicted
",
            );
        }
    }
    // The same run measured the other levels. Print them: the "wrong level" hypothesis is
    // either confirmed or killed here, without a second profiling pass.
    for (level, bytes, metric) in &m.others {
        out.push_str(&format!(
            "  also {level:<5}{:>10}  ({:.4}x analytic, {metric})\n",
            human(*bytes),
            bytes / expected
        ));
    }
    match v {
        TrafficVerdict::Confirmed { .. } => {
            out.push_str(&format!(
                "verdict: CONFIRMED — the byte model describes this kernel at {}\n",
                m.level
            ));
        }
        TrafficVerdict::Inflated { causes, .. } => {
            out.push_str(&format!(
                "verdict: INFLATED — the kernel moves {ratio:.2}x the bytes the accounting counts.\n"
            ));
            out.push_str(
                "  The declared intensity is an upper bound on this machine, not the position\n\
                 \x20 this kernel occupies on the roofline. Shortlist:\n",
            );
            for (i, c) in causes.iter().enumerate() {
                out.push_str(&format!("  [{}] {c}\n", i + 1));
            }
        }
        TrafficVerdict::Absorbed { causes, .. } => {
            out.push_str(&format!(
                "verdict: ABSORBED — only {ratio:.2}x of the counted bytes reached {}.\n",
                m.level
            ));
            out.push_str("  Shortlist:\n");
            for (i, c) in causes.iter().enumerate() {
                out.push_str(&format!("  [{}] {c}\n", i + 1));
            }
        }
    }
    out
}

/// ncu prints the full mangled-ish signature. Keep the name, drop the parameter list.
fn short_kernel(k: &str) -> &str {
    match k.find('(') {
        Some(i) => &k[..i],
        None => k,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CSV: &str = r#""ID","Kernel Name","Metric Name","Metric Unit","Metric Value"
"0","k_integrate(int, int)","dram__bytes.sum","Mbyte","4.75"
"0","k_integrate(int, int)","gpu__time_duration.sum","usecond","120.5"
"1","k_propagate(int)","dram__bytes.sum","Mbyte","2.00"
"#;

    #[test]
    fn picks_the_named_kernel_and_scales_the_unit() {
        let t = parse(CSV, Some("k_integrate"), None).unwrap();
        assert_eq!(t.bytes, 4.75e6);
        assert_eq!(t.launches, 1);
        assert!(t.kernel.contains("k_integrate"));
    }

    #[test]
    fn folding_every_kernel_says_which_ones_it_folded() {
        let t = parse(CSV, None, None).unwrap();
        assert_eq!(t.bytes, 6.75e6);
        assert!(
            t.kernel.contains('+'),
            "folded label should name both kernels"
        );
    }

    #[test]
    fn a_missing_kernel_lists_what_is_there() {
        let e = parse(CSV, Some("k_nope"), None).unwrap_err();
        let msg = e.to_string();
        assert!(
            msg.contains("k_integrate"),
            "error should list kernels present: {msg}"
        );
    }

    #[test]
    fn confirmed_inflated_and_absorbed() {
        let t = parse(CSV, Some("k_integrate"), None).unwrap();
        // 29 bytes per neuron over 163,903 neurons ~ 4.75 MB
        let ok = compare(29.0, 163_903.0, &t, Direction::All, 0.05).unwrap();
        assert!(matches!(ok, TrafficVerdict::Confirmed { .. }), "{ok:?}");

        let low = compare(8.0, 163_903.0, &t, Direction::All, 0.05).unwrap();
        assert!(matches!(low, TrafficVerdict::Inflated { .. }), "{low:?}");

        let high = compare(200.0, 163_903.0, &t, Direction::All, 0.05).unwrap();
        assert!(matches!(high, TrafficVerdict::Absorbed { .. }), "{high:?}");
    }

    /// Verbatim from `ncu --csv` on this machine, Spanish locale: dots group the digits and
    /// the unit is base `byte` because `--csv` implies `--print-units base`.
    const REAL: &str = r#""ID","Process ID","Kernel Name","Section Name","Metric Name","Metric Unit","Metric Value"
"0","19936","k_integrate(int, int, const unsigned int *, int *, float *)","Command line profiler metrics","dram__bytes.sum","byte","2.508.800"
"0","19936","k_integrate(int, int, const unsigned int *, int *, float *)","Command line profiler metrics","dram__bytes_read.sum","","n/a"
"0","19936","k_integrate(int, int, const unsigned int *, int *, float *)","Command line profiler metrics","launch__grid_size","","288"
"#;

    #[test]
    fn a_grouping_dot_is_not_read_as_a_decimal_point() {
        let t = parse(REAL, Some("k_integrate"), None).unwrap();
        assert_eq!(
            t.bytes, 2_508_800.0,
            "2.508.800 byte is 2.5 MB, not 2.5 bytes"
        );
    }

    #[test]
    fn na_cells_are_skipped_not_counted_as_zero() {
        let t = parse(REAL, None, None).unwrap();
        assert_eq!(t.metric, "dram__bytes.sum");
        assert_eq!(
            t.launches, 1,
            "the n/a read metric must not inflate the launch count"
        );
    }

    #[test]
    fn separators_resolve_by_position_then_by_unit() {
        // Both present: the last one is the decimal point, whichever it is.
        assert_eq!(parse_value("1,234.5", "Mbyte"), Some(1234.5e6));
        assert_eq!(parse_value("1.234,5", "Mbyte"), Some(1234.5e6));
        // Repeated: grouping, in either convention.
        assert_eq!(parse_value("2.508.800", "byte"), Some(2_508_800.0));
        assert_eq!(parse_value("2,508,800", "byte"), Some(2_508_800.0));
        // Lone separator, base unit: a fractional byte does not exist, so it groups.
        assert_eq!(parse_value("4.750", "byte"), Some(4750.0));
        // Lone separator, scaled unit: a fractional megabyte does.
        assert_eq!(parse_value("4.750", "Mbyte"), Some(4.75e6));
    }

    /// Verbatim from the sm_120 run that produced the finding: every store retired into L2
    /// and not one was evicted inside the launch.
    const SPLIT: &str = r#""Kernel Name","Metric Name","Metric Unit","Metric Value"
"k_integrate(int, int)","dram__bytes.sum","byte","2.508.800"
"k_integrate(int, int)","dram__bytes_op_read.sum","byte","2.508.800"
"k_integrate(int, int)","dram__bytes_op_write.sum","byte","0"
"#;

    #[test]
    fn a_read_only_measurement_confirms_the_read_half_of_the_accounting() {
        let t = parse(SPLIT, None, None).unwrap();
        assert_eq!(t.read, Some(2_508_800.0));
        assert_eq!(
            t.write,
            Some(0.0),
            "a zero write half is a finding, not a missing value"
        );

        // 29 B/neuron of which 15 are reads (ring 4, refrac 2, adapt 4, v 4, is_stim 1).
        let all = compare(29.0, 166_700.0, &t, Direction::All, 0.05).unwrap();
        assert!(matches!(all, TrafficVerdict::Absorbed { .. }), "{all:?}");

        let read = compare(15.0, 166_700.0, &t, Direction::Read, 0.05).unwrap();
        assert!(
            matches!(read, TrafficVerdict::Confirmed { .. }),
            "the read half is exact; only the write half is invisible: {read:?}"
        );
    }

    #[test]
    fn rw_moves_split_evenly_between_the_halves() {
        assert_eq!(Direction::Read.share_of("rw"), 0.5);
        assert_eq!(Direction::Write.share_of("rw"), 0.5);
        assert_eq!(Direction::Read.share_of("r"), 1.0);
        assert_eq!(Direction::Read.share_of("w"), 0.0);
        assert_eq!(Direction::All.share_of("w"), 1.0);
    }

    #[test]
    fn a_report_without_a_split_refuses_rather_than_inventing_one() {
        let t = parse(CSV, Some("k_integrate"), None).unwrap();
        let e = compare(15.0, 166_700.0, &t, Direction::Read, 0.05).unwrap_err();
        assert!(e.to_string().contains("dram__bytes_op_read"), "{e}");
    }

    #[test]
    fn an_unknown_unit_is_refused_rather_than_guessed() {
        let bad = r#""Kernel Name","Metric Name","Metric Unit","Metric Value"
"k(int)","dram__bytes.sum","furlongs","4.75"
"#;
        assert!(
            parse(bad, None, None).is_err(),
            "must not invent a scale for an unknown unit"
        );
    }
}
