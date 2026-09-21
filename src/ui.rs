//! The game show.

use crate::hardware::Hardware;
use crate::verdict::{is_fast, is_usable, Ranked, Verdict};
use figlet_rs::FIGfont;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::{IsTerminal, Write};
use std::sync::OnceLock;
use std::time::Duration;
use unicode_width::UnicodeWidthStr;

// ---------------------------------------------------------------- styling

pub const RESET: &str = "\x1b[0m";
pub const BOLD: &str = "\x1b[1m";
pub const DIM: &str = "\x1b[38;2;128;128;128m";
pub const ITALIC: &str = "\x1b[3m";
pub const GREEN: &str = "\x1b[38;2;34;197;94m";
pub const AMBER: &str = "\x1b[38;2;245;158;11m";
pub const RED: &str = "\x1b[38;2;239;68;68m";
/// Saffron.
pub const ACCENT: &str = "\x1b[38;2;255;153;51m";

const SAFFRON: (u8, u8, u8) = (255, 153, 51);
const WHITE: (u8, u8, u8) = (255, 255, 255);
const INDIA_GREEN: (u8, u8, u8) = (19, 136, 8);

const DOS_REBEL: &str = include_str!("../fonts/dos_rebel.flf");
const ANSI_REGULAR: &str = include_str!("../fonts/ansi_regular.flf");
const SMALL: &str = include_str!("../fonts/small.flf");

static COLOR: OnceLock<bool> = OnceLock::new();

pub fn color_enabled() -> bool {
    *COLOR.get_or_init(|| {
        if std::env::var_os("NO_COLOR").is_some() {
            return false;
        }
        if std::env::var_os("CLICOLOR_FORCE").is_some() || std::env::var_os("FORCE_COLOR").is_some() {
            return true;
        }
        std::io::stdout().is_terminal()
    })
}

pub fn term_width() -> usize {
    if let Some((terminal_size::Width(w), _)) = terminal_size::terminal_size() {
        return w as usize;
    }
    std::env::var("COLUMNS").ok().and_then(|c| c.parse().ok()).unwrap_or(100)
}

/// A run of styled segments whose visible width we can measure.
#[derive(Clone, Default)]
pub struct Styled {
    parts: Vec<(String, String)>,
}

impl Styled {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn plain(text: impl Into<String>) -> Self {
        Self::new().push(text, "")
    }

    pub fn push(mut self, text: impl Into<String>, style: &str) -> Self {
        self.parts.push((text.into(), style.to_string()));
        self
    }

    pub fn width(&self) -> usize {
        self.parts.iter().map(|(t, _)| t.width()).sum()
    }

    pub fn render(&self) -> String {
        let mut s = String::new();
        for (text, style) in &self.parts {
            if color_enabled() && !style.is_empty() {
                s.push_str(style);
                s.push_str(text);
                s.push_str(RESET);
            } else {
                s.push_str(text);
            }
        }
        s
    }

    pub fn pad_right(self, w: usize) -> Self {
        let n = w.saturating_sub(self.width());
        self.push(" ".repeat(n), "")
    }

    pub fn pad_left(self, w: usize) -> Self {
        let n = w.saturating_sub(self.width());
        let mut out = Styled::plain(" ".repeat(n));
        out.parts.extend(self.parts);
        out
    }
}

fn style2(a: &str, b: &str) -> String {
    format!("{a}{b}")
}

fn out(s: &str) {
    let mut o = std::io::stdout().lock();
    let _ = writeln!(o, "{s}");
}

fn centered(lines: &[Styled]) {
    let tw = term_width();
    let block_w = lines.iter().map(|l| l.width()).max().unwrap_or(0);
    let pad = tw.saturating_sub(block_w) / 2;
    for l in lines {
        out(&format!("{}{}", " ".repeat(pad), l.render()));
    }
}

// ---------------------------------------------------------------- banner

fn lerp(a: (u8, u8, u8), b: (u8, u8, u8), t: f64) -> (u8, u8, u8) {
    let f = |x: u8, y: u8| (x as f64 + (y as f64 - x as f64) * t).round() as u8;
    (f(a.0, b.0), f(a.1, b.1), f(a.2, b.2))
}

/// Saffron -> white -> green across each line, tiranga style.
fn gradient_line(line: &str) -> Styled {
    let chars: Vec<char> = line.chars().collect();
    let n = chars.len().saturating_sub(1).max(1) as f64;
    let mut s = Styled::new();
    for (i, ch) in chars.iter().enumerate() {
        let t = i as f64 / n;
        let (r, g, b) = if t < 0.5 {
            lerp(SAFFRON, WHITE, t * 2.0)
        } else {
            lerp(WHITE, INDIA_GREEN, (t - 0.5) * 2.0)
        };
        s = s.push(ch.to_string(), &format!("\x1b[1;38;2;{r};{g};{b}m"));
    }
    s
}

fn fig(text: &str, font: &str) -> Vec<String> {
    let font = FIGfont::from_content(font).expect("embedded figlet font");
    let art = font.convert(text).map(|f| f.to_string()).unwrap_or_else(|| text.to_string());
    art.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim_end().to_string())
        .collect()
}

/// ascii -> fullwidth unicode, so the line reads about twice as large.
fn fullwidth(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            ' ' => '\u{3000}',
            '!'..='~' => char::from_u32(c as u32 + 0xFEE0).unwrap_or(c),
            c => c,
        })
        .collect()
}

pub fn banner() {
    let w = term_width();
    let big = if w >= 116 {
        fig("TOKENPATI", DOS_REBEL)
    } else if w >= 78 {
        fig("TOKENPATI", ANSI_REGULAR)
    } else {
        fig("TOKENPATI", SMALL)
    };
    out("");
    let lines: Vec<Styled> = big.iter().map(|l| gradient_line(l)).collect();
    centered(&lines);
    centered(&[Styled::new().push(fullwidth("kaun banega tokenpati"), &style2(BOLD, ACCENT))]);
    centered(&[Styled::new().push("which local model will actually run on this thing", &style2(ITALIC, DIM))]);
    out("");
}

// ---------------------------------------------------------------- the dramatic pause

const STAGES: &[&str] = &[
    "scanning the hot seat",
    "inviting the contestants",
    "weighing the weights",
    "counting the KV cache",
    "phoning a friend",
    "audience poll",
    "computerji, lock kiya jaye",
];

pub fn wait_for_it(fast: bool) {
    if !color_enabled() {
        return;
    }
    let pb = ProgressBar::new(STAGES.len() as u64);
    pb.set_style(
        ProgressStyle::with_template("  {spinner:.214} {msg:<28.bold} {bar:30.214/238} {percent:>3}%")
            .unwrap()
            .progress_chars("━━╌")
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    pb.enable_steady_tick(Duration::from_millis(80));
    for stage in STAGES {
        pb.set_message(*stage);
        std::thread::sleep(Duration::from_millis(if fast { 0 } else { 380 }));
        pb.inc(1);
    }
    std::thread::sleep(Duration::from_millis(if fast { 0 } else { 250 }));
    pb.finish_and_clear();
}

// ---------------------------------------------------------------- panels

fn panel(title: &str, subtitle: Option<&str>, body: &[Styled], border: &str, double: bool) {
    let tw = term_width();
    let inner = tw.saturating_sub(2);
    let (tl, tr, bl, br, h, v) = if double {
        ("╔", "╗", "╚", "╝", "═", "║")
    } else {
        ("╭", "╮", "╰", "╯", "─", "│")
    };
    // A rule with an optional centred label, e.g. ──── HOT SEAT ────
    let cap = |left_corner: &str, text: Option<&str>, style: &str, right_corner: &str| -> String {
        let mut s = Styled::new().push(left_corner, border);
        match text {
            Some(t) => {
                let label = format!(" {t} ");
                let lw = label.width();
                let left = inner.saturating_sub(lw) / 2;
                let right = inner.saturating_sub(lw + left);
                s = s.push(h.repeat(left), border).push(label, style).push(h.repeat(right), border);
            }
            None => s = s.push(h.repeat(inner), border),
        }
        s.push(right_corner, border).render()
    };
    let b = |s: &str| Styled::new().push(s, border).render();
    out(&cap(tl, Some(title), &style2(BOLD, border), tr));
    for line in body {
        let padded = line.clone().pad_right(inner);
        out(&format!("{}{}{}", b(v), padded.render(), b(v)));
    }
    out(&cap(bl, subtitle, DIM, br));
}

pub fn hot_seat(hw: &Hardware) {
    let bw = format!(
        "{:.0} GB/s{}",
        hw.bandwidth_gbps,
        if hw.bandwidth_estimated { "  (guess)" } else { "" }
    );
    let left = hw.gpu_name.clone().unwrap_or_else(|| hw.chip.clone());
    let cores = match hw.gpu_cores {
        Some(c) => format!("{c} gpu cores"),
        None => hw.os.clone(),
    };
    let rows: [(&str, String, &str, String); 3] = [
        ("chip", left, "backend", hw.backend.clone()),
        ("memory", hw.memory_label(), "bandwidth", bw),
        ("usable", format!("{:.1} GiB for models", hw.usable_gib), "", cores),
    ];
    let lw = rows.iter().map(|r| r.1.width()).max().unwrap_or(0);
    let body: Vec<Styled> = rows
        .iter()
        .map(|(k1, v1, k2, v2)| {
            Styled::new()
                .push("  ", "")
                .push(format!("{k1:>7}"), DIM)
                .push("   ", "")
                .push(format!("{v1:<lw$}"), BOLD)
                .push("   ", "")
                .push(format!("{k2:>9}"), DIM)
                .push("   ", "")
                .push(v2.clone(), BOLD)
        })
        .collect();
    panel("HOT SEAT", None, &body, ACCENT, false);
}

// ---------------------------------------------------------------- leaderboard

fn memory_bar(hw: &Hardware, weight: f64, kv: f64, fits: bool, width: usize) -> Styled {
    let total = hw.usable_gib;
    if !fits {
        return Styled::new()
            .push("█".repeat(width), RED)
            .push(format!(" {:.0}G", weight + kv), RED);
    }
    let w = ((weight / total * width as f64).round() as usize).min(width);
    let k = ((kv / total * width as f64).round() as usize).min(width - w);
    Styled::new()
        .push("█".repeat(w), GREEN)
        .push("█".repeat(k), AMBER)
        .push("░".repeat(width - w - k), DIM)
}

fn tok_style(tok_s: f64) -> String {
    if is_fast(tok_s) {
        style2(BOLD, GREEN)
    } else if is_usable(tok_s) {
        AMBER.to_string()
    } else {
        RED.to_string()
    }
}

fn ctx(n: u32) -> String {
    if n >= 1024 {
        format!("{}k", n / 1024)
    } else {
        n.to_string()
    }
}

fn verdict_style(v: Verdict) -> String {
    match v {
        Verdict::Daudega => style2(BOLD, GREEN),
        Verdict::Chalega => AMBER.to_string(),
        Verdict::Sutti => RED.to_string(),
    }
}

pub fn leaderboard(hw: &Hardware, ranked: &[Ranked]) {
    let tw = term_width();
    let compact = tw < 100;
    let bar_w = if tw >= 110 { 24 } else if compact { 9 } else { 14 };
    let gap = if compact { "  " } else { "   " };
    let target = ranked.first().map(|r| r.estimate.target_context).unwrap_or(0);
    out(&Styled::new()
        .push(format!("  budgeting for {} tokens of context   (--context to change)", with_commas(target)), DIM)
        .render());
    out("");

    let header: Vec<Styled> = vec![
        Styled::new().push("#", &style2(BOLD, ACCENT)),
        Styled::new().push("contestant", &style2(BOLD, ACCENT)),
        Styled::new().push("quant", &style2(BOLD, ACCENT)),
        Styled::new().push("memory", &style2(BOLD, ACCENT)),
        Styled::new().push("weights", GREEN).push(" ", "").push("kv", AMBER).push(" ", "").push("free", DIM),
        Styled::new().push("tok/s", &style2(BOLD, ACCENT)),
        Styled::new().push("ctx", &style2(BOLD, ACCENT)),
        Styled::new().push("verdict", &style2(BOLD, ACCENT)),
    ];
    let right_align = [true, false, false, true, false, true, true, false];

    let rows: Vec<Vec<Styled>> = ranked
        .iter()
        .map(|r| {
            let e = &r.estimate;
            let dead = r.verdict == Verdict::Sutti;
            let dim_if_dead = |s: &str| if dead { DIM.to_string() } else { s.to_string() };
            let mut name = Styled::new().push(e.model.name, &dim_if_dead(BOLD));
            if e.model.is_moe() {
                name = name.push("  moe", &style2(ITALIC, DIM));
            }
            vec![
                Styled::new().push(r.rank.to_string(), DIM),
                name,
                Styled::new().push(e.quant.label(), &dim_if_dead("")),
                Styled::new().push(format!("{:.1} GiB", e.total_gib), &dim_if_dead("")),
                memory_bar(hw, e.weight_gib, e.kv_gib, e.fits, bar_w),
                if dead {
                    Styled::new().push("-", RED)
                } else {
                    Styled::new().push(format!("{:.0}", e.tok_s), &tok_style(e.tok_s))
                },
                Styled::new().push(if dead { "-".to_string() } else { ctx(e.max_context) }, DIM),
                Styled::new().push(r.verdict.label(), &verdict_style(r.verdict)),
            ]
        })
        .collect();

    // columns dropped in compact mode: memory (3) and ctx (6)
    let keep: Vec<usize> = (0..header.len()).filter(|i| !compact || (*i != 3 && *i != 6)).collect();
    let header: Vec<Styled> = keep.iter().map(|&i| header[i].clone()).collect();
    let right_align: Vec<bool> = keep.iter().map(|&i| right_align[i]).collect();
    let rows: Vec<Vec<Styled>> = rows.into_iter().map(|r| keep.iter().map(|&i| r[i].clone()).collect()).collect();

    let ncol = header.len();
    let mut widths: Vec<usize> = header.iter().map(|h| h.width()).collect();
    for row in &rows {
        for (i, c) in row.iter().enumerate() {
            widths[i] = widths[i].max(c.width());
        }
    }
    let render_row = |cells: &[Styled]| -> String {
        let mut s = String::from("  ");
        for (i, c) in cells.iter().enumerate() {
            let cell = if right_align[i] {
                c.clone().pad_left(widths[i])
            } else if i + 1 < ncol {
                c.clone().pad_right(widths[i])
            } else {
                c.clone()
            };
            s.push_str(&cell.render());
            if i + 1 < ncol {
                s.push_str(gap);
            }
        }
        s
    };
    out(&render_row(&header));
    let rule_w: usize = widths.iter().sum::<usize>() + gap.len() * (ncol - 1);
    out(&format!("  {}", Styled::new().push("─".repeat(rule_w), DIM).render()));
    for row in &rows {
        out(&render_row(row));
    }
}

fn with_commas(n: u32) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

// ---------------------------------------------------------------- final answer

pub fn run_commands(hw: &Hardware, r: &Ranked) -> Vec<(&'static str, String)> {
    let e = &r.estimate;
    let mut cmds = Vec::new();
    if hw.is_apple() {
        if let Some(tag) = e.quant.mlx_tag() {
            cmds.push((
                "fastest on apple silicon",
                format!("pip install mlx-lm && mlx_lm.chat --model mlx-community/{}-{}", e.model.mlx, tag),
            ));
        }
    }
    cmds.push(("easy mode", format!("ollama run {}-{}", e.model.ollama, e.quant.ollama_tag())));
    cmds
}

pub fn final_answer(hw: &Hardware, r: &Ranked) {
    let e = &r.estimate;
    let sep = "  ·  ";
    let head = Styled::new()
        .push("   ", "")
        .push(e.model.name, &style2(BOLD, GREEN))
        .push(sep, DIM)
        .push(e.quant.label(), BOLD)
        .push(sep, DIM)
        .push(hw.backend.as_str(), BOLD)
        .push(sep, DIM)
        .push(format!("~{:.0} tok/s", e.tok_s), &tok_style(e.tok_s))
        .push(sep, DIM)
        .push(format!("up to {} context", ctx(e.max_context)), BOLD);
    let sub = Styled::new().push("   ", "").push(
        format!("{}.  {:.1} GiB of your {:.1} GiB.", e.model.tagline, e.total_gib, hw.usable_gib),
        &style2(ITALIC, DIM),
    );
    let mut body = vec![Styled::new(), head, sub, Styled::new()];
    let cmds = run_commands(hw, r);
    for (i, (label, cmd)) in cmds.iter().enumerate() {
        body.push(Styled::new().push("   ", "").push(*label, DIM));
        body.push(Styled::new().push("     ", "").push(cmd.clone(), &style2(BOLD, ACCENT)));
        if i + 1 < cmds.len() {
            body.push(Styled::new());
        }
    }
    body.push(Styled::new());
    panel("FINAL ANSWER", Some(r.verdict.label()), &body, ACCENT, true);
}

pub fn nobody_wins() {
    let body = vec![
        Styled::new(),
        Styled::new().push("   ghare jake sutti babu. nothing fits. try --context 2048.", RED),
        Styled::new(),
    ];
    panel("FINAL ANSWER", None, &body, RED, true);
}
