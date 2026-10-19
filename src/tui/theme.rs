//! Colours and small drawing helpers. Truecolor when the terminal says so, 256-colour otherwise
//! (macOS Terminal.app mangles RGB).

use crate::show::{lerp, INDIA_GREEN, SAFFRON, WHITE};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use std::sync::OnceLock;

fn truecolor() -> bool {
    static T: OnceLock<bool> = OnceLock::new();
    *T.get_or_init(|| {
        let ct = std::env::var("COLORTERM")
            .unwrap_or_default()
            .to_lowercase();
        ct.contains("truecolor")
            || ct.contains("24bit")
            || std::env::var("TERM_PROGRAM")
                .map(|t| t != "Apple_Terminal")
                .unwrap_or(false)
    })
}

fn c(rgb: (u8, u8, u8), idx: u8) -> Color {
    if truecolor() {
        Color::Rgb(rgb.0, rgb.1, rgb.2)
    } else {
        Color::Indexed(idx)
    }
}

pub fn accent() -> Color {
    c(SAFFRON, 215)
}
pub fn green() -> Color {
    c((34, 197, 94), 41)
}
pub fn amber() -> Color {
    c((245, 158, 11), 214)
}
pub fn red() -> Color {
    c((239, 68, 68), 203)
}
pub fn blue() -> Color {
    c((96, 165, 250), 75)
}
pub fn dim() -> Color {
    c((128, 128, 128), 244)
}
pub fn faint() -> Color {
    c((70, 70, 70), 238)
}
pub fn text() -> Color {
    c((229, 229, 229), 254)
}
pub fn sel_bg() -> Color {
    c((48, 40, 30), 236)
}

pub fn s(fg: Color) -> Style {
    Style::default().fg(fg)
}
pub fn bold(fg: Color) -> Style {
    Style::default().fg(fg).add_modifier(Modifier::BOLD)
}
pub fn italic(fg: Color) -> Style {
    Style::default().fg(fg).add_modifier(Modifier::ITALIC)
}

/// Saffron -> white -> green, one span per character.
pub fn tiranga(text: &str) -> Vec<Span<'static>> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len().saturating_sub(1).max(1) as f64;
    chars
        .iter()
        .enumerate()
        .map(|(i, ch)| {
            let t = i as f64 / n;
            let rgb = if t < 0.5 {
                lerp(SAFFRON, WHITE, t * 2.0)
            } else {
                lerp(WHITE, INDIA_GREEN, (t - 0.5) * 2.0)
            };
            let col = if truecolor() {
                Color::Rgb(rgb.0, rgb.1, rgb.2)
            } else if t < 0.34 {
                Color::Indexed(215)
            } else if t < 0.67 {
                Color::Indexed(255)
            } else {
                Color::Indexed(34)
            };
            Span::styled(
                ch.to_string(),
                Style::default().fg(col).add_modifier(Modifier::BOLD),
            )
        })
        .collect()
}

pub const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// A key hint like `[enter] launch`.
pub fn key<'a>(k: &'a str, what: &'a str) -> Vec<Span<'a>> {
    vec![
        Span::styled(
            format!(" {k} "),
            Style::default().fg(Color::Black).bg(accent()),
        ),
        Span::styled(format!(" {what}  "), s(dim())),
    ]
}

pub fn hints(pairs: &[(&'static str, &'static str)]) -> Line<'static> {
    Line::from(
        pairs
            .iter()
            .flat_map(|(k, w)| key(k, w))
            .collect::<Vec<_>>(),
    )
}

/// Weights | KV | free bar, as spans of `width` cells.
pub fn mem_bar(weights: f64, kv: f64, cap: f64, width: usize, fits: bool) -> Vec<Span<'static>> {
    if width == 0 {
        return vec![];
    }
    if !fits {
        let over = ((weights + kv) / cap.max(0.1) * 100.0).round();
        let mut label = format!(" {over:.0}% of usable ");
        if label.chars().count() + 2 > width {
            label = format!("✗{over:.0}%");
        }
        let n = width.saturating_sub(label.chars().count());
        return vec![
            Span::styled("█".repeat(n), s(red())),
            Span::styled(label, bold(red())),
        ];
    }
    let w = ((weights / cap * width as f64).round() as usize).min(width);
    let k = ((kv / cap * width as f64).ceil() as usize).min(width - w);
    vec![
        Span::styled("█".repeat(w), s(green())),
        Span::styled("█".repeat(k), s(amber())),
        Span::styled("░".repeat(width - w - k), s(faint())),
    ]
}

pub fn ctx_label(n: u32) -> String {
    if n >= 1024 {
        format!("{}k", n / 1024)
    } else {
        n.to_string()
    }
}

/// Base64 for OSC 52 clipboard writes.
pub fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for ch in data.chunks(3) {
        let b = [ch[0], *ch.get(1).unwrap_or(&0), *ch.get(2).unwrap_or(&0)];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        for i in 0..4 {
            if i <= ch.len() {
                out.push(T[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// Put text on the clipboard of whatever terminal we are in, even over SSH.
pub fn copy_to_clipboard(text: &str) {
    use std::io::Write;
    let mut o = std::io::stdout();
    let _ = write!(o, "\x1b]52;c;{}\x07", base64(text.as_bytes()));
    let _ = o.flush();
}

#[cfg(test)]
mod tests {
    #[test]
    fn base64_matches_rfc() {
        assert_eq!(super::base64(b"f"), "Zg==");
        assert_eq!(super::base64(b"fo"), "Zm8=");
        assert_eq!(super::base64(b"foobar"), "Zm9vYmFy");
    }
}
