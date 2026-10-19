//! Drawing. One function per tab, plus modals. Nothing here changes state except hit boxes.

use super::app::*;
use super::theme::*;
use crate::budget;
use crate::engine::{EngineKind, Session};
use crate::net::human_secs;
use crate::state::human_bytes;
use crate::verdict::{self, Verdict};
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, BorderType, Borders, Cell, Clear, Gauge, List, ListItem, ListState, Paragraph, Row,
    Sparkline, Table, TableState, Wrap,
};
use ratatui::Frame;

fn block(title: &str) -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(s(prussian()))
        .title(Span::styled(format!(" {title} "), bold(text())))
}

fn block_hot(title: &str, color: Color) -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(s(color))
        .title(Span::styled(format!(" {title} "), bold(color)))
}

fn kv_line(k: &str, v: impl Into<String>, vstyle: Style, kw: usize) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!(" {k:>kw$}  "), s(dim())),
        Span::styled(v.into(), vstyle),
    ])
}

pub fn centered(area: Rect, w: u16, h: u16) -> Rect {
    let w = w.min(area.width.saturating_sub(2));
    let h = h.min(area.height.saturating_sub(2));
    Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    }
}

fn verdict_style(v: Verdict) -> Style {
    match v {
        Verdict::Daudega => bold(green()),
        Verdict::Chalega => s(amber()),
        Verdict::Sutti => s(red()),
    }
}

fn tok_color(t: f64) -> Color {
    if verdict::is_fast(t) {
        green()
    } else if verdict::is_usable(t) {
        amber()
    } else {
        red()
    }
}

fn spinner(app: &App) -> &'static str {
    SPINNER[(app.tick as usize) % SPINNER.len()]
}

// ---------------------------------------------------------------- frame

pub fn draw(f: &mut Frame, app: &mut App) {
    let area = f.area();
    f.render_widget(Block::default().style(Style::default()), area);
    let [header, body, footer] = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(5),
        Constraint::Length(2),
    ])
    .areas(area);
    draw_header(f, app, header);
    let body = if app.show_log {
        let [b, l] = Layout::vertical([Constraint::Min(5), Constraint::Length(12)]).areas(body);
        draw_log(f, app, l);
        b
    } else {
        body
    };
    match app.tab {
        Tab::Machine => draw_machine(f, app, body),
        Tab::Models => draw_models(f, app, body),
        Tab::Discover => draw_discover(f, app, body),
        Tab::Library => draw_library(f, app, body),
        Tab::Run => draw_run(f, app, body),
        Tab::Chat => draw_chat(f, app, body),
        Tab::Engines => draw_engines(f, app, body),
    }
    draw_footer(f, app, footer);
    match &app.modal {
        Some(Modal::Plan(p)) => draw_plan(f, app, p, area),
        Some(Modal::Confirm(c)) => draw_confirm(f, c, area),
        Some(Modal::Settings { row, editing }) => {
            draw_settings(f, app, *row, editing.as_deref(), area)
        }
        Some(Modal::Help) => draw_help(f, area),
        None => {}
    }
}

fn draw_header(f: &mut Frame, app: &mut App, area: Rect) {
    let [top, tabs, rule] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .areas(area);
    let mut l = vec![Span::raw(" ")];
    l.extend(wordmark("TOKENPATI"));
    if top.width >= 120 {
        l.push(Span::styled("  kaun banega tokenpati", italic(dim())));
    }
    let hw = &app.hw;
    let chip = hw.gpu_name.clone().unwrap_or_else(|| hw.chip.clone());
    let chip: String = chip.chars().take(34).collect();
    l.push(Span::styled("   │ ", s(faint())));
    l.push(Span::styled(chip, s(text())));
    l.push(Span::styled(format!(" · {}", hw.memory_label()), s(dim())));
    let live = &app.live;
    if live.ram_total_gib > 0.0 {
        let pct = live.ram_used_gib / live.ram_total_gib;
        let col = if pct > 0.9 {
            red()
        } else if pct > 0.75 {
            amber()
        } else {
            green()
        };
        l.push(Span::styled("  │ ram ", s(faint())));
        l.push(Span::styled(
            format!("{:.1}/{:.0}G", live.ram_used_gib, live.ram_total_gib),
            s(col),
        ));
        l.push(Span::styled(
            format!("  cpu {:>3.0}%", live.cpu_pct),
            s(dim()),
        ));
        if let (Some(u), Some(t)) = (live.vram_used_gib, live.vram_total_gib) {
            l.push(Span::styled(format!("  vram {u:.1}/{t:.0}G"), s(dim())));
        }
    }
    f.render_widget(Paragraph::new(Line::from(l)), top);

    // tabs, with live status markers
    let mut spans = vec![Span::raw(" ")];
    let mut x = tabs.x + 1;
    app.tab_hits.clear();
    for (i, t) in Tab::ALL.iter().enumerate() {
        let marker = match t {
            Tab::Run => match &app.job {
                Some(j) if j.running() => format!(" {}", spinner(app)),
                Some(j) if j.finished == Some(false) => " ✗".into(),
                Some(_) if app.session.is_some() => " ✓".into(),
                _ => String::new(),
            },
            Tab::Chat if app.session.is_some() => " ●".into(),
            Tab::Engines if app.installing.is_some() => format!(" {}", spinner(app)),
            _ => String::new(),
        };
        let label = format!(" {} {}{} ", i + 1, t.title(), marker);
        let w = unicode_width::UnicodeWidthStr::width(label.as_str()) as u16;
        app.tab_hits.push((x, x + w, *t));
        x += w + 1;
        let st = if *t == app.tab {
            Style::default()
                .fg(Color::Black)
                .bg(accent())
                .add_modifier(Modifier::BOLD)
        } else if marker.contains('✗') {
            s(red())
        } else if !marker.is_empty() {
            s(green())
        } else {
            s(dim())
        };
        spans.push(Span::styled(label, st));
        spans.push(Span::raw(" "));
    }
    if let Some(sess) = &app.session {
        spans.push(Span::styled("   ● ", bold(green())));
        spans.push(Span::styled(sess.name.clone(), bold(ice())));
        spans.push(Span::styled(
            format!(" on {}", sess.engine.name()),
            s(dim()),
        ));
    }
    f.render_widget(Paragraph::new(Line::from(spans)), tabs);
    f.render_widget(
        Paragraph::new(Span::styled("─".repeat(rule.width as usize), s(faint()))),
        rule,
    );
}

fn tab_hints(app: &App) -> Vec<(&'static str, &'static str)> {
    let mut v: Vec<(&'static str, &'static str)> = match app.tab {
        Tab::Machine => vec![
            ("enter", "launch the winner"),
            ("b", "measure bandwidth"),
            ("r", "re-detect"),
        ],
        Tab::Models if app.filtering => {
            vec![("type", "filter"), ("enter", "done"), ("esc", "clear")]
        }
        Tab::Models => vec![
            ("↑↓", "pick"),
            ("←→", "quant"),
            ("enter", "launch…"),
            ("+/-", "context"),
            ("/", "filter"),
            ("f", "fits only"),
        ],
        Tab::Discover => match app.discover.focus {
            DFocus::Search => vec![
                ("type", "search Hugging Face"),
                ("enter", "search"),
                ("esc", "results"),
            ],
            DFocus::Results => vec![("↑↓", "repo"), ("enter/→", "files"), ("/", "search")],
            DFocus::Files => vec![("↑↓", "file"), ("enter", "launch…"), ("←", "repos")],
        },
        Tab::Library => vec![
            ("↑↓", "pick"),
            ("enter", "launch…"),
            ("d", "delete"),
            ("x", "unload"),
            ("r", "refresh"),
        ],
        Tab::Run => match &app.job {
            Some(j) if j.running() => vec![("c", "cancel")],
            Some(j) if j.finished == Some(false) => vec![("r", "retry"), ("L", "full log")],
            Some(_) if app.session.is_some() => {
                vec![("enter", "chat"), ("y", "copy endpoint"), ("x", "unload")]
            }
            _ => vec![],
        },
        Tab::Chat if app.chat.focused => vec![
            ("enter", "send"),
            ("ctrl+j", "newline"),
            (
                "esc",
                if app.chat.generating.is_some() {
                    "stop"
                } else {
                    "unfocus"
                },
            ),
            ("/", "commands"),
        ],
        Tab::Chat => vec![
            ("i", "type"),
            ("↑↓", "scroll"),
            ("t", "thinking"),
            ("x", "unload"),
        ],
        Tab::Engines => vec![
            ("↑↓", "pick"),
            ("i", "install"),
            ("s", "start ollama"),
            ("←→", "llama.cpp build"),
            ("enter", "prefer"),
        ],
    };
    if !app.typing() {
        v.extend([
            ("1-7", "tabs"),
            (",", "settings"),
            ("?", "help"),
            ("q", "quit"),
        ]);
    } else {
        v.push(("tab", "next tab"));
    }
    v
}

fn draw_footer(f: &mut Frame, app: &App, area: Rect) {
    let [status, keys] =
        Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).areas(area);
    let line = match &app.toast {
        Some((t, _)) => Line::from(vec![
            Span::styled(" ◆ ", bold(accent())),
            Span::styled(t.clone(), s(text())),
        ]),
        None => Line::from(Span::styled(format!(" {}", tip(app)), italic(dim()))),
    };
    f.render_widget(Paragraph::new(line), status);
    f.render_widget(Paragraph::new(hints(&tab_hints(app))), keys);
}

fn tip(app: &App) -> String {
    match app.tab {
        Tab::Machine if app.hw.bandwidth_estimated => "bandwidth is a guess for this chip. press b to measure it and every prediction gets sharper.".into(),
        Tab::Machine => format!("every model is budgeted against {:.1} GiB usable and {:.0} GB/s.", app.hw.usable_gib, app.hw.bandwidth_gbps),
        Tab::Models => format!("context budget {} tokens: more context = more KV cache memory. +/- to change.", app.state.settings.context),
        Tab::Discover => "every model on Hugging Face. tokenpati works out fit and speed before anything downloads.".into(),
        Tab::Library => format!("models live in {}", crate::state::models_dir().display()),
        Tab::Run => "every step is shown as it happens. downloads resume if interrupted.".into(),
        Tab::Chat => "the endpoint also works from any OpenAI client while you chat here.".into(),
        Tab::Engines => "tokenpati drives these engines; it does not replace them.".into(),
    }
}

fn draw_log(f: &mut Frame, app: &App, area: Rect) {
    let n = area.height.saturating_sub(2) as usize;
    let lines: Vec<Line> = app
        .log
        .iter()
        .rev()
        .take(n)
        .rev()
        .map(|l| Line::from(Span::styled(l.clone(), s(dim()))))
        .collect();
    f.render_widget(Paragraph::new(lines).block(block("log  (L to hide)")), area);
}

// ---------------------------------------------------------------- 1 machine

fn draw_machine(f: &mut Frame, app: &App, area: Rect) {
    let banner_h = if area.height >= 34 && area.width >= 80 {
        8
    } else {
        0
    };
    let [banner, rest] =
        Layout::vertical([Constraint::Length(banner_h), Constraint::Min(0)]).areas(area);
    if banner_h > 0 {
        let art = crate::show::fig("TOKENPATI", crate::show::ANSI_REGULAR);
        let mut lines: Vec<Line> = vec![Line::raw("")];
        lines.extend(art.iter().map(|l| Line::from(wordmark(l))));
        lines.push(Line::from(Span::styled(
            "which local model will actually run on this thing, and then run it",
            italic(dim()),
        )));
        f.render_widget(Paragraph::new(lines).alignment(Alignment::Center), banner);
    }
    let narrow = rest.width < 100;
    let live_h = if app.hw.has_discrete_gpu() { 10 } else { 8 };
    let (seat, how, live, bets) = if narrow {
        // one column: what you have, then what to run. live stats are in the header already.
        let [seat, bets] =
            Layout::vertical([Constraint::Length(10), Constraint::Min(6)]).areas(rest);
        (seat, Rect::default(), Rect::default(), bets)
    } else {
        let [left, right] =
            Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .areas(rest);
        let [seat, how] =
            Layout::vertical([Constraint::Length(12), Constraint::Min(6)]).areas(left);
        let [live, bets] =
            Layout::vertical([Constraint::Length(live_h), Constraint::Min(6)]).areas(right);
        (seat, how, live, bets)
    };

    // hot seat
    let hw = &app.hw;
    let mut lines = vec![kv_line("chip", hw.chip.clone(), bold(text()), 10)];
    if let Some(g) = &hw.gpu_name {
        lines.push(kv_line("gpu", g.clone(), bold(text()), 10));
    }
    if let Some(c) = hw.gpu_cores {
        lines.push(kv_line("gpu cores", c.to_string(), s(text()), 10));
    }
    lines.push(kv_line("memory", hw.memory_label(), bold(text()), 10));
    lines.push(Line::from(vec![
        Span::styled(format!(" {:>10}  ", "usable"), s(dim())),
        Span::styled(format!("{:.1} GiB", hw.usable_gib), bold(green())),
        Span::styled(
            if hw.is_apple() {
                "  (macOS lets the GPU wire ~75%)".to_string()
            } else if hw.has_discrete_gpu() {
                format!("  VRAM, +{:.0} GiB RAM for offload", hw.offload_gib)
            } else {
                "  (RAM minus 4 GiB for the OS)".to_string()
            },
            s(dim()),
        ),
    ]));
    let bw_note = if app.benchmarking {
        format!("  measuring {}", spinner(app))
    } else if hw.bandwidth_estimated {
        "  guess. press b to measure".into()
    } else if app.state.measured_bandwidth.is_some() && !hw.is_apple() && !hw.has_discrete_gpu() {
        "  measured".into()
    } else {
        "  spec sheet".into()
    };
    lines.push(Line::from(vec![
        Span::styled(format!(" {:>10}  ", "bandwidth"), s(dim())),
        Span::styled(
            format!("{:.0} GB/s", hw.bandwidth_gbps),
            bold(if hw.bandwidth_estimated {
                amber()
            } else {
                text()
            }),
        ),
        Span::styled(bw_note, s(dim())),
    ]));
    if hw.has_discrete_gpu() {
        lines.push(kv_line(
            "ram bw",
            format!("{:.0} GB/s (for offloaded layers)", hw.ram_bandwidth_gbps),
            s(text()),
            10,
        ));
    }
    lines.push(kv_line("backend", hw.backend.clone(), bold(accent()), 10));
    let cal = match app.state.calibration() {
        Some(c) => format!(
            "{:.0}%  calibrated from {} run(s), real/predicted {:.2}",
            hw.efficiency * 100.0,
            app.state.measurements.len(),
            c
        ),
        None => format!(
            "{:.0}%  (default until you run a model)",
            hw.efficiency * 100.0
        ),
    };
    lines.push(kv_line("efficiency", cal, s(text()), 10));
    lines.push(kv_line(
        "threads",
        hw.cpu_threads.to_string(),
        s(text()),
        10,
    ));
    f.render_widget(
        Paragraph::new(lines).block(block_hot("HOT SEAT", accent())),
        seat,
    );

    // how it decides, with this machine's numbers
    let how_lines = vec![
        Line::from(vec![Span::styled(" ① measure ", bold(accent())), Span::styled("memory, bandwidth, GPU. that is all speed depends on.", s(text()))]),
        Line::from(vec![Span::styled(" ② budget  ", bold(accent())), Span::styled(format!("weights + KV cache must fit in {:.1} GiB.", hw.usable_gib), s(text()))]),
        Line::from(vec![Span::styled(" ③ predict ", bold(accent())), Span::styled(format!("tok/s ≈ {:.0} GB/s ÷ bytes per token × {:.0}%.", hw.bandwidth_gbps, hw.efficiency * 100.0), s(text()))]),
        Line::from(vec![Span::styled(" ④ run     ", bold(accent())), Span::styled("download → load → measure real tok/s → calibrate ③.", s(text()))]),
        Line::raw(""),
        Line::from(Span::styled(
            " decode reads every active weight once per token, so memory bandwidth, not compute, sets the speed. MoE models only read their active experts, which is why a 30B-A3B can outrun an 8B.",
            italic(dim()),
        )),
    ];
    if !narrow {
        f.render_widget(
            Paragraph::new(how_lines)
                .wrap(Wrap { trim: false })
                .block(block("HOW IT DECIDES")),
            how,
        );
        draw_live(f, app, live);
    }
    draw_bets(f, app, bets);
}

fn draw_live(f: &mut Frame, app: &App, live: Rect) {
    let hw = &app.hw;
    let b = block("LIVE");
    let inner = b.inner(live);
    f.render_widget(b, live);
    let mut rows = vec![
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(2),
    ];
    if hw.has_discrete_gpu() {
        rows.extend([Constraint::Length(1), Constraint::Length(1)]);
    }
    rows.push(Constraint::Min(0));
    let parts = Layout::vertical(rows).split(inner);
    let l = &app.live;
    let ram_ratio = if l.ram_total_gib > 0.0 {
        (l.ram_used_gib / l.ram_total_gib).clamp(0.0, 1.0)
    } else {
        0.0
    };
    f.render_widget(
        Gauge::default()
            .gauge_style(s(if ram_ratio > 0.9 { red() } else { green() }))
            .ratio(ram_ratio)
            .label(format!(
                "ram {:.1} / {:.1} GiB",
                l.ram_used_gib, l.ram_total_gib
            )),
        parts[0],
    );
    f.render_widget(
        Gauge::default()
            .gauge_style(s(accent()))
            .ratio((l.cpu_pct as f64 / 100.0).clamp(0.0, 1.0))
            .label(format!("cpu {:.0}%", l.cpu_pct)),
        parts[1],
    );
    let model_line = match &app.session {
        Some(sess) => {
            let rss = sess.pid().and(app.live.model_rss_gib);
            Line::from(vec![
                Span::styled(" ● ", bold(green())),
                Span::styled(sess.name.clone(), bold(text())),
                Span::styled(
                    format!(
                        "  {}  {}",
                        sess.engine.name(),
                        rss.map(|r| format!("{r:.1} GiB resident"))
                            .unwrap_or_default()
                    ),
                    s(dim()),
                ),
            ])
        }
        None => Line::from(Span::styled(" no model loaded", s(dim()))),
    };
    f.render_widget(Paragraph::new(model_line), parts[2]);
    let hist: Vec<u64> = app.cpu_hist.iter().copied().collect();
    let start = hist.len().saturating_sub(parts[3].width as usize);
    f.render_widget(
        Sparkline::default()
            .data(&hist[start..])
            .max(100)
            .style(s(accent())),
        parts[3],
    );
    if hw.has_discrete_gpu() {
        let (u, t) = (
            l.vram_used_gib.unwrap_or(0.0),
            l.vram_total_gib.or(hw.vram_gib).unwrap_or(1.0),
        );
        f.render_widget(
            Gauge::default()
                .gauge_style(s(blue()))
                .ratio((u / t).clamp(0.0, 1.0))
                .label(format!("vram {u:.1} / {t:.0} GiB")),
            parts[4],
        );
        let g = l.gpu_util_pct.unwrap_or(0.0);
        f.render_widget(
            Gauge::default()
                .gauge_style(s(blue()))
                .ratio((g as f64 / 100.0).clamp(0.0, 1.0))
                .label(format!("gpu {g:.0}%")),
            parts[5],
        );
    }
}

fn draw_bets(f: &mut Frame, app: &App, bets: Rect) {
    let mut lines = vec![];
    let picks: Vec<_> = app
        .ranked
        .iter()
        .filter(|r| r.verdict != Verdict::Sutti)
        .take(5)
        .collect();
    if picks.is_empty() {
        lines.push(Line::from(Span::styled(
            " nothing fits at this context. press 2 and lower it with -",
            s(red()),
        )));
    }
    for (i, r) in picks.iter().enumerate() {
        let e = &r.estimate;
        let q = app.quant_for(r);
        let crown = if i == 0 { "♛ " } else { "  " };
        lines.push(Line::from(vec![
            Span::styled(format!(" {crown}"), bold(accent())),
            Span::styled(
                format!("{:<22}", e.model.name),
                if i == 0 { bold(ice()) } else { bold(text()) },
            ),
            Span::styled(format!("{:<7}", q.label()), s(dim())),
            Span::styled(format!("~{:>3.0} tok/s ", e.tok_s), s(tok_color(e.tok_s))),
            Span::styled(format!(" {:>5.1} GiB  ", e.total_gib), s(dim())),
            Span::styled(r.verdict.label(), verdict_style(r.verdict)),
        ]));
        if i == 0 {
            lines.push(Line::from(Span::styled(
                format!("     {}", e.model.tagline),
                italic(dim()),
            )));
        }
    }
    lines.push(Line::raw(""));
    lines.push(Line::from(vec![
        Span::styled(" press ", s(dim())),
        Span::styled(" enter ", Style::default().fg(Color::Black).bg(accent())),
        Span::styled(" to launch the winner, or ", s(dim())),
        Span::styled(" 2 ", Style::default().fg(Color::Black).bg(accent())),
        Span::styled(" to see every contestant", s(dim())),
    ]));
    f.render_widget(
        Paragraph::new(lines).block(block_hot("YOUR BEST BETS", ice())),
        bets,
    );
}

// ---------------------------------------------------------------- 2 models

fn draw_models(f: &mut Frame, app: &App, area: Rect) {
    let wide = area.width >= 120;
    let [left, right] = if wide {
        Layout::horizontal([Constraint::Percentage(58), Constraint::Percentage(42)]).areas(area)
    } else {
        Layout::vertical([Constraint::Percentage(55), Constraint::Percentage(45)]).areas(area)
    };
    let models = app.visible_models();
    let bar_w = if left.width > 100 { 14 } else { 8 };
    let header = Row::new(
        [
            "#", "model", "quant", "memory", "", "tok/s", "real", "ctx", "verdict",
        ]
        .map(|h| Cell::from(Span::styled(h, bold(accent())))),
    );
    let rows: Vec<Row> = models
        .iter()
        .map(|r| {
            let q = app.quant_for(r);
            let e = app.estimate_for(&r.estimate.model, q, app.state.settings.context);
            let dead = !e.fits;
            let name_style = if dead { s(dim()) } else { bold(text()) };
            let mut name = vec![Span::styled(e.model.name.to_string(), name_style)];
            if e.model.is_moe() {
                name.push(Span::styled(" moe", italic(dim())));
            }
            if e.offload {
                name.push(Span::styled(" ⇅", s(blue())));
            }
            let real = app
                .measured(e.model.name, q)
                .map(|m| format!("{:.0}", m.tok_s))
                .unwrap_or_default();
            let changed = q != r.estimate.quant;
            Row::new(vec![
                Cell::from(Span::styled(r.rank.to_string(), s(dim()))),
                Cell::from(Line::from(name)),
                Cell::from(Span::styled(
                    format!("{}{}", q.label(), if changed { "*" } else { "" }),
                    if changed { s(accent()) } else { s(text()) },
                )),
                Cell::from(Span::styled(
                    format!("{:>5.1}G", e.total_gib),
                    s(if dead { red() } else { text() }),
                )),
                Cell::from(Line::from(mem_bar(
                    e.weight_gib,
                    e.kv_gib,
                    app.hw.capacity_gib(),
                    bar_w,
                    e.fits,
                ))),
                Cell::from(Span::styled(
                    if dead {
                        "-".into()
                    } else {
                        format!("{:>4.0}", e.tok_s)
                    },
                    s(if dead { red() } else { tok_color(e.tok_s) }),
                )),
                Cell::from(Span::styled(real, bold(blue()))),
                Cell::from(Span::styled(
                    if dead {
                        "-".into()
                    } else {
                        ctx_label(e.max_context)
                    },
                    s(dim()),
                )),
                Cell::from(Span::styled(
                    r.verdict.short(e.fits),
                    verdict_style(r.verdict),
                )),
            ])
        })
        .collect();
    let mut ts = TableState::default().with_selected(Some(app.models_sel));
    let title = if app.filtering || !app.models_filter.is_empty() {
        format!(
            "CONTESTANTS  filter: {}{}",
            app.models_filter,
            if app.filtering { "▌" } else { "" }
        )
    } else if app.only_runnable {
        "CONTESTANTS  (only what fits)".to_string()
    } else {
        format!(
            "CONTESTANTS  ·  {} ctx",
            ctx_label(app.state.settings.context)
        )
    };
    let t = Table::new(
        rows,
        [
            Constraint::Length(3),
            Constraint::Min(18),
            Constraint::Length(7),
            Constraint::Length(7),
            Constraint::Length(bar_w as u16),
            Constraint::Length(5),
            Constraint::Length(4),
            Constraint::Length(5),
            Constraint::Length(10),
        ],
    )
    .header(header)
    .block(block_hot(&title, accent()))
    .row_highlight_style(Style::default().bg(sel_bg()).add_modifier(Modifier::BOLD))
    .highlight_symbol("▶ ");
    f.render_stateful_widget(t, left, &mut ts);

    let Some(r) = models.get(app.models_sel) else {
        f.render_widget(
            Paragraph::new(" nothing matches").block(block("DETAIL")),
            right,
        );
        return;
    };
    draw_model_detail(f, app, r, right);
}

fn draw_model_detail(f: &mut Frame, app: &App, r: &crate::verdict::Ranked, area: Rect) {
    let m = r.estimate.model;
    let ctx = app.state.settings.context;
    let q = app.quant_for(r);
    let e = app.estimate_for(&m, q, ctx);
    let hw = &app.hw;
    let mut lines = vec![
        Line::from(vec![
            Span::styled(format!(" {}", m.name), bold(accent())),
            Span::styled(format!("   {}", m.tagline), italic(dim())),
        ]),
        Line::from(Span::styled(
            format!(
                " {:.1}B params{} · {} layers · {} kv heads × {} · up to {} ctx",
                m.params_b,
                if m.is_moe() {
                    format!(" ({:.1}B active)", m.active_b)
                } else {
                    String::new()
                },
                m.layers,
                m.kv_heads,
                m.head_dim,
                ctx_label(m.max_context)
            ),
            s(dim()),
        )),
        Line::raw(""),
    ];
    let bar_w = area.width.saturating_sub(6) as usize;
    lines.push(Line::from(vec![Span::styled(
        format!(" memory at {} context, {}", ctx_label(ctx), q.tag()),
        bold(text()),
    )]));
    let mut bar = vec![Span::raw(" ")];
    bar.extend(mem_bar(
        e.weight_gib,
        e.kv_gib,
        hw.capacity_gib(),
        bar_w,
        e.fits,
    ));
    lines.push(Line::from(bar));
    lines.push(Line::from(vec![
        Span::styled(" ■ ", s(green())),
        Span::styled(format!("weights {:.1}  ", e.weight_gib), s(text())),
        Span::styled("■ ", s(amber())),
        Span::styled(format!("kv cache {:.2}  ", e.kv_gib), s(text())),
        Span::styled("░ ", s(faint())),
        Span::styled(
            format!(
                "free {:.1} of {:.1} GiB",
                (hw.capacity_gib() - e.total_gib).max(0.0),
                hw.capacity_gib()
            ),
            s(dim()),
        ),
    ]));
    lines.push(Line::raw(""));

    // why
    let why = if !e.fits {
        (
            red(),
            format!(
                "needs {:.1} GiB, you have {:.1}. try a smaller quant (←) or less context (-).",
                e.total_gib,
                hw.capacity_gib()
            ),
        )
    } else if e.offload {
        (blue(), format!("does not fit in VRAM: {} of {} layers on the GPU, the rest in RAM. works, but slower.", e.gpu_layers.unwrap_or(0), m.layers))
    } else if verdict::is_fast(e.tok_s) {
        (
            green(),
            format!(
                "~{:.0} tok/s is faster than you read. comfortable.",
                e.tok_s
            ),
        )
    } else if verdict::is_usable(e.tok_s) {
        (
            amber(),
            format!("~{:.0} tok/s: usable, you will see it type.", e.tok_s),
        )
    } else {
        (
            red(),
            format!(
                "fits, but ~{:.0} tok/s is slower than reading. you can still run it.",
                e.tok_s
            ),
        )
    };
    lines.push(Line::from(vec![
        Span::styled(
            format!(" {} ", r.verdict.label()),
            verdict_style(r.verdict).add_modifier(Modifier::REVERSED),
        ),
        Span::raw(" "),
        Span::styled(why.1, s(why.0)),
    ]));
    lines.push(Line::raw(""));

    // quant matrix
    lines.push(Line::from(Span::styled(
        format!(
            " {:<9}{:>8}{:>9}{:>8}{:>8}   {}",
            "quant", "size", "tok/s", "real", "ctx", ""
        ),
        bold(accent()),
    )));
    for row in budget::quant_matrix(hw, &m, ctx) {
        let selected = row.quant == q;
        let best = row.quant == r.estimate.quant;
        let real = app
            .measured(m.name, row.quant)
            .map(|x| format!("{:.0}", x.tok_s))
            .unwrap_or_else(|| "·".into());
        let st = if selected {
            Style::default()
                .bg(sel_bg())
                .fg(text())
                .add_modifier(Modifier::BOLD)
        } else {
            s(if row.fits { text() } else { dim() })
        };
        lines.push(Line::from(vec![
            Span::styled(if selected { " ▶ " } else { "   " }, bold(accent())),
            Span::styled(format!("{:<6}{:>8.1}G", row.quant.tag(), row.total_gib), st),
            Span::styled(
                format!(
                    "{:>9}",
                    if row.fits {
                        format!("{:.0}", row.tok_s)
                    } else {
                        "-".into()
                    }
                ),
                if row.fits {
                    s(tok_color(row.tok_s))
                } else {
                    s(red())
                },
            ),
            Span::styled(format!("{real:>8}"), s(blue())),
            Span::styled(
                format!(
                    "{:>8}",
                    if row.fits {
                        ctx_label(row.max_context)
                    } else {
                        "-".into()
                    }
                ),
                s(dim()),
            ),
            Span::styled(if best { "  ← pick" } else { "" }, italic(dim())),
        ]));
    }
    lines.push(Line::raw(""));
    let eng = |k: EngineKind| -> Span<'static> {
        let ok = app.engine_info(k).map(|e| e.usable()).unwrap_or(false);
        let possible = k.supported_here() && !(k == EngineKind::Mlx && q.mlx_tag().is_none());
        Span::styled(
            format!(
                " {} {} ",
                if !possible {
                    "–"
                } else if ok {
                    "✓"
                } else {
                    "↓"
                },
                k.name()
            ),
            if !possible {
                s(faint())
            } else if ok {
                s(green())
            } else {
                s(amber())
            },
        )
    };
    lines.push(Line::from(vec![
        Span::styled(" engines ", s(dim())),
        eng(EngineKind::Ollama),
        eng(EngineKind::LlamaCpp),
        eng(EngineKind::Mlx),
    ]));
    lines.push(Line::from(Span::styled(
        "         ✓ ready   ↓ installed on demand   – not here",
        s(faint()),
    )));
    lines.push(Line::raw(""));
    lines.push(Line::from(vec![
        Span::styled(" enter ", Style::default().fg(Color::Black).bg(accent())),
        Span::styled(
            " review the launch plan: engine, quant, context, GPU layers",
            s(dim()),
        ),
    ]));
    f.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(block("DETAIL")),
        area,
    );
}

// ---------------------------------------------------------------- 3 discover

fn draw_discover(f: &mut Frame, app: &App, area: Rect) {
    let d = &app.discover;
    let [search, rest] = Layout::vertical([Constraint::Length(3), Constraint::Min(0)]).areas(area);
    let focus_c = |x: DFocus| if d.focus == x { accent() } else { faint() };
    let q = if d.query.is_empty() && d.focus != DFocus::Search {
        Span::styled(
            "search Hugging Face: qwen3 coder, gemma 3, llama 3.2, deepseek…",
            italic(dim()),
        )
    } else {
        Span::styled(
            format!(
                "{}{}",
                d.query,
                if d.focus == DFocus::Search { "▌" } else { "" }
            ),
            bold(text()),
        )
    };
    let status = if d.searching {
        format!("  {} searching", spinner(app))
    } else {
        String::new()
    };
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" 🔍 ", s(accent())),
            q,
            Span::styled(status, s(dim())),
        ]))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(s(focus_c(DFocus::Search)))
                .title(Span::styled(" SEARCH  (/) ", bold(focus_c(DFocus::Search)))),
        ),
        search,
    );
    let [list, files] =
        Layout::horizontal([Constraint::Percentage(40), Constraint::Percentage(60)]).areas(rest);

    let items: Vec<ListItem> = d
        .results
        .iter()
        .map(|h| {
            let (owner, name) = h.id.split_once('/').unwrap_or(("", &h.id));
            ListItem::new(vec![
                Line::from(vec![Span::styled(name.to_string(), bold(text()))]),
                Line::from(vec![
                    Span::styled(format!("  {owner}"), s(dim())),
                    Span::styled(format!("   ↓ {}", short_num(h.downloads)), s(green())),
                    Span::styled(format!("  ♥ {}", short_num(h.likes)), s(red())),
                ]),
            ])
        })
        .collect();
    let title = if d.query.is_empty() {
        "POPULAR ON HUGGING FACE".to_string()
    } else {
        format!("RESULTS  {}", d.results.len())
    };
    let lb = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(s(focus_c(DFocus::Results)))
        .title(Span::styled(
            format!(" {title} "),
            bold(focus_c(DFocus::Results)),
        ));
    if let Some(e) = &d.error {
        f.render_widget(
            Paragraph::new(Span::styled(format!(" {e}"), s(red())))
                .wrap(Wrap { trim: false })
                .block(lb),
            list,
        );
    } else if d.results.is_empty() {
        let msg = if d.searching {
            format!(" {} asking huggingface.co…", spinner(app))
        } else {
            " type a search above and press enter".into()
        };
        f.render_widget(Paragraph::new(Span::styled(msg, s(dim()))).block(lb), list);
    } else {
        let mut st = ListState::default().with_selected(Some(d.sel));
        f.render_stateful_widget(
            List::new(items)
                .block(lb)
                .highlight_style(Style::default().bg(sel_bg()))
                .highlight_symbol("▶ "),
            list,
            &mut st,
        );
    }

    let fb = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(s(focus_c(DFocus::Files)))
        .title(Span::styled(
            " SIZES  (smaller = faster, bigger = smarter) ",
            bold(focus_c(DFocus::Files)),
        ));
    let Some(hit) = d.results.get(d.sel) else {
        f.render_widget(Paragraph::new("").block(fb), files);
        return;
    };
    match d.files.get(&hit.id) {
        None => f.render_widget(
            Paragraph::new(Span::styled(" press enter to list files", s(dim()))).block(fb),
            files,
        ),
        Some(Err(e)) if e.starts_with("loading") => f.render_widget(
            Paragraph::new(Span::styled(
                format!(" {} listing {}…", spinner(app), hit.id),
                s(dim()),
            ))
            .block(fb),
            files,
        ),
        Some(Err(e)) => f.render_widget(
            Paragraph::new(Span::styled(format!(" {e}"), s(red())))
                .wrap(Wrap { trim: false })
                .block(fb),
            files,
        ),
        Some(Ok(choices)) if choices.is_empty() => f.render_widget(
            Paragraph::new(Span::styled(" nothing tokenpati can run in this repo. try another result", s(dim()))).block(fb),
            files,
        ),
        Some(Ok(choices)) => {
            let inner = fb.inner(files);
            f.render_widget(fb, files);
            let [tbl, info] =
                Layout::vertical([Constraint::Min(5), Constraint::Length(8)]).areas(inner);
            let cap = app.hw.capacity_gib();
            // every quant in a repo shares one architecture, so one header covers them all
            let repo_meta = app.repo_meta(choices);
            let ctx = app.state.settings.context as f64;
            let rows: Vec<Row> = choices
                .iter()
                .map(|c| {
                    let gib = c.size() as f64 / (1u64 << 30) as f64;
                    let meta = repo_meta;
                    let kv = meta
                        .and_then(|m| m.kv_bytes_per_token())
                        .map(|b| b * ctx / (1u64 << 30) as f64)
                        .unwrap_or(gib * 0.12 * ctx / 8192.0);
                    let active = gib * meta.map(|m| m.active_fraction()).unwrap_or(1.0);
                    let total = gib * 1.03 + kv;
                    let fits = total <= cap;
                    let tok = budget::predicted_tok_s(
                        &app.hw,
                        active,
                        kv * budget::SPEED_CONTEXT as f64 / ctx,
                    );
                    let badge = if !fits {
                        Span::styled("too big", s(red()))
                    } else if total > app.hw.usable_gib {
                        Span::styled("offload", s(blue()))
                    } else if verdict::is_fast(tok) {
                        Span::styled("fast", bold(green()))
                    } else if verdict::is_usable(tok) {
                        Span::styled("ok", s(amber()))
                    } else {
                        Span::styled("slow", s(red()))
                    };
                    Row::new(vec![
                        Cell::from(Span::styled(
                            bits_label(c.quant.as_deref()),
                            bold(text()),
                        )),
                        Cell::from(Span::styled(human_bytes(c.size()), s(text()))),
                        Cell::from(Span::styled(
                            if fits {
                                format!("~{tok:.0}")
                            } else {
                                "-".into()
                            },
                            s(if fits { tok_color(tok) } else { red() }),
                        )),
                        Cell::from(badge),
                        Cell::from(Span::styled(c.label(), s(dim()))),
                    ])
                })
                .collect();
            let mut st =
                TableState::default().with_selected(Some(d.file_sel.min(choices.len() - 1)));
            f.render_stateful_widget(
                Table::new(
                    rows,
                    [
                        Constraint::Length(8),
                        Constraint::Length(9),
                        Constraint::Length(6),
                        Constraint::Length(8),
                        Constraint::Min(10),
                    ],
                )
                .header(Row::new(
                    ["quality", "size", "tok/s", "fit", "file"]
                        .map(|h| Cell::from(Span::styled(h, bold(accent())))),
                ))
                .row_highlight_style(Style::default().bg(sel_bg()))
                .highlight_symbol("▶ "),
                tbl,
                &mut st,
            );
            // selected file detail from its GGUF header
            let mut lines = vec![Line::from(Span::styled(
                "─".repeat(info.width as usize),
                s(faint()),
            ))];
            if let Some(c) = choices.get(d.file_sel) {
                let header: Option<Result<&crate::gguf::Meta, &String>> =
                    match (repo_meta, d.metas.get(&c.url(0))) {
                        (Some(m), _) => Some(Ok(m)),
                        (None, Some(r)) => Some(r.as_ref()),
                        (None, None) => None,
                    };
                match header {
                    Some(Ok(m)) => {
                        lines.push(Line::from(vec![
                            Span::styled(" header ", s(dim())),
                            Span::styled(
                                format!("{}  ", m.name.clone().unwrap_or_else(|| m.arch.clone())),
                                bold(text()),
                            ),
                            Span::styled(
                                format!(
                                    "arch {} · {} layers · ctx {}",
                                    m.arch,
                                    m.layers.unwrap_or(0),
                                    ctx_label(m.context_length.unwrap_or(0))
                                ),
                                s(dim()),
                            ),
                        ]));
                        if let Some(b) = m.kv_bytes_per_token() {
                            lines.push(Line::from(Span::styled(
                                format!(
                                    " kv cache {:.0} KiB/token → {:.2} GiB at {} ctx{}",
                                    b / 1024.0,
                                    b * app.state.settings.context as f64 / (1u64 << 30) as f64,
                                    ctx_label(app.state.settings.context),
                                    if m.active_fraction() < 1.0 {
                                        format!(
                                            " · MoE {}/{} experts active",
                                            m.experts_used.unwrap_or(0),
                                            m.experts.unwrap_or(0)
                                        )
                                    } else {
                                        String::new()
                                    }
                                ),
                                s(dim()),
                            )));
                        }
                    }
                    Some(Err(e)) if e.starts_with("reading") => {
                        lines.push(Line::from(Span::styled(
                            format!(
                                " {} reading the model's shape for exact numbers…",
                                spinner(app)
                            ),
                            s(dim()),
                        )))
                    }
                    Some(Err(e)) => lines.push(Line::from(Span::styled(
                        format!(" header unavailable ({e}); numbers are estimates"),
                        s(dim()),
                    ))),
                    None => {}
                }
                lines.push(Line::from(vec![
                    Span::styled(" enter ", Style::default().fg(Color::Black).bg(accent())),
                    Span::styled(
                        format!(
                            " launch plan for {}  ({})",
                            c.label(),
                            human_bytes(c.size())
                        ),
                        s(dim()),
                    ),
                ]));
            }
            f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), info);
        }
    }
}

/// "Q4_K_M" -> "4-bit". People think in bits; the exact tag is in the file column.
fn bits_label(q: Option<&str>) -> String {
    let Some(q) = q else { return "?".into() };
    let digits: String = q.trim_start_matches('I').trim_start_matches(['Q', 'B', 'F', 'M', 'X']).chars().take_while(|c| c.is_ascii_digit()).collect();
    match (q, digits.as_str()) {
        (_, "") => q.to_string(),
        (_, d) => format!("{d}-bit"),
    }
}

fn short_num(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

// ---------------------------------------------------------------- 4 library

fn draw_library(f: &mut Frame, app: &App, area: Rect) {
    let [now, list] = Layout::vertical([Constraint::Length(4), Constraint::Min(5)]).areas(area);
    let now_lines = match &app.session {
        Some(sess) => session_lines(sess),
        None => vec![Line::from(Span::styled(
            " nothing loaded. pick a model below and press enter.",
            s(dim()),
        ))],
    };
    f.render_widget(
        Paragraph::new(now_lines).block(block_hot("LOADED NOW", ice())),
        now,
    );

    let total: u64 = app.library.iter().filter(|m| m.ours).map(|m| m.size).sum();
    let title = format!(
        "ON DISK  {} models · {}{}",
        app.library.len(),
        human_bytes(total),
        if app.library_loading {
            format!("  {}", spinner(app))
        } else {
            String::new()
        }
    );
    if app.library.is_empty() {
        let msg = vec![
            Line::raw(""),
            Line::from(Span::styled(" no models downloaded yet.", bold(text()))),
            Line::from(Span::styled(
                " launching one from Models (2) or Search (3) downloads it; it shows up here.",
                s(dim()),
            )),
            Line::from(Span::styled(
                " tokenpati also finds ollama's models and LM Studio's models.",
                s(dim()),
            )),
        ];
        f.render_widget(Paragraph::new(msg).block(block_hot(&title, accent())), list);
        return;
    }
    let rows: Vec<Row> = app
        .library
        .iter()
        .map(|m| {
            let live = app
                .session
                .as_ref()
                .map(|s| s.model_id == m.id || s.model_id.ends_with(&m.name))
                .unwrap_or(false)
                || m.loaded;
            Row::new(vec![
                Cell::from(Span::styled(if live { "●" } else { " " }, bold(green()))),
                Cell::from(Span::styled(m.name.clone(), bold(text()))),
                Cell::from(Span::styled(m.engine.name(), s(accent()))),
                Cell::from(Span::styled(
                    format!("{:>9}", human_bytes(m.size)),
                    s(text()),
                )),
                Cell::from(Span::styled(m.detail.clone(), s(dim()))),
            ])
        })
        .collect();
    let mut st = TableState::default().with_selected(Some(app.library_sel));
    f.render_stateful_widget(
        Table::new(
            rows,
            [
                Constraint::Length(2),
                Constraint::Percentage(40),
                Constraint::Length(10),
                Constraint::Length(10),
                Constraint::Min(10),
            ],
        )
        .header(Row::new(
            ["", "model", "engine", "size", "detail"]
                .map(|h| Cell::from(Span::styled(h, bold(accent())))),
        ))
        .block(block_hot(&title, accent()))
        .row_highlight_style(Style::default().bg(sel_bg()))
        .highlight_symbol("▶ "),
        list,
        &mut st,
    );
}

fn session_lines(sess: &Session) -> Vec<Line<'static>> {
    let up = sess.started.elapsed().as_secs();
    vec![
        Line::from(vec![
            Span::styled(" ● ", bold(green())),
            Span::styled(sess.name.clone(), bold(text())),
            Span::styled(
                format!(
                    "   {} · {} ctx · up {}",
                    sess.engine.name(),
                    ctx_label(sess.context),
                    human_secs(up)
                ),
                s(dim()),
            ),
        ]),
        Line::from(vec![
            Span::styled("   endpoint ", s(dim())),
            Span::styled(sess.openai_url(), bold(accent())),
            Span::styled("   (OpenAI-compatible)", s(dim())),
        ]),
    ]
}

// ---------------------------------------------------------------- 5 run

fn draw_run(f: &mut Frame, app: &App, area: Rect) {
    let Some(job) = &app.job else {
        let lines = vec![
            Line::raw(""),
            Line::from(Span::styled("nothing launched yet", bold(text()))),
            Line::raw(""),
            Line::from(Span::styled(
                "pick a model on Models (2), Search (3) or Library (4) and press enter.",
                s(dim()),
            )),
            Line::from(Span::styled(
                "you will review a launch plan first; nothing downloads until you say go.",
                s(dim()),
            )),
            Line::raw(""),
            Line::from(Span::styled(
                "then every step shows up here:  engine → weights → load → warm-up → ready",
                s(accent()),
            )),
        ];
        f.render_widget(
            Paragraph::new(lines)
                .alignment(Alignment::Center)
                .block(block("LAUNCH")),
            area,
        );
        return;
    };
    let [summary, stepper, steps] = Layout::vertical([
        Constraint::Length(4),
        Constraint::Length(2),
        Constraint::Min(5),
    ])
    .areas(area);

    let p = &job.plan;
    let mut sum = vec![
        Line::from(vec![
            Span::styled(format!(" {}", p.name), bold(accent())),
            Span::styled(
                format!("   via {}   {} ctx", p.engine.name(), ctx_label(p.context)),
                s(text()),
            ),
            Span::styled(
                p.gpu_layers
                    .map(|n| format!("   {n} GPU layers"))
                    .unwrap_or_default(),
                s(text()),
            ),
            Span::styled(
                format!("   {}", human_secs(job.started.elapsed().as_secs())),
                s(dim()),
            ),
        ]),
        Line::from(vec![
            Span::styled(" weights  ", s(dim())),
            Span::styled(p.artifact(), s(text())),
            Span::styled(
                p.download_bytes
                    .map(|b| format!("   ~{}", human_bytes(b)))
                    .unwrap_or_default(),
                s(dim()),
            ),
        ]),
    ];
    let mut pred = vec![Span::styled(" predicted ", s(dim()))];
    if let Some(g) = p.predicted_gib {
        pred.push(Span::styled(format!("{g:.1} GiB"), s(text())));
    }
    if let Some(t) = p.predicted_tok_s {
        pred.push(Span::styled(format!("  ~{t:.0} tok/s"), s(tok_color(t))));
    }
    sum.push(Line::from(pred));
    let cancelled = job.cancel.load(std::sync::atomic::Ordering::Relaxed);
    let border = match job.finished {
        None => accent(),
        Some(true) => green(),
        Some(false) if cancelled => amber(),
        Some(false) => red(),
    };
    let title = match job.finished {
        None => format!("LAUNCHING {}", spinner(app)),
        Some(true) => "LAUNCHED".into(),
        Some(false) if cancelled => "CANCELLED · r to resume".into(),
        Some(false) => "LAUNCH FAILED".into(),
    };
    f.render_widget(
        Paragraph::new(sum).block(block_hot(&title, border)),
        summary,
    );

    // ① Engine ━━ ② Weights ━━ …
    let mut st = vec![Span::raw(" ")];
    for (i, step) in job.steps.iter().enumerate() {
        let (icon, col) = step_icon(app, step.status);
        st.push(Span::styled(
            format!("{icon} {}", step.title),
            if step.status == StepStatus::Running {
                bold(col)
            } else {
                s(col)
            },
        ));
        if i + 1 < job.steps.len() {
            let done = matches!(step.status, StepStatus::Done | StepStatus::Skipped);
            st.push(Span::styled(
                " ━━━━ ",
                s(if done { green() } else { faint() }),
            ));
        }
    }
    f.render_widget(Paragraph::new(Line::from(st)), stepper);

    // one block per step; the active one gets the room
    let active = job.current();
    let heights: Vec<Constraint> = job
        .steps
        .iter()
        .enumerate()
        .map(|(i, s)| match s.status {
            StepStatus::Running => Constraint::Min(6),
            StepStatus::Failed => Constraint::Min(6),
            StepStatus::Pending => Constraint::Length(3),
            _ if i == 4 => Constraint::Min(6),
            _ if i == active => Constraint::Min(5),
            _ => Constraint::Length(3 + s.lines.len().min(1) as u16),
        })
        .collect();
    let areas = Layout::vertical(heights).split(steps);
    for (i, step) in job.steps.iter().enumerate() {
        draw_step(f, app, i, step, areas[i]);
    }
}

fn step_icon(app: &App, st: StepStatus) -> (&'static str, Color) {
    match st {
        StepStatus::Pending => ("○", faint()),
        StepStatus::Running => (spinner(app), accent()),
        StepStatus::Done => ("✓", green()),
        StepStatus::Skipped => ("↷", blue()),
        StepStatus::Failed => ("✗", red()),
    }
}

fn step_help(i: usize) -> &'static str {
    match i {
        0 => "the program that runs the model: installed, started, or found",
        1 => "the model file(s): skipped if already on disk, resumable if not",
        2 => "server starts and reads the weights into RAM / VRAM",
        3 => "a first reply, timed: the real tok/s on your machine",
        4 => "an OpenAI-compatible endpoint you can chat with",
        _ => "",
    }
}

fn draw_step(f: &mut Frame, app: &App, i: usize, step: &StepState, area: Rect) {
    let (icon, col) = step_icon(app, step.status);
    let took = match (step.started, step.ended) {
        (Some(a), Some(b)) => format!(" {:.1}s ", (b - a).as_secs_f64()),
        (Some(a), None) => format!(" {:.0}s ", a.elapsed().as_secs_f64()),
        _ => String::new(),
    };
    let b = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(s(if step.status == StepStatus::Pending {
            faint()
        } else {
            col
        }))
        .title(Line::from(vec![Span::styled(
            format!(" {icon} {}. {} ", i + 1, step.title),
            bold(col),
        )]))
        .title(Line::from(Span::styled(took, s(dim()))).alignment(Alignment::Right));
    let inner = b.inner(area);
    f.render_widget(b, area);
    let mut lines: Vec<Line> = vec![];
    match step.status {
        StepStatus::Pending => lines.push(Line::from(Span::styled(
            format!(" {}", step_help(i)),
            s(faint()),
        ))),
        _ => {
            if !step.detail.is_empty() {
                let st = match step.status {
                    StepStatus::Failed => bold(red()),
                    StepStatus::Running => s(text()),
                    _ => s(text()),
                };
                lines.push(Line::from(Span::styled(format!(" {}", step.detail), st)));
            }
        }
    }
    let mut rest = inner;
    if let Some((p, label)) = &step.progress {
        let [txt, gauge, after] = Layout::vertical([
            Constraint::Length(lines.len() as u16),
            Constraint::Length(1),
            Constraint::Min(0),
        ])
        .areas(inner);
        f.render_widget(Paragraph::new(lines.clone()), txt);
        lines.clear();
        if p.total > 0 {
            let ratio = (p.done as f64 / p.total as f64).clamp(0.0, 1.0);
            let eta = p
                .eta_secs()
                .map(|s| format!("  eta {}", human_secs(s)))
                .unwrap_or_default();
            let rate = if p.rate > 0.0 {
                format!("  {}/s", human_bytes(p.rate as u64))
            } else {
                String::new()
            };
            f.render_widget(
                Gauge::default()
                    .gauge_style(Style::default().fg(prussian()).bg(faint()))
                    .ratio(ratio)
                    .label(Span::styled(
                        format!(
                            "{} / {}  {:.0}%{rate}{eta}  {label}",
                            human_bytes(p.done),
                            human_bytes(p.total),
                            ratio * 100.0
                        ),
                        bold(Color::White),
                    )),
                Rect {
                    x: gauge.x + 1,
                    width: gauge.width.saturating_sub(2),
                    ..gauge
                },
            );
        } else {
            f.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled(format!(" {} ", spinner(app)), bold(accent())),
                    Span::styled(label.clone(), s(text())),
                ])),
                gauge,
            );
        }
        rest = after;
    }
    // tail of this step's log lines
    let room = rest.height as usize;
    let show_lines =
        matches!(step.status, StepStatus::Running | StepStatus::Failed) || i == 4 || i == 3;
    if show_lines && room > lines.len() {
        let tail = room - lines.len();
        for l in step
            .lines
            .iter()
            .skip(step.lines.len().saturating_sub(tail))
        {
            let st = if l.starts_with('$') {
                s(accent())
            } else if l.to_lowercase().contains("error") {
                s(red())
            } else {
                s(dim())
            };
            let mut text_l = format!("   {l}");
            if text_l.chars().count() > rest.width as usize {
                text_l = text_l
                    .chars()
                    .take(rest.width as usize - 1)
                    .collect::<String>()
                    + "…";
            }
            lines.push(Line::from(Span::styled(text_l, st)));
        }
    }
    if !show_lines && matches!(step.status, StepStatus::Done | StepStatus::Skipped) {
        if let Some(l) = step.lines.back() {
            lines.push(Line::from(Span::styled(format!("   {l}"), s(dim()))));
        }
    }
    if step.status == StepStatus::Failed {
        lines.push(Line::from(vec![
            Span::styled("   r ", Style::default().fg(Color::Black).bg(accent())),
            Span::styled(" retry   ", s(dim())),
            Span::styled(" L ", Style::default().fg(Color::Black).bg(accent())),
            Span::styled(
                " full log   or pick another engine/quant in the plan",
                s(dim()),
            ),
        ]));
    }
    if i == 4 && step.status == StepStatus::Done {
        if let Some(sess) = &app.session {
            lines.push(Line::from(vec![
                Span::styled("   try it: ", s(dim())),
                Span::styled(
                    format!("curl {}/chat/completions -d '{{\"model\":\"{}\",\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]}}'", sess.openai_url(), short_model(&sess.model_id)),
                    s(text()),
                ),
            ]));
            lines.push(Line::from(vec![
                Span::styled("   enter ", Style::default().fg(Color::Black).bg(green())),
                Span::styled(" chat here   ", s(dim())),
                Span::styled(" y ", Style::default().fg(Color::Black).bg(accent())),
                Span::styled(" copy endpoint   ", s(dim())),
                Span::styled(" x ", Style::default().fg(Color::Black).bg(accent())),
                Span::styled(" unload", s(dim())),
            ]));
        }
    }
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), rest);
}

/// Word-wrap `text` to `width` columns, every line starting with `indent`. Long words are hard-broken.
fn wrap_indented(text: &str, width: usize, indent: &str, style: Style) -> Vec<Line<'static>> {
    use unicode_width::UnicodeWidthStr;
    let room = width.saturating_sub(indent.width()).max(8);
    let mut out = vec![];
    for para in text.split('\n') {
        let mut line = String::new();
        let mut w = 0usize;
        // keep the paragraph's own leading spaces (code, lists)
        let lead: String = para.chars().take_while(|c| *c == ' ').collect();
        let words = para.trim_start_matches(' ').split(' ');
        line.push_str(&lead);
        w += lead.width();
        for word in words {
            let ww = word.width();
            if w > lead.width() && w + 1 + ww > room {
                out.push(Line::from(Span::styled(format!("{indent}{line}"), style)));
                line = lead.clone();
                w = lead.width();
            }
            if w > lead.width() {
                line.push(' ');
                w += 1;
            }
            if ww > room {
                for ch in word.chars() {
                    let cw = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
                    if w + cw > room {
                        out.push(Line::from(Span::styled(format!("{indent}{line}"), style)));
                        line.clear();
                        w = 0;
                    }
                    line.push(ch);
                    w += cw;
                }
            } else {
                line.push_str(word);
                w += ww;
            }
        }
        out.push(Line::from(Span::styled(format!("{indent}{line}"), style)));
    }
    out
}

/// A path is noise in a curl example (llama.cpp ignores the field); a tag must stay whole.
fn short_model(id: &str) -> String {
    if id.starts_with('/') || id.contains('\\') {
        id.rsplit(['/', '\\']).next().unwrap_or(id).to_string()
    } else {
        id.to_string()
    }
}

// ---------------------------------------------------------------- 6 chat

fn draw_chat(f: &mut Frame, app: &App, area: Rect) {
    let input_lines = app.chat.input.split('\n').count().clamp(1, 6) as u16;
    let [top, msgs, input] = Layout::vertical([
        Constraint::Length(2),
        Constraint::Min(3),
        Constraint::Length(input_lines + 2),
    ])
    .areas(area);

    // status strip
    let Some(sess) = &app.session else {
        let lines = vec![
            Line::raw(""),
            Line::from(Span::styled("no model loaded", bold(text()))),
            Line::raw(""),
            Line::from(Span::styled(
                "press 1 and enter to launch the winner, or pick one on Models (2).",
                s(dim()),
            )),
        ];
        f.render_widget(Paragraph::new(lines).alignment(Alignment::Center), msgs);
        return;
    };
    let used = app.chat_tokens();
    let ratio = (used as f64 / sess.context.max(1) as f64).min(1.0);
    let last = app.chat.entries.iter().rev().find_map(|e| e.stats.clone());
    let mut l1 = vec![
        Span::styled(" ● ", bold(green())),
        Span::styled(sess.name.clone(), bold(text())),
        Span::styled(
            format!("  {}  {}", sess.engine.name(), sess.openai_url()),
            s(dim()),
        ),
    ];
    if let Some(st) = &last {
        l1.push(Span::styled(
            format!("   {:.1} tok/s", st.tok_s),
            bold(tok_color(st.tok_s)),
        ));
        l1.push(Span::styled(format!("  ttft {} ms", st.ttft_ms), s(dim())));
        if let Some(p) = sess.predicted_tok_s {
            l1.push(Span::styled(format!("  (predicted {p:.0})"), s(dim())));
        }
    }
    let bar_w = 20usize;
    let filled = (ratio * bar_w as f64).round() as usize;
    let ccol = if ratio > 0.9 {
        red()
    } else if ratio > 0.7 {
        amber()
    } else {
        green()
    };
    let l2 = Line::from(vec![
        Span::styled(" context ", s(dim())),
        Span::styled("█".repeat(filled), s(ccol)),
        Span::styled("░".repeat(bar_w - filled), s(faint())),
        Span::styled(format!(" ~{} / {} tokens", used, sess.context), s(dim())),
        Span::styled(
            format!("   temp {:.1}", app.state.settings.temperature),
            s(dim()),
        ),
        Span::styled(
            if ratio > 0.9 {
                "   nearly full: /clear to start over"
            } else {
                ""
            },
            s(red()),
        ),
    ]);
    f.render_widget(Paragraph::new(vec![Line::from(l1), l2]), top);

    // transcript
    let wid = msgs.width as usize;
    let mut lines: Vec<Line> = vec![];
    if app.chat.entries.is_empty() {
        lines.push(Line::raw(""));
        lines.push(Line::from(Span::styled(
            format!("  {} is loaded. say something.", sess.name),
            s(dim()),
        )));
        lines.push(Line::from(Span::styled(
            "  /system <prompt>  /temp 0.3  /think  /copy  /save  /clear",
            s(faint()),
        )));
    }
    let n = app.chat.entries.len();
    for (i, e) in app.chat.entries.iter().enumerate() {
        lines.push(Line::raw(""));
        if e.role == "user" {
            lines.push(Line::from(Span::styled(" you ›", bold(accent()))));
            lines.extend(wrap_indented(&e.content, wid, "   ", s(text())));
            continue;
        }
        lines.push(Line::from(Span::styled(
            format!(" {} ›", sess.name),
            bold(green()),
        )));
        if !e.thinking.trim().is_empty() {
            if app.state.settings.show_thinking {
                lines.push(Line::from(Span::styled("   ┌ thinking", italic(dim()))));
                lines.extend(wrap_indented(
                    e.thinking.trim(),
                    wid,
                    "   │ ",
                    italic(dim()),
                ));
                lines.push(Line::from(Span::styled("   └", italic(dim()))));
            } else {
                lines.push(Line::from(Span::styled(
                    format!(
                        "   (thought for {} chars · /think to show)",
                        e.thinking.len()
                    ),
                    italic(faint()),
                )));
            }
        }
        let generating = i + 1 == n && app.chat.generating.is_some();
        if e.content.is_empty() && generating {
            let what = if e.thinking.is_empty() {
                "reading your prompt"
            } else {
                "thinking"
            };
            lines.push(Line::from(Span::styled(
                format!("   {} {what}…", spinner(app)),
                s(dim()),
            )));
        }
        if !e.content.is_empty() {
            let mut body = wrap_indented(e.content.trim_start_matches('\n'), wid, "   ", s(text()));
            if generating {
                if let Some(last) = body.last_mut() {
                    last.spans.push(Span::styled("▌", bold(accent())));
                }
            }
            lines.extend(body);
        }
        if let Some(err) = &e.error {
            lines.extend(wrap_indented(&format!("✗ {err}"), wid, "   ", s(red())));
        }
        if let Some(st) = &e.stats {
            let mut t = format!(
                "   {:.1} tok/s · {} tokens · first token {} ms",
                st.tok_s, st.tokens, st.ttft_ms
            );
            if let (Some(pn), Some(pt)) = (st.prompt_tokens, st.prompt_tok_s) {
                t.push_str(&format!(" · read {pn} prompt tokens at {pt:.0}/s"));
            }
            if !st.exact {
                t.push_str(" · timed by tokenpati");
            }
            lines.push(Line::from(Span::styled(t, s(faint()))));
        }
    }
    let total = lines.len() as u16;
    let para = Paragraph::new(lines);
    let h = msgs.height;
    let max_scroll = total.saturating_sub(h);
    let up = app.chat.scroll.min(max_scroll);
    f.render_widget(para.scroll((max_scroll - up, 0)), msgs);
    if up > 0 {
        let tag = format!(" ↑ {up} lines up · PgDn to follow ");
        let w = tag.chars().count() as u16;
        f.render_widget(
            Paragraph::new(Span::styled(
                tag,
                Style::default().fg(Color::Black).bg(amber()),
            )),
            Rect {
                x: msgs.right().saturating_sub(w + 1),
                y: msgs.y,
                width: w,
                height: 1,
            },
        );
    }

    // input
    let focus = app.chat.focused;
    let title = if app.chat.generating.is_some() {
        format!(" {} generating · esc to stop ", spinner(app))
    } else if focus {
        " message · enter send · ctrl+j newline · ↑ history ".to_string()
    } else {
        " press i to type ".to_string()
    };
    let ib = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(s(if focus { accent() } else { faint() }))
        .title(Span::styled(title, s(if focus { accent() } else { dim() })));
    let inner = ib.inner(input);
    f.render_widget(ib, input);
    let text_lines: Vec<&str> = app.chat.input.split('\n').collect();
    let start = text_lines.len().saturating_sub(inner.height as usize);
    let shown: Vec<Line> = text_lines[start..]
        .iter()
        .map(|l| Line::from(Span::styled(l.to_string(), s(text()))))
        .collect();
    // horizontal scroll so the cursor stays visible on long lines
    let before = &app.chat.input[..app.chat.cursor];
    let row = before.matches('\n').count();
    let col = unicode_width::UnicodeWidthStr::width(before.rsplit('\n').next().unwrap_or(""));
    let hscroll = col.saturating_sub(inner.width.saturating_sub(2) as usize) as u16;
    f.render_widget(Paragraph::new(shown).scroll((0, hscroll)), inner);
    if focus && app.modal.is_none() && row >= start {
        f.set_cursor_position((
            inner.x + (col as u16 - hscroll).min(inner.width.saturating_sub(1)),
            inner.y + (row - start) as u16,
        ));
    }
}

// ---------------------------------------------------------------- 7 engines

fn draw_engines(f: &mut Frame, app: &App, area: Rect) {
    let [cards, guide] = if area.width >= 110 {
        Layout::horizontal([Constraint::Percentage(62), Constraint::Percentage(38)]).areas(area)
    } else {
        Layout::vertical([Constraint::Min(10), Constraint::Length(0)]).areas(area)
    };
    if app.engines.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled(
                format!(" {} detecting engines…", spinner(app)),
                s(dim()),
            ))
            .block(block("SETUP")),
            cards,
        );
        return;
    }
    let areas = Layout::vertical(
        app.engines
            .iter()
            .map(|e| Constraint::Length(if e.kind == EngineKind::LlamaCpp { 9 } else { 7 }))
            .chain([Constraint::Min(0)]),
    )
    .split(cards);
    for (i, e) in app.engines.iter().enumerate() {
        let selected = i == app.engines_sel;
        let preferred = app.state.settings.engine == e.kind.name();
        let (badge, bcol) = if !e.kind.supported_here() {
            ("not for this machine", faint())
        } else if e.running {
            ("● running", green())
        } else if e.installed {
            ("✓ installed", green())
        } else {
            ("✗ not installed", amber())
        };
        let b = Block::default()
            .borders(Borders::ALL)
            .border_type(if selected {
                BorderType::Thick
            } else {
                BorderType::Rounded
            })
            .border_style(s(if selected { accent() } else { faint() }))
            .title(Line::from(vec![
                Span::styled(
                    format!(" {} ", e.kind.name()),
                    bold(if selected { accent() } else { text() }),
                ),
                Span::styled(format!(" {badge} "), bold(bcol)),
                Span::styled(if preferred { " ★ preferred " } else { "" }, bold(accent())),
            ]));
        let mut lines = vec![Line::from(Span::styled(
            format!(" {}", e.kind.blurb()),
            italic(dim()),
        ))];
        if let Some(v) = &e.version {
            lines.push(kv_line("version", v.clone(), s(text()), 7));
        }
        if let Some(p) = &e.path {
            lines.push(kv_line("path", p.display().to_string(), s(text()), 7));
        }
        lines.push(kv_line("status", e.detail.clone(), s(text()), 7));
        if e.kind == EngineKind::LlamaCpp {
            let vs = crate::engine::llamacpp::Variant::choices();
            let mut sp = vec![Span::styled(format!(" {:>7}  ", "build"), s(dim()))];
            for v in vs {
                sp.push(Span::styled(
                    format!(" {} ", v.name()),
                    if v == app.variant {
                        Style::default().fg(Color::Black).bg(accent())
                    } else {
                        s(dim())
                    },
                ));
            }
            sp.push(Span::styled("  ←/→", s(faint())));
            lines.push(Line::from(sp));
            if let Some((p, log)) = &app.installing {
                let t = match p {
                    Some(p) if p.total > 0 => format!(
                        " {} installing {:.0}%  {}",
                        spinner(app),
                        p.done as f64 / p.total as f64 * 100.0,
                        log.last().cloned().unwrap_or_default()
                    ),
                    _ => format!(
                        " {} {}",
                        spinner(app),
                        log.last().cloned().unwrap_or_else(|| "starting".into())
                    ),
                };
                lines.push(Line::from(Span::styled(t, bold(accent()))));
            }
        }
        f.render_widget(Paragraph::new(lines).block(b), areas[i]);
    }
    if guide.width > 0 {
        let g = vec![
            Line::from(Span::styled(" which one?", bold(accent()))),
            Line::raw(""),
            Line::from(Span::styled(" new to this: ollama", bold(text()))),
            Line::from(Span::styled(
                "  one install, then any model is a pull away.",
                s(dim()),
            )),
            Line::raw(""),
            Line::from(Span::styled(" want control: llama.cpp", bold(text()))),
            Line::from(Span::styled(
                "  any model on Hugging Face, exact quant, choose",
                s(dim()),
            )),
            Line::from(Span::styled(
                "  how many layers go on the GPU. tokenpati installs",
                s(dim()),
            )),
            Line::from(Span::styled("  it for you with i, no sudo.", s(dim()))),
            Line::raw(""),
            Line::from(Span::styled(" on a Mac: mlx", bold(text()))),
            Line::from(Span::styled(
                "  usually the fastest on Apple silicon.",
                s(dim()),
            )),
            Line::raw(""),
            Line::from(Span::styled(
                " all three end up as an OpenAI-compatible server,",
                s(dim()),
            )),
            Line::from(Span::styled(
                " so your editor, scripts and agents can use the",
                s(dim()),
            )),
            Line::from(Span::styled(" model tokenpati loaded.", s(dim()))),
            Line::raw(""),
            Line::from(Span::styled(
                " enter marks an engine as preferred; plans default to it.",
                s(faint()),
            )),
        ];
        f.render_widget(
            Paragraph::new(g)
                .wrap(Wrap { trim: false })
                .block(block("GUIDE")),
            guide,
        );
    }
}

// ---------------------------------------------------------------- modals

fn draw_plan(f: &mut Frame, app: &App, p: &PlanModal, area: Rect) {
    let nums = app.plan_numbers(p);
    let rows = app.plan_rows(p);
    let r = centered(
        area,
        92,
        14 + rows.len() as u16
            + app.session.is_some() as u16
            + nums.download.is_some() as u16
            + nums.tok_s.is_some() as u16,
    );
    f.render_widget(Clear, r);
    let engine = p.engines[p.engine_idx];
    let b = block_hot(&format!("LAUNCH PLAN · {}", p.name), accent());
    let inner = b.inner(r);
    f.render_widget(b, r);
    let mut lines = vec![
        Line::from(Span::styled(
            " nothing happens until you press enter. ↑↓ pick a row, ←→ change it.",
            italic(dim()),
        )),
        Line::raw(""),
    ];
    for (i, row) in rows.iter().enumerate() {
        let sel = i == p.row.min(rows.len() - 1);
        let (label, value, note): (&str, String, String) = match row {
            PlanRow::Quant => {
                let q = p.quants[p.quant_idx];
                (
                    "quant",
                    q.tag().to_string(),
                    format!(
                        "{}  ·  {:.2} bytes per weight",
                        q.label(),
                        q.bytes_per_param()
                    ),
                )
            }
            PlanRow::Engine => {
                let info = app.engine_info(engine);
                let note = match info {
                    Some(e) if e.usable() => {
                        format!("ready  {}", e.version.clone().unwrap_or_default())
                    }
                    Some(_) if engine == EngineKind::LlamaCpp => format!(
                        "not installed yet: tokenpati fetches the {} build first",
                        app.variant.name()
                    ),
                    Some(_) if engine == EngineKind::Ollama => {
                        "not installed: install it from Setup (7) first".into()
                    }
                    Some(_) => "not installed: see Setup (7)".into(),
                    None => "checking…".into(),
                };
                ("engine", engine.name().to_string(), note)
            }
            PlanRow::Context => (
                "context",
                format!("{} tokens", p.context),
                format!("KV cache {:.2} GiB", nums.kv_gib),
            ),
            PlanRow::GpuLayers => (
                "gpu layers",
                format!(
                    "{}/{}",
                    nums.gpu_layers.unwrap_or(0),
                    nums.max_layers.unwrap_or(0)
                ),
                if p.gpu_layers.is_some() {
                    "manual".into()
                } else {
                    "auto: as many as fit in VRAM".into()
                },
            ),
        };
        let vstyle = if sel {
            Style::default()
                .fg(Color::Black)
                .bg(accent())
                .add_modifier(Modifier::BOLD)
        } else {
            bold(text())
        };
        lines.push(Line::from(vec![
            Span::styled(if sel { " ▶ " } else { "   " }, bold(accent())),
            Span::styled(format!("{label:<11}"), s(dim())),
            Span::styled(if sel { "◀ " } else { "  " }, s(accent())),
            Span::styled(format!(" {value} "), vstyle),
            Span::styled(if sel { " ▶" } else { "  " }, s(accent())),
            Span::styled(format!("   {note}"), s(dim())),
        ]));
    }
    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        " ─ what this means ".to_string() + &"─".repeat(inner.width.saturating_sub(20) as usize),
        s(faint()),
    )));
    let bar_w = inner.width.saturating_sub(16) as usize;
    let mut bar = vec![Span::styled(format!(" {:<11}  ", "memory"), s(dim()))];
    bar.extend(mem_bar(
        nums.weights_gib,
        nums.kv_gib,
        app.hw.capacity_gib(),
        bar_w,
        nums.fits,
    ));
    lines.push(Line::from(bar));
    lines.push(Line::from(vec![
        Span::styled(format!(" {:<11}  ", ""), s(dim())),
        Span::styled(
            format!(
                "{:.1} GiB of {:.1} GiB",
                nums.total_gib,
                app.hw.capacity_gib()
            ),
            bold(if nums.fits { green() } else { red() }),
        ),
        Span::styled(
            if !nums.fits {
                "   will not fit. it may crash or swap; go smaller".to_string()
            } else if nums.offload {
                "   spills into RAM (partial offload)".to_string()
            } else {
                String::new()
            },
            s(if nums.fits { blue() } else { red() }),
        ),
        Span::styled(if nums.exact { "" } else { "   (estimate)" }, s(dim())),
    ]));
    if let Some(t) = nums.tok_s {
        lines.push(Line::from(vec![
            Span::styled(format!(" {:<11}  ", "speed"), s(dim())),
            Span::styled(format!("~{t:.0} tok/s"), bold(tok_color(t))),
            Span::styled(
                if verdict::is_fast(t) {
                    "   faster than you read"
                } else if verdict::is_usable(t) {
                    "   you will see it type"
                } else {
                    "   slow; fine for batch jobs"
                },
                s(dim()),
            ),
        ]));
    }
    if let Some(d) = nums.download {
        let free = crate::state::free_disk_bytes(&crate::state::models_dir());
        lines.push(Line::from(vec![
            Span::styled(format!(" {:<11}  ", "download"), s(dim())),
            Span::styled(format!("up to {}", human_bytes(d)), s(text())),
            Span::styled(
                free.map(|f| format!("   {} free on disk", human_bytes(f)))
                    .unwrap_or_default(),
                s(if free.map(|f| f < d).unwrap_or(false) {
                    red()
                } else {
                    dim()
                }),
            ),
            Span::styled("   skipped if already there", s(faint())),
        ]));
    }
    // what will actually be pulled / run
    let plan_preview = LaunchPreview {
        app,
        p,
        engine,
        gpu_layers: nums.gpu_layers,
    };
    lines.push(Line::from(vec![
        Span::styled(format!(" {:<11}  ", "weights"), s(dim())),
        Span::styled(plan_preview.artifact(), s(text())),
    ]));
    lines.push(Line::from(vec![
        Span::styled(format!(" {:<11}  ", "runs"), s(dim())),
        Span::styled(plan_preview.command(), s(accent())),
    ]));
    if let Some(sess) = &app.session {
        lines.push(Line::from(Span::styled(
            format!(" {:<11}  {} will be unloaded first", "note", sess.name),
            s(amber()),
        )));
    }
    lines.push(Line::raw(""));
    lines.push(Line::from(vec![
        Span::raw(" "),
        Span::styled(
            " enter ",
            Style::default()
                .fg(Color::Black)
                .bg(green())
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" launch   ", bold(green())),
        Span::styled(" esc ", Style::default().fg(Color::Black).bg(dim())),
        Span::styled(" back", s(dim())),
    ]));
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
}

struct LaunchPreview<'a> {
    app: &'a App,
    p: &'a PlanModal,
    engine: EngineKind,
    gpu_layers: Option<u32>,
}

impl LaunchPreview<'_> {
    fn source(&self) -> Option<crate::pipeline::Source> {
        match (&self.p.fixed, self.p.model) {
            (Some(s), _) => Some(s.clone()),
            (None, Some(m)) => Some(crate::pipeline::Source::Catalog {
                model: m,
                quant: self.p.quants[self.p.quant_idx],
            }),
            _ => None,
        }
    }
    fn artifact(&self) -> String {
        let Some(source) = self.source() else {
            return String::new();
        };
        crate::pipeline::LaunchPlan {
            name: String::new(),
            engine: self.engine,
            source,
            context: self.p.context,
            gpu_layers: None,
            predicted_tok_s: None,
            predicted_gib: None,
            download_bytes: None,
            measure_key: None,
            port: 0,
            variant: self.app.variant,
            temperature: 0.0,
        }
        .artifact()
    }
    fn command(&self) -> String {
        match self.engine {
            EngineKind::Ollama => format!("ollama serve  +  num_ctx {}", self.p.context),
            EngineKind::LlamaCpp => format!(
                "llama-server -m <model file> -c {}{} --port {}",
                self.p.context,
                self.gpu_layers
                    .map(|n| format!(" -ngl {n}"))
                    .unwrap_or_default(),
                self.app.state.settings.port
            ),
            EngineKind::Mlx => format!(
                "mlx_lm.server --model <dir> --port {}",
                self.app.state.settings.port
            ),
        }
    }
}

fn draw_confirm(f: &mut Frame, c: &Confirm, area: Rect) {
    let w = 76u16.min(area.width.saturating_sub(2));
    let room = w.saturating_sub(3).max(10) as usize;
    let rows: usize = c
        .body
        .iter()
        .map(|l| {
            (unicode_width::UnicodeWidthStr::width(l.as_str()) + 1)
                .div_ceil(room)
                .max(1)
        })
        .sum();
    let r = centered(area, w, rows as u16 + 6);
    f.render_widget(Clear, r);
    let mut lines = vec![Line::raw("")];
    for l in &c.body {
        lines.push(Line::from(Span::styled(format!(" {l}"), s(text()))));
    }
    lines.push(Line::raw(""));
    let mut keys = vec![
        Span::raw(" "),
        Span::styled(
            " y ",
            Style::default()
                .fg(Color::Black)
                .bg(accent())
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!(" {}   ", c.yes), bold(accent())),
    ];
    if matches!(c.action, ConfirmAction::Quit) && c.body.iter().any(|l| l.starts_with("k ")) {
        keys.push(Span::styled(
            " k ",
            Style::default().fg(Color::Black).bg(green()),
        ));
        keys.push(Span::styled(" keep server   ", s(green())));
    }
    keys.push(Span::styled(
        " n ",
        Style::default().fg(Color::Black).bg(dim()),
    ));
    keys.push(Span::styled(" never mind", s(dim())));
    lines.push(Line::from(keys));
    f.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(block_hot(&c.title, accent())),
        r,
    );
}

fn draw_settings(f: &mut Frame, app: &App, row: usize, editing: Option<&str>, area: Rect) {
    let r = centered(area, 84, App::SETTINGS.len() as u16 + 6);
    f.render_widget(Clear, r);
    let mut lines = vec![
        Line::from(Span::styled(
            " saved to ~/.tokenpati/state.json as you change them",
            italic(dim()),
        )),
        Line::raw(""),
    ];
    for (i, name) in App::SETTINGS.iter().enumerate() {
        let sel = i == row;
        let value = match (sel, editing) {
            (true, Some(buf)) => format!("{buf}▌"),
            _ => {
                let v = app.setting_value(i);
                if v.chars().count() > 50 {
                    v.chars().take(49).collect::<String>() + "…"
                } else {
                    v
                }
            }
        };
        lines.push(Line::from(vec![
            Span::styled(if sel { " ▶ " } else { "   " }, bold(accent())),
            Span::styled(format!("{name:<16}"), s(dim())),
            Span::styled(
                format!(" {value} "),
                if sel {
                    Style::default().fg(Color::Black).bg(accent())
                } else {
                    bold(text())
                },
            ),
        ]));
    }
    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        " ↑↓ pick  ←→ change  enter edit text  esc close",
        s(dim()),
    )));
    f.render_widget(
        Paragraph::new(lines).block(block_hot("SETTINGS", accent())),
        r,
    );
}

fn draw_help(f: &mut Frame, area: Rect) {
    let r = centered(area, 90, 34);
    f.render_widget(Clear, r);
    let h = |k: &str, d: &str| {
        Line::from(vec![
            Span::styled(format!("  {k:<16}"), bold(accent())),
            Span::styled(d.to_string(), s(text())),
        ])
    };
    let sec = |t: &str| Line::from(Span::styled(format!(" {t}"), bold(text())));
    let lines = vec![
        sec("the flow"),
        Line::from(Span::styled(
            "  Machine shows what you have → Models ranks what fits → enter opens a launch plan →",
            s(dim()),
        )),
        Line::from(Span::styled(
            "  Launch shows each step as it happens → Chat talks to it. Everything also serves an",
            s(dim()),
        )),
        Line::from(Span::styled(
            "  OpenAI-compatible endpoint other apps can use.",
            s(dim()),
        )),
        Line::raw(""),
        sec("everywhere"),
        h(
            "1-7, tab",
            "switch tabs (tab and alt+1-7 work even while typing)",
        ),
        h("?", "this help"),
        h(
            ",",
            "settings: context, temperature, system prompt, engine, port…",
        ),
        h("L", "toggle the log panel"),
        h("q, ctrl+c", "quit (asks before stopping a running model)"),
        Line::raw(""),
        sec("models"),
        h("↑↓ / j k", "move"),
        h("← → / h l", "change quant for the selected model"),
        h("+ / -", "double / halve the context budget"),
        h("/  f", "filter by name · show only what fits"),
        h("enter", "launch plan"),
        Line::raw(""),
        sec("launch plan"),
        h(
            "↑↓  ← →",
            "pick a row · change quant, engine, context, GPU layers",
        ),
        h("enter / esc", "launch / back"),
        Line::raw(""),
        sec("run · chat · library"),
        h("c  r", "cancel / retry a launch"),
        h("y  x", "copy endpoint / unload the model"),
        h(
            "enter",
            "send · ctrl+j newline · esc stop or unfocus · ↑ history",
        ),
        h("/clear /system", "/temp /think /copy /save"),
        h("d", "delete a downloaded model (asks first)"),
        Line::raw(""),
        Line::from(Span::styled(
            "  files: ~/.tokenpati   ·   gated HF repos: export HF_TOKEN=…   ·   esc to close",
            s(faint()),
        )),
    ];
    f.render_widget(Paragraph::new(lines).block(block_hot("HELP", accent())), r);
}
