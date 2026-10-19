//! The full-screen app: `tokenpati` with no arguments.

mod app;
mod draw;
mod theme;

use app::{App, Ev};
use crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event, KeyEventKind,
};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::io::{stdout, Write};
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::{Duration, Instant};

type Term = Terminal<CrosstermBackend<std::io::Stdout>>;

fn enter(mouse: bool) -> std::io::Result<Term> {
    enable_raw_mode()?;
    let mut out = stdout();
    out.execute(EnterAlternateScreen)?;
    out.execute(EnableBracketedPaste)?;
    if mouse {
        out.execute(EnableMouseCapture)?;
    }
    let mut t = Terminal::new(CrosstermBackend::new(stdout()))?;
    t.clear()?;
    Ok(t)
}

fn leave() {
    let mut out = stdout();
    let _ = out.execute(DisableMouseCapture);
    let _ = out.execute(DisableBracketedPaste);
    let _ = out.execute(LeaveAlternateScreen);
    let _ = disable_raw_mode();
    let _ = out.execute(crossterm::cursor::Show);
}

/// Hand the real terminal to a command (installers may prompt for a password), then come back.
fn run_in_shell(cmd: &str) {
    leave();
    println!("\n\x1b[1;38;5;215m▶ tokenpati is running:\x1b[0m {cmd}\n");
    let status = std::process::Command::new("sh").arg("-c").arg(cmd).status();
    match status {
        Ok(s) if s.success() => println!("\n\x1b[1;32m✓ done.\x1b[0m"),
        Ok(s) => println!("\n\x1b[1;31m✗ exited with {s}.\x1b[0m"),
        Err(e) => println!("\n\x1b[1;31m✗ could not run it: {e}\x1b[0m"),
    }
    print!("press enter to go back to tokenpati… ");
    let _ = std::io::stdout().flush();
    let mut s = String::new();
    let _ = std::io::stdin().read_line(&mut s);
}

pub fn run(hw: crate::hardware::Hardware) -> std::io::Result<()> {
    let state = crate::state::State::load();
    let mouse = state.settings.mouse;
    let (tx, rx) = mpsc::channel::<Ev>();
    let mut app = App::new(tx.clone(), hw.clone(), state);

    // Never leave someone's terminal in raw mode, even on a panic.
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        leave();
        hook(info);
    }));

    // Live stats in the background so a slow nvidia-smi never stutters the UI.
    {
        let tx = tx.clone();
        let pid = app.pid_watch.clone();
        std::thread::spawn(move || {
            let mut sampler = crate::hardware::Sampler::new(&hw);
            loop {
                let mut live = sampler.sample();
                let p = pid.load(Ordering::Relaxed);
                if p != 0 {
                    live.model_rss_gib = crate::hardware::process_rss_gib(p);
                }
                if tx.send(Ev::Live(live)).is_err() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(1000));
            }
        });
    }

    let mut term = enter(mouse)?;
    let mut mouse_on = mouse;
    let mut last_tick = Instant::now();
    loop {
        term.draw(|f| draw::draw(f, &mut app))?;
        if event::poll(Duration::from_millis(60))? {
            // Drain everything queued (fast typing, pastes) before redrawing.
            loop {
                match event::read()? {
                    Event::Key(k) if k.kind != KeyEventKind::Release => app.on_key(k),
                    Event::Mouse(m) => app.on_mouse(m),
                    Event::Paste(p) => app.on_paste(&p),
                    _ => {}
                }
                if !event::poll(Duration::from_millis(0))? {
                    break;
                }
            }
        }
        while let Ok(ev) = rx.try_recv() {
            app.on_event(ev);
        }
        if last_tick.elapsed() >= Duration::from_millis(100) {
            app.on_tick();
            last_tick = Instant::now();
        }
        if let Some(cmd) = app.pending_shell.take() {
            run_in_shell(&cmd);
            term = enter(mouse_on)?;
            app.refresh_engines();
        }
        if app.state.settings.mouse != mouse_on {
            mouse_on = app.state.settings.mouse;
            let mut out = stdout();
            if mouse_on {
                let _ = out.execute(EnableMouseCapture);
            } else {
                let _ = out.execute(DisableMouseCapture);
            }
        }
        if app.quit {
            break;
        }
    }
    leave();
    app.stop_generation();
    if let Some(mut s) = app.session.take() {
        if app.keep_running {
            if let Some(c) = s.child.take() {
                println!(
                    "{} keeps running (pid {}) at {}",
                    s.name,
                    c.id(),
                    s.openai_url()
                );
                std::mem::forget(c);
            } else {
                println!(
                    "{} stays loaded in {} at {}",
                    s.name,
                    s.engine.name(),
                    s.openai_url()
                );
            }
        } else {
            println!("stopping {}…", s.name);
            s.stop();
        }
    }
    Ok(())
}
