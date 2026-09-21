//! tokenpati: read the machine, run the show, print the answer.

mod budget;
mod catalog;
mod hardware;
mod ui;
mod verdict;

use clap::Parser;
use serde::Serialize;

/// kaun banega tokenpati: which local LLM will actually run on this machine
#[derive(Parser)]
#[command(name = "tokenpati", version)]
struct Args {
    /// Context length to budget memory for
    #[arg(long, default_value_t = 8192)]
    context: u32,
    /// Skip the dramatic pause
    #[arg(long)]
    fast: bool,
    /// Machine-readable output
    #[arg(long)]
    json: bool,
}

#[derive(Serialize)]
struct Row<'a> {
    rank: usize,
    model: &'a str,
    quant: &'a str,
    weight_gib: f64,
    kv_gib: f64,
    fits: bool,
    tok_s: f64,
    max_context: u32,
    verdict: &'a str,
}

#[derive(Serialize)]
struct Report<'a> {
    hardware: &'a hardware::Hardware,
    winner: Option<&'a str>,
    ranked: Vec<Row<'a>>,
}

fn main() {
    let args = Args::parse();
    let hw = hardware::detect();
    let estimates = budget::estimate_all(&hw, catalog::CATALOG, args.context);
    let ranked = verdict::rank(verdict::best_quant_per_model(&estimates));
    let win = verdict::winner(&ranked);

    if args.json {
        let report = Report {
            hardware: &hw,
            winner: win.map(|r| r.estimate.model.name),
            ranked: ranked
                .iter()
                .map(|r| Row {
                    rank: r.rank,
                    model: r.estimate.model.name,
                    quant: r.estimate.quant.tag(),
                    weight_gib: (r.estimate.weight_gib * 100.0).round() / 100.0,
                    kv_gib: (r.estimate.kv_gib * 100.0).round() / 100.0,
                    fits: r.estimate.fits,
                    tok_s: (r.estimate.tok_s * 10.0).round() / 10.0,
                    max_context: r.estimate.max_context,
                    verdict: r.verdict.label(),
                })
                .collect(),
        };
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
        return;
    }

    ui::banner();
    ui::wait_for_it(args.fast);
    ui::hot_seat(&hw);
    println!();
    ui::leaderboard(&hw, &ranked);
    println!();
    match win {
        Some(r) => ui::final_answer(&hw, r),
        None => ui::nobody_wins(),
    }
    println!();
}
