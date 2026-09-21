//! daudega / chalega / ghare jake sutti babu.

use crate::budget::Estimate;
use serde::Serialize;
use std::collections::HashMap;

/// Below this chat feels broken.
pub const SLOW_TOK_S: f64 = 10.0;
/// Above this it reads faster than you do.
pub const FAST_TOK_S: f64 = 18.0;
pub const DAUDEGA_MAX: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Verdict {
    Daudega,
    Chalega,
    Sutti,
}

impl Verdict {
    pub fn label(self) -> &'static str {
        match self {
            Verdict::Daudega => "daudega",
            Verdict::Chalega => "chalega",
            Verdict::Sutti => "ghare jake sutti babu",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Ranked {
    pub estimate: Estimate,
    pub verdict: Verdict,
    pub rank: usize,
}

pub fn is_fast(tok_s: f64) -> bool {
    tok_s.round() >= FAST_TOK_S
}

pub fn is_usable(tok_s: f64) -> bool {
    tok_s.round() >= SLOW_TOK_S
}

/// Per model: highest precision that clears FAST; else the fastest quant that clears SLOW;
/// else the smallest quant, so the table can show how far off it is.
pub fn best_quant_per_model(estimates: &[Estimate]) -> Vec<Estimate> {
    let mut order: Vec<&str> = Vec::new();
    let mut by_model: HashMap<&str, Vec<&Estimate>> = HashMap::new();
    for e in estimates {
        if !by_model.contains_key(e.model.name) {
            order.push(e.model.name);
        }
        by_model.entry(e.model.name).or_default().push(e);
    }
    order
        .iter()
        .map(|name| {
            let mut group = by_model[name].clone();
            group.sort_by(|a, b| a.weight_gib.partial_cmp(&b.weight_gib).unwrap());
            let ok: Vec<&Estimate> = group.iter().copied().filter(|e| e.fits).collect();
            let fast: Vec<&Estimate> = ok.iter().copied().filter(|e| is_fast(e.tok_s)).collect();
            let usable: Vec<&Estimate> = ok.iter().copied().filter(|e| is_usable(e.tok_s)).collect();
            if let Some(e) = fast.last() {
                (*e).clone()
            } else if let Some(e) = usable.first() {
                (*e).clone()
            } else {
                group[0].clone()
            }
        })
        .collect()
}

/// Bigger model wins among those that run; daudega for the top 3 that are also fast.
pub fn rank(estimates: Vec<Estimate>) -> Vec<Ranked> {
    let (mut runs, mut dead): (Vec<Estimate>, Vec<Estimate>) =
        estimates.into_iter().partition(|e| e.fits && is_usable(e.tok_s));
    runs.sort_by(|a, b| {
        b.model.params_b.partial_cmp(&a.model.params_b).unwrap()
            .then(b.tok_s.partial_cmp(&a.tok_s).unwrap())
    });
    dead.sort_by(|a, b| a.model.params_b.partial_cmp(&b.model.params_b).unwrap());

    let mut daudega_left = DAUDEGA_MAX;
    let mut ranked: Vec<Ranked> = runs
        .into_iter()
        .map(|e| {
            let verdict = if daudega_left > 0 && is_fast(e.tok_s) {
                daudega_left -= 1;
                Verdict::Daudega
            } else {
                Verdict::Chalega
            };
            Ranked { estimate: e, verdict, rank: 0 }
        })
        .collect();
    ranked.sort_by(|a, b| {
        (a.verdict != Verdict::Daudega).cmp(&(b.verdict != Verdict::Daudega))
            .then(b.estimate.model.params_b.partial_cmp(&a.estimate.model.params_b).unwrap())
    });
    ranked.extend(dead.into_iter().map(|e| Ranked { estimate: e, verdict: Verdict::Sutti, rank: 0 }));
    for (i, r) in ranked.iter_mut().enumerate() {
        r.rank = i + 1;
    }
    ranked
}

pub fn winner(ranked: &[Ranked]) -> Option<&Ranked> {
    ranked.iter().find(|r| r.verdict != Verdict::Sutti)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::estimate_all;
    use crate::catalog::CATALOG;
    use crate::hardware::Hardware;

    #[test]
    fn m4_16gb_picks_something_sane() {
        let hw = Hardware {
            chip: "Apple M4".into(), os: "macos".into(), memory_gib: 16.0, usable_gib: 12.0,
            bandwidth_gbps: 120.0, bandwidth_estimated: false, backend: "mlx".into(),
            vram_gib: None, gpu_name: None, gpu_cores: Some(10),
        };
        let ranked = rank(best_quant_per_model(&estimate_all(&hw, CATALOG, 8192)));
        let win = winner(&ranked).expect("someone should win");
        assert!(win.estimate.fits);
        assert!(win.estimate.model.params_b <= 15.0);
        assert!(ranked.iter().any(|r| r.verdict == Verdict::Sutti));
        assert!(ranked.iter().filter(|r| r.verdict == Verdict::Daudega).count() <= DAUDEGA_MAX);
    }
}
