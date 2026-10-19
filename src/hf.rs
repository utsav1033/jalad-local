//! Hugging Face: search GGUF repos, list their files, pick the right one for a quant.

use crate::net;
use serde::Deserialize;

/// HF_ENDPOINT points at a mirror (same variable huggingface_hub uses).
pub fn base() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| s.trim_end_matches('/').to_string())
        .unwrap_or_else(|| "https://huggingface.co".into())
}

#[derive(Debug, Clone, Deserialize)]
pub struct RepoHit {
    pub id: String,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub likes: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TreeEntry {
    #[serde(rename = "type")]
    pub kind: String,
    pub path: String,
    #[serde(default)]
    pub size: u64,
}

/// One downloadable variant of a repo: a single .gguf, or the parts of a split one.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufChoice {
    pub repo: String,
    /// All parts, first part first. llama.cpp is pointed at the first one.
    pub files: Vec<(String, u64)>,
    pub quant: Option<String>,
}

impl GgufChoice {
    pub fn size(&self) -> u64 {
        self.files.iter().map(|f| f.1).sum()
    }
    pub fn label(&self) -> String {
        let first = &self.files[0].0;
        let name = first.rsplit('/').next().unwrap_or(first);
        if self.files.len() > 1 {
            format!("{name}  (+{} parts)", self.files.len() - 1)
        } else {
            name.to_string()
        }
    }
    pub fn url(&self, i: usize) -> String {
        file_url(&self.repo, &self.files[i].0)
    }
}

pub fn file_url(repo: &str, path: &str) -> String {
    format!("{}/{repo}/resolve/main/{path}", base())
}

pub fn search(query: &str) -> Result<Vec<RepoHit>, String> {
    let agent = net::agent_for(&base());
    let resp = net::hf_get(&agent, &format!("{}/api/models", base()))
        .query("search", query)
        .query("filter", "gguf")
        .query("sort", "downloads")
        .query("direction", "-1")
        .query("limit", "40")
        .call()
        .map_err(net::err_text)?;
    resp.into_json()
        .map_err(|e| format!("bad search response: {e}"))
}

pub fn tree(repo: &str) -> Result<Vec<TreeEntry>, String> {
    let agent = net::agent_for(&base());
    let resp = net::hf_get(&agent, &format!("{}/api/models/{repo}/tree/main", base()))
        .query("recursive", "true")
        .call()
        .map_err(net::err_text)?;
    resp.into_json().map_err(|e| format!("bad file list: {e}"))
}

/// "x-00001-of-00003.gguf" -> Some(("x", 1, 3))
fn split_part(path: &str) -> Option<(String, u32, u32)> {
    let stem = path.strip_suffix(".gguf")?;
    let (head, of) = stem.rsplit_once("-of-")?;
    let (base, idx) = head.rsplit_once('-')?;
    if idx.len() != 5 || of.len() != 5 {
        return None;
    }
    Some((base.to_string(), idx.parse().ok()?, of.parse().ok()?))
}

/// Group a repo's .gguf files into downloadable choices, smallest first.
/// Skips mmproj (vision projector) and imatrix files, which are not models on their own.
pub fn gguf_choices(repo: &str, entries: &[TreeEntry]) -> Vec<GgufChoice> {
    let mut singles = Vec::new();
    let mut splits: std::collections::BTreeMap<String, Vec<(u32, String, u64)>> =
        Default::default();
    for e in entries {
        if e.kind != "file" || !e.path.ends_with(".gguf") {
            continue;
        }
        let lower = e.path.to_lowercase();
        if lower.contains("mmproj") || lower.contains("imatrix") {
            continue;
        }
        match split_part(&e.path) {
            Some((base, idx, _)) => {
                splits
                    .entry(base)
                    .or_default()
                    .push((idx, e.path.clone(), e.size))
            }
            None => singles.push(GgufChoice {
                repo: repo.to_string(),
                quant: crate::budget::quant_from_filename(&e.path),
                files: vec![(e.path.clone(), e.size)],
            }),
        }
    }
    for (base, mut parts) in splits {
        parts.sort();
        singles.push(GgufChoice {
            repo: repo.to_string(),
            quant: crate::budget::quant_from_filename(&base),
            files: parts.into_iter().map(|(_, p, s)| (p, s)).collect(),
        });
    }
    singles.sort_by_key(|c| c.size());
    singles
}

/// The choice whose quant matches `tag` (e.g. "Q4_K_M"), preferring a single file.
pub fn pick(choices: &[GgufChoice], tag: &str) -> Option<GgufChoice> {
    let tag = tag.to_uppercase();
    let mut matching: Vec<&GgufChoice> = choices
        .iter()
        .filter(|c| {
            c.quant.as_deref() == Some(tag.as_str())
                || (tag == "FP16" && c.quant.as_deref() == Some("F16"))
        })
        .collect();
    matching.sort_by_key(|c| c.files.len());
    matching.first().map(|c| (*c).clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(p: &str, s: u64) -> TreeEntry {
        TreeEntry {
            kind: "file".into(),
            path: p.into(),
            size: s,
        }
    }

    #[test]
    fn groups_splits_and_skips_junk() {
        let es = vec![
            f("README.md", 1),
            f("M-Q4_K_M.gguf", 40),
            f("M-Q8_0/M-Q8_0-00002-of-00002.gguf", 30),
            f("M-Q8_0/M-Q8_0-00001-of-00002.gguf", 50),
            f("mmproj-M-f16.gguf", 2),
            f("M.imatrix.gguf", 2),
        ];
        let cs = gguf_choices("a/b", &es);
        assert_eq!(cs.len(), 2);
        assert_eq!(cs[0].quant.as_deref(), Some("Q4_K_M"));
        let q8 = pick(&cs, "Q8_0").unwrap();
        assert_eq!(q8.files[0].0, "M-Q8_0/M-Q8_0-00001-of-00002.gguf");
        assert_eq!(q8.size(), 80);
        assert!(pick(&cs, "Q6_K").is_none());
    }

    #[test]
    fn urls() {
        std::env::remove_var("HF_ENDPOINT");
        assert_eq!(
            file_url("a/b", "x.gguf"),
            "https://huggingface.co/a/b/resolve/main/x.gguf"
        );
    }
}
