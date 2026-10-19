//! llama.cpp: tokenpati installs the official prebuilt `llama-server`, downloads GGUFs itself,
//! and runs one server per loaded model.

use super::{EngineInfo, EngineKind};
use crate::net;
use serde::Deserialize;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

const RELEASES: &str = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=8";

fn exe() -> &'static str {
    if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    }
}

pub fn install_root() -> PathBuf {
    crate::state::bin_dir().join("llama.cpp")
}

fn find_named(root: &Path, name: &str, depth: usize) -> Option<PathBuf> {
    if depth == 0 {
        return None;
    }
    let rd = std::fs::read_dir(root).ok()?;
    let mut dirs = Vec::new();
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            dirs.push(p);
        } else if p.file_name().and_then(|n| n.to_str()) == Some(name) {
            return Some(p);
        }
    }
    dirs.sort();
    dirs.reverse(); // newest build tag first
    dirs.iter().find_map(|d| find_named(d, name, depth - 1))
}

/// Ours first (we know how it was built), then whatever is on PATH.
pub fn server_binary() -> Option<PathBuf> {
    find_named(&install_root(), exe(), 5)
        .filter(|p| super::is_executable(p))
        .or_else(|| super::which(exe()))
}

pub fn version_of(bin: &Path) -> Option<String> {
    let out = Command::new(bin)
        .arg("--version")
        .stdin(Stdio::null())
        .output()
        .ok()?;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    text.lines()
        .find_map(|l| l.trim().strip_prefix("version:"))
        .map(|v| format!("b{}", v.split_whitespace().next().unwrap_or("?")))
}

pub fn detect() -> EngineInfo {
    let path = server_binary();
    let version = path.as_deref().and_then(version_of);
    EngineInfo {
        kind: EngineKind::LlamaCpp,
        installed: path.is_some(),
        running: false,
        detail: match &path {
            Some(p) if p.starts_with(install_root()) => "installed by tokenpati".into(),
            Some(_) => "found on PATH".into(),
            None => {
                "not installed. press i: tokenpati downloads the official build (~30-200 MB)".into()
            }
        },
        version,
        path,
    }
}

/// Which prebuilt to fetch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// macOS build, Metal on Apple silicon.
    Mac,
    Cpu,
    Vulkan,
    Cuda,
    Rocm,
}

impl Variant {
    pub fn name(self) -> &'static str {
        match self {
            Variant::Mac => "macos (metal)",
            Variant::Cpu => "cpu",
            Variant::Vulkan => "vulkan",
            Variant::Cuda => "cuda",
            Variant::Rocm => "rocm",
        }
    }

    pub fn choices() -> Vec<Variant> {
        if cfg!(target_os = "macos") {
            vec![Variant::Mac]
        } else if cfg!(target_arch = "aarch64") {
            vec![Variant::Cpu, Variant::Vulkan, Variant::Cuda]
        } else {
            vec![Variant::Cpu, Variant::Vulkan, Variant::Cuda, Variant::Rocm]
        }
    }

    pub fn default_for(hw: &crate::hardware::Hardware) -> Variant {
        if cfg!(target_os = "macos") {
            return Variant::Mac;
        }
        if hw.backend.contains("cuda") {
            return Variant::Cuda;
        }
        if Path::new("/dev/kfd").exists() {
            return Variant::Rocm;
        }
        if Path::new("/dev/dri").exists() && has_vulkan_loader() {
            return Variant::Vulkan;
        }
        Variant::Cpu
    }
}

fn has_vulkan_loader() -> bool {
    [
        "/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
        "/usr/lib/aarch64-linux-gnu/libvulkan.so.1",
        "/usr/lib/libvulkan.so.1",
        "/usr/lib64/libvulkan.so.1",
    ]
    .iter()
    .any(|p| Path::new(p).exists())
}

#[derive(Debug, Clone, Deserialize)]
pub struct Asset {
    pub name: String,
    pub browser_download_url: String,
    #[serde(default)]
    pub size: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Release {
    pub tag_name: String,
    #[serde(default)]
    pub assets: Vec<Asset>,
}

fn arch() -> &'static str {
    if cfg!(target_arch = "aarch64") {
        "arm64"
    } else {
        "x64"
    }
}

/// Does `name` look like the prebuilt we want? Release names look like
/// llama-b6000-bin-ubuntu-vulkan-x64.tar.gz, llama-b6000-bin-macos-arm64.tar.gz,
/// llama-b6000-bin-ubuntu-cuda-12.8-x64.tar.gz.
pub fn asset_matches(name: &str, v: Variant, arch: &str) -> bool {
    if !name.starts_with("llama-") || !(name.ends_with(".tar.gz") || name.ends_with(".zip")) {
        return false;
    }
    let os_part = |s: &str| name.contains(&format!("-bin-{s}-{arch}."));
    match v {
        Variant::Mac => os_part("macos"),
        Variant::Cpu => os_part("ubuntu"),
        Variant::Vulkan => os_part("ubuntu-vulkan"),
        Variant::Cuda => name.contains("-bin-ubuntu-cuda-") && name.contains(&format!("-{arch}.")),
        Variant::Rocm => name.contains("-bin-ubuntu-rocm-") && name.contains(&format!("-{arch}.")),
    }
}

/// Pick the asset from the newest release that has one for us. For CUDA, prefer the oldest
/// CUDA major in that release (12.x runs on more drivers than 13.x). Returns the matching
/// cudart runtime asset too, when there is one.
pub fn pick_asset(
    releases: &[Release],
    v: Variant,
    arch: &str,
) -> Option<(String, Asset, Option<Asset>)> {
    for r in releases {
        let mut hits: Vec<&Asset> = r
            .assets
            .iter()
            .filter(|a| asset_matches(&a.name, v, arch) && a.name.ends_with(".tar.gz"))
            .collect();
        hits.sort_by(|a, b| a.name.cmp(&b.name));
        if let Some(a) = hits.first() {
            let cudart = if v == Variant::Cuda {
                let ver = a
                    .name
                    .split("-bin-ubuntu-cuda-")
                    .nth(1)
                    .and_then(|s| s.split('-').next())
                    .unwrap_or("");
                r.assets
                    .iter()
                    .find(|c| {
                        c.name.starts_with("cudart-")
                            && c.name.contains(&format!("cuda-{ver}-{arch}"))
                    })
                    .cloned()
            } else {
                None
            };
            return Some((r.tag_name.clone(), (*a).clone(), cudart));
        }
    }
    None
}

fn extract_tgz(archive: &Path, dest: &Path, flatten: bool) -> Result<(), String> {
    let f = std::fs::File::open(archive).map_err(|e| e.to_string())?;
    let mut ar = tar::Archive::new(flate2::read::GzDecoder::new(f));
    ar.set_preserve_permissions(true);
    std::fs::create_dir_all(dest).map_err(|e| e.to_string())?;
    if !flatten {
        return ar.unpack(dest).map_err(|e| format!("extract: {e}"));
    }
    for entry in ar.entries().map_err(|e| e.to_string())? {
        let mut entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path().map_err(|e| e.to_string())?.into_owned();
        if let Some(name) = path.file_name() {
            if entry.header().entry_type().is_dir() {
                continue;
            }
            entry
                .unpack(dest.join(name))
                .map_err(|e| format!("extract: {e}"))?;
        }
    }
    Ok(())
}

/// Download and unpack the official build. Steps report through `log`; bytes through `progress`.
pub fn install(
    v: Variant,
    cancel: &Arc<AtomicBool>,
    mut log: impl FnMut(String),
    mut progress: impl FnMut(net::Progress),
) -> Result<PathBuf, String> {
    let agent = net::agent_for(RELEASES);
    log("asking GitHub for the latest llama.cpp release".into());
    let releases: Vec<Release> = agent
        .get(RELEASES)
        .set("Accept", "application/vnd.github+json")
        .call()
        .map_err(net::err_text)?
        .into_json()
        .map_err(|e| format!("bad release list: {e}"))?;
    let (tag, asset, cudart) = pick_asset(&releases, v, arch()).ok_or_else(|| {
        format!(
            "no prebuilt {} build for {} {}. on macOS: brew install llama.cpp",
            v.name(),
            std::env::consts::OS,
            arch()
        )
    })?;
    log(format!("release {tag}: {}", asset.name));

    let dest = install_root().join(&tag);
    let tmp = install_root().join("download");
    let archive = tmp.join(&asset.name);
    let total = asset.size + cudart.as_ref().map(|c| c.size).unwrap_or(0);
    let base = asset.size;
    net::download(
        &agent,
        &asset.browser_download_url,
        &archive,
        Some(asset.size).filter(|s| *s > 0),
        cancel,
        |p| {
            progress(net::Progress {
                done: p.done,
                total,
                rate: p.rate,
            })
        },
    )?;
    log(format!("unpacking into {}", dest.display()));
    extract_tgz(&archive, &dest, false)?;
    let bin = find_named(&dest, exe(), 5).ok_or("the archive had no llama-server in it")?;
    if let Some(c) = cudart {
        log(format!("fetching CUDA runtime: {}", c.name));
        let car = tmp.join(&c.name);
        net::download(
            &agent,
            &c.browser_download_url,
            &car,
            Some(c.size).filter(|s| *s > 0),
            cancel,
            |p| {
                progress(net::Progress {
                    done: base + p.done,
                    total,
                    rate: p.rate,
                })
            },
        )?;
        extract_tgz(&car, bin.parent().unwrap_or(&dest), true)?;
    }
    let _ = std::fs::remove_dir_all(&tmp);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755));
    }
    match version_of(&bin) {
        Some(ver) => log(format!("llama-server {ver} works")),
        None => log("installed, but `llama-server --version` did not answer. missing GPU drivers? try the cpu build".into()),
    }
    Ok(bin)
}

pub struct ServeOpts<'a> {
    pub model: &'a Path,
    pub context: u32,
    pub gpu_layers: Option<u32>,
    pub port: u16,
    pub threads: Option<usize>,
}

pub fn serve_args(o: &ServeOpts) -> Vec<String> {
    let mut a: Vec<String> = vec![
        "-m".into(),
        o.model.display().to_string(),
        "-c".into(),
        o.context.to_string(),
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        o.port.to_string(),
        "--jinja".into(),
    ];
    if let Some(n) = o.gpu_layers {
        a.push("-ngl".into());
        a.push(n.to_string());
    }
    if let Some(t) = o.threads {
        a.push("-t".into());
        a.push(t.to_string());
    }
    a
}

/// Start llama-server. Every output line goes to `on_line` (from a background thread) and to the log file.
pub fn serve(
    bin: &Path,
    o: &ServeOpts,
    on_line: impl Fn(String) + Send + Sync + 'static,
) -> Result<Child, String> {
    let args = serve_args(o);
    on_line(format!("$ {} {}", bin.display(), args.join(" ")));
    let mut child = Command::new(bin)
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not start llama-server: {e}"))?;
    pipe_lines(&mut child, "llama-server.log", on_line);
    Ok(child)
}

/// Forward a child's stdout and stderr, line by line, to a callback and a log file.
pub fn pipe_lines(
    child: &mut Child,
    log_name: &str,
    on_line: impl Fn(String) + Send + Sync + 'static,
) {
    let _ = std::fs::create_dir_all(crate::state::logs_dir());
    let logf = std::fs::File::create(crate::state::logs_dir().join(log_name)).ok();
    let logf = Arc::new(std::sync::Mutex::new(logf));
    let on_line = Arc::new(on_line);
    let out = child
        .stdout
        .take()
        .map(|s| Box::new(s) as Box<dyn std::io::Read + Send>);
    let err = child
        .stderr
        .take()
        .map(|s| Box::new(s) as Box<dyn std::io::Read + Send>);
    for stream in [out, err].into_iter().flatten() {
        let on_line = on_line.clone();
        let logf = logf.clone();
        std::thread::spawn(move || {
            for line in std::io::BufReader::new(stream)
                .lines()
                .map_while(Result::ok)
            {
                if let Ok(mut g) = logf.lock() {
                    if let Some(f) = g.as_mut() {
                        use std::io::Write;
                        let _ = writeln!(f, "{line}");
                    }
                }
                on_line(line);
            }
        });
    }
}

/// Every model .gguf under `root`: skips vision projectors, partial downloads, and non-first split parts.
pub fn find_ggufs(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk(root, 6, &mut out);
    out.sort();
    out
}

fn walk(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth == 0 {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, depth - 1, out);
            continue;
        }
        let name = p
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_lowercase();
        if !name.ends_with(".gguf") || name.contains("mmproj") {
            continue;
        }
        if let Some(i) = name.find("-of-") {
            // keep only part 00001
            if !name[..i].ends_with("-00001") {
                continue;
            }
        }
        out.push(p);
    }
}

/// For "x-00001-of-00003.gguf" all three parts; otherwise just the file.
pub fn split_siblings(p: &Path) -> Vec<PathBuf> {
    let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let Some(i) = name.find("-00001-of-") else {
        return vec![p.to_path_buf()];
    };
    let of = &name[i + 10..i + 15];
    let Ok(n) = of.parse::<u32>() else {
        return vec![p.to_path_buf()];
    };
    let prefix = &name[..i];
    (1..=n)
        .map(|k| p.with_file_name(format!("{prefix}-{k:05}-of-{of}.gguf")))
        .filter(|q| q.exists())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a(name: &str) -> Asset {
        Asset {
            name: name.into(),
            browser_download_url: format!("https://x/{name}"),
            size: 1,
        }
    }

    fn release() -> Release {
        Release {
            tag_name: "b11669".into(),
            assets: vec![
                a("cudart-llama-b11669-bin-ubuntu-cuda-12.8-x64.tar.gz"),
                a("cudart-llama-b11669-bin-ubuntu-cuda-13.4-x64.tar.gz"),
                a("llama-b11669-bin-macos-arm64.tar.gz"),
                a("llama-b11669-bin-ubuntu-arm64.tar.gz"),
                a("llama-b11669-bin-ubuntu-cuda-12.8-x64.tar.gz"),
                a("llama-b11669-bin-ubuntu-cuda-13.4-x64.tar.gz"),
                a("llama-b11669-bin-ubuntu-vulkan-x64.tar.gz"),
                a("llama-b11669-bin-ubuntu-x64.tar.gz"),
                a("llama-b11669-bin-win-cpu-x64.zip"),
            ],
        }
    }

    #[test]
    fn picks_the_right_prebuilt() {
        let rs = vec![
            Release {
                tag_name: "v0.7.0".into(),
                assets: vec![a("nightly-tag.txt")],
            },
            release(),
        ];
        let (tag, cpu, _) = pick_asset(&rs, Variant::Cpu, "x64").unwrap();
        assert_eq!(tag, "b11669");
        assert_eq!(cpu.name, "llama-b11669-bin-ubuntu-x64.tar.gz");
        assert_eq!(
            pick_asset(&rs, Variant::Vulkan, "x64").unwrap().1.name,
            "llama-b11669-bin-ubuntu-vulkan-x64.tar.gz"
        );
        assert_eq!(
            pick_asset(&rs, Variant::Mac, "arm64").unwrap().1.name,
            "llama-b11669-bin-macos-arm64.tar.gz"
        );
        assert_eq!(
            pick_asset(&rs, Variant::Cpu, "arm64").unwrap().1.name,
            "llama-b11669-bin-ubuntu-arm64.tar.gz"
        );
        let (_, cuda, rt) = pick_asset(&rs, Variant::Cuda, "x64").unwrap();
        assert_eq!(cuda.name, "llama-b11669-bin-ubuntu-cuda-12.8-x64.tar.gz");
        assert_eq!(
            rt.unwrap().name,
            "cudart-llama-b11669-bin-ubuntu-cuda-12.8-x64.tar.gz"
        );
        assert!(pick_asset(&rs, Variant::Rocm, "x64").is_none());
    }

    #[test]
    fn split_files() {
        let d = std::env::temp_dir().join(format!("tp-split-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        for n in [
            "m-00001-of-00002.gguf",
            "m-00002-of-00002.gguf",
            "solo.gguf",
            "mmproj-f16.gguf",
            "x.gguf.part",
        ] {
            std::fs::write(d.join(n), b"x").unwrap();
        }
        let found = find_ggufs(&d);
        let names: Vec<_> = found
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(names, vec!["m-00001-of-00002.gguf", "solo.gguf"]);
        assert_eq!(split_siblings(&found[0]).len(), 2);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn unpacks_release_and_finds_server() {
        let d = std::env::temp_dir().join(format!("tp-tgz-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let tgz = d.join("llama-b1-bin-ubuntu-x64.tar.gz");
        {
            let f = std::fs::File::create(&tgz).unwrap();
            let mut b = tar::Builder::new(flate2::write::GzEncoder::new(
                f,
                flate2::Compression::fast(),
            ));
            for (path, mode) in [
                ("build/bin/llama-server", 0o755),
                ("build/bin/libggml.so", 0o644),
            ] {
                let mut h = tar::Header::new_gnu();
                h.set_size(4);
                h.set_mode(mode);
                h.set_cksum();
                b.append_data(&mut h, path, &b"\x7fELF"[..]).unwrap();
            }
            b.into_inner().unwrap().finish().unwrap();
        }
        let out = d.join("out");
        extract_tgz(&tgz, &out, false).unwrap();
        let bin = find_named(&out, "llama-server", 5).unwrap();
        assert!(bin.ends_with("build/bin/llama-server"));
        assert!(super::super::is_executable(&bin));
        // cudart archives are flattened next to the binary
        extract_tgz(&tgz, &d.join("flat"), true).unwrap();
        assert!(d.join("flat").join("libggml.so").exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn args_carry_offload() {
        let o = ServeOpts {
            model: Path::new("/m.gguf"),
            context: 4096,
            gpu_layers: Some(20),
            port: 9000,
            threads: None,
        };
        let a = serve_args(&o).join(" ");
        assert!(a.contains("-c 4096") && a.contains("-ngl 20") && a.contains("--port 9000"));
    }
}
