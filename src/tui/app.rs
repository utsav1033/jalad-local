//! Everything the TUI knows, and how keys and background events change it.

use crate::budget::{self, Estimate, Quant};
use crate::catalog::{Model, CATALOG};
use crate::engine::{
    self, llamacpp::Variant, Chunk, EngineInfo, EngineKind, LocalModel, Msg, Session, Stats,
};
use crate::gguf::Meta;
use crate::hardware::{self, Hardware, Live};
use crate::hf::{self, GgufChoice, RepoHit};
use crate::net::Progress;
use crate::pipeline::{self, JobEvent, LaunchPlan, Source};
use crate::state::{self, Measurement, State};
use crate::verdict::{self, Ranked};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::Instant;

// ---------------------------------------------------------------- events from worker threads

pub enum Ev {
    Job(u64, JobEvent),
    Engines(Vec<EngineInfo>),
    Library(Vec<LocalModel>),
    Search(u64, Result<Vec<RepoHit>, String>),
    Files(String, Result<Vec<GgufChoice>, String>),
    Meta(String, Result<Meta, String>),
    Bandwidth(f64),
    Live(Live),
    Chat(u64, ChatEv),
    Install(InstallEv),
    Toast(String),
}

pub enum ChatEv {
    Chunk(Chunk),
    Done(Stats),
    Error(String),
}

pub enum InstallEv {
    Line(String),
    Progress(Progress),
    Done(Result<PathBuf, String>),
}

// ---------------------------------------------------------------- tabs

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Machine,
    Models,
    Discover,
    Library,
    Run,
    Chat,
    Engines,
}

impl Tab {
    pub const ALL: [Tab; 7] = [
        Tab::Machine,
        Tab::Models,
        Tab::Discover,
        Tab::Library,
        Tab::Run,
        Tab::Chat,
        Tab::Engines,
    ];
    pub fn title(self) -> &'static str {
        match self {
            Tab::Machine => "Machine",
            Tab::Models => "Models",
            Tab::Discover => "Search",
            Tab::Library => "Library",
            Tab::Run => "Launch",
            Tab::Chat => "Chat",
            Tab::Engines => "Setup",
        }
    }
    pub fn index(self) -> usize {
        Tab::ALL.iter().position(|t| *t == self).unwrap_or(0)
    }
}

// ---------------------------------------------------------------- launch jobs

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepStatus {
    Pending,
    Running,
    Done,
    Skipped,
    Failed,
}

pub struct StepState {
    pub title: &'static str,
    pub status: StepStatus,
    pub detail: String,
    pub lines: VecDeque<String>,
    pub progress: Option<(Progress, String)>,
    pub started: Option<Instant>,
    pub ended: Option<Instant>,
}

pub struct Job {
    pub id: u64,
    pub plan: LaunchPlan,
    pub steps: Vec<StepState>,
    pub cancel: Arc<AtomicBool>,
    pub finished: Option<bool>,
    pub started: Instant,
}

impl Job {
    pub fn running(&self) -> bool {
        self.finished.is_none()
    }
    pub fn current(&self) -> usize {
        self.steps
            .iter()
            .position(|s| s.status == StepStatus::Running || s.status == StepStatus::Failed)
            .unwrap_or_else(|| {
                self.steps
                    .iter()
                    .rposition(|s| s.status != StepStatus::Pending)
                    .unwrap_or(0)
            })
    }
}

// ---------------------------------------------------------------- modals

/// The screen between "I want this model" and "go": every knob, with the consequences recomputed live.
pub struct PlanModal {
    pub name: String,
    /// Catalog model, if this came from the leaderboard (enables the quant row).
    pub model: Option<Model>,
    pub quants: Vec<Quant>,
    pub quant_idx: usize,
    /// Discover / library sources are fixed.
    pub fixed: Option<Source>,
    pub engines: Vec<EngineKind>,
    pub engine_idx: usize,
    pub context: u32,
    /// None = auto (all that fit).
    pub gpu_layers: Option<u32>,
    pub row: usize,
    /// Size of the file, for Discover sources without a catalog entry.
    pub file_bytes: Option<u64>,
    pub meta: Option<Meta>,
}

pub enum PlanRow {
    Quant,
    Engine,
    Context,
    GpuLayers,
}

pub struct Confirm {
    pub title: String,
    pub body: Vec<String>,
    pub yes: &'static str,
    pub action: ConfirmAction,
}

pub enum ConfirmAction {
    Delete(LocalModel),
    CancelJob,
    StopSession,
    Shell(String),
    InstallLlama(Variant),
    Quit,
}

pub enum Modal {
    Plan(Box<PlanModal>),
    Confirm(Confirm),
    Settings { row: usize, editing: Option<String> },
    Help,
}

// ---------------------------------------------------------------- chat

pub struct ChatEntry {
    pub role: String,
    pub content: String,
    pub thinking: String,
    pub stats: Option<Stats>,
    pub error: Option<String>,
}

#[derive(Default)]
pub struct Chat {
    pub entries: Vec<ChatEntry>,
    pub input: String,
    pub cursor: usize,
    pub focused: bool,
    pub generating: Option<(u64, Arc<AtomicBool>)>,
    /// Lines scrolled up from the bottom. 0 = follow the stream.
    pub scroll: u16,
    pub history: Vec<String>,
    pub history_idx: Option<usize>,
}

// ---------------------------------------------------------------- discover

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DFocus {
    Search,
    Results,
    Files,
}

pub struct Discover {
    pub query: String,
    pub focus: DFocus,
    pub results: Vec<RepoHit>,
    pub sel: usize,
    pub searching: bool,
    pub seq: u64,
    pub error: Option<String>,
    pub files: HashMap<String, Result<Vec<GgufChoice>, String>>,
    pub file_sel: usize,
    pub metas: HashMap<String, Result<Meta, String>>,
    pub loaded_once: bool,
}

// ---------------------------------------------------------------- the app

pub struct App {
    pub tx: Sender<Ev>,
    pub tab: Tab,
    pub state: State,
    pub hw: Hardware,
    pub ranked: Vec<Ranked>,
    pub models_sel: usize,
    pub quant_override: HashMap<&'static str, Quant>,
    pub models_filter: String,
    pub filtering: bool,
    pub only_runnable: bool,

    pub engines: Vec<EngineInfo>,
    pub engines_sel: usize,
    pub variant: Variant,
    pub installing: Option<(Option<Progress>, Vec<String>)>,

    pub library: Vec<LocalModel>,
    pub library_sel: usize,
    pub library_loading: bool,

    pub discover: Discover,

    pub job: Option<Job>,
    pub next_job: u64,
    pub session: Option<Session>,
    pub chat: Chat,
    pub next_chat: u64,

    pub live: Live,
    pub cpu_hist: VecDeque<u64>,
    pub benchmarking: bool,

    pub modal: Option<Modal>,
    pub toast: Option<(String, Instant)>,
    pub log: VecDeque<String>,
    pub show_log: bool,
    pub tick: u64,
    pub quit: bool,
    /// A command to run in the real terminal (installers that may prompt for a password).
    pub pending_shell: Option<String>,
    /// Detach a running server instead of killing it on exit.
    pub keep_running: bool,
    /// Tab bar hit boxes, filled by draw for mouse clicks.
    pub tab_hits: Vec<(u16, u16, Tab)>,
    /// pid of the server we spawned, for the live sampler (0 = none).
    pub pid_watch: Arc<std::sync::atomic::AtomicU32>,
}

impl App {
    pub fn new(tx: Sender<Ev>, mut hw: Hardware, state: State) -> App {
        hw.apply_state(&state);
        let variant = Variant::default_for(&hw);
        let mut app = App {
            tx,
            tab: Tab::Machine,
            state,
            hw,
            ranked: vec![],
            models_sel: 0,
            quant_override: HashMap::new(),
            models_filter: String::new(),
            filtering: false,
            only_runnable: false,
            engines: vec![],
            engines_sel: 0,
            variant,
            installing: None,
            library: vec![],
            library_sel: 0,
            library_loading: false,
            discover: Discover {
                query: String::new(),
                focus: DFocus::Search,
                results: vec![],
                sel: 0,
                searching: false,
                seq: 0,
                error: None,
                files: HashMap::new(),
                file_sel: 0,
                metas: HashMap::new(),
                loaded_once: false,
            },
            job: None,
            next_job: 1,
            session: None,
            chat: Chat {
                focused: true,
                ..Default::default()
            },
            next_chat: 1,
            live: Live::default(),
            cpu_hist: VecDeque::new(),
            benchmarking: false,
            modal: None,
            toast: None,
            log: VecDeque::new(),
            show_log: false,
            tick: 0,
            quit: false,
            pending_shell: None,
            keep_running: false,
            tab_hits: vec![],
            pid_watch: Arc::new(std::sync::atomic::AtomicU32::new(0)),
        };
        app.recompute();
        app.refresh_engines();
        app.refresh_library();
        if app.state.measurements.is_empty() {
            app.toast = Some(("new here? enter launches the winner below, 2 shows every model, ? explains every key".into(), Instant::now()));
        }
        app
    }

    // ------------------------------------------------------------ derived data

    pub fn recompute(&mut self) {
        let keep = self
            .ranked
            .get(self.models_sel)
            .map(|r| r.estimate.model.name);
        let est = budget::estimate_all(&self.hw, CATALOG, self.state.settings.context);
        self.ranked = verdict::rank(verdict::best_quant_per_model(&est));
        if let Some(name) = keep {
            if let Some(i) = self
                .visible_models()
                .iter()
                .position(|r| r.estimate.model.name == name)
            {
                self.models_sel = i;
            }
        }
    }

    pub fn visible_models(&self) -> Vec<&Ranked> {
        let f = self.models_filter.to_lowercase();
        self.ranked
            .iter()
            .filter(|r| {
                f.is_empty()
                    || r.estimate.model.name.to_lowercase().contains(&f)
                    || r.estimate.model.tagline.contains(&f)
            })
            .filter(|r| !self.only_runnable || r.estimate.fits)
            .collect()
    }

    pub fn winner(&self) -> Option<&Ranked> {
        verdict::winner(&self.ranked)
    }

    /// The quant shown for a model: what you picked, else the leaderboard's choice.
    pub fn quant_for(&self, r: &Ranked) -> Quant {
        *self
            .quant_override
            .get(r.estimate.model.name)
            .unwrap_or(&r.estimate.quant)
    }

    pub fn estimate_for(&self, model: &Model, q: Quant, ctx: u32) -> Estimate {
        budget::estimate(&self.hw, model, q, ctx)
    }

    pub fn measured(&self, model: &str, quant: Quant) -> Option<&Measurement> {
        self.state.measurements.get(&State::key(model, quant.tag()))
    }

    pub fn engine_info(&self, k: EngineKind) -> Option<&EngineInfo> {
        self.engines.iter().find(|e| e.kind == k)
    }

    /// The engine to suggest for a source: your setting if usable, else the best one installed.
    fn preferred_engine(&self, options: &[EngineKind]) -> usize {
        if let Some(k) = EngineKind::from_name(&self.state.settings.engine) {
            if let Some(i) = options.iter().position(|o| *o == k) {
                return i;
            }
        }
        let usable = |k: EngineKind| self.engine_info(k).map(|e| e.usable()).unwrap_or(false);
        let order: Vec<EngineKind> = if self.hw.is_apple() {
            vec![EngineKind::Mlx, EngineKind::Ollama, EngineKind::LlamaCpp]
        } else {
            vec![EngineKind::Ollama, EngineKind::LlamaCpp]
        };
        for k in order.iter().filter(|k| usable(**k)) {
            if let Some(i) = options.iter().position(|o| o == k) {
                return i;
            }
        }
        // Nothing installed: llama.cpp, since tokenpati can install it without asking for a password.
        options
            .iter()
            .position(|o| *o == EngineKind::LlamaCpp)
            .unwrap_or(0)
    }

    // ------------------------------------------------------------ messages

    pub fn toast(&mut self, s: impl Into<String>) {
        let s = s.into();
        self.logln(s.clone());
        self.toast = Some((s, Instant::now()));
    }

    pub fn logln(&mut self, s: impl Into<String>) {
        self.log.push_back(s.into());
        while self.log.len() > 800 {
            self.log.pop_front();
        }
    }

    // ------------------------------------------------------------ background work

    pub fn refresh_engines(&mut self) {
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let _ = tx.send(Ev::Engines(engine::detect_all()));
        });
    }

    pub fn refresh_library(&mut self) {
        self.library_loading = true;
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let _ = tx.send(Ev::Library(engine::list_local()));
        });
    }

    pub fn search(&mut self) {
        let d = &mut self.discover;
        d.seq += 1;
        d.searching = true;
        d.error = None;
        d.loaded_once = true;
        let (q, seq, tx) = (d.query.clone(), d.seq, self.tx.clone());
        std::thread::spawn(move || {
            let _ = tx.send(Ev::Search(seq, hf::search(&q)));
        });
    }

    fn load_files(&mut self) {
        let Some(hit) = self.discover.results.get(self.discover.sel) else {
            return;
        };
        let repo = hit.id.clone();
        if self.discover.files.contains_key(&repo) {
            return;
        }
        self.discover
            .files
            .insert(repo.clone(), Err("loading…".into()));
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let r = hf::tree(&repo).map(|t| hf::gguf_choices(&repo, &t));
            let _ = tx.send(Ev::Files(repo, r));
        });
    }

    pub fn selected_choice(&self) -> Option<&GgufChoice> {
        let hit = self.discover.results.get(self.discover.sel)?;
        match self.discover.files.get(&hit.id)? {
            Ok(v) => v.get(self.discover.file_sel),
            Err(_) => None,
        }
    }

    /// Any header we have read from this repo's files.
    pub fn repo_meta<'a>(&'a self, choices: &[GgufChoice]) -> Option<&'a Meta> {
        choices.iter().find_map(|c| {
            self.discover
                .metas
                .get(&c.url(0))
                .and_then(|r| r.as_ref().ok())
        })
    }

    fn load_meta(&mut self) {
        let Some(c) = self.selected_choice() else {
            return;
        };
        let url = c.url(0);
        if self.discover.metas.contains_key(&url) {
            return;
        }
        // one header per repo is enough
        let siblings: Vec<GgufChoice> = self
            .discover
            .results
            .get(self.discover.sel)
            .and_then(|h| self.discover.files.get(&h.id))
            .and_then(|r| r.as_ref().ok())
            .cloned()
            .unwrap_or_default();
        if self.repo_meta(&siblings).is_some() {
            return;
        }
        self.discover
            .metas
            .insert(url.clone(), Err("reading header…".into()));
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let r = crate::gguf::read_meta_url(&url);
            let _ = tx.send(Ev::Meta(url, r));
        });
    }

    fn bench(&mut self) {
        if self.benchmarking {
            return;
        }
        self.benchmarking = true;
        self.toast("measuring memory bandwidth: reading a big buffer from every core…");
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let _ = tx.send(Ev::Bandwidth(hardware::bench_bandwidth()));
        });
    }

    // ------------------------------------------------------------ launching

    pub fn open_plan_catalog(&mut self, model: Model, quant: Quant) {
        let quants: Vec<Quant> = Quant::ALL
            .iter()
            .copied()
            .filter(|q| !self.hw.is_apple() || q.mlx_tag().is_some())
            .collect();
        let src = Source::Catalog { model, quant };
        let engines = LaunchPlan::engines_for(&src);
        let engine_idx = self.preferred_engine(&engines);
        self.modal = Some(Modal::Plan(Box::new(PlanModal {
            name: model.name.to_string(),
            model: Some(model),
            quant_idx: quants.iter().position(|q| *q == quant).unwrap_or(0),
            quants,
            fixed: None,
            engines,
            engine_idx,
            context: self.state.settings.context.min(model.max_context),
            gpu_layers: None,
            row: 0,
            file_bytes: None,
            meta: None,
        })));
    }

    pub fn open_plan_source(
        &mut self,
        name: String,
        src: Source,
        file_bytes: Option<u64>,
        meta: Option<Meta>,
    ) {
        let engines = LaunchPlan::engines_for(&src);
        let engine_idx = self.preferred_engine(&engines);
        let ctx = meta
            .as_ref()
            .and_then(|m| m.context_length)
            .map(|c| c.min(self.state.settings.context))
            .unwrap_or(self.state.settings.context);
        self.modal = Some(Modal::Plan(Box::new(PlanModal {
            name,
            model: None,
            quants: vec![],
            quant_idx: 0,
            fixed: Some(src),
            engines,
            engine_idx,
            context: ctx,
            gpu_layers: None,
            row: 0,
            file_bytes,
            meta,
        })));
    }

    /// What the plan modal shows: memory, speed, layers, download.
    pub fn plan_numbers(&self, p: &PlanModal) -> PlanNumbers {
        let engine = p.engines[p.engine_idx];
        if let Some(m) = p.model {
            let q = p.quants[p.quant_idx];
            let e = self.estimate_for(&m, q, p.context);
            let max_ngl = m.layers;
            let gpu_layers = p.gpu_layers.or(e.gpu_layers);
            let mut tok_s = e.tok_s;
            if let (Some(user), true) = (p.gpu_layers, self.hw.has_discrete_gpu()) {
                let frac = user as f64 / m.layers as f64;
                tok_s = budget::offload_tok_s(
                    &self.hw,
                    budget::active_weight_gib(&m, q) + budget::kv_gib(&m, budget::SPEED_CONTEXT),
                    frac,
                );
            }
            return PlanNumbers {
                total_gib: e.total_gib,
                weights_gib: e.weight_gib,
                kv_gib: e.kv_gib,
                fits: e.fits,
                offload: e.offload,
                tok_s: Some(tok_s),
                gpu_layers: if engine == EngineKind::LlamaCpp && self.hw.has_discrete_gpu() {
                    gpu_layers
                } else {
                    None
                },
                max_layers: Some(max_ngl),
                download: Some((m.params_b * 1e9 * q.bytes_per_param()) as u64),
                exact: true,
            };
        }
        // A file from Hugging Face or disk: size is known, shape comes from the GGUF header if we read it.
        let bytes = p.file_bytes.unwrap_or(0) as f64;
        let gib = bytes / (1u64 << 30) as f64;
        let (kv, active, layers, exact) = match &p.meta {
            Some(m) => (
                m.kv_bytes_per_token()
                    .map(|b| b * p.context as f64 / (1u64 << 30) as f64)
                    .unwrap_or(gib * 0.12),
                gib * m.active_fraction(),
                m.layers,
                m.kv_bytes_per_token().is_some(),
            ),
            None => (gib * 0.12 * p.context as f64 / 8192.0, gib, None, false),
        };
        let total = gib * 1.03 + kv;
        let fits = total <= self.hw.capacity_gib();
        let in_vram = total <= self.hw.usable_gib;
        let kv2k = kv * budget::SPEED_CONTEXT as f64 / p.context.max(1) as f64;
        let (tok_s, ngl) = match (self.hw.has_discrete_gpu(), in_vram, layers) {
            (true, false, Some(l)) => {
                let n = p
                    .gpu_layers
                    .unwrap_or_else(|| budget::gpu_layers_that_fit(&self.hw, l, total));
                (
                    budget::offload_tok_s(&self.hw, active + kv2k, n as f64 / l as f64),
                    Some(n),
                )
            }
            (true, _, Some(l)) => (
                budget::predicted_tok_s(&self.hw, active, kv2k),
                Some(p.gpu_layers.unwrap_or(l)),
            ),
            _ => (budget::predicted_tok_s(&self.hw, active, kv2k), None),
        };
        PlanNumbers {
            total_gib: total,
            weights_gib: gib * 1.03,
            kv_gib: kv,
            fits,
            offload: fits && !in_vram && self.hw.has_discrete_gpu(),
            tok_s: (bytes > 0.0).then_some(tok_s),
            gpu_layers: if engine == EngineKind::LlamaCpp && self.hw.has_discrete_gpu() {
                ngl
            } else {
                None
            },
            max_layers: layers,
            // only Hugging Face sources download anything
            download: if matches!(p.fixed, Some(Source::HfGguf { .. })) {
                p.file_bytes
            } else {
                None
            },
            exact,
        }
    }

    pub fn plan_rows(&self, p: &PlanModal) -> Vec<PlanRow> {
        let mut v = vec![];
        if p.model.is_some() {
            v.push(PlanRow::Quant);
        }
        v.push(PlanRow::Engine);
        v.push(PlanRow::Context);
        if p.engines[p.engine_idx] == EngineKind::LlamaCpp && self.hw.has_discrete_gpu() {
            v.push(PlanRow::GpuLayers);
        }
        v
    }

    fn start_plan(&mut self) {
        let Some(Modal::Plan(p)) = self.modal.take() else {
            return;
        };
        let nums = self.plan_numbers(&p);
        let engine = p.engines[p.engine_idx];
        let source = match (&p.fixed, p.model) {
            (Some(s), _) => s.clone(),
            (None, Some(m)) => Source::Catalog {
                model: m,
                quant: p.quants[p.quant_idx],
            },
            _ => return,
        };
        let measure_key = p
            .model
            .map(|m| State::key(m.name, p.quants[p.quant_idx].tag()));
        if let (Some(m), Source::Catalog { quant, .. }) = (p.model, &source) {
            self.quant_override.insert(m.name, *quant);
        }
        let plan = LaunchPlan {
            name: match &source {
                Source::Catalog { model, quant } => format!("{} {}", model.name, quant.tag()),
                _ => p.name.clone(),
            },
            engine,
            source,
            context: p.context,
            gpu_layers: nums.gpu_layers,
            predicted_tok_s: nums.tok_s,
            predicted_gib: Some(nums.total_gib),
            download_bytes: nums.download,
            measure_key,
            port: self.state.settings.port,
            variant: self.variant,
            temperature: self.state.settings.temperature,
        };
        self.launch(plan);
    }

    pub fn launch(&mut self, plan: LaunchPlan) {
        if let Some(j) = &self.job {
            if j.running() {
                self.toast("a launch is already running. cancel it first (c on the Launch tab)");
                return;
            }
        }
        self.stop_generation();
        let previous = self.session.take();
        let id = self.next_job;
        self.next_job += 1;
        let cancel = Arc::new(AtomicBool::new(false));
        self.job = Some(Job {
            id,
            plan: plan.clone(),
            steps: pipeline::STEPS
                .iter()
                .map(|t| StepState {
                    title: t,
                    status: StepStatus::Pending,
                    detail: String::new(),
                    lines: VecDeque::new(),
                    progress: None,
                    started: None,
                    ended: None,
                })
                .collect(),
            cancel: cancel.clone(),
            finished: None,
            started: Instant::now(),
        });
        self.logln(format!("launch: {} with {}", plan.name, plan.engine.name()));
        self.tab = Tab::Run;
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let emit_tx = tx.clone();
            let emit: Arc<dyn Fn(JobEvent) + Send + Sync> = Arc::new(move |e| {
                let _ = emit_tx.send(Ev::Job(id, e));
            });
            if let Some(mut prev) = previous {
                emit(JobEvent::Line(0, format!("unloading {} first", prev.name)));
                prev.stop();
            }
            pipeline::run(plan, cancel, emit);
        });
    }

    fn retry(&mut self) {
        if let Some(j) = &self.job {
            if !j.running() {
                let plan = j.plan.clone();
                self.launch(plan);
            }
        }
    }

    pub fn stop_session(&mut self) {
        self.stop_generation();
        if let Some(mut s) = self.session.take() {
            let name = s.name.clone();
            std::thread::spawn(move || s.stop());
            self.toast(format!("stopped {name}"));
            self.refresh_library();
        }
    }

    // ------------------------------------------------------------ chat

    pub fn send_chat(&mut self) {
        let text = self.chat.input.trim().to_string();
        if text.is_empty() {
            return;
        }
        if let Some(cmd) = text.strip_prefix('/') {
            self.chat.input.clear();
            self.chat.cursor = 0;
            self.chat_command(cmd);
            return;
        }
        let Some(sess) = &self.session else {
            self.toast("no model is loaded. pick one on the Models tab (2) and press enter");
            return;
        };
        if self.chat.generating.is_some() {
            return;
        }
        self.chat.history.push(text.clone());
        self.chat.history_idx = None;
        self.chat.input.clear();
        self.chat.cursor = 0;
        self.chat.scroll = 0;
        self.chat.entries.push(ChatEntry {
            role: "user".into(),
            content: text,
            thinking: String::new(),
            stats: None,
            error: None,
        });
        let mut msgs = vec![];
        if !self.state.settings.system_prompt.trim().is_empty() {
            msgs.push(Msg::new(
                "system",
                self.state.settings.system_prompt.clone(),
            ));
        }
        for e in &self.chat.entries {
            if e.error.is_none() && !(e.role == "assistant" && e.content.is_empty()) {
                msgs.push(Msg::new(&e.role, e.content.clone()));
            }
        }
        self.chat.entries.push(ChatEntry {
            role: "assistant".into(),
            content: String::new(),
            thinking: String::new(),
            stats: None,
            error: None,
        });
        let id = self.next_chat;
        self.next_chat += 1;
        let cancel = Arc::new(AtomicBool::new(false));
        self.chat.generating = Some((id, cancel.clone()));
        let (api, base, model) = (
            sess.api.clone(),
            sess.base_url.clone(),
            sess.model_id.clone(),
        );
        let opts = engine::ChatOpts {
            temperature: self.state.settings.temperature,
            num_ctx: sess.context,
            max_tokens: None,
        };
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let t2 = tx.clone();
            let r = engine::chat(&api, &base, &model, &msgs, &opts, &cancel, move |c| {
                let _ = t2.send(Ev::Chat(id, ChatEv::Chunk(c)));
            });
            let _ = tx.send(Ev::Chat(
                id,
                match r {
                    Ok(s) => ChatEv::Done(s),
                    Err(e) => ChatEv::Error(e),
                },
            ));
        });
    }

    fn chat_command(&mut self, cmd: &str) {
        let (name, arg) = cmd
            .split_once(' ')
            .map(|(a, b)| (a, b.trim()))
            .unwrap_or((cmd, ""));
        match name {
            "clear" => {
                self.chat.entries.clear();
                self.toast("chat cleared");
            }
            "system" if !arg.is_empty() => {
                self.state.settings.system_prompt = arg.to_string();
                self.state.save();
                self.toast("system prompt set for the next message");
            }
            "system" => self.toast(format!(
                "system prompt: {}",
                self.state.settings.system_prompt
            )),
            "temp" => match arg.parse::<f32>() {
                Ok(t) if (0.0..=2.0).contains(&t) => {
                    self.state.settings.temperature = t;
                    self.state.save();
                    self.toast(format!("temperature {t}"));
                }
                _ => self.toast("usage: /temp 0.7  (0 to 2)"),
            },
            "save" => match self.save_chat() {
                Ok(p) => self.toast(format!("saved {}", p.display())),
                Err(e) => self.toast(format!("save failed: {e}")),
            },
            "think" => {
                self.state.settings.show_thinking = !self.state.settings.show_thinking;
                self.state.save();
                self.toast(if self.state.settings.show_thinking {
                    "showing thinking"
                } else {
                    "hiding thinking"
                });
            }
            "copy" => {
                if let Some(e) = self
                    .chat
                    .entries
                    .iter()
                    .rev()
                    .find(|e| e.role == "assistant")
                {
                    theme_copy(&e.content);
                    self.toast("last reply copied to the clipboard");
                }
            }
            _ => {
                self.toast("commands: /clear  /system <prompt>  /temp <0-2>  /think  /copy  /save")
            }
        }
    }

    fn save_chat(&self) -> Result<PathBuf, String> {
        let dir = state::home().join("chats");
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let p = dir.join(format!("chat-{}.md", state::now()));
        let model = self
            .session
            .as_ref()
            .map(|s| s.name.clone())
            .unwrap_or_default();
        let mut md = format!("# chat with {model}\n\n");
        for e in &self.chat.entries {
            md.push_str(&format!(
                "**{}**\n\n{}\n\n",
                if e.role == "user" { "you" } else { &model },
                e.content
            ));
        }
        std::fs::write(&p, md).map_err(|e| e.to_string())?;
        Ok(p)
    }

    pub fn stop_generation(&mut self) {
        if let Some((_, c)) = self.chat.generating.take() {
            c.store(true, Ordering::Relaxed);
        }
    }

    /// Rough tokens in the conversation (4 chars per token), for the context gauge.
    pub fn chat_tokens(&self) -> usize {
        let chars: usize = self
            .chat
            .entries
            .iter()
            .map(|e| e.content.len() + 8)
            .sum::<usize>()
            + self.state.settings.system_prompt.len();
        chars / 4
    }

    // ------------------------------------------------------------ events

    pub fn on_event(&mut self, ev: Ev) {
        match ev {
            Ev::Job(id, e) => self.on_job(id, e),
            Ev::Engines(v) => {
                self.engines = v;
                if let Some(s) = &self.session {
                    for e in self.engines.iter_mut() {
                        if e.kind == s.engine {
                            e.running = true;
                        }
                    }
                }
            }
            Ev::Library(v) => {
                self.library = v;
                self.library_loading = false;
                self.library_sel = self.library_sel.min(self.library.len().saturating_sub(1));
            }
            Ev::Search(seq, r) => {
                if seq != self.discover.seq {
                    return;
                }
                self.discover.searching = false;
                match r {
                    Ok(v) => {
                        self.discover.results = v;
                        self.discover.sel = 0;
                        self.discover.file_sel = 0;
                        if self.discover.results.is_empty() {
                            self.discover.error = Some("nothing on Hugging Face matches. try a model family: qwen, llama, gemma, mistral, phi".into());
                        } else {
                            self.load_files();
                        }
                    }
                    Err(e) => self.discover.error = Some(e),
                }
            }
            Ev::Files(repo, r) => {
                self.discover.files.insert(repo, r);
                self.load_meta();
            }
            Ev::Meta(url, r) => {
                self.discover.metas.insert(url, r);
            }
            Ev::Bandwidth(bw) => {
                self.benchmarking = false;
                self.state.measured_bandwidth = Some(bw);
                self.state.save();
                self.hw.apply_state(&self.state);
                self.recompute();
                self.toast(format!(
                    "measured {bw:.0} GB/s of memory bandwidth. predictions updated"
                ));
            }
            Ev::Live(l) => {
                push_hist(&mut self.cpu_hist, l.cpu_pct.round() as u64);
                self.live = l;
            }
            Ev::Chat(id, e) => self.on_chat(id, e),
            Ev::Install(e) => match e {
                InstallEv::Line(l) => {
                    self.logln(format!("llama.cpp install: {l}"));
                    if let Some((_, lines)) = &mut self.installing {
                        lines.push(l);
                    }
                }
                InstallEv::Progress(p) => {
                    if let Some((prog, _)) = &mut self.installing {
                        *prog = Some(p);
                    }
                }
                InstallEv::Done(r) => {
                    self.installing = None;
                    match r {
                        Ok(p) => self.toast(format!("llama.cpp installed: {}", p.display())),
                        Err(e) => self.toast(format!("llama.cpp install failed: {e}")),
                    }
                    self.refresh_engines();
                }
            },
            Ev::Toast(s) => self.toast(s),
        }
    }

    fn on_job(&mut self, id: u64, e: JobEvent) {
        // Events from a job we already replaced: only a session is worth keeping (to stop it).
        if self.job.as_ref().map(|j| j.id) != Some(id) {
            if let JobEvent::Ready(mut s) = e {
                std::thread::spawn(move || s.stop());
            }
            return;
        }
        let mut log = None;
        let e_kind = if matches!(e, JobEvent::Skip(..)) {
            Kind::Skip
        } else {
            Kind::Other
        };
        let Some(job) = self.job.as_mut() else { return };
        match e {
            JobEvent::Start(i, d) => {
                let st = &mut job.steps[i];
                st.status = StepStatus::Running;
                st.detail = d;
                st.started = Some(Instant::now());
            }
            JobEvent::Progress(i, p, label) => {
                let st = &mut job.steps[i];
                if st.status == StepStatus::Pending {
                    st.status = StepStatus::Running;
                    st.started = Some(Instant::now());
                }
                st.progress = Some((p, label));
            }
            // the server keeps talking after it loaded; that belongs in the log, not the Load block
            JobEvent::Line(i, l) if job.steps[i].status == StepStatus::Done => log = Some(l),
            JobEvent::Line(i, l) => {
                let st = &mut job.steps[i];
                st.lines.push_back(l.clone());
                while st.lines.len() > 400 {
                    st.lines.pop_front();
                }
                log = Some(l);
            }
            JobEvent::Done(i, d) | JobEvent::Skip(i, d) if i < job.steps.len() => {
                let skipped = matches!(e_kind, Kind::Skip);
                let st = &mut job.steps[i];
                st.status = if skipped {
                    StepStatus::Skipped
                } else {
                    StepStatus::Done
                };
                st.detail = d.clone();
                st.progress = None;
                st.ended = Some(Instant::now());
                log = Some(format!("{}: {d}", st.title));
            }
            JobEvent::Done(..) | JobEvent::Skip(..) => {}
            JobEvent::Fail(i, err) => {
                let st = &mut job.steps[i];
                st.status = StepStatus::Failed;
                st.detail = err.clone();
                st.progress = None;
                st.ended = Some(Instant::now());
                log = Some(format!("{} failed: {err}", st.title));
            }
            JobEvent::Measured {
                tok_s,
                predicted,
                key,
                engine,
            } => {
                if let (Some(k), Some(pred)) = (key, predicted) {
                    // Store the prediction at base efficiency so calibration does not feed on itself.
                    let base_pred = pred * budget::EFFICIENCY / self.hw.efficiency;
                    self.state.measurements.insert(
                        k,
                        Measurement {
                            tok_s,
                            predicted: base_pred,
                            engine,
                            when: state::now(),
                        },
                    );
                    self.state.save();
                    self.hw.efficiency = budget::EFFICIENCY;
                    self.hw.apply_state(&self.state);
                    self.recompute();
                }
            }
            JobEvent::Ready(s) => {
                let name = s.name.clone();
                let url = s.openai_url();
                self.session = Some(s);
                self.chat.entries.clear();
                self.chat.focused = true;
                self.toast(format!("{name} is ready at {url}. press enter to chat"));
                for e in self.engines.iter_mut() {
                    e.running =
                        e.running || Some(e.kind) == self.session.as_ref().map(|s| s.engine);
                }
            }
            JobEvent::Finished(ok) => {
                job.finished = Some(ok);
                if !ok && job.cancel.load(Ordering::Relaxed) {
                    log = Some("launch cancelled".into());
                }
                self.refresh_library();
            }
        }
        if let Some(l) = log {
            self.logln(l);
        }
    }

    fn on_chat(&mut self, id: u64, e: ChatEv) {
        let current = self.chat.generating.as_ref().map(|g| g.0) == Some(id);
        let Some(last) = self.chat.entries.last_mut() else {
            return;
        };
        if last.role != "assistant" {
            return;
        }
        match e {
            ChatEv::Chunk(Chunk::Content(t)) if current => last.content.push_str(&t),
            ChatEv::Chunk(Chunk::Thinking(t)) if current => last.thinking.push_str(&t),
            ChatEv::Chunk(_) => {}
            ChatEv::Done(st) => {
                if current || last.stats.is_none() {
                    last.stats = Some(st.clone());
                }
                if current {
                    self.chat.generating = None;
                }
                // Long replies are the best speed measurements we get.
                if st.tokens >= 32 && st.tok_s > 0.0 {
                    if let Some(s) = &self.session {
                        if let (Some(k), Some(pred)) = (&s.measure_key, s.predicted_tok_s) {
                            let base_pred =
                                pred * budget::EFFICIENCY / self.hw.efficiency.max(0.01);
                            self.state.measurements.insert(
                                k.clone(),
                                Measurement {
                                    tok_s: st.tok_s,
                                    predicted: base_pred,
                                    engine: s.engine.name().into(),
                                    when: state::now(),
                                },
                            );
                            self.state.save();
                        }
                    }
                }
            }
            ChatEv::Error(err) => {
                last.error = Some(err);
                if current {
                    self.chat.generating = None;
                }
            }
        }
    }

    // ------------------------------------------------------------ keys

    pub fn typing(&self) -> bool {
        (self.tab == Tab::Chat && self.chat.focused)
            || (self.tab == Tab::Discover && self.discover.focus == DFocus::Search)
            || (self.tab == Tab::Models && self.filtering)
            || matches!(
                self.modal,
                Some(Modal::Settings {
                    editing: Some(_),
                    ..
                })
            )
    }

    pub fn on_key(&mut self, k: KeyEvent) {
        if k.modifiers.contains(KeyModifiers::CONTROL) && k.code == KeyCode::Char('c') {
            if self.tab == Tab::Chat && self.chat.generating.is_some() {
                self.stop_generation();
                return;
            }
            self.ask_quit();
            return;
        }
        if self.modal.is_some() {
            self.on_modal_key(k);
            return;
        }
        // alt+1..7 switches tabs from anywhere, even mid-sentence
        if k.modifiers.contains(KeyModifiers::ALT) {
            if let KeyCode::Char(c @ '1'..='7') = k.code {
                self.goto(Tab::ALL[(c as u8 - b'1') as usize]);
                return;
            }
            if matches!(k.code, KeyCode::Char(_)) {
                return; // never type alt-chords into an input
            }
        }
        if !self.typing() {
            match k.code {
                KeyCode::Char(c @ '1'..='7') => {
                    self.goto(Tab::ALL[(c as u8 - b'1') as usize]);
                    return;
                }
                KeyCode::Char('q') => {
                    self.ask_quit();
                    return;
                }
                KeyCode::Char('?') => {
                    self.modal = Some(Modal::Help);
                    return;
                }
                KeyCode::Char(',') => {
                    self.modal = Some(Modal::Settings {
                        row: 0,
                        editing: None,
                    });
                    return;
                }
                KeyCode::Char('L') => {
                    self.show_log = !self.show_log;
                    return;
                }
                _ => {}
            }
        }
        match k.code {
            KeyCode::Tab => {
                self.goto(Tab::ALL[(self.tab.index() + 1) % Tab::ALL.len()]);
                return;
            }
            KeyCode::BackTab => {
                self.goto(Tab::ALL[(self.tab.index() + Tab::ALL.len() - 1) % Tab::ALL.len()]);
                return;
            }
            _ => {}
        }
        match self.tab {
            Tab::Machine => self.key_machine(k),
            Tab::Models => self.key_models(k),
            Tab::Discover => self.key_discover(k),
            Tab::Library => self.key_library(k),
            Tab::Run => self.key_run(k),
            Tab::Chat => self.key_chat(k),
            Tab::Engines => self.key_engines(k),
        }
    }

    /// Pasted text lands in whatever is being typed into, newlines and all, without sending anything.
    pub fn on_paste(&mut self, text: &str) {
        let text = text.replace("\r\n", "\n").replace('\r', "\n");
        if let Some(Modal::Settings {
            editing: Some(buf), ..
        }) = &mut self.modal
        {
            buf.push_str(&text.replace('\n', " "));
            return;
        }
        if self.modal.is_some() {
            return;
        }
        match self.tab {
            Tab::Chat => {
                self.chat.focused = true;
                self.chat.input.insert_str(self.chat.cursor, &text);
                self.chat.cursor += text.len();
            }
            Tab::Discover => {
                self.discover.focus = DFocus::Search;
                self.discover
                    .query
                    .push_str(text.lines().next().unwrap_or("").trim());
            }
            Tab::Models if self.filtering => self
                .models_filter
                .push_str(text.lines().next().unwrap_or("").trim()),
            _ => {}
        }
    }

    pub fn goto(&mut self, t: Tab) {
        self.tab = t;
        match t {
            Tab::Library => self.refresh_library(),
            Tab::Engines => self.refresh_engines(),
            Tab::Discover if !self.discover.loaded_once => {
                self.discover.focus = DFocus::Results;
                self.search();
            }
            Tab::Chat => self.chat.focused = true,
            _ => {}
        }
    }

    fn ask_quit(&mut self) {
        let running = self
            .session
            .as_ref()
            .filter(|s| s.child.is_some())
            .map(|s| s.name.clone());
        let busy = self.job.as_ref().map(|j| j.running()).unwrap_or(false);
        if running.is_none() && !busy {
            self.quit = true;
            return;
        }
        let mut body = vec![];
        if let Some(n) = running {
            body.push(format!("{n} is loaded and serving."));
            body.push("y  stop it and quit".into());
            body.push("k  quit but keep the server running in the background".into());
        }
        if busy {
            body.push(
                "a launch is in progress; it will be cancelled (downloads resume next time)."
                    .into(),
            );
        }
        self.modal = Some(Modal::Confirm(Confirm {
            title: "Quit tokenpati?".into(),
            body,
            yes: "quit",
            action: ConfirmAction::Quit,
        }));
    }

    fn key_machine(&mut self, k: KeyEvent) {
        match k.code {
            KeyCode::Char('b') => self.bench(),
            KeyCode::Enter => {
                if let Some(w) = self.winner() {
                    let (m, q) = (w.estimate.model, self.quant_for(w));
                    self.open_plan_catalog(m, q);
                } else {
                    self.goto(Tab::Models);
                }
            }
            KeyCode::Char('r') => {
                self.hw = hardware::detect();
                self.hw.apply_state(&self.state);
                self.recompute();
                self.toast("hardware re-detected");
            }
            _ => {}
        }
    }

    fn change_context(&mut self, up: bool) {
        let c = self.state.settings.context;
        self.state.settings.context = if up {
            (c * 2).min(262_144)
        } else {
            (c / 2).max(512)
        };
        self.state.save();
        self.recompute();
        self.toast(format!(
            "budgeting for {} tokens of context",
            self.state.settings.context
        ));
    }

    fn key_models(&mut self, k: KeyEvent) {
        if self.filtering {
            match k.code {
                KeyCode::Esc => {
                    self.filtering = false;
                    self.models_filter.clear();
                }
                KeyCode::Enter => self.filtering = false,
                KeyCode::Backspace => {
                    self.models_filter.pop();
                }
                KeyCode::Char(c) => {
                    self.models_filter.push(c);
                    self.models_sel = 0;
                }
                KeyCode::Down => self.models_sel += 1,
                KeyCode::Up => self.models_sel = self.models_sel.saturating_sub(1),
                _ => {}
            }
            self.models_sel = self
                .models_sel
                .min(self.visible_models().len().saturating_sub(1));
            return;
        }
        let n = self.visible_models().len();
        match k.code {
            KeyCode::Down | KeyCode::Char('j') => {
                self.models_sel = (self.models_sel + 1).min(n.saturating_sub(1))
            }
            KeyCode::Up | KeyCode::Char('k') => self.models_sel = self.models_sel.saturating_sub(1),
            KeyCode::PageDown => self.models_sel = (self.models_sel + 10).min(n.saturating_sub(1)),
            KeyCode::PageUp => self.models_sel = self.models_sel.saturating_sub(10),
            KeyCode::Home | KeyCode::Char('g') => self.models_sel = 0,
            KeyCode::End | KeyCode::Char('G') => self.models_sel = n.saturating_sub(1),
            KeyCode::Char('/') => {
                self.filtering = true;
                self.models_filter.clear();
            }
            KeyCode::Char('f') => {
                self.only_runnable = !self.only_runnable;
                self.models_sel = 0;
            }
            KeyCode::Char('+') | KeyCode::Char('=') | KeyCode::Char(']') => {
                self.change_context(true)
            }
            KeyCode::Char('-') | KeyCode::Char('[') => self.change_context(false),
            KeyCode::Left | KeyCode::Right | KeyCode::Char('h') | KeyCode::Char('l') => {
                let Some(r) = self
                    .visible_models()
                    .get(self.models_sel)
                    .map(|r| (*r).clone())
                else {
                    return;
                };
                let qs: Vec<Quant> = Quant::ALL
                    .iter()
                    .copied()
                    .filter(|q| !self.hw.is_apple() || q.mlx_tag().is_some())
                    .collect();
                let cur = qs
                    .iter()
                    .position(|q| *q == self.quant_for(&r))
                    .unwrap_or(0);
                let next = if matches!(k.code, KeyCode::Right | KeyCode::Char('l')) {
                    (cur + 1).min(qs.len() - 1)
                } else {
                    cur.saturating_sub(1)
                };
                self.quant_override.insert(r.estimate.model.name, qs[next]);
            }
            KeyCode::Enter => {
                if let Some(r) = self
                    .visible_models()
                    .get(self.models_sel)
                    .map(|r| (*r).clone())
                {
                    let q = self.quant_for(&r);
                    self.open_plan_catalog(r.estimate.model, q);
                }
            }
            KeyCode::Esc => {
                self.models_filter.clear();
                self.only_runnable = false;
            }
            _ => {}
        }
    }

    fn key_discover(&mut self, k: KeyEvent) {
        let d = &mut self.discover;
        match d.focus {
            DFocus::Search => match k.code {
                KeyCode::Enter => {
                    d.focus = DFocus::Results;
                    self.search();
                }
                KeyCode::Esc | KeyCode::Down => d.focus = DFocus::Results,
                KeyCode::Backspace => {
                    d.query.pop();
                }
                KeyCode::Char(c) => d.query.push(c),
                _ => {}
            },
            DFocus::Results => match k.code {
                KeyCode::Char('/') | KeyCode::Char('s') => d.focus = DFocus::Search,
                KeyCode::Down | KeyCode::Char('j') => {
                    if d.sel + 1 < d.results.len() {
                        d.sel += 1;
                        d.file_sel = 0;
                        self.load_files();
                        self.load_meta();
                    }
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    if d.sel > 0 {
                        d.sel -= 1;
                        d.file_sel = 0;
                        self.load_files();
                        self.load_meta();
                    } else {
                        d.focus = DFocus::Search;
                    }
                }
                KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                    d.focus = DFocus::Files;
                    self.load_files();
                    self.load_meta();
                }
                KeyCode::Char('r') => self.search(),
                _ => {}
            },
            DFocus::Files => {
                let n = d
                    .results
                    .get(d.sel)
                    .and_then(|h| d.files.get(&h.id))
                    .and_then(|r| r.as_ref().ok())
                    .map(|v| v.len())
                    .unwrap_or(0);
                match k.code {
                    KeyCode::Esc | KeyCode::Left | KeyCode::Char('h') => d.focus = DFocus::Results,
                    KeyCode::Down | KeyCode::Char('j') => {
                        d.file_sel = (d.file_sel + 1).min(n.saturating_sub(1));
                        self.load_meta();
                    }
                    KeyCode::Up | KeyCode::Char('k') => {
                        d.file_sel = d.file_sel.saturating_sub(1);
                        self.load_meta();
                    }
                    KeyCode::Char('/') => d.focus = DFocus::Search,
                    KeyCode::Enter => {
                        if let Some(c) = self.selected_choice().cloned() {
                            let all: Vec<GgufChoice> = self
                                .discover
                                .results
                                .get(self.discover.sel)
                                .and_then(|h| self.discover.files.get(&h.id))
                                .and_then(|r| r.as_ref().ok())
                                .cloned()
                                .unwrap_or_default();
                            let meta = self.repo_meta(&all).cloned();
                            let name = format!(
                                "{} {}",
                                c.repo
                                    .rsplit('/')
                                    .next()
                                    .unwrap_or(&c.repo)
                                    .trim_end_matches("-GGUF"),
                                c.quant.clone().unwrap_or_default()
                            );
                            self.open_plan_source(
                                name,
                                Source::HfGguf { choice: c.clone() },
                                Some(c.size()),
                                meta,
                            );
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    fn key_library(&mut self, k: KeyEvent) {
        let n = self.library.len();
        match k.code {
            KeyCode::Down | KeyCode::Char('j') => {
                self.library_sel = (self.library_sel + 1).min(n.saturating_sub(1))
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.library_sel = self.library_sel.saturating_sub(1)
            }
            KeyCode::Char('r') => {
                self.refresh_library();
                self.toast("refreshing library");
            }
            KeyCode::Char('x') => self.stop_session(),
            KeyCode::Char('d') | KeyCode::Delete => {
                if let Some(m) = self.library.get(self.library_sel).cloned() {
                    if !m.ours {
                        self.toast("that file belongs to LM Studio; tokenpati only reads it");
                        return;
                    }
                    self.modal = Some(Modal::Confirm(Confirm {
                        title: format!("Delete {}?", m.name),
                        body: vec![
                            format!("engine  {}", m.engine.name()),
                            format!("frees   {}", state::human_bytes(m.size)),
                            format!("where   {}", m.id),
                            String::new(),
                            "you can download it again any time.".into(),
                        ],
                        yes: "delete",
                        action: ConfirmAction::Delete(m),
                    }));
                }
            }
            KeyCode::Enter => {
                let Some(m) = self.library.get(self.library_sel).cloned() else {
                    return;
                };
                let src = match m.engine {
                    EngineKind::Ollama => Source::OllamaTag { tag: m.id.clone() },
                    EngineKind::LlamaCpp => Source::LocalGguf {
                        path: PathBuf::from(&m.id),
                    },
                    EngineKind::Mlx => Source::LocalMlx {
                        path: PathBuf::from(&m.id),
                    },
                };
                let meta = (m.engine == EngineKind::LlamaCpp)
                    .then(|| crate::gguf::read_meta_file(std::path::Path::new(&m.id)).ok())
                    .flatten();
                self.open_plan_source(m.name.clone(), src, Some(m.size), meta);
            }
            _ => {}
        }
    }

    fn key_run(&mut self, k: KeyEvent) {
        match k.code {
            KeyCode::Char('c') | KeyCode::Esc => {
                if self.job.as_ref().map(|j| j.running()).unwrap_or(false) {
                    self.modal = Some(Modal::Confirm(Confirm {
                        title: "Cancel this launch?".into(),
                        body: vec!["partial downloads are kept and resume next time.".into()],
                        yes: "cancel launch",
                        action: ConfirmAction::CancelJob,
                    }));
                }
            }
            KeyCode::Char('r') => self.retry(),
            KeyCode::Enter => {
                if self.session.is_some() {
                    self.goto(Tab::Chat);
                }
            }
            KeyCode::Char('y') => {
                if let Some(s) = &self.session {
                    let url = s.openai_url();
                    theme_copy(&url);
                    self.toast(format!("copied {url}"));
                }
            }
            KeyCode::Char('x') if self.session.is_some() => {
                self.modal = Some(Modal::Confirm(Confirm {
                    title: "Unload the model?".into(),
                    body: vec!["frees its memory. the weights stay on disk.".into()],
                    yes: "unload",
                    action: ConfirmAction::StopSession,
                }));
            }
            _ => {}
        }
    }

    fn key_chat(&mut self, k: KeyEvent) {
        let c = &mut self.chat;
        if !c.focused {
            match k.code {
                KeyCode::Enter | KeyCode::Char('i') => c.focused = true,
                KeyCode::Up | KeyCode::Char('k') => c.scroll = c.scroll.saturating_add(1),
                KeyCode::Down | KeyCode::Char('j') => c.scroll = c.scroll.saturating_sub(1),
                KeyCode::PageUp => c.scroll = c.scroll.saturating_add(10),
                KeyCode::PageDown => c.scroll = c.scroll.saturating_sub(10),
                KeyCode::Char('x') => self.stop_session(),
                KeyCode::Char('t') => self.chat_command("think"),
                _ => {}
            }
            return;
        }
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        match k.code {
            KeyCode::Esc => {
                if c.generating.is_some() {
                    self.stop_generation();
                } else {
                    c.focused = false;
                }
            }
            KeyCode::Enter
                if k.modifiers.contains(KeyModifiers::ALT)
                    || k.modifiers.contains(KeyModifiers::SHIFT) =>
            {
                c.input.insert(c.cursor, '\n');
                c.cursor += 1;
            }
            KeyCode::Char('j') if ctrl => {
                c.input.insert(c.cursor, '\n');
                c.cursor += 1;
            }
            KeyCode::Enter => self.send_chat(),
            KeyCode::Char('l') if ctrl => self.chat_command("clear"),
            KeyCode::Char('u') if ctrl => {
                c.input.clear();
                c.cursor = 0;
            }
            KeyCode::Backspace => {
                if c.cursor > 0 {
                    let prev = c.input[..c.cursor]
                        .chars()
                        .last()
                        .map(|ch| ch.len_utf8())
                        .unwrap_or(1);
                    c.cursor -= prev;
                    c.input.remove(c.cursor);
                }
            }
            KeyCode::Delete => {
                if c.cursor < c.input.len() {
                    c.input.remove(c.cursor);
                }
            }
            KeyCode::Left => {
                if c.cursor > 0 {
                    c.cursor -= c.input[..c.cursor]
                        .chars()
                        .last()
                        .map(|ch| ch.len_utf8())
                        .unwrap_or(1);
                }
            }
            KeyCode::Right => {
                if c.cursor < c.input.len() {
                    c.cursor += c.input[c.cursor..]
                        .chars()
                        .next()
                        .map(|ch| ch.len_utf8())
                        .unwrap_or(1);
                }
            }
            KeyCode::Home => c.cursor = 0,
            KeyCode::End => c.cursor = c.input.len(),
            KeyCode::Up if c.input.is_empty() || c.history_idx.is_some() => {
                if c.history.is_empty() {
                    return;
                }
                let i = c
                    .history_idx
                    .map(|i| i.saturating_sub(1))
                    .unwrap_or(c.history.len() - 1);
                c.history_idx = Some(i);
                c.input = c.history[i].clone();
                c.cursor = c.input.len();
            }
            KeyCode::Down if c.history_idx.is_some() => {
                let i = c.history_idx.unwrap() + 1;
                if i >= c.history.len() {
                    c.history_idx = None;
                    c.input.clear();
                } else {
                    c.history_idx = Some(i);
                    c.input = c.history[i].clone();
                }
                c.cursor = c.input.len();
            }
            KeyCode::PageUp => c.scroll = c.scroll.saturating_add(10),
            KeyCode::PageDown => c.scroll = c.scroll.saturating_sub(10),
            KeyCode::Char(ch) => {
                c.input.insert(c.cursor, ch);
                c.cursor += ch.len_utf8();
            }
            _ => {}
        }
    }

    fn key_engines(&mut self, k: KeyEvent) {
        let n = self.engines.len();
        match k.code {
            KeyCode::Down | KeyCode::Char('j') => {
                self.engines_sel = (self.engines_sel + 1).min(n.saturating_sub(1))
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.engines_sel = self.engines_sel.saturating_sub(1)
            }
            KeyCode::Char('r') => {
                self.refresh_engines();
                self.toast("re-detecting engines");
            }
            KeyCode::Left | KeyCode::Right | KeyCode::Char('v') => {
                if self.engines.get(self.engines_sel).map(|e| e.kind) == Some(EngineKind::LlamaCpp)
                {
                    let vs = Variant::choices();
                    let i = vs.iter().position(|v| *v == self.variant).unwrap_or(0);
                    let next = if k.code == KeyCode::Left {
                        (i + vs.len() - 1) % vs.len()
                    } else {
                        (i + 1) % vs.len()
                    };
                    self.variant = vs[next];
                }
            }
            KeyCode::Enter => {
                if let Some(e) = self.engines.get(self.engines_sel) {
                    let name = e.kind.name().to_string();
                    self.state.settings.engine = if self.state.settings.engine == name {
                        "auto".into()
                    } else {
                        name
                    };
                    self.state.save();
                    self.toast(format!("preferred engine: {}", self.state.settings.engine));
                }
            }
            KeyCode::Char('s') => {
                if self.engines.get(self.engines_sel).map(|e| e.kind) == Some(EngineKind::Ollama) {
                    let tx = self.tx.clone();
                    std::thread::spawn(move || {
                        let msg = match engine::ollama::ensure_server(|_| {}) {
                            Ok(v) => format!("ollama {v} is running"),
                            Err(e) => e,
                        };
                        let _ = tx.send(Ev::Toast(msg));
                        let _ = tx.send(Ev::Engines(engine::detect_all()));
                    });
                }
            }
            KeyCode::Char('i') => {
                let Some(e) = self.engines.get(self.engines_sel) else {
                    return;
                };
                if !e.kind.supported_here() {
                    self.toast(format!("{} does not run on this machine", e.kind.name()));
                    return;
                }
                match e.kind {
                    EngineKind::LlamaCpp => {
                        self.modal = Some(Modal::Confirm(Confirm {
                            title: format!("Install llama.cpp ({})?", self.variant.name()),
                            body: vec![
                                "downloads the official prebuilt llama-server from".into(),
                                "github.com/ggml-org/llama.cpp/releases into".into(),
                                format!("{}", engine::llamacpp::install_root().display()),
                                String::new(),
                                "no sudo, nothing outside that folder. ←/→ on the card picks the build.".into(),
                            ],
                            yes: "install",
                            action: ConfirmAction::InstallLlama(self.variant),
                        }));
                    }
                    EngineKind::Ollama => {
                        let cmd = if cfg!(target_os = "macos") && engine::which("brew").is_some() {
                            "brew install ollama".to_string()
                        } else if cfg!(target_os = "macos") {
                            "open https://ollama.com/download".to_string()
                        } else {
                            "curl -fsSL https://ollama.com/install.sh | sh".to_string()
                        };
                        self.confirm_shell("Install ollama?", cmd);
                    }
                    EngineKind::Mlx => {
                        self.confirm_shell("Install mlx-lm?", engine::mlx::install_command())
                    }
                }
            }
            _ => {}
        }
    }

    fn confirm_shell(&mut self, title: &str, cmd: String) {
        self.modal = Some(Modal::Confirm(Confirm {
            title: title.into(),
            body: vec![
                "tokenpati will step aside and run this in your terminal:".into(),
                String::new(),
                format!("  $ {cmd}"),
                String::new(),
                "it may ask for your password. you come back here when it ends.".into(),
            ],
            yes: "run it",
            action: ConfirmAction::Shell(cmd),
        }));
    }

    fn install_llama(&mut self, v: Variant) {
        if self.installing.is_some() {
            return;
        }
        self.installing = Some((None, vec![]));
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let cancel = Arc::new(AtomicBool::new(false));
            let t1 = tx.clone();
            let t2 = tx.clone();
            let r = engine::llamacpp::install(
                v,
                &cancel,
                move |l| {
                    let _ = t1.send(Ev::Install(InstallEv::Line(l)));
                },
                move |p| {
                    let _ = t2.send(Ev::Install(InstallEv::Progress(p)));
                },
            );
            let _ = tx.send(Ev::Install(InstallEv::Done(r)));
        });
    }

    // ------------------------------------------------------------ modal keys

    fn on_modal_key(&mut self, k: KeyEvent) {
        let Some(modal) = self.modal.take() else {
            return;
        };
        match modal {
            Modal::Help => {
                if !matches!(
                    k.code,
                    KeyCode::Esc | KeyCode::Char('?') | KeyCode::Char('q') | KeyCode::Enter
                ) {
                    self.modal = Some(Modal::Help);
                }
            }
            Modal::Confirm(c) => match k.code {
                KeyCode::Char('y') | KeyCode::Enter => self.confirmed(c.action),
                KeyCode::Char('k') if matches!(c.action, ConfirmAction::Quit) => {
                    self.keep_running = true;
                    self.quit = true;
                }
                KeyCode::Char('n') | KeyCode::Esc => {}
                _ => self.modal = Some(Modal::Confirm(c)),
            },
            Modal::Plan(mut p) => {
                let rows = self.plan_rows(&p);
                match k.code {
                    KeyCode::Esc => return,
                    KeyCode::Enter => {
                        self.modal = Some(Modal::Plan(p));
                        self.start_plan();
                        return;
                    }
                    KeyCode::Down | KeyCode::Char('j') => p.row = (p.row + 1).min(rows.len() - 1),
                    KeyCode::Up | KeyCode::Char('k') => p.row = p.row.saturating_sub(1),
                    KeyCode::Left | KeyCode::Right | KeyCode::Char('h') | KeyCode::Char('l') => {
                        let right = matches!(k.code, KeyCode::Right | KeyCode::Char('l'));
                        match rows.get(p.row.min(rows.len() - 1)) {
                            Some(PlanRow::Quant) => {
                                p.quant_idx = if right {
                                    (p.quant_idx + 1).min(p.quants.len() - 1)
                                } else {
                                    p.quant_idx.saturating_sub(1)
                                };
                                // mlx has no 5-bit; drop it from the engine list when it cannot run this quant
                                if let Some(m) = p.model {
                                    let cur = p.engines[p.engine_idx];
                                    p.engines = LaunchPlan::engines_for(&Source::Catalog {
                                        model: m,
                                        quant: p.quants[p.quant_idx],
                                    });
                                    p.engine_idx =
                                        p.engines.iter().position(|e| *e == cur).unwrap_or(0);
                                }
                                p.gpu_layers = None;
                            }
                            Some(PlanRow::Engine) => {
                                let n = p.engines.len();
                                p.engine_idx = if right {
                                    (p.engine_idx + 1) % n
                                } else {
                                    (p.engine_idx + n - 1) % n
                                };
                            }
                            Some(PlanRow::Context) => {
                                let max = p
                                    .model
                                    .map(|m| m.max_context)
                                    .or(p.meta.as_ref().and_then(|m| m.context_length))
                                    .unwrap_or(262_144);
                                p.context = if right {
                                    (p.context * 2).min(max)
                                } else {
                                    (p.context / 2).max(512)
                                };
                                p.gpu_layers = None;
                            }
                            Some(PlanRow::GpuLayers) => {
                                let nums = self.plan_numbers(&p);
                                let max = nums.max_layers.unwrap_or(99);
                                let cur = p.gpu_layers.or(nums.gpu_layers).unwrap_or(max);
                                p.gpu_layers = Some(if right {
                                    (cur + 1).min(max)
                                } else {
                                    cur.saturating_sub(1)
                                });
                            }
                            None => {}
                        }
                    }
                    _ => {}
                }
                self.modal = Some(Modal::Plan(p));
            }
            Modal::Settings { row, editing } => self.settings_key(k, row, editing),
        }
    }

    fn confirmed(&mut self, a: ConfirmAction) {
        match a {
            ConfirmAction::Delete(m) => {
                if self
                    .session
                    .as_ref()
                    .map(|s| s.model_id == m.id || s.name == m.name)
                    .unwrap_or(false)
                {
                    self.stop_session();
                }
                match engine::delete(&m) {
                    Ok(()) => self.toast(format!(
                        "deleted {}, freed {}",
                        m.name,
                        state::human_bytes(m.size)
                    )),
                    Err(e) => self.toast(format!("delete failed: {e}")),
                }
                self.refresh_library();
            }
            ConfirmAction::CancelJob => {
                if let Some(j) = &self.job {
                    j.cancel.store(true, Ordering::Relaxed);
                }
                self.toast("cancelling…");
            }
            ConfirmAction::StopSession => self.stop_session(),
            ConfirmAction::Shell(cmd) => self.pending_shell = Some(cmd),
            ConfirmAction::InstallLlama(v) => self.install_llama(v),
            ConfirmAction::Quit => {
                if let Some(j) = &self.job {
                    j.cancel.store(true, Ordering::Relaxed);
                }
                self.quit = true;
            }
        }
    }

    pub const SETTINGS: [&'static str; 8] = [
        "context",
        "temperature",
        "system prompt",
        "engine",
        "first port",
        "show thinking",
        "use calibration",
        "mouse",
    ];

    pub fn setting_value(&self, i: usize) -> String {
        let s = &self.state.settings;
        match i {
            0 => format!("{} tokens", s.context),
            1 => format!("{:.2}", s.temperature),
            2 => s.system_prompt.clone(),
            3 => s.engine.clone(),
            4 => s.port.to_string(),
            5 => if s.show_thinking { "on" } else { "off" }.into(),
            6 => match self.state.calibration() {
                Some(c) => format!(
                    "{}  (measured speed is {:.0}% of predicted)",
                    if s.use_calibration { "on" } else { "off" },
                    c * 100.0
                ),
                None => format!(
                    "{}  (no measurements yet)",
                    if s.use_calibration { "on" } else { "off" }
                ),
            },
            7 => if s.mouse { "on" } else { "off" }.into(),
            _ => String::new(),
        }
    }

    fn settings_key(&mut self, k: KeyEvent, mut row: usize, editing: Option<String>) {
        if let Some(mut buf) = editing {
            match k.code {
                KeyCode::Enter => {
                    match row {
                        2 => self.state.settings.system_prompt = buf,
                        4 => {
                            if let Ok(p) = buf.trim().parse() {
                                self.state.settings.port = p
                            }
                        }
                        _ => {}
                    }
                    self.state.save();
                    self.modal = Some(Modal::Settings { row, editing: None });
                }
                KeyCode::Esc => self.modal = Some(Modal::Settings { row, editing: None }),
                KeyCode::Backspace => {
                    buf.pop();
                    self.modal = Some(Modal::Settings {
                        row,
                        editing: Some(buf),
                    });
                }
                KeyCode::Char(c) => {
                    buf.push(c);
                    self.modal = Some(Modal::Settings {
                        row,
                        editing: Some(buf),
                    });
                }
                _ => {
                    self.modal = Some(Modal::Settings {
                        row,
                        editing: Some(buf),
                    })
                }
            }
            return;
        }
        let s = &mut self.state.settings;
        let right = matches!(
            k.code,
            KeyCode::Right | KeyCode::Char('l') | KeyCode::Enter | KeyCode::Char(' ')
        );
        match k.code {
            KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char(',') => {
                self.state.save();
                self.hw.efficiency = budget::EFFICIENCY;
                self.hw.apply_state(&self.state);
                self.recompute();
                return;
            }
            KeyCode::Down | KeyCode::Char('j') => row = (row + 1).min(Self::SETTINGS.len() - 1),
            KeyCode::Up | KeyCode::Char('k') => row = row.saturating_sub(1),
            KeyCode::Left
            | KeyCode::Right
            | KeyCode::Char('h')
            | KeyCode::Char('l')
            | KeyCode::Enter
            | KeyCode::Char(' ') => match row {
                0 => {
                    s.context = if right {
                        (s.context * 2).min(262_144)
                    } else {
                        (s.context / 2).max(512)
                    }
                }
                1 => {
                    s.temperature =
                        ((s.temperature + if right { 0.1 } else { -0.1 }).clamp(0.0, 2.0) * 10.0)
                            .round()
                            / 10.0
                }
                2 | 4 if k.code == KeyCode::Enter => {
                    let cur = if row == 2 {
                        s.system_prompt.clone()
                    } else {
                        s.port.to_string()
                    };
                    self.modal = Some(Modal::Settings {
                        row,
                        editing: Some(cur),
                    });
                    return;
                }
                3 => {
                    let opts = ["auto", "ollama", "llama.cpp", "mlx"];
                    let i = opts.iter().position(|o| *o == s.engine).unwrap_or(0);
                    s.engine = opts[if right { (i + 1) % 4 } else { (i + 3) % 4 }].into();
                }
                5 => s.show_thinking = !s.show_thinking,
                6 => s.use_calibration = !s.use_calibration,
                7 => s.mouse = !s.mouse,
                _ => {}
            },
            _ => {}
        }
        self.state.save();
        self.modal = Some(Modal::Settings { row, editing: None });
    }

    // ------------------------------------------------------------ mouse

    pub fn on_mouse(&mut self, m: MouseEvent) {
        match m.kind {
            MouseEventKind::Down(_) => {
                if self.modal.is_none() {
                    if let Some((_, _, t)) = self
                        .tab_hits
                        .iter()
                        .find(|(x0, x1, _)| m.row <= 2 && m.column >= *x0 && m.column < *x1)
                    {
                        self.goto(*t);
                    }
                }
            }
            MouseEventKind::ScrollUp | MouseEventKind::ScrollDown => {
                let up = matches!(m.kind, MouseEventKind::ScrollUp);
                let code = if up { KeyCode::Up } else { KeyCode::Down };
                match self.tab {
                    Tab::Chat => {
                        self.chat.scroll = if up {
                            self.chat.scroll.saturating_add(3)
                        } else {
                            self.chat.scroll.saturating_sub(3)
                        };
                    }
                    Tab::Models | Tab::Library | Tab::Engines => {
                        self.on_key(KeyEvent::new(code, KeyModifiers::NONE))
                    }
                    Tab::Discover if self.discover.focus != DFocus::Search => {
                        self.on_key(KeyEvent::new(code, KeyModifiers::NONE))
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    pub fn on_tick(&mut self) {
        self.tick += 1;
        self.pid_watch.store(
            self.session.as_ref().and_then(|s| s.pid()).unwrap_or(0),
            Ordering::Relaxed,
        );
        if let Some((_, t)) = &self.toast {
            if t.elapsed().as_secs() >= 6 {
                self.toast = None;
            }
        }
        // A server we spawned can die on its own (OOM, driver crash). Notice it.
        let died = self
            .session
            .as_mut()
            .and_then(|s| s.child.as_mut())
            .and_then(|c| c.try_wait().ok().flatten());
        if let Some(status) = died {
            let name = self
                .session
                .as_ref()
                .map(|s| s.name.clone())
                .unwrap_or_default();
            if let Some(s) = self.session.as_mut() {
                s.child = None;
            }
            self.session = None;
            self.chat.generating = None;
            self.toast(format!(
                "{name} stopped unexpectedly ({status}). out of memory? check the Launch tab log"
            ));
        }
    }
}

pub struct PlanNumbers {
    pub total_gib: f64,
    pub weights_gib: f64,
    pub kv_gib: f64,
    pub fits: bool,
    pub offload: bool,
    pub tok_s: Option<f64>,
    pub gpu_layers: Option<u32>,
    pub max_layers: Option<u32>,
    pub download: Option<u64>,
    /// Shape known exactly (catalog or GGUF header) rather than guessed from file size.
    pub exact: bool,
}

enum Kind {
    Skip,
    Other,
}

fn push_hist(h: &mut VecDeque<u64>, v: u64) {
    h.push_back(v);
    while h.len() > 120 {
        h.pop_front();
    }
}

fn theme_copy(s: &str) {
    super::theme::copy_to_clipboard(s);
}
