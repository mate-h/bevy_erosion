//! Headless file runner: erode a heightfield read from a file, without a window, and write the
//! eroded heights and KTT's fields as raw f32 files plus a `run.json`.
//!
//! ```text
//! cargo run --release --example headless -- --height in/dem.f32 --out sim_raw \
//!     [--grid in/dem.json] [--params params.json] [--softness in/soft.f32] \
//!     [--erosion-mask in/erode.f32] [--outflow in/outflow.f32] [--uplift in/uplift.f32] \
//!     [--iters-per-frame 64] [--no-repro-check]
//! ```
//!
//! Inputs are row-major little-endian f32, one value per texel, first row first. The grid sidecar
//! (`--grid`, else `<height>.json`, else `grid.json` next to the heights) holds `width`,
//! `height`, the texel size as `metres_per_texel` (or `texel_size_m` / `dx`) and, optionally,
//! `origin` and `crs`, which are copied to `run.json`. The masks are on the same grid.
//!
//! The params JSON sets any [`ErodeParams`] field by name, plus `seed` (the trails' random start
//! points, default 1), `base_m` (the height subtracted before the simulation, default 10 m below
//! the lowest texel), `bedrock_depth_m` and `outflow_sea_level_m` (metres, converted to height
//! units). Unknown keys are an error. See the README for the outputs.

use bevy::{
    app::PluginsState,
    prelude::*,
    render::{
        RenderApp,
        gpu_readback::{Readback, ReadbackComplete},
        render_resource::PollType,
        renderer::{RenderAdapterInfo, RenderDevice},
    },
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_erosion::{drainage::DrainageArea, *};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    process::ExitCode,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

/// Largest dz RMS (m) between two runs on the same input that still counts as reproducible.
const REPRO_RMS_LIMIT_M: f64 = 0.01;
/// Default trail seed.
const DEFAULT_SEED: u32 = 1;
/// Headroom (m) below the lowest texel: the simulation keeps heights at or above 0.
const BASE_HEADROOM_M: f32 = 10.0;
/// Frames before a stalled cascade or readback counts as a failure.
const MAX_FRAMES: u32 = 200_000;

struct Args {
    height: PathBuf,
    grid: Option<PathBuf>,
    params: Option<PathBuf>,
    softness: Option<PathBuf>,
    erosion_mask: Option<PathBuf>,
    outflow: Option<PathBuf>,
    uplift: Option<PathBuf>,
    out: PathBuf,
    iters_per_frame: u32,
    repro_check: bool,
}

const USAGE: &str = "usage: headless --height H.f32 --out DIR [--grid G.json] [--params P.json] \
[--softness S.f32] [--erosion-mask E.f32] [--outflow O.f32] [--uplift U.f32] \
[--iters-per-frame N] [--no-repro-check]";

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let mut a = Args {
        height: PathBuf::new(),
        grid: None,
        params: None,
        softness: None,
        erosion_mask: None,
        outflow: None,
        uplift: None,
        out: PathBuf::new(),
        iters_per_frame: 64,
        repro_check: true,
    };
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--height" => a.height = value()?.into(),
            "--grid" => a.grid = Some(value()?.into()),
            "--params" => a.params = Some(value()?.into()),
            "--softness" => a.softness = Some(value()?.into()),
            "--erosion-mask" => a.erosion_mask = Some(value()?.into()),
            "--outflow" => a.outflow = Some(value()?.into()),
            "--uplift" => a.uplift = Some(value()?.into()),
            "--out" => a.out = value()?.into(),
            "--iters-per-frame" => {
                a.iters_per_frame = value()?
                    .parse()
                    .map_err(|e| format!("--iters-per-frame: {e}"))?
            }
            "--no-repro-check" => a.repro_check = false,
            "-h" | "--help" => return Err(USAGE.into()),
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    if a.height.as_os_str().is_empty() || a.out.as_os_str().is_empty() {
        return Err(USAGE.into());
    }
    Ok(a)
}

/// The grid sidecar.
struct Grid {
    width: u32,
    height: u32,
    dx: f32,
    json: Value,
}

fn read_grid(args: &Args) -> Result<(Grid, PathBuf), String> {
    let path = args.grid.clone().unwrap_or_else(|| {
        let own = args.height.with_extension("json");
        if own.exists() {
            own
        } else {
            args.height.with_file_name("grid.json")
        }
    });
    let json: Value = serde_json::from_slice(
        &std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?,
    )
    .map_err(|e| format!("{}: {e}", path.display()))?;
    let num = |keys: &[&str]| {
        keys.iter()
            .find_map(|k| json.get(*k).and_then(Value::as_f64))
    };
    let width = num(&["width"]).ok_or("grid: no width")? as u32;
    let height = num(&["height"]).ok_or("grid: no height")? as u32;
    let dx = num(&["metres_per_texel", "texel_size_m", "dx"]).ok_or("grid: no texel size")? as f32;
    if width < 8 || height < 8 || !(dx > 0.0) {
        return Err(format!(
            "grid: bad size {width}x{height} or texel size {dx}"
        ));
    }
    Ok((
        Grid {
            width,
            height,
            dx,
            json,
        },
        path,
    ))
}

fn read_f32(path: &Path, texels: usize) -> Result<(Vec<f32>, String), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() != texels * 4 {
        return Err(format!(
            "{}: {} bytes, expected {} ({} texels of f32)",
            path.display(),
            bytes.len(),
            texels * 4,
            texels
        ));
    }
    let values: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    if let Some(i) = values.iter().position(|v| !v.is_finite()) {
        return Err(format!("{}: texel {i} is not finite", path.display()));
    }
    Ok((values, sha256(&bytes)))
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Sets the [`ErodeParams`] fields named in a JSON object; `None` for an unknown key.
macro_rules! erode_fields {
    ($p:ident, $key:ident, $val:ident; f32: $($f:ident),* ; u32: $($u:ident),* $(,)?) => {{
        match $key {
            $(stringify!($f) => $val.as_f64().map(|v| $p.$f = v as f32),)*
            $(stringify!($u) => $val.as_u64().or($val.as_bool().map(u64::from)).map(|v| $p.$u = v as u32),)*
            _ => None,
        }
    }};
    (dump $p:ident; f32: $($f:ident),* ; u32: $($u:ident),* $(,)?) => {{
        let mut m = Map::new();
        $(m.insert(stringify!($f).into(), json!($p.$f));)*
        $(m.insert(stringify!($u).into(), json!($p.$u));)*
        m
    }};
}

macro_rules! with_erode_fields {
    ($mac:ident!($($pre:tt)*)) => {
        $mac!($($pre)*
            f32: detail_scale, erosion_strength, rock_softness, sediment_compaction,
            compaction_threshold, channeling, channeling_character, sediment_removal,
            removal_character, wear_angle, talus_angle, max_deposit_angle, flow_length,
            ridge_softening_amount, erosion_amount, ridge_erosion_amount, friction, rock_friction,
            flow_volume, velocity_randomness, velocity_randomness_refinement, suspended_load,
            river_scale, river_character, river_friction_reduction, river_volume,
            river_channeling, momentum_coherence, meander_longevity, uplift_river_carving, uplift,
            feature_size, source_detail_smoothing, geological_age, erosion_cutoff_angle,
            hydro_conditioning_search_distance, outflow_sea_level, outflow_border, bedrock_depth,
            granularity_amount, granularity_scale, granularity_strength, age_strength,
            river_threshold, trail_damping;
            u32: compute_ridge_erosion, ridge_erosion_steps, do_rivers, clamp_river_channeling,
            use_hydro_conditioning, use_softness_mask, use_outflow, outflow_integration,
            use_erosion_mask, use_uplift_map, use_bedrock, use_bedrock_map, additive_sediment,
            normalize_age, bedrock_mode, noise_seed, mask_source, trail_seed,
        )
    };
}

fn set_field(p: &mut ErodeParams, key: &str, val: &Value) -> Option<()> {
    with_erode_fields!(erode_fields!(p, key, val;))
}

fn dump_params(p: &ErodeParams) -> Map<String, Value> {
    with_erode_fields!(erode_fields!(dump p;))
}

/// Simulation settings resolved from the params JSON and the inputs.
struct Setup {
    params: ErodeParams,
    seed: u32,
    base_m: f32,
    metres_per_height: f32,
    /// The params file as given.
    given: Value,
}

fn resolve_params(
    file: Option<&Path>,
    grid: &Grid,
    heights: &[f32],
    masks: &Masks,
) -> Result<Setup, String> {
    let given: Value = match file {
        Some(path) => serde_json::from_slice(
            &std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?,
        )
        .map_err(|e| format!("{}: {e}", path.display()))?,
        None => json!({}),
    };
    let obj = given.as_object().ok_or("params: not a JSON object")?;
    let metres_per_height = CELLS_PER_HEIGHT * grid.dx;
    let size = UVec2::new(grid.width, grid.height);
    let mut p = ErodeParams {
        map_size: size,
        num_particles: size.x * size.y,
        padded_size: size,
        // No procedural sea unless asked for: the outflow comes from the mask and the border.
        outflow_sea_level: 0.0,
        ..default()
    };
    // Masks given switch their KTT toggles on (the params file can still turn them off).
    p.mask_source = u32::from(masks.any());
    p.use_softness_mask = u32::from(masks.softness.is_some());
    p.use_erosion_mask = u32::from(masks.erosion.is_some());
    p.use_outflow = u32::from(masks.outflow.is_some());
    p.use_uplift_map = u32::from(masks.uplift.is_some());
    let min = heights.iter().copied().fold(f32::INFINITY, f32::min);
    let mut base_m = (min - BASE_HEADROOM_M).floor();
    let mut seed = DEFAULT_SEED;
    for (key, val) in obj {
        let num = || {
            val.as_f64()
                .ok_or(format!("params: {key} must be a number"))
        };
        match key.as_str() {
            "seed" => seed = num()? as u32,
            "base_m" => base_m = num()? as f32,
            "bedrock_depth_m" => p.bedrock_depth = num()? as f32 / metres_per_height,
            // Converted below, once `base_m` is known.
            "outflow_sea_level_m" => {
                num()?;
            }
            "_comment" | "comment" => {}
            _ => set_field(&mut p, key, val)
                .ok_or(format!("params: unknown or mistyped key {key}"))?,
        }
    }
    // `outflow_sea_level_m` needs the final base.
    if let Some(sea) = obj.get("outflow_sea_level_m").and_then(Value::as_f64) {
        p.outflow_sea_level = (sea as f32 - base_m).max(0.0) / metres_per_height;
    }
    if base_m > min {
        return Err(format!(
            "params: base_m {base_m} is above the lowest texel {min}"
        ));
    }
    p.trail_seed = seed;
    Ok(Setup {
        params: p,
        seed,
        base_m,
        metres_per_height,
        given,
    })
}

#[derive(Default)]
struct Masks {
    softness: Option<Vec<f32>>,
    outflow: Option<Vec<f32>>,
    erosion: Option<Vec<f32>>,
    uplift: Option<Vec<f32>>,
}

impl Masks {
    fn any(&self) -> bool {
        self.softness.is_some()
            || self.outflow.is_some()
            || self.erosion.is_some()
            || self.uplift.is_some()
    }

    /// Rgba32Float texels for [`ErosionImages::source_masks`]; unused masks hold the identity.
    fn texels(&self, n: usize) -> Vec<u8> {
        let at =
            |m: &Option<Vec<f32>>, i: usize, default: f32| m.as_ref().map_or(default, |m| m[i]);
        let mut out = Vec::with_capacity(n * 16);
        for i in 0..n {
            for v in [
                at(&self.softness, i, 1.0),
                at(&self.outflow, i, 0.0),
                at(&self.erosion, i, 1.0),
                at(&self.uplift, i, 1.0),
            ] {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }
}

/// The fields of one run, on the base map.
struct Fields {
    /// Eroded height in metres.
    height_m: Vec<f32>,
    deposition: Vec<f32>,
    erosion: Vec<f32>,
    flow: Vec<f32>,
    age: Vec<f32>,
    momentum_x: Vec<f32>,
    momentum_y: Vec<f32>,
    rivers: Vec<f32>,
    drainage_area: Vec<f32>,
}

/// One cascade's timing.
struct RunStats {
    frames: u32,
    cascade_seconds: f64,
    levels: Vec<PyramidLevel>,
}

struct Runner {
    app: App,
}

impl Runner {
    fn new(
        grid: &Grid,
        setup: &Setup,
        heights_units: Vec<f32>,
        masks: Vec<u8>,
        iters: u32,
    ) -> Self {
        let size = UVec2::new(grid.width, grid.height);
        let mut app = App::new();
        app.insert_resource(ErosionConfig {
            map_size: size,
            ..default()
        })
        .insert_resource(WorldScale {
            metres_per_texel: grid.dx,
        })
        .add_plugins(
            DefaultPlugins
                .build()
                .disable::<WinitPlugin>()
                .disable::<bevy::render::pipelined_rendering::PipelinedRenderingPlugin>()
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .set(AssetPlugin {
                    file_path: concat!(env!("CARGO_MANIFEST_DIR"), "/assets").into(),
                    ..default()
                })
                .set(bevy::log::LogPlugin {
                    filter: "wgpu=error,naga=warn,bevy_render=warn,bevy_erosion=info".into(),
                    ..default()
                }),
        )
        .add_plugins(ErosionComputePlugin);
        let params = setup.params.clone();
        let heights = Arc::new(Mutex::new(Some(heights_units)));
        let masks = Arc::new(Mutex::new(Some(masks)));
        app.add_systems(
            PostStartup,
            move |mut commands: Commands,
                  erosion_images: Res<ErosionImages>,
                  mut images: ResMut<Assets<Image>>,
                  mut sim: ResMut<SimControl>,
                  mut water: ResMut<WaterControl>| {
                // The terrain: flatten weight 1 onto the file's heights replaces the noise exactly.
                if let Some(h) = heights.lock().unwrap().take() {
                    let mut image = images.get_mut(&erosion_images.terrain_edit).unwrap();
                    let data = image.data.as_mut().unwrap();
                    for (i, v) in h.iter().enumerate() {
                        let texel = [0.0f32, 1.0, *v, 0.0];
                        for (k, c) in texel.iter().enumerate() {
                            data[i * 16 + 4 * k..i * 16 + 4 * k + 4]
                                .copy_from_slice(&c.to_le_bytes());
                        }
                    }
                }
                if let Some(m) = masks.lock().unwrap().take() {
                    let mut image = images.get_mut(&erosion_images.source_masks).unwrap();
                    image.data = Some(m);
                }
                commands.insert_resource(FieldExport::new(&mut images, size));
                commands.insert_resource(params.clone());
                sim.iterations_per_frame = iters.max(1);
                sim.continue_after_cascade = false;
                // Erosion only: no water, rain or landslides change the terrain afterwards.
                water.landslide.auto = false;
                water.rainfall = 0.0;
                water.springs.clear();
                water.bed_load.live = false;
            },
        );
        while app.plugins_state() == PluginsState::Adding {
            bevy::tasks::tick_global_task_pools_on_main_thread();
            std::thread::sleep(Duration::from_millis(1));
        }
        app.finish();
        app.cleanup();
        Self { app }
    }

    fn frame(&mut self) {
        self.app.update();
        let device = self
            .app
            .sub_app(RenderApp)
            .world()
            .resource::<RenderDevice>()
            .clone();
        device.poll(PollType::wait_indefinitely()).ok();
    }

    fn progress(&self) -> PyramidProgressData {
        self.app.world().resource::<PyramidProgress>().get()
    }

    fn adapter(&self) -> Value {
        let info = self
            .app
            .sub_app(RenderApp)
            .world()
            .resource::<RenderAdapterInfo>();
        json!({
            "name": info.name,
            "backend": format!("{:?}", info.backend),
            "device_type": format!("{:?}", info.device_type),
            "driver": info.driver,
            "driver_info": info.driver_info,
            "vendor": info.vendor,
            "device": info.device,
        })
    }

    /// Runs one cascade to its end; `restart` re-erodes the same source terrain first.
    fn cascade(&mut self, restart: bool) -> Result<RunStats, String> {
        if restart {
            let mut reset = self.app.world_mut().resource_mut::<ResetSim>();
            reset.generation = reset.generation.wrapping_add(1);
            // Let the render world see the reset before waiting for the finish.
            for _ in 0..2 {
                self.frame();
            }
        }
        let mut frames = 0;
        let mut started: Option<Instant> = None;
        let mut last_level = usize::MAX;
        loop {
            self.frame();
            frames += 1;
            let p = self.progress();
            if p.level_count > 0 && started.is_none() {
                started = Some(Instant::now());
            }
            if p.level_count > 0 && p.level != last_level {
                last_level = p.level;
                info!(
                    "level {}/{}: {}x{}",
                    p.level + 1,
                    p.level_count,
                    p.size.x,
                    p.size.y
                );
            }
            if p.finished && started.is_some() {
                break;
            }
            if frames > MAX_FRAMES {
                return Err("cascade did not finish".into());
            }
        }
        let params = self.app.world().resource::<ErodeParams>().clone();
        Ok(RunStats {
            frames,
            cascade_seconds: started.map_or(0.0, |t| t.elapsed().as_secs_f64()),
            levels: params.pyramid_levels(),
        })
    }

    fn readback(&mut self, handle: Handle<Image>) -> Result<Vec<u8>, String> {
        let captured = Arc::new(Mutex::new(None));
        let sink = captured.clone();
        self.app
            .world_mut()
            .spawn(Readback::texture(handle))
            .observe(move |event: On<ReadbackComplete>, mut commands: Commands| {
                *sink.lock().unwrap() = Some(event.data.clone());
                commands.entity(event.entity).try_despawn();
            });
        for _ in 0..1000 {
            self.frame();
            if let Some(bytes) = captured.lock().unwrap().take() {
                return Ok(bytes);
            }
        }
        Err("readback timed out".into())
    }

    /// Reads the fields of the finished cascade; `drainage_after` is the drainage generation
    /// before it ran (the drainage area is computed on the CPU once it finishes).
    fn fields(
        &mut self,
        grid: &Grid,
        setup: &Setup,
        drainage_after: u32,
    ) -> Result<Fields, String> {
        let export = self.app.world().resource::<FieldExport>().clone();
        let a = self.readback(export.a)?;
        let b = self.readback(export.b)?;
        let mut frames = 0;
        while self.app.world().resource::<DrainageArea>().generation == drainage_after {
            self.frame();
            frames += 1;
            if frames > 10_000 {
                return Err("drainage area did not arrive".into());
            }
        }
        let (w, h) = (grid.width as usize, grid.height as usize);
        let channels = |bytes: &[u8]| -> [Vec<f32>; 4] {
            let stride = bytes.len() / h;
            let mut out: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::with_capacity(w * h));
            for y in 0..h {
                let row = &bytes[y * stride..y * stride + w * 16];
                for texel in row.as_chunks::<16>().0 {
                    for (k, ch) in out.iter_mut().enumerate() {
                        ch.push(f32::from_le_bytes(
                            texel[4 * k..4 * k + 4].try_into().unwrap(),
                        ));
                    }
                }
            }
            out
        };
        let [height, deposition, erosion, flow] = channels(&a);
        let [age, momentum_x, momentum_y, rivers] = channels(&b);
        let drainage = self.app.world().resource::<DrainageArea>();
        if drainage.area.len() != w * h {
            return Err("drainage area has the wrong size".into());
        }
        Ok(Fields {
            height_m: height
                .iter()
                .map(|u| u * setup.metres_per_height + setup.base_m)
                .collect(),
            deposition,
            erosion,
            flow,
            age,
            momentum_x,
            momentum_y,
            rivers,
            drainage_area: drainage.area.as_ref().clone(),
        })
    }

    fn drainage_generation(&self) -> u32 {
        self.app.world().resource::<DrainageArea>().generation
    }
}

fn git_state() -> Value {
    let dir = env!("CARGO_MANIFEST_DIR");
    let git = |args: &[&str]| {
        std::process::Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    json!({
        "sha": git(&["rev-parse", "HEAD"]),
        "dirty": git(&["status", "--porcelain", "--untracked-files=no"]).map(|s| !s.is_empty()),
        "branch": git(&["rev-parse", "--abbrev-ref", "HEAD"]),
    })
}

fn write_f32(dir: &Path, name: &str, values: &[f32]) -> Result<String, String> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let path = dir.join(format!("{name}.f32"));
    std::fs::write(&path, &bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(sha256(&bytes))
}

/// RMS and largest magnitude.
fn rms_max(v: &[f32]) -> (f64, f64) {
    let (mut sq, mut max) = (0.0f64, 0.0f64);
    for &x in v {
        let x = x as f64;
        sq += x * x;
        max = max.max(x.abs());
    }
    ((sq / v.len().max(1) as f64).sqrt(), max)
}

/// Three box blurs of radius `r` (close to a Gaussian of sigma ~ r), edges clamped.
fn box_blur3(v: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let mut a = v.to_vec();
    let mut b = vec![0.0f32; v.len()];
    let pass = |src: &[f32],
                dst: &mut [f32],
                len: usize,
                count: usize,
                at: &dyn Fn(usize, usize) -> usize| {
        let mut line = vec![0.0f64; len + 1];
        for c in 0..count {
            for i in 0..len {
                line[i + 1] = line[i] + src[at(c, i)] as f64;
            }
            for i in 0..len {
                let lo = i.saturating_sub(r);
                let hi = (i + r + 1).min(len);
                dst[at(c, i)] = ((line[hi] - line[lo]) / (hi - lo) as f64) as f32;
            }
        }
    };
    for _ in 0..3 {
        pass(&a, &mut b, w, h, &|y, x| y * w + x);
        pass(&b, &mut a, h, w, &|x, y| y * w + x);
    }
    a
}

fn stats(v: &[f32]) -> Value {
    let (mut lo, mut hi, mut sum) = (f64::INFINITY, f64::NEG_INFINITY, 0.0);
    for &x in v {
        let x = x as f64;
        lo = lo.min(x);
        hi = hi.max(x);
        sum += x;
    }
    json!({ "min": lo, "max": hi, "mean": sum / v.len().max(1) as f64 })
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("headless: {e}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<ExitCode, String> {
    let wall = Instant::now();
    let args = parse_args()?;
    let (grid, grid_path) = read_grid(&args)?;
    let n = (grid.width * grid.height) as usize;
    let mut inputs = BTreeMap::new();
    let mut input = |name: &str, path: &Path, sha: String| {
        inputs.insert(
            name.to_string(),
            json!({ "path": path.canonicalize().unwrap_or(path.into()), "sha256": sha }),
        );
    };
    let grid_bytes = std::fs::read(&grid_path).map_err(|e| e.to_string())?;
    input("grid", &grid_path, sha256(&grid_bytes));
    let (heights, sha) = read_f32(&args.height, n)?;
    input("height", &args.height, sha);
    let mut masks = Masks::default();
    for (name, path, slot) in [
        ("softness", &args.softness, &mut masks.softness),
        ("erosion_mask", &args.erosion_mask, &mut masks.erosion),
        ("outflow", &args.outflow, &mut masks.outflow),
        ("uplift", &args.uplift, &mut masks.uplift),
    ] {
        if let Some(path) = path {
            let (values, sha) = read_f32(path, n)?;
            input(name, path, sha);
            *slot = Some(values);
        }
    }
    if let Some(path) = &args.params {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        input("params", path, sha256(&bytes));
    }
    let setup = resolve_params(args.params.as_deref(), &grid, &heights, &masks)?;
    let heights_units: Vec<f32> = heights
        .iter()
        .map(|h| (h - setup.base_m) / setup.metres_per_height)
        .collect();
    std::fs::create_dir_all(&args.out).map_err(|e| format!("{}: {e}", args.out.display()))?;

    let mut runner = Runner::new(
        &grid,
        &setup,
        heights_units,
        masks.texels(n),
        args.iters_per_frame,
    );
    let adapter = runner.adapter();
    info!("adapter: {adapter}");
    let drainage_before = runner.drainage_generation();
    let run1 = runner.cascade(false)?;
    info!(
        "cascade finished in {:.2} s ({} frames)",
        run1.cascade_seconds, run1.frames
    );
    let fields = runner.fields(&grid, &setup, drainage_before)?;

    let repro = if args.repro_check {
        let drainage_before = runner.drainage_generation();
        let run2 = runner.cascade(true)?;
        let second = runner.fields(&grid, &setup, drainage_before)?;
        let diff: Vec<f32> = fields
            .height_m
            .iter()
            .zip(&second.height_m)
            .map(|(a, b)| a - b)
            .collect();
        let (rms, max) = rms_max(&diff);
        // How much of the spread is fine-scale: the difference after a low-pass at these scales.
        let mut lowpass = Map::new();
        for metres in [8.0f32, 32.0, 128.0] {
            let r = (metres / grid.dx / 2.0).round().max(1.0) as usize;
            let smooth = box_blur3(&diff, grid.width as usize, grid.height as usize, r);
            lowpass.insert(format!("{metres}m"), json!(rms_max(&smooth).0));
        }
        info!("repro: dz rms {rms:.4} m, max {max:.4} m, low-passed {lowpass:?}");
        let mut field_rms = Map::new();
        for (name, a, b) in [
            ("deposition", &fields.deposition, &second.deposition),
            ("flow", &fields.flow, &second.flow),
            (
                "drainage_area",
                &fields.drainage_area,
                &second.drainage_area,
            ),
        ] {
            let d: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
            field_rms.insert(
                name.into(),
                json!({ "diff_rms": rms_max(&d).0, "rms": rms_max(a).0 }),
            );
        }
        Some(json!({
            "runs": 2,
            "dz_rms_m": rms,
            "dz_max_m": max,
            "limit_rms_m": REPRO_RMS_LIMIT_M,
            "pass": rms < REPRO_RMS_LIMIT_M,
            "dz_rms_lowpass_m": lowpass,
            "fields": field_rms,
            "second_cascade_seconds": run2.cascade_seconds,
        }))
    } else {
        None
    };

    let mut outputs = BTreeMap::new();
    for (name, values) in [
        ("eroded_height", &fields.height_m),
        ("deposition", &fields.deposition),
        ("erosion", &fields.erosion),
        ("flow", &fields.flow),
        ("drainage_area", &fields.drainage_area),
        ("age", &fields.age),
        ("momentum_x", &fields.momentum_x),
        ("momentum_y", &fields.momentum_y),
        ("rivers", &fields.rivers),
    ] {
        let sha = write_f32(&args.out, name, values)?;
        outputs.insert(
            format!("{name}.f32"),
            json!({ "sha256": sha, "stats": stats(values) }),
        );
    }
    let p = &setup.params;
    let dz: Vec<f32> = fields
        .height_m
        .iter()
        .zip(&heights)
        .map(|(a, b)| a - b)
        .collect();
    let pad = pyramid_padding(p.map_size, &run1.levels);
    let pass = repro.as_ref().is_none_or(|r| r["pass"] == json!(true));
    let run = json!({
        // Experiment 8's keys (read by terracraft's erosion_upres post.py).
        "width": grid.width,
        "height": grid.height,
        "dx": grid.dx,
        "base_m": setup.base_m,
        "metres_per_height": setup.metres_per_height,
        "feature_size": p.feature_size,
        "erosion_amount": p.erosion_amount,
        "bedrock_depth_m": p.bedrock_depth * setup.metres_per_height,
        "rivers": p.do_rivers != 0,
        "softness": p.use_softness_mask != 0,
        "levels": run1.levels.iter().map(|l| l.size.x).collect::<Vec<_>>(),
        "frames": run1.frames,
        "cascade_seconds": run1.cascade_seconds,
        "total_seconds": wall.elapsed().as_secs_f64(),
        "drainage": true,
        // Provenance.
        "runner": "bevy_erosion examples/headless.rs",
        "seed": setup.seed,
        "git": git_state(),
        "adapter": adapter,
        "grid": grid.json,
        "inputs": inputs,
        "params_given": setup.given,
        "params": dump_params(p),
        "pyramid": run1.levels.iter().map(|l| json!({
            "size": [l.size.x, l.size.y],
            "voxel_size": l.voxel_size,
            "iterations": l.iterations,
            "trail_weight": l.trail_weight,
        })).collect::<Vec<_>>(),
        "pad_offset": [pad.x, pad.y],
        "iters_per_frame": args.iters_per_frame,
        "outputs": outputs,
        "units": {
            "eroded_height": "m",
            "deposition": "height units (x metres_per_height for m); loose sediment, signed",
            "erosion": "height units (x metres_per_height for m); cut accumulated by the trails",
            "flow": "KTT flow accumulation (trail deposition steps, unitless)",
            "drainage_area": "m2 (CPU MFD on the eroded heights, map border as the outlet)",
            "age": "KTT __Age: base voxels of erosion since the cell last changed",
            "momentum_x": "KTT river momentum __m.x",
            "momentum_y": "KTT river momentum __m.y (0 without momentum coherence)",
            "rivers": "KTT river momentum magnitude (0 unless do_rivers)",
        },
        "dz_m": stats(&dz),
        "repro": repro,
    });
    let path = args.out.join("run.json");
    std::fs::write(&path, serde_json::to_string_pretty(&run).unwrap())
        .map_err(|e| format!("{}: {e}", path.display()))?;
    println!("{}", path.display());
    if !pass {
        eprintln!("headless: two runs differ by more than {REPRO_RMS_LIMIT_M} m RMS");
        return Ok(ExitCode::from(3));
    }
    Ok(ExitCode::SUCCESS)
}
