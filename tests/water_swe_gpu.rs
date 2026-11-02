//! GPU checks of the water solvers on analytic beds (Phase 4a of `docs/WATER_PLAN.md`).
//!
//! Each test flattens the source terrain to a bed through the terrain edit layer, turns the KTT
//! erosion off so the bed stays as drawn, and runs the water headlessly with the GPU synced every
//! frame. They need a graphics adapter, so they are ignored by default:
//! `cargo test --release --test water_swe_gpu -- --ignored --test-threads=1 --nocapture`.

use bevy::{
    app::PluginsState,
    prelude::*,
    render::{
        RenderApp,
        gpu_readback::{Readback, ReadbackComplete},
        render_resource::PollType,
        renderer::RenderDevice,
        storage::ShaderBuffer,
    },
    time::TimeUpdateStrategy,
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_erosion::*;
use std::{
    f32::consts::PI,
    sync::{Arc, Mutex},
    time::Duration,
};

const N: u32 = 256;
const FPS: f32 = 60.0;
/// Solver step in seconds (`WATER_STEP`): velocities in the water texture are texels per step.
const STEP: f32 = 1.0 / 480.0;
/// Height units to solver cells (`CELLS_PER_HEIGHT` in `water.wgsl`).
const CELLS_PER_HEIGHT: f32 = 100.0;

fn f16_to_f32(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    match e {
        0 => s * m * 2f32.powi(-24),
        31 if m == 0.0 => s * f32::INFINITY,
        31 => f32::NAN,
        _ => s * (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

/// One texel of the water texture: depth (height units), velocity (texels per second).
#[derive(Clone, Copy, Default)]
struct Water {
    depth: f32,
    velocity: Vec2,
}

struct Harness {
    app: App,
    fps: f32,
}

impl Harness {
    /// A headless app whose terrain is `bed` (height units, per texel), with erosion off.
    fn new(solver: WaterSolver, bed: impl Fn(u32, u32) -> f32) -> Self {
        let mut h = Self::natural(solver);
        {
            let mut params = h.app.world_mut().resource_mut::<ErodeParams>();
            params.erosion_amount = 0.0;
            params.erosion_strength = 0.0;
            params.ridge_erosion_amount = 0.0;
            params.compute_ridge_erosion = 0;
            params.do_rivers = 0;
            params.uplift = 0.0;
        }
        h.set_bed(bed);
        h
    }

    /// A headless app on the default eroded terrain, once its cascade has finished.
    fn natural(solver: WaterSolver) -> Self {
        let mut app = App::new();
        app.insert_resource(ErosionConfig {
            map_size: UVec2::splat(N),
            ..default()
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
                }),
        )
        .add_plugins(ErosionComputePlugin)
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
            1.0 / FPS,
        )));
        while app.plugins_state() == PluginsState::Adding {
            bevy::tasks::tick_global_task_pools_on_main_thread();
            std::thread::sleep(Duration::from_millis(1));
        }
        app.finish();
        app.cleanup();
        app.update();

        let mut h = Self { app, fps: FPS };
        {
            let mut water = h.app.world_mut().resource_mut::<WaterControl>();
            water.solver = solver;
            water.evaporation = 0.0;
            water.sediment_capacity = 0.0;
            water.bed_load.live = false;
            if let Some(g) = std::env::var("TEST_GRAVITY")
                .ok()
                .and_then(|v| v.parse().ok())
            {
                water.gravity = g;
            }
            if let Some(n) = std::env::var("TEST_MANNING")
                .ok()
                .and_then(|v| v.parse().ok())
            {
                water.manning_n = n;
            }
        }
        h.settle_cascade();
        h
    }

    /// Redraw the source terrain as `bed` and re-run the cascade (the water stays).
    fn set_bed(&mut self, bed: impl Fn(u32, u32) -> f32) {
        let handle = self
            .app
            .world()
            .resource::<ErosionImages>()
            .terrain_edit
            .clone();
        {
            let mut images = self.app.world_mut().resource_mut::<Assets<Image>>();
            let mut image = images.get_mut(&handle).unwrap();
            let data = image.data.as_mut().unwrap();
            for y in 0..N {
                for x in 0..N {
                    // Flatten weight 1 onto the target: the edit replaces the noise.
                    let texel = [0.0f32, 1.0, bed(x, y), 0.0];
                    let i = ((y * N + x) * 16) as usize;
                    for (k, v) in texel.iter().enumerate() {
                        data[i + 4 * k..i + 4 * k + 4].copy_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }
        let mut reset = self.app.world_mut().resource_mut::<ResetSim>();
        reset.generation = reset.generation.wrapping_add(1);
        self.settle_cascade();
    }

    fn settle_cascade(&mut self) {
        for _ in 0..3 {
            self.frame();
        }
        for _ in 0..5000 {
            if self
                .app
                .world()
                .resource::<PyramidProgress>()
                .get()
                .finished
            {
                break;
            }
            self.frame();
        }
        assert!(
            self.app
                .world()
                .resource::<PyramidProgress>()
                .get()
                .finished,
            "cascade did not finish"
        );
    }

    fn water(&mut self) -> Mut<'_, WaterControl> {
        self.app.world_mut().resource_mut::<WaterControl>()
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

    fn run(&mut self, seconds: f32) {
        for _ in 0..(seconds * self.fps).round() as u32 {
            self.frame();
        }
    }

    /// Read an `Rgba16Float` image; the water is paused meanwhile.
    fn read(&mut self, handle: Handle<Image>) -> Vec<[f32; 4]> {
        self.app.world_mut().resource_mut::<SimControl>().paused = true;
        let captured = Arc::new(Mutex::new(None));
        let sink = captured.clone();
        self.app
            .world_mut()
            .spawn(Readback::texture(handle))
            .observe(move |event: On<ReadbackComplete>, mut commands: Commands| {
                *sink.lock().unwrap() = Some(event.data.clone());
                commands.entity(event.entity).try_despawn();
            });
        for _ in 0..200 {
            self.frame();
            if captured.lock().unwrap().is_some() {
                break;
            }
        }
        self.app.world_mut().resource_mut::<SimControl>().paused = false;
        let bytes = captured.lock().unwrap().take().expect("readback timed out");
        bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| {
                std::array::from_fn(|k| f16_to_f32(u16::from_le_bytes([c[2 * k], c[2 * k + 1]])))
            })
            .collect()
    }

    /// Read a storage buffer of `vec4<f32>`; the water is paused meanwhile.
    fn read_vec4(&mut self, handle: Handle<ShaderBuffer>) -> Vec<Vec4> {
        self.app.world_mut().resource_mut::<SimControl>().paused = true;
        let captured = Arc::new(Mutex::new(None));
        let sink = captured.clone();
        self.app
            .world_mut()
            .spawn(Readback::buffer(handle))
            .observe(move |event: On<ReadbackComplete>, mut commands: Commands| {
                *sink.lock().unwrap() = Some(event.data.clone());
                commands.entity(event.entity).try_despawn();
            });
        for _ in 0..200 {
            self.frame();
            if captured.lock().unwrap().is_some() {
                break;
            }
        }
        self.app.world_mut().resource_mut::<SimControl>().paused = false;
        let bytes = captured.lock().unwrap().take().expect("readback timed out");
        bytes
            .as_chunks::<16>()
            .0
            .iter()
            .map(|c| {
                Vec4::from_array(std::array::from_fn(|k| {
                    f32::from_le_bytes(c[4 * k..4 * k + 4].try_into().unwrap())
                }))
            })
            .collect()
    }

    /// The shared discharge field ([`WaterFields::discharge`]).
    fn read_discharge(&mut self) -> Vec<Vec4> {
        let handle = self.app.world().resource::<WaterFields>().discharge.clone();
        self.read_vec4(handle)
    }

    /// Simulated frames per second from now on.
    fn set_fps(&mut self, fps: f32) {
        self.fps = fps;
        self.app
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
                1.0 / fps,
            )));
    }

    fn pause(&mut self, paused: bool) {
        self.app.world_mut().resource_mut::<SimControl>().paused = paused;
    }

    fn read_water(&mut self) -> Vec<Water> {
        let handle = self.app.world().resource::<ErosionImages>().water.clone();
        let texels = self.read(handle);
        assert!(
            texels.iter().flatten().all(|v| v.is_finite()),
            "water texture holds NaN or infinity"
        );
        texels
            .iter()
            .map(|t| Water {
                depth: t[0],
                velocity: Vec2::new(t[1], t[2]) / STEP,
            })
            .collect()
    }

    fn read_heights(&mut self) -> Vec<f32> {
        let handle = self.app.world().resource::<ErosionImages>().height.clone();
        self.read(handle).iter().map(|t| t[0]).collect()
    }

    /// Hold the brush over a disc for `seconds`.
    fn pour(&mut self, center: Vec2, radius: f32, rate: f32, seconds: f32) {
        {
            let mut water = self.water();
            water.stroke_active = true;
            water.center = center;
            water.radius = radius;
            water.rate = rate;
        }
        self.run(seconds);
        self.water().stroke_active = false;
    }
}

fn idx(x: u32, y: u32) -> usize {
    (y * N + x) as usize
}

/// Volume the brush adds: the (1 − r²)² disc integrates to π R² / 3.
fn poured(radius: f32, rate: f32, seconds: f32) -> f32 {
    rate * PI * radius * radius / 3.0 * seconds
}

fn assert_bed(heights: &[f32], bed: impl Fn(u32, u32) -> f32) {
    let worst = (0..N)
        .flat_map(|y| (0..N).map(move |x| (x, y)))
        .map(|(x, y)| (heights[idx(x, y)] - bed(x, y)).abs())
        .fold(0.0f32, f32::max);
    // Rgba16Float resolves about 0.00025 at these heights.
    assert!(
        worst < 0.001,
        "bed differs from the drawn terrain by {worst}"
    );
}

/// A mound of water on a flat bed spreads as a radial dam break: it must stay mirror and
/// diagonal symmetric and keep its volume.
#[test]
#[ignore = "needs a graphics adapter"]
fn dam_break_spreads_symmetrically() {
    let bed = |_, _| 0.3;
    let mut h = Harness::new(WaterSolver::Swe, bed);
    let heights = h.read_heights();
    assert_bed(&heights, bed);

    let center = Vec2::splat(N as f32 / 2.0);
    let (radius, rate, time) = (16.0, 0.06, 0.5);
    h.pour(center, radius, rate, time);
    let peak_before = h.read_water().iter().map(|w| w.depth).fold(0.0, f32::max);
    h.run(0.5);
    let water = h.read_water();

    let depth = |x: u32, y: u32| water[idx(x, y)].depth;
    let peak = water.iter().map(|w| w.depth).fold(0.0f32, f32::max);
    let (mut mirror, mut diagonal) = (0.0f32, 0.0f32);
    for y in 0..N {
        for x in 0..N {
            mirror = mirror
                .max((depth(x, y) - depth(N - 1 - x, y)).abs())
                .max((depth(x, y) - depth(x, N - 1 - y)).abs());
            diagonal = diagonal.max((depth(x, y) - depth(y, x)).abs());
        }
    }
    let reach = (0..N / 2)
        .rev()
        .find(|&r| depth(N / 2 + r, N / 2) > 1e-4)
        .unwrap_or(0);
    let volume: f32 = water.iter().map(|w| w.depth).sum();
    let added = poured(radius, rate, time);
    println!(
        "dam break: peak {peak_before:.4} -> {peak:.4}, front at {reach} texels (pour radius {radius}), \
         mirror {:.2}% diagonal {:.2}% of peak, volume {volume:.3} of {added:.3}",
        100.0 * mirror / peak,
        100.0 * diagonal / peak
    );
    assert!(reach as f32 > radius + 4.0, "the mound did not spread");
    assert!(mirror < 0.02 * peak, "mirror asymmetry {mirror}");
    assert!(diagonal < 0.02 * peak, "diagonal asymmetry {diagonal}");
    assert!(
        (volume - added).abs() < 0.01 * added,
        "volume {volume} of {added}"
    );
}

/// Steady rain on a tilted plane: the sheet flow speed must follow Manning,
/// u = h^(2/3) S^(1/2) / n in cells.
#[test]
#[ignore = "needs a graphics adapter"]
fn flume_follows_manning() {
    let slope = 0.001; // height units per texel: a tangent of 0.1 in cells
    let bed = move |x: u32, _| 0.3 + slope * (N - 1 - x) as f32;
    let mut h = Harness::new(WaterSolver::Swe, bed);
    assert_bed(&h.read_heights(), bed);

    let n = h.water().manning_n;
    let s = slope * CELLS_PER_HEIGHT;
    // Rain for a sheet about 0.3 cells deep in the middle of the plane.
    let q_mid = 0.3f32.powf(5.0 / 3.0) * s.sqrt() / n;
    let rain = q_mid / (CELLS_PER_HEIGHT * (N / 2) as f32);
    h.water().rainfall = rain;
    // Long enough for water from the top edge to reach the middle and settle.
    let u_mid = 0.3f32.powf(2.0 / 3.0) * s.sqrt() / n;
    let settle = (3.0 * (N / 2) as f32 / u_mid).min(90.0);
    h.run(settle);
    let water = h.read_water();

    let mut ratios = Vec::new();
    let mut depth_ratios = Vec::new();
    for y in (N / 4)..(3 * N / 4) {
        for x in (3 * N / 8)..(5 * N / 8) {
            let w = water[idx(x, y)];
            let h_cells = w.depth * CELLS_PER_HEIGHT;
            let manning = h_cells.powf(2.0 / 3.0) * s.sqrt() / n;
            ratios.push(w.velocity.x / manning);
            // Steady state: the flow rate carries all the rain from upslope.
            let q = rain * CELLS_PER_HEIGHT * (x + 1) as f32;
            let h_manning = (q * n / s.sqrt()).powf(3.0 / 5.0);
            depth_ratios.push(h_cells / h_manning);
        }
    }
    let median = |v: &mut Vec<f32>| {
        v.sort_by(f32::total_cmp);
        v[v.len() / 2]
    };
    let (u_ratio, h_ratio) = (median(&mut ratios), median(&mut depth_ratios));
    println!(
        "flume: n {n}, slope {s} (cells), rain {rain:.2e}/s for {settle:.0} s: speed / Manning {u_ratio:.3}, \
         depth / steady Manning {h_ratio:.3}, Manning speed mid-plane {u_mid:.1} texels/s"
    );
    assert!(
        (u_ratio - 1.0).abs() < 0.15,
        "speed is {u_ratio} of Manning"
    );
    assert!(
        (h_ratio - 1.0).abs() < 0.15,
        "depth is {h_ratio} of steady Manning"
    );
}

/// Land at 0.5 with a basin (bed 0.40) and a channel sloping into it from the west.
fn lake_and_channel(x: u32, y: u32) -> f32 {
    let in_basin = (120..236).contains(&x) && (40..216).contains(&y);
    let in_channel = (16..120).contains(&x) && y.abs_diff(N / 2) < 8;
    if in_basin {
        0.40
    } else if in_channel {
        0.48 - (x - 16) as f32 * (0.06 / 104.0)
    } else {
        0.5
    }
}

/// Speed along the jet's centreline: at the mouth, and 0.5, 2 and 4 channel widths into the lake,
/// plus the mean speed far to the side.
fn lake_jet(solver: WaterSolver) -> [f32; 5] {
    let mut h = Harness::new(solver, lake_and_channel);
    assert_bed(&h.read_heights(), lake_and_channel);
    // Fill the basin to about 0.46, then let the slosh die down.
    h.pour(Vec2::new(178.0, 128.0), 60.0, 0.33, 1.0);
    h.run(8.0);
    h.water().springs.push(Spring {
        center: Vec2::new(24.0, 128.0),
        radius: 6.0,
        rate: 0.05,
    });
    h.run(12.0);
    let water = h.read_water();
    let speed = |x: u32| {
        (N / 2 - 2..N / 2 + 2)
            .map(|y| water[idx(x, y)].velocity.length())
            .sum::<f32>()
            / 4.0
    };
    let mut side = 0.0;
    let mut count = 0.0;
    for y in 50..80 {
        for x in 140..220 {
            side += water[idx(x, y)].velocity.length();
            count += 1.0;
        }
    }
    [speed(112), speed(128), speed(152), speed(184), side / count]
}

/// A river entering a lake keeps a jet a few channel widths past the mouth.
#[test]
#[ignore = "needs a graphics adapter"]
fn river_jet_runs_into_the_lake() {
    let swe = lake_jet(WaterSolver::Swe);
    let pipes = lake_jet(WaterSolver::Pipes);
    for (name, s) in [("swe", swe), ("pipes", pipes)] {
        println!(
            "lake jet {name}: channel {:.2}, +0.5W {:.2}, +2W {:.2}, +4W {:.2}, lake side {:.3} texels/s",
            s[0], s[1], s[2], s[3], s[4]
        );
    }
    assert!(swe[0] > 0.5, "no flow in the channel");
    assert!(
        swe[2] > 0.4 * swe[0] && swe[2] > 2.0 * swe[4],
        "jet two widths into the lake ({}) has died against the channel ({}) or the lake ({})",
        swe[2],
        swe[0],
        swe[4]
    );
    assert!(
        swe[2] / swe[0] > pipes[2] / pipes[0],
        "the pipes keep more of the jet"
    );
}

/// Bowl of radius 80 around the map centre, rim at 0.5, floor at 0.4 (plus `raise` inside r < 40).
fn bowl(raise: f32) -> impl Fn(u32, u32) -> f32 {
    move |x, y| {
        let r = Vec2::new(x as f32 + 0.5, y as f32 + 0.5).distance(Vec2::splat(N as f32 / 2.0));
        let t = (r / 80.0).min(1.0);
        let lift = raise * (1.0 - (r / 40.0).min(1.0)).powi(2);
        0.4 + 0.1 * t * t + lift
    }
}

/// Lake surface: mean, RMS deviation, volume and wet cells, over cells deeper than `min_depth`.
fn lake_stats(heights: &[f32], water: &[Water], min_depth: f32) -> (f32, f32, f32, usize) {
    let wet: Vec<f32> = water
        .iter()
        .zip(heights)
        .filter(|(w, _)| w.depth > min_depth)
        .map(|(w, g)| g + w.depth)
        .collect();
    let mean = wet.iter().sum::<f32>() / wet.len() as f32;
    let rms = (wet.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / wet.len() as f32).sqrt();
    let volume = water.iter().map(|w| w.depth).sum();
    (mean, rms, volume, wet.len())
}

/// Raising the bed under a lake moves the bed without resetting the water: the volume stays and
/// the surface rises by the displaced volume.
#[test]
#[ignore = "needs a graphics adapter"]
fn bed_moves_under_a_lake() {
    let mut h = Harness::new(WaterSolver::Swe, bowl(0.0));
    assert_bed(&h.read_heights(), bowl(0.0));
    h.pour(Vec2::splat(N as f32 / 2.0), 50.0, 0.3, 1.0);
    h.run(12.0);
    let (heights0, water0) = (h.read_heights(), h.read_water());
    let (mean0, rms0, volume0, wet0) = lake_stats(&heights0, &water0, 0.004);

    let raise = 0.02;
    h.set_bed(bowl(raise));
    h.run(10.0);
    let (heights1, water1) = (h.read_heights(), h.read_water());
    assert_bed(&heights1, bowl(raise));
    let (mean1, rms1, volume1, wet1) = lake_stats(&heights1, &water1, 0.004);

    // Volume the raised floor displaces, spread over the lake.
    let displaced: f32 = heights1.iter().zip(&heights0).map(|(a, b)| a - b).sum();
    let expected = displaced / wet0 as f32;
    println!(
        "bed under lake: surface {mean0:.4} (rms {rms0:.5}, {wet0} cells) -> {mean1:.4} \
         (rms {rms1:.5}, {wet1} cells), rise {:.4} for {expected:.4} expected, volume {volume0:.2} -> {volume1:.2}",
        mean1 - mean0
    );
    assert!(
        (volume1 - volume0).abs() < 0.01 * volume0,
        "volume {volume0} -> {volume1}"
    );
    // A seiche of a few hundredths of a cell lingers; height readback resolves about 0.00025.
    assert!(rms0 < 0.001 && rms1 < 0.001, "lake surface is not flat");
    assert!(
        ((mean1 - mean0) - expected).abs() < 0.25 * expected,
        "surface rose {} for {expected}",
        mean1 - mean0
    );
}

/// Ridge with its crest along x at y = N/2, flanks falling 0.003 per texel.
fn ridge(_x: u32, y: u32) -> f32 {
    0.4 - 0.003 * (y as f32 + 0.5 - N as f32 / 2.0).abs()
}

/// Power of the bed change along x (across the runoff) on both flanks, by period in texels.
fn cut_spectrum(before: &[f32], after: &[f32]) -> Vec<(f32, f32)> {
    let (x0, len) = (64u32, 128usize);
    let mut power = vec![0.0f32; len / 2 + 1];
    let rows = (N / 2 - 60..N / 2 - 8).chain(N / 2 + 8..N / 2 + 60);
    for y in rows {
        let cut: Vec<f32> = (x0..x0 + len as u32)
            .map(|x| after[idx(x, y)] - before[idx(x, y)])
            .collect();
        let mean = cut.iter().sum::<f32>() / len as f32;
        for (k, p) in power.iter_mut().enumerate().skip(1) {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for (j, c) in cut.iter().enumerate() {
                let a = 2.0 * PI * (k * j) as f32 / len as f32;
                re += (c - mean) * a.cos();
                im += (c - mean) * a.sin();
            }
            *p += re * re + im * im;
        }
    }
    power
        .iter()
        .enumerate()
        .skip(1)
        .map(|(k, p)| (len as f32 / k as f32, *p))
        .collect()
}

/// Runoff on a smooth ridge must leave its flanks alone. A truncating height write in
/// `water_erode` once lowered every wet cell by up to 1e-6 per step, depending on its exact
/// height, which cut stripes along the contours.
#[test]
#[ignore = "needs a graphics adapter"]
fn ridge_runoff_leaves_the_flanks_alone() {
    let solver = match std::env::var("TEST_SOLVER").as_deref() {
        Ok("pipes") => WaterSolver::Pipes,
        _ => WaterSolver::Swe,
    };
    let mut h = Harness::new(solver, ridge);
    h.water().sediment_capacity = WaterControl::default().sediment_capacity;
    let before = h.read_heights();
    assert_bed(&before, ridge);
    h.pour(Vec2::new(N as f32 / 2.0, N as f32 / 2.0), 40.0, 0.05, 6.0);
    if std::env::var("TEST_DEPTH_SPECTRUM").is_ok() {
        {
            let mut water = h.water();
            water.stroke_active = true;
        }
        let water = h.read_water();
        h.water().stroke_active = false;
        let depth: Vec<f32> = water.iter().map(|w| w.depth).collect();
        let zero = vec![0.0; depth.len()];
        let across = cut_spectrum(&zero, &depth);
        // Along the flow (y) on the lower flank: transpose and reuse.
        let t: Vec<f32> = (0..N * N).map(|i| depth[idx(i / N, i % N)]).collect();
        let along = cut_spectrum(&zero, &t);
        for (name, mut sp) in [("across", across), ("along", along)] {
            let total: f32 = sp.iter().map(|(_, p)| p).sum();
            sp.sort_by(|a, b| b.1.total_cmp(&a.1));
            let top: Vec<String> = sp[..8]
                .iter()
                .map(|(p, w)| format!("{p:.1} ({:.0}%)", 100.0 * w / total))
                .collect();
            println!("depth {name}: {}", top.join(", "));
        }
        let speeds: Vec<f32> = water.iter().map(|w| w.velocity.length()).collect();
        let fr_max = water
            .iter()
            .map(|w| {
                let hc = (w.depth * CELLS_PER_HEIGHT).max(1e-4);
                w.velocity.length() / (h.app.world().resource::<WaterControl>().gravity * hc).sqrt()
            })
            .filter(|f| f.is_finite())
            .fold(0.0f32, f32::max);
        println!(
            "max speed {:.1} texels/s, max Froude {fr_max:.2}",
            speeds.iter().fold(0.0f32, |a, b| a.max(*b))
        );
    }
    h.run(2.0);
    let after = h.read_heights();
    if let Ok(dir) = std::env::var("TEST_IMAGE_DIR") {
        let cut: Vec<f32> = after.iter().zip(&before).map(|(a, b)| a - b).collect();
        let m = cut.iter().fold(1e-6f32, |a, b| a.max(b.abs()));
        write_bmp(
            &format!("{dir}/ridge_flank_cut.bmp"),
            &cut,
            N as usize,
            N as usize,
            -m,
            m,
            2,
        );
    }
    let mut spectrum = cut_spectrum(&before, &after);
    let total: f32 = spectrum
        .iter()
        .map(|(_, p)| p)
        .sum::<f32>()
        .max(f32::MIN_POSITIVE);
    let cut_rms = {
        let d: Vec<f32> = before.iter().zip(&after).map(|(b, a)| a - b).collect();
        (d.iter().map(|v| v * v).sum::<f32>() / d.len() as f32).sqrt()
    };
    let share = |period: f32| {
        spectrum
            .iter()
            .filter(|(p, _)| (p - period).abs() < 0.01)
            .map(|(_, w)| w / total)
            .sum::<f32>()
    };
    let (p8, p4, p2) = (share(8.0), share(4.0), share(2.0));
    spectrum.sort_by(|a, b| b.1.total_cmp(&a.1));
    let top: Vec<String> = spectrum[..6]
        .iter()
        .map(|(p, w)| format!("{p:.1} ({:.0}%)", 100.0 * w / total))
        .collect();
    println!(
        "ridge runoff {solver:?}: cut rms {cut_rms:.5}, strongest periods {}, period 8 {:.1}% 4 {:.1}% 2 {:.1}%",
        top.join(", "),
        100.0 * p8,
        100.0 * p4,
        100.0 * p2
    );
    assert!(
        cut_rms < 0.0002,
        "the runoff cut {cut_rms} (rms) into a smooth ridge"
    );
}

/// Write a greyscale BMP of `values` (`w` × `h`, row 0 at the top) mapped from `lo..hi`, each
/// texel drawn as a `scale` × `scale` block.
fn write_bmp(path: &str, values: &[f32], w: usize, h: usize, lo: f32, hi: f32, scale: usize) {
    let (sw, sh) = (w * scale, h * scale);
    let row = (sw * 3).next_multiple_of(4);
    let mut out = Vec::with_capacity(54 + row * sh);
    let size = (54 + row * sh) as u32;
    out.extend_from_slice(b"BM");
    out.extend_from_slice(&size.to_le_bytes());
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(&54u32.to_le_bytes());
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&(sw as i32).to_le_bytes());
    out.extend_from_slice(&(-(sh as i32)).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    out.extend_from_slice(&[0; 24]);
    for y in 0..sh {
        let start = out.len();
        for x in 0..sw {
            let v = values[(y / scale) * w + x / scale];
            let g = (((v - lo) / (hi - lo)).clamp(0.0, 1.0) * 255.0) as u8;
            out.extend_from_slice(&[g, g, g]);
        }
        out.resize(start + row, 0);
    }
    std::fs::write(path, out).unwrap();
}

/// Pour on the sharpest ridge of the default terrain and save the depth mid-pour and the cut
/// it leaves, around the pour, to `TEST_IMAGE_DIR`.
#[test]
#[ignore = "needs a graphics adapter"]
fn natural_ridge_pour_images() {
    let solver = match std::env::var("TEST_SOLVER").as_deref() {
        Ok("pipes") => WaterSolver::Pipes,
        _ => WaterSolver::Swe,
    };
    let Ok(dir) = std::env::var("TEST_IMAGE_DIR") else {
        println!("set TEST_IMAGE_DIR to write the images");
        return;
    };
    let mut h = Harness::natural(solver);
    h.water().sediment_capacity = WaterControl::default().sediment_capacity;
    let before = h.read_heights();
    // Sharpest crest: height above the 17 × 17 box mean, away from the border.
    let (r, m) = (8i32, 40u32);
    let mut best = (0.0f32, N / 2, N / 2);
    for y in m..N - m {
        for x in m..N - m {
            let mut sum = 0.0;
            for dy in -r..=r {
                for dx in -r..=r {
                    sum += before[idx((x as i32 + dx) as u32, (y as i32 + dy) as u32)];
                }
            }
            let crest = before[idx(x, y)] - sum / ((2 * r + 1) * (2 * r + 1)) as f32;
            if crest > best.0 {
                best = (crest, x, y);
            }
        }
    }
    let (_, cx, cy) = best;
    {
        let mut water = h.water();
        water.stroke_active = true;
        water.center = Vec2::new(cx as f32, cy as f32);
        water.radius = 8.0;
        water.rate = 0.05;
    }
    let pour_seconds = std::env::var("TEST_POUR_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4.0);
    h.run(pour_seconds);
    let water = h.read_water();
    h.water().stroke_active = false;
    h.run(2.0);
    let after = h.read_heights();

    let half = 32u32;
    let window = |f: &dyn Fn(u32, u32) -> f32| -> Vec<f32> {
        (cy - half..cy + half)
            .flat_map(|y| (cx - half..cx + half).map(move |x| (x, y)))
            .map(|(x, y)| f(x, y))
            .collect()
    };
    let depth = window(&|x, y| water[idx(x, y)].depth);
    let cut = window(&|x, y| after[idx(x, y)] - before[idx(x, y)]);
    let speed = window(&|x, y| water[idx(x, y)].velocity.length());
    let bed = window(&|x, y| before[idx(x, y)]);
    let (dmax, cmax) = (
        depth.iter().fold(0.0f32, |a, b| a.max(*b)),
        cut.iter().fold(0.0f32, |a, b| a.max(b.abs())),
    );
    let (blo, bhi) = bed
        .iter()
        .fold((f32::MAX, f32::MIN), |(l, h), v| (l.min(*v), h.max(*v)));
    let w = (2 * half) as usize;
    write_bmp(
        &format!("{dir}/ridge_depth.bmp"),
        &depth,
        w,
        w,
        0.0,
        dmax,
        8,
    );
    write_bmp(&format!("{dir}/ridge_cut.bmp"), &cut, w, w, -cmax, cmax, 8);
    write_bmp(&format!("{dir}/ridge_bed.bmp"), &bed, w, w, blo, bhi, 8);
    // Shaded relief of a wider area before and after, lit from the north-west.
    let wide = 64u32;
    let (x0, y0) = (
        cx.saturating_sub(wide).max(1),
        cy.saturating_sub(wide).max(1),
    );
    let (x1, y1) = ((cx + wide).min(N - 1), (cy + wide).min(N - 1));
    for (name, hs) in [("before", &before), ("after", &after)] {
        let mut shade = Vec::new();
        for y in y0..y1 {
            for x in x0..x1 {
                let dx = (hs[idx(x + 1, y)] - hs[idx(x - 1, y)]) * 50.0;
                let dy = (hs[idx(x, y + 1)] - hs[idx(x, y - 1)]) * 50.0;
                let n = Vec3::new(-dx, 1.0, -dy).normalize();
                shade.push(n.dot(Vec3::new(-0.5, 1.0, -0.5).normalize()));
            }
        }
        let (sw, sh) = ((x1 - x0) as usize, (y1 - y0) as usize);
        write_bmp(
            &format!("{dir}/relief_{name}.bmp"),
            &shade,
            sw,
            sh,
            0.3,
            1.0,
            4,
        );
    }
    let smax = speed.iter().fold(0.0f32, |a, b| a.max(*b));
    write_bmp(
        &format!("{dir}/ridge_speed.bmp"),
        &speed,
        w,
        w,
        0.0,
        smax,
        8,
    );
    // Checkerboard share of the speed: |s - mean of 4 neighbours| relative to s, on wet cells.
    let mut odd = 0.0;
    let mut wet = 0.0;
    for y in 1..w - 1 {
        for x in 1..w - 1 {
            let c = speed[y * w + x];
            if depth[y * w + x] < 0.0005 {
                continue;
            }
            let n4 = (speed[y * w + x - 1]
                + speed[y * w + x + 1]
                + speed[(y - 1) * w + x]
                + speed[(y + 1) * w + x])
                / 4.0;
            odd += (c - n4).abs() / c.max(1e-3);
            wet += 1.0;
        }
    }
    println!(
        "speed max {smax:.1} texels/s, mean relative 4-neighbour jump {:.3}",
        odd / wet
    );
    // Speckle in the cut: mean |c - mean of 4 neighbours| over the mean |c|.
    let (mut jump, mut size) = (0.0, 0.0);
    for y in 1..w - 1 {
        for x in 1..w - 1 {
            let c = cut[y * w + x];
            let n4 = (cut[y * w + x - 1]
                + cut[y * w + x + 1]
                + cut[(y - 1) * w + x]
                + cut[(y + 1) * w + x])
                / 4.0;
            jump += (c - n4).abs();
            size += c.abs();
        }
    }
    println!(
        "natural ridge {solver:?} at ({cx}, {cy}): depth max {dmax:.4}, cut max {cmax:.5}, cut speckle {:.3}",
        jump / size
    );
    // Does the water follow and deepen grooves that were already there? Correlate the fine part
    // (minus the 4-neighbour mean) of the cut and of the depth with the bed's fine part.
    let fine = |f: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0; w * w];
        for y in 1..w - 1 {
            for x in 1..w - 1 {
                out[y * w + x] = f[y * w + x]
                    - (f[y * w + x - 1]
                        + f[y * w + x + 1]
                        + f[(y - 1) * w + x]
                        + f[(y + 1) * w + x])
                        / 4.0;
            }
        }
        out
    };
    let corr = |a: &[f32], b: &[f32]| {
        let n = a.len() as f32;
        let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
        let cov: f32 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
        let va: f32 = a.iter().map(|x| (x - ma).powi(2)).sum();
        let vb: f32 = b.iter().map(|y| (y - mb).powi(2)).sum();
        cov / (va * vb).sqrt()
    };
    let bed_fine = fine(&bed);
    println!(
        "groove correlation: cut {:.2}, depth {:.2} (negative = the water wears small bumps down)",
        corr(&fine(&cut), &bed_fine),
        corr(&fine(&depth), &bed_fine)
    );
}

/// Valley along x, falling toward +x, with its thalweg at y = N/2; low, so the f16 height
/// readback resolves about 3e-5.
fn valley(x: u32, y: u32) -> f32 {
    0.03 + 0.0003 * (N - 1 - x) as f32 + 0.0001 * (y as f32 + 0.5 - N as f32 / 2.0).abs()
}

/// Shape Beds from discharge: a spring-fed stream gets its channel on the first press, and a
/// second press changes nothing.
#[test]
#[ignore = "needs a graphics adapter"]
fn shape_beds_converges() {
    let mut h = Harness::new(WaterSolver::Swe, valley);
    h.water().springs.push(Spring {
        center: Vec2::new(24.0, N as f32 / 2.0),
        radius: 6.0,
        rate: 0.05,
    });
    h.run(40.0);
    let q = h.read_discharge();
    let water = h.read_water();
    let before = h.read_heights();
    let mid = N / 2;
    let q_mid = q[idx(N / 2, mid)].truncate().truncate().length();
    let channel = h.app.world().resource::<WaterControl>().bed.channel_discharge;
    let flowing = q
        .iter()
        .filter(|v| v.truncate().truncate().length() > channel)
        .count();
    println!(
        "valley stream: depth {:.4} speed {:.1} texels/s at the middle, unit discharge {q_mid:.3}, \
         {flowing} cells above the channel threshold",
        water[idx(N / 2, mid)].depth,
        water[idx(N / 2, mid)].velocity.length()
    );

    let press = |h: &mut Harness| {
        h.pause(true);
        let mut water = h.water();
        water.shape_generation = water.shape_generation.wrapping_add(1);
        h.frame();
        h.frame();
        h.read_heights()
    };
    let first = press(&mut h);
    if std::env::var("TEST_PROFILE").is_ok() {
        for x in (8..N - 8).step_by(16) {
            let w = q[idx(x, mid)];
            let wet = (0..N).filter(|&y| q[idx(x, y)].truncate().truncate().length() > channel).count();
            println!(
                "x {x}: cut {:.4} q {:.3} depth {:.4} flowing width {wet}",
                before[idx(x, mid)] - first[idx(x, mid)],
                w.truncate().truncate().length(),
                w.z
            );
        }
    }
    let second = press(&mut h);
    // Below the spring's pool, which is shaped as still water.
    let cut: Vec<f32> = (64..N - 40)
        .map(|x| before[idx(x, mid)] - first[idx(x, mid)])
        .collect();
    let mean_cut = cut.iter().sum::<f32>() / cut.len() as f32;
    let min_cut = cut.iter().fold(f32::MAX, |a, b| a.min(*b));
    let (again, at) = first
        .iter()
        .zip(&second)
        .enumerate()
        .fold((0.0f32, 0), |a, (i, (x, y))| if (x - y).abs() > a.0 { ((x - y).abs(), i) } else { a });
    println!("largest second-press change at ({}, {})", at as u32 % N, at as u32 / N);
    // Cross-section at the middle: the thalweg is the lowest point and the banks rise from it.
    let section: Vec<f32> = (mid - 6..=mid + 6)
        .map(|y| first[idx(N / 2, y)] - before[idx(N / 2, y)])
        .collect();
    println!(
        "shape beds: thalweg cut mean {mean_cut:.4} min {min_cut:.4}, second press changes {again:.6}, \
         cross-section change {:?}",
        section.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>()
    );
    // Keep shaping with the water running between presses: the cut settles.
    h.pause(false);
    h.run(10.0);
    let third = press(&mut h);
    let drift = (64..N - 40)
        .map(|x| (third[idx(x, mid)] - first[idx(x, mid)]).abs())
        .fold(0.0f32, f32::max);
    println!("after 10 s of flow, another press moves the thalweg by up to {drift:.5}");
    assert!(min_cut > 0.002, "the stream got no channel ({min_cut})");
    assert!(again < 1e-4, "a second press changed the bed by {again}");
    assert!(
        section[6] <= section[0] && section[6] <= section[12],
        "the thalweg is not the deepest cut"
    );
    assert!(
        drift < 0.4 * mean_cut,
        "pressing again after the flow settled cut {drift} deeper"
    );
}

/// The shared discharge field: on the default terrain with KTT rivers, the solver's discharge
/// from a spring runs where KTT's rivers are. Reports their correlation and the calibration
/// of `KTT_DISCHARGE_SCALE`.
#[test]
#[ignore = "needs a graphics adapter"]
fn discharge_matches_ktt() {
    let mut h = Harness::natural(WaterSolver::Swe);
    {
        let mut params = h.app.world_mut().resource_mut::<ErodeParams>();
        params.do_rivers = 1;
        params.momentum_coherence = 0.5;
    }
    let mut reset = h.app.world_mut().resource_mut::<ResetSim>();
    reset.generation = reset.generation.wrapping_add(1);
    h.settle_cascade();
    // One frame of water so the discharge field picks up KTT's rivers.
    h.water().rainfall = 1e-9;
    h.run(0.1);
    let heights = h.read_heights();
    let ktt: Vec<f32> = h.read_discharge().iter().map(|v| v.w).collect();
    // Spring on a strong river high up: the largest KTT discharge above the median height.
    let mut sorted = heights.clone();
    sorted.sort_by(f32::total_cmp);
    let median = sorted[sorted.len() / 2];
    let m = 24;
    let (mut best, mut at) = (0.0f32, (N / 2, N / 2));
    for y in m..N - m {
        for x in m..N - m {
            let i = idx(x, y);
            if heights[i] > median && ktt[i] > best {
                best = ktt[i];
                at = (x, y);
            }
        }
    }
    h.water().rainfall = 0.0;
    h.water().springs.push(Spring {
        center: Vec2::new(at.0 as f32, at.1 as f32),
        radius: 6.0,
        rate: 0.05,
    });
    h.run(20.0);
    let d = h.read_discharge();
    let solver: Vec<f32> = d.iter().map(|v| v.truncate().truncate().length()).collect();
    // Cells the spring's water reaches.
    let cells: Vec<usize> = (0..d.len()).filter(|&i| solver[i] > 0.01).collect();
    let mean = |f: &dyn Fn(usize) -> f32| cells.iter().map(|&i| f(i)).sum::<f32>() / cells.len() as f32;
    let (ms, mk) = (mean(&|i| solver[i]), mean(&|i| ktt[i]));
    let cov = mean(&|i| (solver[i] - ms) * (ktt[i] - mk));
    let (vs, vk) = (
        mean(&|i| (solver[i] - ms).powi(2)),
        mean(&|i| (ktt[i] - mk).powi(2)),
    );
    let corr = cov / (vs * vk).sqrt().max(1e-12);
    // Share of the solver's strongest discharge that sits on KTT's upper-quartile rivers.
    let mut k_sorted: Vec<f32> = ktt.clone();
    k_sorted.sort_by(f32::total_cmp);
    let k75 = k_sorted[k_sorted.len() * 3 / 4];
    let mut top = cells.clone();
    top.sort_by(|&a, &b| solver[b].total_cmp(&solver[a]));
    top.truncate(200);
    let on_river = top.iter().filter(|&&i| ktt[i] > k75).count() as f32 / top.len().max(1) as f32;
    let mut ratios: Vec<f32> = top
        .iter()
        .filter(|&&i| ktt[i] > 0.0)
        .map(|&i| solver[i] / ktt[i])
        .collect();
    ratios.sort_by(f32::total_cmp);
    let ratio = ratios.get(ratios.len() / 2).copied().unwrap_or(0.0);
    println!(
        "discharge: spring at {at:?} (KTT {best:.3}), {} wet cells, correlation {corr:.2}, \
         {:.0}% of the 200 strongest on KTT's upper-quartile rivers, solver / KTT {ratio:.2} \
         (scale KTT_DISCHARGE_SCALE by this)",
        cells.len(),
        100.0 * on_river
    );
    assert!(cells.len() > 100, "the spring did not flow");
    assert!(on_river > 0.5, "the spring's water does not follow KTT's rivers");
    assert!(d.iter().all(|v| v.is_finite()), "discharge holds NaN");
}

/// Straight channel along x, `width` texels wide, falling `slope` per texel toward +x, between
/// walls 0.1 high.
fn flume(slope: f32, width: f32) -> impl Fn(u32, u32) -> f32 {
    move |x, y| {
        let floor = 0.02 + slope * (N - 1 - x) as f32;
        if (y as f32 + 0.5 - N as f32 / 2.0).abs() < width / 2.0 {
            floor
        } else {
            floor + 0.1
        }
    }
}

struct FlumeRun {
    before: Vec<f32>,
    after: Vec<f32>,
    water: Vec<Water>,
}

/// Live bed load on a flume fed by a spring at its top, for `seconds`.
fn run_flume(bed: impl Fn(u32, u32) -> f32, radius: f32, rate: f32, seconds: f32) -> FlumeRun {
    let mut h = Harness::new(WaterSolver::Swe, bed);
    {
        let mut water = h.water();
        water.bed_load.live = true;
        if let Some(a) = std::env::var("TEST_SLOPE_EFFECT").ok().and_then(|v| v.parse().ok()) {
            water.bed_load.slope_effect = a;
        }
        water.springs.push(Spring {
            center: Vec2::new(12.0, N as f32 / 2.0),
            radius,
            rate,
        });
    }
    // Let the flow reach the far end before the bed starts to count.
    h.water().bed_load.live = false;
    h.run(6.0);
    let before = h.read_heights();
    h.water().bed_load.live = true;
    h.run(seconds);
    if let Some(steps) = std::env::var("TEST_REFINE").ok().and_then(|v| v.parse().ok()) {
        h.water().bed_load.refine_steps = steps;
        {
            let mut water = h.water();
            water.refine_generation = water.refine_generation.wrapping_add(1);
        }
        h.run(steps as f32 / 64.0 / FPS + 1.0);
    }
    let water = h.read_water();
    let after = h.read_heights();
    assert!(
        after.iter().all(|v| v.is_finite()),
        "heights hold NaN or infinity"
    );
    FlumeRun {
        before,
        after,
        water,
    }
}

/// Dominant period (texels) of `v` after removing its linear trend, that period's share of the
/// power, the share of periods 8 to 40, and the RMS of the detrended profile.
fn dominant_period(v: &[f32]) -> (f32, f32, f32, f32) {
    let n = v.len();
    let xm = (n as f32 - 1.0) / 2.0;
    let ym = v.iter().sum::<f32>() / n as f32;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (i, y) in v.iter().enumerate() {
        sxy += (i as f32 - xm) * (y - ym);
        sxx += (i as f32 - xm).powi(2);
    }
    let b = sxy / sxx;
    let r: Vec<f32> = v
        .iter()
        .enumerate()
        .map(|(i, y)| y - ym - b * (i as f32 - xm))
        .collect();
    let mut best = (0.0, 0.0);
    let mut total = 0.0;
    let mut band = 0.0;
    for k in 1..n / 2 {
        let (mut re, mut im) = (0.0f32, 0.0f32);
        for (j, c) in r.iter().enumerate() {
            let a = 2.0 * PI * (k * j) as f32 / n as f32;
            re += c * a.cos();
            im += c * a.sin();
        }
        let p = re * re + im * im;
        total += p;
        if (8.0..=40.0).contains(&(n as f32 / k as f32)) {
            band += p;
        }
        if p > best.1 {
            best = (n as f32 / k as f32, p);
        }
    }
    let total = total.max(f32::MIN_POSITIVE);
    let rms = (r.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt();
    (best.0, best.1 / total, band / total, rms)
}

/// Bed load on a gentle and a steep flume: the gentle bed lowers without grid grooves; the steep
/// one, in supercritical flow, grows steps.
#[test]
#[ignore = "needs a graphics adapter"]
fn bed_load_on_flumes() {
    let mid = N / 2;
    let g = WaterControl::default().gravity;
    for (name, slope, width) in [
        ("gentle", 0.0003, 20.0f32),
        ("steep", 0.004, 20.0),
        ("narrow", 0.006, 6.0),
    ] {
        let seconds = std::env::var("TEST_FLUME_SECONDS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(20.0);
        // The spring fits inside the channel; narrower channels get a stronger one.
        let radius = (width / 2.0 - 1.0).min(6.0);
        let r = run_flume(flume(slope, width), radius, 0.05 * 36.0 / (radius * radius), seconds);
        let xs = 48..N - 24;
        let cut: Vec<f32> = xs
            .clone()
            .map(|x| r.after[idx(x, mid)] - r.before[idx(x, mid)])
            .collect();
        let mean_cut = cut.iter().sum::<f32>() / cut.len() as f32;
        // Cross-channel unevenness: the cut minus its row mean, RMS over the channel.
        let mut across = 0.0;
        let mut count = 0.0;
        for x in xs.clone() {
            let row: Vec<f32> = (mid - 6..mid + 6)
                .map(|y| r.after[idx(x, y)] - r.before[idx(x, y)])
                .collect();
            if width < 12.0 {
                continue;
            }
            let m = row.iter().sum::<f32>() / row.len() as f32;
            for v in &row {
                across += (v - m).powi(2);
                count += 1.0;
            }
        }
        let across = (across / count).sqrt();
        let profile: Vec<f32> = xs.clone().map(|x| r.after[idx(x, mid)]).collect();
        if std::env::var("TEST_PROFILE").is_ok() {
            let detrended: Vec<String> = profile
                .iter()
                .zip(xs.clone())
                .step_by(2)
                .map(|(z, x)| format!("{:.4}", z - r.before[idx(x, mid)]))
                .collect();
            println!("{name} bed change along the centreline: {}", detrended.join(" "));
        }
        let (period, share, steps, relief) = dominant_period(&profile);
        // Deepest cell across the channel at the middle (the thalweg wanders between bars).
        let w = (mid - 10..mid + 10)
            .map(|y| r.water[idx(N / 2, y)])
            .max_by(|a, b| a.depth.total_cmp(&b.depth))
            .unwrap();
        let hc = w.depth * CELLS_PER_HEIGHT;
        let froude = w.velocity.length() / (g * hc.max(1e-4)).sqrt();
        let theta = {
            let n = WaterControl::default().manning_n;
            let grain = BedLoad::default().grain;
            n * n * w.velocity.length().powi(2) / (grain * hc.max(1e-4).cbrt())
        };
        println!(
            "{name} flume: depth {:.2} cells, speed {:.1}, Froude {froude:.2}, Shields {theta:.3}; \
             mean cut {mean_cut:.5}, cross-channel rms {across:.5}, long profile period {period:.1} \
             ({:.0}% of the detrended power, periods 8-40 {:.0}%, relief rms {relief:.4})",
            hc,
            w.velocity.length(),
            100.0 * share,
            100.0 * steps
        );
        // Dry ground far from the channel is left alone.
        let dry = (0..N)
            .flat_map(|x| (0..mid - 40).map(move |y| idx(x, y)))
            .map(|i| (r.after[i] - r.before[i]).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(dry, 0.0, "{name}: dry ground changed");
        match name {
            "gentle" => assert!(across < 0.0005, "grooves across the gentle flume ({across})"),
            "narrow" => assert!(
                steps > 0.5 && relief > 0.002,
                "no steps on the steep narrow flume"
            ),
            _ => {}
        }
        if let Ok(dir) = std::env::var("TEST_IMAGE_DIR") {
            let d: Vec<f32> = r.after.iter().zip(&r.before).map(|(a, b)| a - b).collect();
            let m = d.iter().fold(1e-6f32, |a, b| a.max(b.abs()));
            write_bmp(&format!("{dir}/flume_{name}_cut.bmp"), &d, N as usize, N as usize, -m, m, 2);
            let depth: Vec<f32> = r.water.iter().map(|w| w.depth).collect();
            let dm = depth.iter().fold(1e-6f32, |a, b| a.max(*b));
            write_bmp(&format!("{dir}/flume_{name}_depth.bmp"), &depth, N as usize, N as usize, 0.0, dm, 2);
        }
    }
}

/// A steep flume running into a closed basin: bed load is conserved (nothing reaches the map
/// border) and drops where the jet slows, building a bar at the mouth.
#[test]
#[ignore = "needs a graphics adapter"]
fn bed_load_conserves_and_builds_a_delta() {
    let mouth = 170u32;
    let basin = move |x: u32, y: u32| {
        let dy = (y as f32 + 0.5 - N as f32 / 2.0).abs();
        if (mouth..mouth + 60).contains(&x) && dy < 40.0 {
            0.02
        } else if x < mouth && dy < 6.0 {
            0.06 + 0.002 * (mouth - x) as f32
        } else {
            0.5
        }
    };
    let r = run_flume(basin, 5.0, 0.07, 30.0);
    let dh: Vec<f32> = r.after.iter().zip(&r.before).map(|(a, b)| a - b).collect();
    let net: f32 = dh.iter().sum();
    let moved: f32 = dh.iter().map(|v| v.abs()).sum::<f32>() / 2.0;
    let mid = N / 2;
    let near: f32 = (mouth..mouth + 15)
        .flat_map(|x| (mid - 8..mid + 8).map(move |y| (x, y)))
        .map(|(x, y)| dh[idx(x, y)])
        .sum::<f32>()
        / (15.0 * 16.0);
    let far: f32 = (mouth + 40..mouth + 55)
        .flat_map(|x| (mid - 8..mid + 8).map(move |y| (x, y)))
        .map(|(x, y)| dh[idx(x, y)])
        .sum::<f32>()
        / (15.0 * 16.0);
    println!(
        "delta: moved {moved:.4}, net {net:.5} ({:.2}%), mean change at the mouth {near:.5}, \
         far in the basin {far:.5}",
        100.0 * net / moved.max(1e-9)
    );
    if let Ok(dir) = std::env::var("TEST_IMAGE_DIR") {
        let m = dh.iter().fold(1e-6f32, |a, b| a.max(b.abs()));
        write_bmp(&format!("{dir}/delta_change.bmp"), &dh, N as usize, N as usize, -m, m, 2);
    }
    assert!(moved > 0.05, "the flow moved no bed ({moved})");
    assert!(net.abs() < 0.02 * moved, "bed volume changed by {net} of {moved} moved");
    assert!(near > 2.0 * far.max(0.0) && near > 0.0, "no bar at the mouth");
}

/// Meandering channel: centreline y = N/2 + 24 sin(2π x / 128), 16 texels wide, falling toward
/// +x, with banks rising 0.0025 per texel (below the wet slumping angle).
fn s_bend(x: u32, y: u32) -> f32 {
    let c = N as f32 / 2.0 + 24.0 * (2.0 * PI * x as f32 / 128.0).sin();
    let floor = 0.02 + 0.0015 * (N - 1 - x) as f32;
    floor + 0.0025 * ((y as f32 + 0.5 - c).abs() - 8.0).max(0.0)
}

/// Where the bed was cut at the bend apexes (x = 32 + 64k): the cut-weighted offset from the
/// centreline toward the outer bank, per apex.
fn apex_scour(helical: f32) -> Vec<f32> {
    let mut h = Harness::new(WaterSolver::Swe, s_bend);
    {
        let mut water = h.water();
        water.bed_load.helical_strength = helical;
        water.springs.push(Spring {
            center: Vec2::new(10.0, N as f32 / 2.0 + 24.0 * (2.0 * PI * 10.0 / 128.0).sin()),
            radius: 6.0,
            rate: 0.05,
        });
    }
    h.run(6.0);
    let before = h.read_heights();
    h.water().bed_load.live = true;
    h.run(25.0);
    let after = h.read_heights();
    [96u32, 160]
        .iter()
        .map(|&x| {
            let s = (2.0 * PI * x as f32 / 128.0).sin();
            let c = N as f32 / 2.0 + 24.0 * s;
            // The bend curves around -s: its outer bank is on the side of +s.
            let (mut moment, mut total) = (0.0, 0.0);
            // Inside the channel, relative to the section's mean change.
            let ys: Vec<u32> = ((c - 7.0).ceil() as u32..=(c + 6.0).floor() as u32).collect();
            let dzs: Vec<f32> = ys.iter().map(|&y| after[idx(x, y)] - before[idx(x, y)]).collect();
            let mean = dzs.iter().sum::<f32>() / dzs.len() as f32;
            let section: Vec<String> = dzs.iter().map(|d| format!("{d:.4}")).collect();
            for (&y, dz) in ys.iter().zip(&dzs) {
                let cut = (mean - dz).max(0.0);
                moment += cut * (y as f32 + 0.5 - c) * s.signum();
                total += cut;
            }
            println!("apex x {x} (outer side {}): {}", if s > 0.0 { "+y" } else { "-y" }, section.join(" "));
            moment / total.max(1e-9)
        })
        .collect()
}

/// Helical flow moves bed load toward the inner banks of bends: the outer bank scours, on
/// opposite sides in successive bends.
#[test]
#[ignore = "needs a graphics adapter"]
fn bends_scour_their_outer_banks() {
    let strength = std::env::var("TEST_HELICAL").ok().and_then(|v| v.parse().ok()).unwrap_or(BedLoad::default().helical_strength);
    let with = apex_scour(strength);
    let without = apex_scour(0.0);
    println!(
        "bend scour toward the outer bank (texels): helical {strength}: {with:?}, without: {without:?}"
    );
    for (w, wo) in with.iter().zip(&without) {
        assert!(*w > 0.5, "the scour is not on the outer bank ({w})");
        assert!(
            *w > wo + 0.2,
            "helical flow did not move the scour outward ({w} against {wo})"
        );
    }
}

/// Refine Beds runs its steps with the morphological factor, also while paused: the bed moves
/// many times more than in the same number of live steps.
#[test]
#[ignore = "needs a graphics adapter"]
fn refine_fast_forwards_the_bed() {
    // Banks below the slumping angle, so only bed load moves the bed.
    let mut h = Harness::new(WaterSolver::Swe, s_bend);
    h.water().springs.push(Spring {
        center: Vec2::new(10.0, N as f32 / 2.0 + 24.0 * (2.0 * PI * 10.0 / 128.0).sin()),
        radius: 6.0,
        rate: 0.05,
    });
    h.run(6.0);
    let start = h.read_heights();
    let steps = BedLoad::default().refine_steps;
    // Live: the same number of steps (8 per frame at 60 fps).
    h.water().bed_load.live = true;
    for _ in 0..steps / 8 {
        h.frame();
    }
    h.water().bed_load.live = false;
    let live = h.read_heights();
    h.pause(true);
    {
        let mut water = h.water();
        water.refine_generation = water.refine_generation.wrapping_add(1);
    }
    for _ in 0..steps.div_ceil(64) + 2 {
        h.frame();
    }
    let refined = h.read_heights();
    let moved = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>();
    let (live_moved, refine_moved) = (moved(&live, &start), moved(&refined, &live));
    println!(
        "refine: {steps} live steps moved {live_moved:.3}, {steps} refine steps (factor {}) moved {refine_moved:.3}",
        BedLoad::default().morph_factor
    );
    assert!(refined.iter().all(|v| v.is_finite()), "NaN after refining");
    assert!(
        refine_moved > 5.0 * live_moved,
        "refining did not fast-forward the bed"
    );
}

/// Hold the rain brush over `center` for `seconds` at `fps`, as the demo does each frame.
fn rain(h: &mut Harness, center: Vec2, seconds: f32, fps: f32) {
    h.set_fps(fps);
    let brush = bevy_erosion::brush::WaterBrush::default();
    for _ in 0..(seconds * fps).round() as u32 {
        *h.app.world_mut().resource_mut::<BrushStroke>() = brush.rain_stroke(center, 1.0 / fps);
        h.frame();
    }
    *h.app.world_mut().resource_mut::<BrushStroke>() = BrushStroke::default();
    h.set_fps(FPS);
}

/// The rain brush leaves water along its trails, the same amount at any frame rate.
#[test]
#[ignore = "needs a graphics adapter"]
fn rain_wets_the_trails() {
    let mut volumes = Vec::new();
    for fps in [30.0, 120.0] {
        let mut h = Harness::new(WaterSolver::Swe, bowl(0.0));
        rain(&mut h, Vec2::new(N as f32 / 2.0 + 60.0, N as f32 / 2.0), 1.0, fps);
        h.run(0.5);
        let water = h.read_water();
        volumes.push(water.iter().map(|w| w.depth).sum::<f32>());
    }
    println!("rain water: {:.4} at 30 fps, {:.4} at 120 fps", volumes[0], volumes[1]);
    assert!(volumes[0] > 0.01, "the rain brush left no water");
    assert!(
        (volumes[0] - volumes[1]).abs() < 0.1 * volumes[0],
        "the water rain leaves depends on the frame rate"
    );
}

/// Trails that reach a lake hand it their sediment, which settles in the lake; Clear Water
/// settles what is still suspended instead of deleting it.
#[test]
#[ignore = "needs a graphics adapter"]
fn trail_sediment_reaches_the_lake() {
    let mut h = Harness::new(WaterSolver::Swe, bowl(0.0));
    {
        let mut params = h.app.world_mut().resource_mut::<ErodeParams>();
        // Trails carve again (the harness turns erosion off to keep its beds).
        params.erosion_strength = 0.5;
    }
    h.water().rain_water = 0.0;
    // The capacity model keeps some of the load suspended for Clear Water to settle.
    h.water().sediment_capacity = WaterControl::default().sediment_capacity;
    h.pour(Vec2::splat(N as f32 / 2.0), 40.0, 0.3, 1.0);
    h.run(4.0);
    let heights0 = h.read_heights();
    let lake: Vec<bool> = h.read_water().iter().map(|w| w.depth > 0.004).collect();
    rain(&mut h, Vec2::new(N as f32 / 2.0 + 62.0, N as f32 / 2.0), 2.0, FPS);
    let read_sediment = |h: &mut Harness| -> Vec<f32> {
        let handle = h.app.world().resource::<ErosionImages>().water.clone();
        h.read(handle).iter().map(|t| t[3]).collect()
    };
    let sediment = read_sediment(&mut h);
    let heights1 = h.read_heights();
    let suspended: f32 = sediment.iter().sum();
    let cut: f32 = heights0.iter().zip(&heights1).map(|(a, b)| (a - b).max(0.0)).sum();
    let in_lake: f32 = (0..heights0.len())
        .filter(|&i| lake[i])
        .map(|i| (heights1[i] - heights0[i]).max(0.0))
        .sum();
    h.run(4.0);
    let heights2 = h.read_heights();
    let settled: f32 = (0..heights0.len())
        .filter(|&i| lake[i])
        .map(|i| (heights2[i] - heights0[i]).max(0.0))
        .sum();
    let sediment2 = read_sediment(&mut h);
    // The water's deposits count as KTT's loose sediment (analysis G).
    let handle = h.app.world().resource::<ErosionImages>().analysis.clone();
    let analysis = h.read(handle);
    let shown: f32 = (0..heights0.len())
        .filter(|&i| lake[i] && heights2[i] > heights0[i])
        .map(|i| analysis[i][1])
        .sum();
    println!("loose sediment shown under the lake deposits: {shown:.3}");
    assert!(shown > 0.0, "the water's deposits are not in KTT's deposition buffer");
    let before_clear: f32 = h.read_heights().iter().sum();
    {
        let mut water = h.water();
        water.clear_generation = water.clear_generation.wrapping_add(1);
    }
    h.run(0.1);
    let after_clear: f32 = h.read_heights().iter().sum();
    let suspended2: f32 = sediment2.iter().sum();
    println!(
        "trail sediment: trails cut {cut:.3}; right after the rain {suspended:.3} suspended and {in_lake:.3} \
         settled in the lake; 4 s later {settled:.3} settled and {suspended2:.3} suspended; \
         Clear Water settled {:.3}",
        after_clear - before_clear
    );
    assert!(cut > 0.01, "the rain carved nothing");
    assert!(
        settled + suspended2 > 0.3 * cut,
        "little of the carved sediment reached the lake"
    );
    assert!(suspended2 > 0.001, "nothing stayed suspended to check Clear Water with");
    // Heights read back as f16 (about 0.00024 apart here), so the sum of many small deposits is
    // only good to tens of percent; before this, Clear Water deleted the load (0 settled).
    assert!(
        after_clear - before_clear > 0.5 * suspended2,
        "Clear Water lost the suspended load"
    );
}

/// KTT trails on a plane with a spring-fed stream: their river momentum (the Rivers view)
/// inside the stream against the plane around it, and the stream's wet cells.
fn trail_cut_along_stream(steer: bool) -> (f32, f32) {
    let plane = |x: u32, _y: u32| 0.03 + 0.001 * (N - 1 - x) as f32;
    let mut h = Harness::new(WaterSolver::Swe, plane);
    h.water().steer_rivers = steer;
    h.water().springs.push(Spring {
        center: Vec2::new(24.0, N as f32 / 2.0),
        radius: 4.0,
        rate: 0.05,
    });
    h.run(6.0);
    {
        let mut params = h.app.world_mut().resource_mut::<ErodeParams>();
        params.erosion_amount = 0.1;
        params.erosion_strength = 0.5;
        params.do_rivers = 1;
        params.momentum_coherence = 0.5;
    }
    h.app
        .world_mut()
        .resource_mut::<SimControl>()
        .continue_after_cascade = true;
    h.run(6.0);
    let q = h.read_discharge();
    let handle = h.app.world().resource::<ErosionImages>().height.clone();
    let analysis = h.read(handle);
    let (mut inside, mut n_in, mut outside, mut n_out) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for y in N / 2 - 40..N / 2 + 40 {
        for x in 48..N - 24 {
            let i = idx(x, y);
            if q[i].truncate().truncate().length() > 0.02 {
                inside += analysis[i][2];
                n_in += 1.0;
            } else {
                outside += analysis[i][2];
                n_out += 1.0;
            }
        }
    }
    (inside / n_in.max(1.0) / (outside / n_out.max(1.0)).max(1e-9), n_in)
}

/// With KTT's rivers on, the water's discharge steers the trails: they cut along the stream
/// rather than across the whole plane. (On a uniform plane without river momentum, trails drop
/// their load as fast as they pick it up and cut nothing anywhere.)
#[test]
#[ignore = "needs a graphics adapter"]
fn water_steers_ktt_rivers() {
    let (with, wet) = trail_cut_along_stream(true);
    let (without, _) = trail_cut_along_stream(false);
    println!(
        "trail cut in the stream ({wet} cells) against the plane around it: {with:.2}× steered, {without:.2}× not"
    );
    assert!(wet > 50.0, "no stream formed");
    assert!(
        with > 2.0 && with > 1.5 * without,
        "the water did not steer the trails"
    );
}

/// Live bed load on the steep 20-texel flume, with KTT's params changed by `setup` (the cascade
/// reruns so the masks and bedrock follow them): the deepest cut and the total bed moved.
fn flume_cut(slope: f32, setup: impl Fn(&mut ErodeParams), hardness: f32) -> (f32, f32) {
    let bed = flume(slope, 20.0);
    let mut h = Harness::new(WaterSolver::Swe, &bed);
    setup(&mut h.app.world_mut().resource_mut::<ErodeParams>());
    h.set_bed(&bed);
    {
        let mut water = h.water();
        water.bed_load.rock_hardness = hardness;
        water.springs.push(Spring {
            center: Vec2::new(12.0, N as f32 / 2.0),
            radius: 6.0,
            rate: 0.05,
        });
    }
    h.run(6.0);
    let before = h.read_heights();
    h.water().bed_load.live = std::env::var("TEST_NO_BED_LOAD").is_err();
    h.run(20.0);
    let after = h.read_heights();
    let mid = N / 2;
    // The deepest cut below the drawn bed (the walls slump into the channel and raise it first),
    // and the bed moved while bed load ran.
    let (mut deepest, mut moved) = (0.0f32, 0.0f32);
    for y in mid - 9..mid + 9 {
        for x in 24..N - 8 {
            deepest = deepest.max(bed(x, y) - after[idx(x, y)]);
            moved += (after[idx(x, y)] - before[idx(x, y)]).abs();
        }
    }
    (deepest, moved)
}

/// Bed load respects KTT's rock: it never cuts below the bedrock layer, and hard rock in the
/// softness mask moves less.
#[test]
#[ignore = "needs a graphics adapter"]
fn bed_load_respects_rock() {
    let depth = 0.001;
    let with_bedrock = |p: &mut ErodeParams| {
        p.use_bedrock = 1;
        p.bedrock_depth = depth;
    };
    let (free, _) = flume_cut(0.002, |_| {}, 0.0);
    let (floored, _) = flume_cut(0.002, with_bedrock, 0.0);
    // A gentler flume, where the flow is near the threshold of motion.
    let softness = |p: &mut ErodeParams| p.use_softness_mask = 1;
    let (_, soft) = flume_cut(0.0008, softness, 0.0);
    let (_, hard) = flume_cut(0.0008, softness, BedLoad::default().rock_hardness);
    println!(
        "rock: deepest cut {free:.4} free, {floored:.4} above bedrock {depth} below; bed moved \
         {soft:.2} with the softness mask at hardness 0, {hard:.2} at hardness {}",
        BedLoad::default().rock_hardness
    );
    assert!(free > 2.0 * depth, "the free bed did not cut deep enough to test the bedrock");
    // f16 readback resolves about 0.00025 at these heights.
    assert!(floored < depth + 0.0005, "bed load cut {floored} below the bedrock");
    assert!(hard < 0.8 * soft, "hard rock moved as much as soft");
}

// ---- Validation sweeps against the literature (docs/VALIDATION.md). Each writes a CSV to
// `TEST_DATA_DIR` (default `target/validation`) and prints its rows.

fn data_file(name: &str, header: &str) -> std::fs::File {
    use std::io::Write;
    let dir = std::env::var("TEST_DATA_DIR").unwrap_or_else(|_| "target/validation".into());
    std::fs::create_dir_all(&dir).unwrap();
    let mut f = std::fs::File::create(format!("{dir}/{name}.csv")).unwrap();
    writeln!(f, "{header}").unwrap();
    f
}

fn row(f: &mut std::fs::File, values: &[f32]) {
    use std::io::Write;
    let line: Vec<String> = values.iter().map(|v| format!("{v:.5}")).collect();
    println!("  {}", line.join(", "));
    writeln!(f, "{}", line.join(",")).unwrap();
}

/// Detrended coefficient of variation of `v` (std of the residual from a linear fit / mean).
fn detrended_cv(v: &[f32]) -> f32 {
    let n = v.len() as f32;
    let xm = (n - 1.0) / 2.0;
    let ym = v.iter().sum::<f32>() / n;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (i, y) in v.iter().enumerate() {
        sxy += (i as f32 - xm) * (y - ym);
        sxx += (i as f32 - xm).powi(2);
    }
    let b = sxy / sxx;
    let var = v
        .iter()
        .enumerate()
        .map(|(i, y)| (y - ym - b * (i as f32 - xm)).powi(2))
        .sum::<f32>()
        / n;
    var.sqrt() / ym.abs().max(1e-9)
}

/// Roll waves: Manning theory (Jeffreys 1925; Vedernikov) puts their onset in a wide channel at
/// Froude 1.5. Sweep the Froude number with Manning n on a fixed-bed flume and measure the depth
/// unevenness along the flow.
#[test]
#[ignore = "needs a graphics adapter"]
fn validate_roll_wave_onset() {
    let mut f = data_file(
        "roll_waves",
        "manning_n,depth_cells,speed,froude,depth_cv,cv_upstream,cv_downstream",
    );
    let g = WaterControl::default().gravity;
    let mid = N / 2;
    for n in [0.016, 0.012, 0.0095, 0.008, 0.007, 0.006, 0.005, 0.0042, 0.0035, 0.0028, 0.0022, 0.0017, 0.0013] {
        let mut h = Harness::new(WaterSolver::Swe, flume(0.002, 24.0));
        h.water().manning_n = n;
        h.water().second_order_depth = std::env::var("TEST_SECOND_ORDER").is_ok();
        h.water().springs.push(Spring {
            center: Vec2::new(12.0, N as f32 / 2.0),
            radius: 6.0,
            rate: 0.05,
        });
        // Long enough for the front to leave the flume at the slowest flow.
        h.run(25.0);
        let (mut cv, mut up, mut down, mut depth, mut speed) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for _ in 0..3 {
            let water = h.read_water();
            let line: Vec<f32> = (64..N - 16).map(|x| water[idx(x, mid)].depth).collect();
            cv += detrended_cv(&line) / 3.0;
            // An instability grows as it travels: compare the two halves of the reach.
            up += detrended_cv(&line[..line.len() / 2]) / 3.0;
            down += detrended_cv(&line[line.len() / 2..]) / 3.0;
            depth += line.iter().sum::<f32>() / line.len() as f32 * CELLS_PER_HEIGHT / 3.0;
            speed += (64..N - 16)
                .map(|x| water[idx(x, mid)].velocity.length())
                .sum::<f32>()
                / (N - 80) as f32
                / 3.0;
            h.run(0.3);
        }
        let froude = speed / (g * depth).sqrt();
        row(&mut f, &[n, depth, speed, froude, cv, up, down]);
    }
}

/// Residual of `v` from its linear fit, and the fitted slope per sample.
fn detrend(v: &[f32]) -> (Vec<f32>, f32) {
    let n = v.len() as f32;
    let xm = (n - 1.0) / 2.0;
    let ym = v.iter().sum::<f32>() / n;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (i, y) in v.iter().enumerate() {
        sxy += (i as f32 - xm) * (y - ym);
        sxx += (i as f32 - xm).powi(2);
    }
    let b = sxy / sxx;
    let r = v
        .iter()
        .enumerate()
        .map(|(i, y)| y - ym - b * (i as f32 - xm))
        .collect();
    (r, b)
}

/// Shift in samples that best maps `a` onto `b`, from the peak of their normalised
/// cross-correlation within `max_lag`, refined with a parabola through the peak.
fn profile_shift(a: &[f32], b: &[f32], max_lag: i32) -> f32 {
    let n = a.len() as i32;
    let corr = |lag: i32| {
        let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
        for i in 0.max(-lag)..n.min(n - lag) {
            let (x, y) = (a[i as usize], b[(i + lag) as usize]);
            ab += x * y;
            aa += x * x;
            bb += y * y;
        }
        ab / (aa * bb).sqrt().max(f32::MIN_POSITIVE)
    };
    let c: Vec<f32> = (-max_lag..=max_lag).map(corr).collect();
    let k = (0..c.len()).max_by(|&i, &j| c[i].total_cmp(&c[j])).unwrap();
    let mut lag = k as f32 - max_lag as f32;
    if k > 0 && k + 1 < c.len() {
        let d = c[k - 1] - 2.0 * c[k] + c[k + 1];
        if d < 0.0 {
            lag += 0.5 * (c[k - 1] - c[k + 1]) / d;
        }
    }
    lag
}

/// Step migration and mean slope: Parker & Izumi (2000) and Sun & Parker (2005) predict that
/// cyclic steps migrate upstream, and that steps usually let the same transport run at a lower
/// mean slope (S_e / S_n < 1). Track the bed profile of the stepped flumes over a few seconds.
#[test]
#[ignore = "needs a graphics adapter"]
fn validate_step_migration() {
    let mut f = data_file(
        "step_migration",
        "slope_cells,wavelength,speed_texels_per_s,slope_ratio",
    );
    let mid = N / 2;
    let (x0, x1) = (48, N - 24);
    for slope in [0.006f32, 0.009, 0.013] {
        let mut h = Harness::new(WaterSolver::Swe, flume(slope, 6.0));
        h.water().second_order_depth = std::env::var("TEST_SECOND_ORDER").is_ok();
        h.water().springs.push(Spring {
            center: Vec2::new(12.0, N as f32 / 2.0),
            radius: 2.0,
            rate: 0.45,
        });
        h.run(6.0);
        let start = h.read_heights();
        h.water().bed_load.live = true;
        h.run(25.0);
        let profile = |heights: &[f32]| -> Vec<f32> {
            (x0..x1).map(|x| heights[idx(x, mid)]).collect()
        };
        let (_, slope_before) = detrend(&profile(&start));
        let mut prev = detrend(&profile(&h.read_heights())).0;
        let (period, ..) = dominant_period(&prev);
        let (mut shift, mut slope_after) = (0.0, 0.0);
        let (snapshots, dt) = (8, 0.5);
        for _ in 0..snapshots {
            h.run(dt);
            let (r, b) = detrend(&profile(&h.read_heights()));
            shift += profile_shift(&prev, &r, (period / 2.0).floor().max(1.0) as i32);
            slope_after += b / snapshots as f32;
            prev = r;
        }
        row(
            &mut f,
            &[
                slope * CELLS_PER_HEIGHT,
                period,
                shift / (snapshots as f32 * dt),
                slope_after / slope_before,
            ],
        );
    }
}

/// Plane tilted along x with a faint V across it, so a stream stays near y = N/2.
fn tilted_plane(x: u32, y: u32) -> f32 {
    0.02 + 0.002 * (N - 1 - x) as f32 + 0.0001 * (y as f32 + 0.5 - N as f32 / 2.0).abs()
}

/// Emergent hydraulic geometry: springs of different discharge carve their own channels in an
/// erodible plane with bed load (fast-forwarded by Refine Beds). Width and depth against the
/// discharge are compared with Leopold and Maddock (1953): width ∝ Q^0.5, depth ∝ Q^0.4.
#[test]
#[ignore = "needs a graphics adapter"]
fn validate_hydraulic_geometry() {
    let mut f = data_file(
        "hydraulic_geometry",
        "discharge,wet_width,mean_depth_cells,max_depth_cells,incision,channel_width,cut_width",
    );
    let mut points = Vec::new();
    for factor in [0.25f32, 0.5, 1.0, 2.0, 4.0] {
        let mut h = Harness::new(WaterSolver::Swe, tilted_plane);
        if let Some(a) = std::env::var("TEST_SLOPE_EFFECT").ok().and_then(|v| v.parse().ok()) {
            h.water().bed_load.slope_effect = a;
        }
        h.water().springs.push(Spring {
            center: Vec2::new(16.0, N as f32 / 2.0),
            radius: 6.0,
            rate: 0.05 * factor,
        });
        h.run(8.0);
        let before = h.read_heights();
        h.water().bed_load.refine_steps = 3000;
        {
            let mut water = h.water();
            water.refine_generation = water.refine_generation.wrapping_add(1);
        }
        h.run(1.5);
        h.water().bed_load.live = true;
        h.run(6.0);
        let water = h.read_water();
        let q = h.read_discharge();
        let after = h.read_heights();
        let (mut width, mut depth, mut deepest, mut incision, mut discharge, mut count) =
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let (mut channel, mut cut_width) = (0.0, 0.0);
        for x in (96..208).step_by(8) {
            let wet: Vec<u32> = (0..N).filter(|&y| water[idx(x, y)].depth > 0.002).collect();
            if wet.is_empty() {
                continue;
            }
            width += wet.len() as f32;
            // Channel: deeper than half the section's deepest water; cut: lowered by more than
            // half the section's deepest cut.
            let top = wet.iter().map(|&y| water[idx(x, y)].depth).fold(0.0, f32::max);
            channel += wet.iter().filter(|&&y| water[idx(x, y)].depth > 0.5 * top).count() as f32;
            let cuts: Vec<f32> = (0..N).map(|y| before[idx(x, y)] - after[idx(x, y)]).collect();
            let most = cuts.iter().fold(0.0f32, |a, b| a.max(*b));
            cut_width += cuts.iter().filter(|&&c| most > 0.0 && c > 0.5 * most).count() as f32;
            depth += wet.iter().map(|&y| water[idx(x, y)].depth).sum::<f32>() / wet.len() as f32;
            deepest += wet.iter().map(|&y| water[idx(x, y)].depth).fold(0.0, f32::max);
            incision += (0..N)
                .map(|y| before[idx(x, y)] - after[idx(x, y)])
                .fold(0.0, f32::max);
            discharge += (0..N).map(|y| q[idx(x, y)].x).sum::<f32>();
            count += 1.0;
        }
        let r = [
            discharge / count,
            width / count,
            depth / count * CELLS_PER_HEIGHT,
            deepest / count * CELLS_PER_HEIGHT,
            incision / count,
            channel / count,
            cut_width / count,
        ];
        row(&mut f, &r);
        points.push(r);
    }
    let fit = |k: usize| {
        let xs: Vec<f32> = points.iter().map(|p| p[0].ln()).collect();
        let ys: Vec<f32> = points.iter().map(|p| p[k].max(1e-6).ln()).collect();
        let (xm, ym) = (
            xs.iter().sum::<f32>() / xs.len() as f32,
            ys.iter().sum::<f32>() / ys.len() as f32,
        );
        xs.iter().zip(&ys).map(|(x, y)| (x - xm) * (y - ym)).sum::<f32>()
            / xs.iter().map(|x| (x - xm).powi(2)).sum::<f32>()
    };
    println!(
        "hydraulic geometry exponents: wet width {:.2}, channel width {:.2}, cut width {:.2}, \
         mean depth {:.2}, max depth {:.2}",
        fit(1),
        fit(5),
        fit(6),
        fit(2),
        fit(3)
    );
}

/// Alternate bars: theory and flume data (Redolfi et al. 2020; Crosato and Mosselman 2009) form
/// bars only above a critical half-width-to-depth ratio β = W / 2h, with wavelengths of several
/// widths. Sweep the flume width at a fixed slope and discharge.
#[test]
#[ignore = "needs a graphics adapter"]
fn validate_alternate_bars() {
    let mut f = data_file(
        "alternate_bars",
        "width,depth_cells,beta,shields,thalweg_std,wavelength,wavelength_over_width",
    );
    let mid = N / 2;
    let g = WaterControl::default();
    for width in [6.0f32, 10.0, 16.0, 24.0, 36.0, 52.0] {
        let radius = (width / 2.0 - 1.0).min(6.0);
        let slope = std::env::var("TEST_BAR_SLOPE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.003);
        let r = run_flume(
            flume(slope, width),
            radius,
            0.05 * 36.0 / (radius * radius),
            30.0,
        );
        let half = (width / 2.0) as u32 - 1;
        let thalweg: Vec<f32> = (40..N - 16)
            .map(|x| {
                (mid - half..mid + half)
                    .min_by(|&a, &b| r.after[idx(x, a)].total_cmp(&r.after[idx(x, b)]))
                    .unwrap() as f32
            })
            .collect();
        let mean = thalweg.iter().sum::<f32>() / thalweg.len() as f32;
        let spread = (thalweg.iter().map(|t| (t - mean).powi(2)).sum::<f32>()
            / thalweg.len() as f32)
            .sqrt();
        if let Ok(dir) = std::env::var("TEST_IMAGE_DIR") {
            // Detrended bed of the channel, stretched across.
            let rows = 2 * half;
            let bed: Vec<f32> = (0..rows)
                .flat_map(|j| (0..N).map(move |x| (x, mid - half + j)))
                .map(|(x, y)| r.after[idx(x, y)] - r.before[idx(x, y)])
                .collect();
            let m = bed.iter().fold(1e-6f32, |a, b| a.max(b.abs()));
            write_bmp(
                &format!("{dir}/bars_w{}.bmp", width as u32),
                &bed,
                N as usize,
                rows as usize,
                -m,
                m,
                3,
            );
        }
        // The fundamental alternate-bar mode: bed on the left half minus the right half.
        let asymmetry: Vec<f32> = (40..N - 16)
            .map(|x| {
                let side = |ys: std::ops::Range<u32>| {
                    ys.clone().map(|y| r.after[idx(x, y)]).sum::<f32>() / ys.len() as f32
                };
                side(mid - half..mid) - side(mid..mid + half)
            })
            .collect();
        let (period, ..) = dominant_period(&asymmetry);
        let wet: Vec<f32> = (64..N - 24)
            .flat_map(|x| (mid - half..mid + half).map(move |y| (x, y)))
            .map(|(x, y)| r.water[idx(x, y)])
            .filter(|w| w.depth > 0.0005)
            .map(|w| w.depth)
            .collect();
        let depth = wet.iter().sum::<f32>() / wet.len().max(1) as f32 * CELLS_PER_HEIGHT;
        let speed = (64..N - 24)
            .flat_map(|x| (mid - half..mid + half).map(move |y| (x, y)))
            .map(|(x, y)| r.water[idx(x, y)])
            .filter(|w| w.depth > 0.0005)
            .map(|w| w.velocity.length())
            .sum::<f32>()
            / wet.len().max(1) as f32;
        let shields = g.manning_n.powi(2) * speed * speed
            / (g.bed_load.grain * depth.max(1e-4).cbrt());
        row(
            &mut f,
            &[
                width,
                depth,
                width / (2.0 * depth),
                shields,
                spread,
                period,
                period / width,
            ],
        );
    }
}

/// Cyclic steps: supercritical flow over an erodible bed forms steps (Parker and Izumi 2000;
/// Balmforth and Vakil 2012). Sweep the slope of a narrow flume and record the Froude number,
/// the step wavelength in flow depths and the relief.
#[test]
#[ignore = "needs a graphics adapter"]
fn validate_cyclic_steps() {
    let mut f = data_file(
        "cyclic_steps",
        "slope_cells,depth_cells,froude,wavelength,wavelength_over_depth,relief_cells,band_share",
    );
    let mid = N / 2;
    let g = WaterControl::default().gravity;
    for slope in [0.001f32, 0.002, 0.004, 0.006, 0.009, 0.013] {
        let r = run_flume(flume(slope, 6.0), 2.0, 0.45, 25.0);
        let profile: Vec<f32> = (48..N - 24).map(|x| r.after[idx(x, mid)]).collect();
        let (period, _, band, relief) = dominant_period(&profile);
        let line: Vec<Water> = (64..N - 24).map(|x| r.water[idx(x, mid)]).collect();
        let depth = line.iter().map(|w| w.depth).sum::<f32>() / line.len() as f32 * CELLS_PER_HEIGHT;
        let speed = line.iter().map(|w| w.velocity.length()).sum::<f32>() / line.len() as f32;
        let froude = speed / (g * depth.max(1e-4)).sqrt();
        row(
            &mut f,
            &[
                slope * CELLS_PER_HEIGHT,
                depth,
                froude,
                period,
                period / depth.max(1e-4),
                relief * CELLS_PER_HEIGHT,
                band,
            ],
        );
    }
}

/// Height less its 9 × 9 box mean: the small-scale relief erosion carves.
fn detail(h: &[f32]) -> Vec<f32> {
    let r = 4i32;
    let mut out = vec![0.0; h.len()];
    for y in 0..N as i32 {
        for x in 0..N as i32 {
            let mut sum = 0.0;
            for dy in -r..=r {
                for dx in -r..=r {
                    let (sx, sy) = ((x + dx).clamp(0, N as i32 - 1), (y + dy).clamp(0, N as i32 - 1));
                    sum += h[idx(sx as u32, sy as u32)];
                }
            }
            out[idx(x as u32, y as u32)] = h[idx(x as u32, y as u32)] - sum / 81.0;
        }
    }
    out
}

/// Lambert hillshade of `h` (height units) lit from the north-west.
fn hillshade(h: &[f32]) -> Vec<f32> {
    let light = Vec3::new(-1.0, -1.0, 1.5).normalize();
    (0..N * N)
        .map(|i| {
            let (x, y) = (i % N, i / N);
            let at = |x: u32, y: u32| h[idx(x.min(N - 1), y.min(N - 1))] * CELLS_PER_HEIGHT;
            let gx = at(x + 1, y) - at(x.saturating_sub(1), y);
            let gy = at(x, y + 1) - at(x, y.saturating_sub(1));
            Vec3::new(-gx, -gy, 2.0).normalize().dot(light).max(0.0)
        })
        .collect()
}

/// KTT rivers against plain KTT erosion on the default terrain. Channels are the cells with the
/// top 10 % of river momentum at the default River Scale. Reports, per setting, the RMS of the
/// small-scale relief on the hillslopes (the rest) and in the channels, and how far each sits
/// below the rivers-off terrain. Set `TEST_IMAGE_DIR` for a hillshade of each run.
#[test]
#[ignore = "needs a graphics adapter"]
fn validate_ktt_rivers() {
    let mut f = data_file(
        "ktt_rivers",
        "rivers,river_scale,hill_detail_rms,channel_detail_rms,hill_lowering,channel_lowering",
    );
    let dir = std::env::var("TEST_IMAGE_DIR").ok();
    let scales: Vec<f32> = std::env::var("TEST_RIVER_SCALES")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.parse().ok()).collect())
        .unwrap_or_else(|| vec![0.25, 0.1, 0.03]);
    let mut h = Harness::natural(WaterSolver::Swe);
    let mut run = |h: &mut Harness, rivers: bool, scale: f32| {
        {
            let mut params = h.app.world_mut().resource_mut::<ErodeParams>();
            params.do_rivers = u32::from(rivers);
            params.river_scale = scale;
            if let Some(t) = std::env::var("TEST_RIVER_THRESHOLD").ok().and_then(|v| v.parse().ok()) {
                params.river_threshold = t;
            }
            if let Some(d) = std::env::var("TEST_TRAIL_DAMPING").ok().and_then(|v| v.parse().ok()) {
                params.trail_damping = d;
            }
            if let Some(a) = std::env::var("TEST_MAX_DEPOSIT").ok().and_then(|v| v.parse().ok()) {
                params.max_deposit_angle = a;
            }
        }
        let mut reset = h.app.world_mut().resource_mut::<ResetSim>();
        reset.generation = reset.generation.wrapping_add(1);
        h.settle_cascade();
        let heights = h.read_heights();
        let handle = h.app.world().resource::<ErosionImages>().height.clone();
        let momentum: Vec<f32> = h.read(handle).iter().map(|t| t[2] / 0.02).collect();
        h.pause(false);
        (heights, momentum)
    };
    let (off, _) = run(&mut h, false, 0.25);
    let mut angles: Vec<f32> = (0..N * N)
        .map(|i| {
            let (x, y) = (i % N, i / N);
            let at = |x: u32, y: u32| off[idx(x.min(N - 1), y.min(N - 1))] * CELLS_PER_HEIGHT;
            let gx = (at(x + 1, y) - at(x.saturating_sub(1), y)) / 2.0;
            let gy = (at(x, y + 1) - at(x, y.saturating_sub(1))) / 2.0;
            gx.hypot(gy).atan().to_degrees()
        })
        .collect();
    angles.sort_by(f32::total_cmp);
    let pct = |p: usize| angles[angles.len() * p / 100];
    println!(
        "slope angles (deg) 10/50/90/99: {:.1} {:.1} {:.1} {:.1}",
        pct(10),
        pct(50),
        pct(90),
        pct(99)
    );
    let (_, momentum) = run(&mut h, true, 0.25);
    let mut sorted = momentum.clone();
    sorted.sort_by(f32::total_cmp);
    let cut = sorted[sorted.len() * 9 / 10];
    let m = 8;
    let inner = |i: usize| {
        let (x, y) = (i as u32 % N, i as u32 / N);
        x >= m && y >= m && x < N - m && y < N - m
    };
    let channel: Vec<bool> = momentum.iter().map(|&v| v > cut).collect();
    println!(
        "momentum percentiles 50/75/90/99: {} {} {} {}",
        sorted[sorted.len() / 2],
        sorted[sorted.len() * 3 / 4],
        cut,
        sorted[sorted.len() * 99 / 100]
    );
    if let Some(dir) = &dir {
        let log: Vec<f32> = momentum.iter().map(|v| (1.0 + v.max(0.0)).ln()).collect();
        let top = log.iter().cloned().fold(0.0, f32::max);
        write_bmp(&format!("{dir}/ktt_momentum.bmp"), &log, N as usize, N as usize, 0.0, top, 2);
    }
    let stats = |heights: &[f32]| {
        let d = detail(heights);
        let (mut hill, mut chan, mut hill_low, mut chan_low, mut nh, mut nc) =
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for i in (0..heights.len()).filter(|&i| inner(i)) {
            let low = (off[i] - heights[i]) * CELLS_PER_HEIGHT;
            if channel[i] {
                chan += d[i] * d[i];
                chan_low += low;
                nc += 1.0;
            } else {
                hill += d[i] * d[i];
                hill_low += low;
                nh += 1.0;
            }
        }
        let k = CELLS_PER_HEIGHT;
        ((hill / nh).sqrt() * k, (chan / nc).sqrt() * k, hill_low / nh, chan_low / nc)
    };
    let mut report = |name: &str, rivers: bool, scale: f32, heights: &[f32]| {
        let (a, b, c, d) = stats(heights);
        row(&mut f, &[f32::from(u8::from(rivers)), scale, a, b, c, d]);
        if let Some(dir) = &dir {
            write_bmp(&format!("{dir}/ktt_{name}.bmp"), &hillshade(heights), N as usize, N as usize, 0.0, 1.0, 2);
        }
    };
    report("off", false, 0.0, &off);
    for scale in scales {
        let (heights, _) = run(&mut h, true, scale);
        report(&format!("rivers_{scale}"), true, scale, &heights);
    }
}
