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
}

impl Harness {
    /// A headless app whose terrain is `bed` (height units, per texel), with erosion off.
    fn new(solver: WaterSolver, bed: impl Fn(u32, u32) -> f32) -> Self {
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

        let mut h = Self { app };
        {
            let mut params = h.app.world_mut().resource_mut::<ErodeParams>();
            params.erosion_amount = 0.0;
            params.erosion_strength = 0.0;
            params.ridge_erosion_amount = 0.0;
            params.compute_ridge_erosion = 0;
            params.do_rivers = 0;
            params.uplift = 0.0;
        }
        {
            let mut water = h.app.world_mut().resource_mut::<WaterControl>();
            water.solver = solver;
            water.evaporation = 0.0;
            water.sediment_capacity = 0.0;
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
        h.set_bed(bed);
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
        for _ in 0..(seconds * FPS).round() as u32 {
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
            .chunks_exact(8)
            .map(|c| {
                std::array::from_fn(|k| f16_to_f32(u16::from_le_bytes([c[2 * k], c[2 * k + 1]])))
            })
            .collect()
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
