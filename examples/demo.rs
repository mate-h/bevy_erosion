//! Erosion demo.
//!
//! `--simple` (or `EROSION_SIMPLE=1`) renders Lambert-shaded terrain with a flat water tint and
//! no atmosphere, environment map, bloom, auto exposure or shadows, with vsync off, and logs the
//! mean frame time every few seconds, so the frame is mostly the simulations.
//! `EROSION_BENCH_SECONDS=<n>` logs the mean frame time after the cascade settles and quits after
//! n seconds; `EROSION_BENCH_SPRING=1` places a spring at the map centre so the water solver runs,
//! `EROSION_BENCH_PIPES=1` switches it to the pipe model, and `EROSION_NO_ENV_FREEZE=1` keeps
//! re-filtering the environment map every frame. `EROSION_BENCH_POUR=x,y` places a spring at
//! texel (x, y) and moves the camera close to it (or to `EROSION_BENCH_TARGET=x,y`;
//! `EROSION_BENCH_DISTANCE`, `_PITCH` and `_YAW`,
//! in degrees, frame it), and `EROSION_BENCH_RAIN=<‰/s>` rains on the whole map. `EROSION_BENCH_SCREENSHOT=<path>` hides the settings panel and saves
//! the window just before the bench quits.

use bevy::{
    core_pipeline::tonemapping::Tonemapping,
    curve::cubic_splines::LinearSpline,
    light::{
        Atmosphere, AtmosphereEnvironmentMapLight, EnvironmentMapLight,
        GeneratedEnvironmentMapLight, atmosphere::ScatteringMedium, light_consts::lux,
    },
    material::OpaqueRendererMethod,
    math::vec2,
    pbr::{AtmosphereSettings, ExtendedMaterial, MaterialPlugin, StandardMaterial},
    post_process::{
        auto_exposure::{AutoExposure, AutoExposureCompensationCurve, AutoExposurePlugin},
        bloom::Bloom,
    },
    prelude::*,
    render::{
        Render, RenderApp,
        render_resource::PipelineCache,
        view::screenshot::{Screenshot, save_to_disk},
    },
    window::PresentMode,
};
#[cfg(feature = "ui")]
use bevy_erosion::ui::ErosionParamsPlugin;
use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering},
};
use bevy_erosion::{
    ErosionConfig,
    brush::{BrushMode, TerrainSurface, WaterBrush, WaterBrushPlugin},
    camera::{OrbitCameraPlugin, OrbitController},
    sun::{Sun, SunController, SunPlugin},
    *,
};

/// Map size (width and height). Customize this to change erosion resolution.
const MAP_SIZE: UVec2 = UVec2::new(512, 512);
// Controls how much to under-expose (in F-stops). Negative values = under-expose, positive = over-expose
const UNDER_EXPOSURE_AMOUNT: f32 = -2.0;
/// Render-ready frames the atmosphere environment map gets to bake before it is frozen.
const ENV_MAP_BAKE_FRAMES: u32 = 6;

fn simple_mode() -> bool {
    std::env::args().any(|a| a == "--simple")
        || std::env::var("EROSION_SIMPLE").is_ok_and(|v| v != "0")
}

fn bench_seconds() -> Option<f32> {
    std::env::var("EROSION_BENCH_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok())
}

/// `EROSION_SEED=<n>` starts on another landscape (the Seed slider under Terrain Source).
fn seed_from_env(mut params: ResMut<ErodeParams>) {
    if let Some(seed) = std::env::var("EROSION_SEED").ok().and_then(|v| v.parse().ok()) {
        params.noise_seed = seed;
    }
}

fn main() {
    let simple = simple_mode();
    let present_mode = if simple || bench_seconds().is_some() {
        PresentMode::AutoNoVsync
    } else {
        PresentMode::AutoVsync
    };
    let ready_frames = RenderReadyFrames::default();
    let mut app = App::new();
    app.insert_resource(ClearColor(Color::BLACK))
        .insert_resource(ready_frames.clone())
        .insert_resource(ErosionConfig {
            map_size: MAP_SIZE,
            ..Default::default()
        })
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    present_mode,
                    ..default()
                }),
                ..default()
            }),
            ErosionComputePlugin,
            MaterialPlugin::<TerrainMaterial>::default(),
            OrbitCameraPlugin,
            SunPlugin,
            AutoExposurePlugin,
            WaterBrushPlugin,
            #[cfg(feature = "ui")]
            ErosionParamsPlugin,
        ))
        .add_systems(Startup, (setup, print_controls))
        .add_systems(PostStartup, (spawn_terrain, seed_from_env))
        .add_systems(
            Update,
            (
                handle_reset_input,
                handle_sim_input,
                handle_preview_mode_input,
                handle_brush_input,
                sync_sea_level,
                freeze_atmosphere_env_map,
                log_frame_times,
            ),
        );
    app.sub_app_mut(RenderApp)
        .insert_resource(ready_frames)
        .add_systems(Render, count_ready_render_frames);
    app.run();
}

/// Render frames so far with no pipeline still compiling (shared with the render world).
#[derive(Resource, Clone, Default)]
struct RenderReadyFrames(Arc<AtomicU32>);

fn count_ready_render_frames(pipeline_cache: Res<PipelineCache>, ready: Res<RenderReadyFrames>) {
    if pipeline_cache.waiting_pipelines().count() == 0 {
        ready.0.fetch_add(1, Ordering::Relaxed);
    }
}

/// The baked map, kept while [`GeneratedEnvironmentMapLight`] is off the camera.
#[derive(Component)]
struct CachedEnvMap(GeneratedEnvironmentMapLight);

/// Bevy re-filters the atmosphere environment map (cubemap downsample plus diffuse and specular
/// convolution) every frame. The sky only changes with the sun, so let it bake for a few frames,
/// then take [`GeneratedEnvironmentMapLight`] off the camera (its [`EnvironmentMapLight`] stays)
/// and bake again once the sun has moved and settled. Same approach as flex's
/// `freeze_atmosphere_env_map`.
#[allow(clippy::type_complexity)]
fn freeze_atmosphere_env_map(
    mut commands: Commands,
    ready: Res<RenderReadyFrames>,
    sun: Query<&Transform, With<Sun>>,
    cameras: Query<
        (
            Entity,
            Option<&GeneratedEnvironmentMapLight>,
            Option<&CachedEnvMap>,
        ),
        With<EnvironmentMapLight>,
    >,
    mut bake_start: Local<Option<u32>>,
    mut baked_sun: Local<Vec3>,
) {
    if std::env::var("EROSION_NO_ENV_FREEZE").is_ok_and(|v| v != "0") {
        return;
    }
    let (Ok((camera, generated, cached)), Ok(sun)) = (cameras.single(), sun.single()) else {
        return;
    };
    let ready = ready.0.load(Ordering::Relaxed);
    let sun_dir = *sun.forward();
    // Re-bake once the sun direction moves by more than about half a degree.
    let sun_moved = sun_dir.dot(*baked_sun) < 0.99996;
    if let Some(generated) = generated {
        if sun_moved {
            *baked_sun = sun_dir;
            *bake_start = Some(ready);
        } else if ready.saturating_sub(*bake_start.get_or_insert(ready)) >= ENV_MAP_BAKE_FRAMES {
            commands
                .entity(camera)
                .insert(CachedEnvMap(generated.clone()))
                .remove::<GeneratedEnvironmentMapLight>();
        }
    } else if let Some(cached) = cached
        && sun_moved
    {
        commands
            .entity(camera)
            .insert(cached.0.clone())
            .remove::<CachedEnvMap>();
    }
}

/// In simple or bench mode, log the mean frame time every few seconds once the cascade is done
/// (and, for a bench, quit after `EROSION_BENCH_SECONDS`).
#[allow(clippy::too_many_arguments)]
fn log_frame_times(
    time: Res<Time<Real>>,
    progress: Res<PyramidProgress>,
    mut water: ResMut<WaterControl>,
    mut exit: MessageWriter<AppExit>,
    mut window: Local<(f32, u32)>,
    mut bench: Local<Option<f32>>,
    mut shot_taken: Local<bool>,
    mut commands: Commands,
    mut cameras: Query<&mut OrbitController>,
    mut panels: Query<&mut Visibility, (With<Node>, Without<ChildOf>)>,
) {
    let bench_len = bench_seconds();
    if !(simple_mode() || bench_len.is_some()) || !progress.get().finished {
        return;
    }
    if bench.is_none() {
        *bench = Some(0.0);
        if std::env::var("EROSION_BENCH_PIPES").is_ok_and(|v| v != "0") {
            water.solver = WaterSolver::Pipes;
        }
        if std::env::var("EROSION_BENCH_SPRING").is_ok_and(|v| v != "0") {
            let center = progress.get().size.as_vec2() * 0.5;
            water.springs.push(Spring {
                center,
                radius: 12.0,
                rate: 0.007,
            });
        }
        // `x,y` in texels: pour there and look at it from close by.
        if let Some(rain) = std::env::var("EROSION_BENCH_RAIN")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
        {
            water.rainfall = rain / 1000.0;
        }
        if let Ok(pour) = std::env::var("EROSION_BENCH_POUR")
            && let Some((x, y)) = pour.split_once(',')
            && let (Ok(x), Ok(y)) = (x.parse::<f32>(), y.parse::<f32>())
        {
            water.springs.push(Spring {
                center: Vec2::new(x, y),
                radius: 8.0,
                rate: 0.05,
            });
            let size = progress.get().size.as_vec2();
            let half = size.x * 0.25;
            let env = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<f32>().ok());
            let (tx, ty) = std::env::var("EROSION_BENCH_TARGET")
                .ok()
                .and_then(|t| {
                    let (a, b) = t.split_once(',')?;
                    Some((a.parse::<f32>().ok()?, b.parse::<f32>().ok()?))
                })
                .unwrap_or((x, y));
            for mut orbit in &mut cameras {
                orbit.target = Vec3::new(tx / size.x * 2.0 * half - half, 20.0, ty / size.y * 2.0 * half - half);
                orbit.distance = env("EROSION_BENCH_DISTANCE").unwrap_or(45.0);
                if let Some(pitch) = env("EROSION_BENCH_PITCH") {
                    orbit.pitch = pitch.to_radians();
                    orbit.target_pitch = orbit.pitch;
                }
                if let Some(yaw) = env("EROSION_BENCH_YAW") {
                    orbit.yaw = yaw.to_radians();
                    orbit.target_yaw = orbit.yaw;
                }
            }
        }
    }
    let dt = time.delta_secs();
    window.0 += dt;
    window.1 += 1;
    let elapsed = bench.as_mut().unwrap();
    *elapsed += dt;
    if window.0 >= 3.0 {
        info!(
            "frame {:.2} ms ({} frames, water solver {:?})",
            window.0 * 1000.0 / window.1 as f32,
            window.1,
            water.solver
        );
        *window = (0.0, 0);
    }
    if let Some(len) = bench_len {
        let shot = std::env::var("EROSION_BENCH_SCREENSHOT").ok();
        if *elapsed >= len - 0.5
            && let Some(path) = shot
            && !*shot_taken
        {
            *shot_taken = true;
            // The settings panel would cover part of the shot.
            for mut visibility in &mut panels {
                *visibility = Visibility::Hidden;
            }
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk(path));
        }
        if *elapsed >= len {
            exit.write(AppExit::Success);
        }
    }
}

#[derive(Resource, Default)]
struct CurrentPreviewMode {
    mode: PreviewMode,
}

#[derive(Component)]
struct DemoAtmosphere;

#[derive(Resource)]
struct DemoEarthMedium(Handle<ScatteringMedium>);

fn setup(mut commands: Commands) {
    commands.insert_resource(CurrentPreviewMode::default());
    commands.insert_resource(GlobalAmbientLight::NONE);
}

fn print_controls() {
    println!("Controls:");
    println!("  R - reset simulation");
    println!("  Space - pause/resume");
    println!("  E - step one iteration (when paused)");
    println!("  1 - PBR shading mode (default)");
    println!("  2 - Flow map preview");
    println!("  3 - Sediment mask preview");
    println!("  4 - Erosion mask preview");
    println!("  5 - Height map preview");
    println!("  6 - View-space normals preview");
    println!("  7 - Curvature map preview");
    println!("  8 - Age map preview");
    println!("  9 - Simple shading (no atmosphere or water optics; see --simple)");
    println!("  B - toggle the brush (left drag paints, Shift + drag erases, right drag orbits)");
    println!("  V - cycle brush: rain / terrain / water body");
    println!("  T - cycle terrain tool: raise / lower / flatten / smooth");
    println!("  [ / ] - brush radius");
    println!("  Alt + click (water body) - place a spring, Alt + Shift + click removes springs");
    println!("  C - clear water and springs   X - reset terrain edits");
    println!("  G - shape lake and river beds under the current water");
    println!("  M - refine beds: fast-forward bed load under the current flow");
}

fn spawn_terrain(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut compensation_curves: ResMut<Assets<AutoExposureCompensationCurve>>,
    mut scattering_mediums: ResMut<Assets<ScatteringMedium>>,
    erosion_images: Res<ErosionImages>,
    config: Res<ErosionConfig>,
) {
    // Spawn a subdivided plane with UVs (match map size for correct 1:1 UVs)
    let size = config.map_size.x as f32;
    let resolution = (config.map_size.x - 1) as u32; // subdivisions
    let half_size = size * 0.5;
    let plane = Mesh::from(
        Plane3d::default()
            .mesh()
            .subdivisions(resolution)
            .size(half_size, half_size),
    );
    let mesh_handle = meshes.add(plane);
    let height_scale = 50.0;

    let mat_handle = materials.add(ExtendedMaterial {
        base: StandardMaterial {
            base_color: Color::srgb(1.0, 1.0, 1.0),
            perceptual_roughness: 1.0,
            metallic: 0.0,
            opaque_render_method: OpaqueRendererMethod::Auto,
            ..Default::default()
        },
        extension: TerrainExtension {
            height_tex: erosion_images.height.clone(),
            color_tex: erosion_images.color.clone(),
            analysis_tex: erosion_images.analysis.clone(),
            ao_tex: erosion_images.ao.clone(),
            height_scale,
            preview_mode: if simple_mode() { 8 } else { 0 }, // PBR, or Simple
            water_tex: erosion_images.water.clone(),
            sea_level: 0.0,
        },
    });

    commands.spawn((
        Mesh3d(mesh_handle),
        MeshMaterial3d(mat_handle),
        Transform::from_xyz(0.0, 0.0, 0.0),
        TerrainSurface {
            size: Vec2::splat(half_size),
            height_scale,
        },
    ));

    let earth_medium = scattering_mediums.add(ScatteringMedium::earth(256, 256));
    commands.insert_resource(DemoEarthMedium(earth_medium.clone()));

    let mut atmosphere = Atmosphere::earth(earth_medium);
    atmosphere.ground_albedo = Vec3::splat(0.3);
    let visibility = if simple_mode() {
        Visibility::Hidden
    } else {
        Visibility::Visible
    };
    commands.spawn((atmosphere, visibility, DemoAtmosphere));

    // Create a compensation curve that slightly under-exposes
    // Negative compensation values = darker/under-exposed
    // The curve shape applies more under-exposure in darker scenes and less in brighter scenes
    let under_expose_curve = compensation_curves.add(
        AutoExposureCompensationCurve::from_curve(LinearSpline::new([
            vec2(-4.0, UNDER_EXPOSURE_AMOUNT * 1.67), // Dark scenes: more under-exposure
            vec2(-2.0, UNDER_EXPOSURE_AMOUNT * 1.33), // Medium-dark: moderate under-exposure
            vec2(0.0, UNDER_EXPOSURE_AMOUNT),         // Middle gray: base under-exposure
            vec2(2.0, UNDER_EXPOSURE_AMOUNT * 0.67),  // Medium-bright: less under-exposure
            vec2(4.0, UNDER_EXPOSURE_AMOUNT * 0.33),  // Bright scenes: minimal under-exposure
        ]))
        .expect("Failed to create compensation curve"),
    );

    // 3D camera
    let mut camera = commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(150.0, 50.0, 150.0).looking_at(Vec3::ZERO, Vec3::Y),
        OrbitController {
            target: Vec3::ZERO,
            distance: 225.0,
            ..Default::default()
        },
    ));
    if simple_mode() {
        camera.insert((Tonemapping::None, Bloom { intensity: 0.0, ..Bloom::NATURAL }));
    } else {
        camera.insert((
            AtmosphereSettings::default(),
            AtmosphereEnvironmentMapLight::default(),
            Tonemapping::AcesFitted,
            Bloom::NATURAL,
            AutoExposure {
                compensation_curve: under_expose_curve,
                ..Default::default()
            },
        ));
    }

    // Directional light for PBR shading
    let initial_sun_pos = Vec3::new(1.0, 0.1, 0.0);
    let initial_sun_dir = -initial_sun_pos.normalize(); // Direction from origin to sun
    commands.spawn((
        DirectionalLight {
            shadow_maps_enabled: !simple_mode(),
            illuminance: lux::RAW_SUNLIGHT,
            ..default()
        },
        Transform::from_translation(initial_sun_pos).looking_at(Vec3::ZERO, Vec3::Y),
        Sun,
        SunController::new(initial_sun_dir),
    ));

    // sprite for the color image
    commands.spawn((
        Sprite::from_image(erosion_images.color.clone()),
        Transform::from_xyz(-300.0, 0.0, 0.0),
    ));

    commands.spawn((
        Sprite::from_image(erosion_images.height.clone()),
        Transform::from_xyz(300.0, 0.0, 0.0),
    ));

    // 2d camera
    // commands.spawn((Camera2d, Transform::from_xyz(0.0, 0.0, 0.0)));
}

fn handle_reset_input(keys: Res<ButtonInput<KeyCode>>, mut reset: ResMut<ResetSim>) {
    if keys.just_pressed(KeyCode::KeyR) {
        reset.generation = reset.generation.wrapping_add(1);
    }
}

fn handle_sim_input(keys: Res<ButtonInput<KeyCode>>, mut sim: ResMut<SimControl>) {
    if keys.just_pressed(KeyCode::Space) {
        sim.paused = !sim.paused;
    }
    if sim.paused && keys.just_pressed(KeyCode::KeyE) {
        sim.step_counter = sim.step_counter.wrapping_add(1);
    }
}

fn handle_brush_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut brush: ResMut<WaterBrush>,
    mut water: ResMut<WaterControl>,
) {
    if keys.just_pressed(KeyCode::KeyG) {
        water.shape_generation = water.shape_generation.wrapping_add(1);
        println!("Shaping lake and river beds");
    }
    if keys.just_pressed(KeyCode::KeyM) {
        water.refine_generation = water.refine_generation.wrapping_add(1);
        println!("Refining beds");
    }
    if keys.just_pressed(KeyCode::KeyB) {
        brush.enabled = !brush.enabled;
        println!("Water brush: {}", if brush.enabled { "on" } else { "off" });
    }
    if keys.just_pressed(KeyCode::KeyV) {
        brush.mode = brush.mode.next();
        println!("Brush: {}", brush.mode.label());
    }
    if keys.just_pressed(KeyCode::KeyT) {
        brush.mode = match brush.mode {
            BrushMode::Terrain(tool) => BrushMode::Terrain(tool.next()),
            _ => BrushMode::Terrain(Default::default()),
        };
        println!("Brush: {}", brush.mode.label());
    }
    if keys.just_pressed(KeyCode::KeyC) {
        brush.clear_water = true;
    }
    if keys.just_pressed(KeyCode::KeyX) {
        brush.clear_terrain = true;
    }
    if keys.just_pressed(KeyCode::BracketLeft) {
        brush.radius = (brush.radius / 1.25).max(2.0);
    }
    if keys.just_pressed(KeyCode::BracketRight) {
        brush.radius = (brush.radius * 1.25).min(128.0);
    }
}

/// The ocean renders flat at the sea level while procedural outflow is on.
fn sync_sea_level(params: Res<ErodeParams>, mut materials: ResMut<Assets<TerrainMaterial>>) {
    if !params.is_changed() {
        return;
    }
    let level = if params.use_outflow != 0 {
        params.outflow_sea_level
    } else {
        0.0
    };
    for (_, mat) in materials.iter_mut() {
        if mat.extension.sea_level != level {
            mat.extension.sea_level = level;
        }
    }
}

fn handle_preview_mode_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut preview_mode: ResMut<CurrentPreviewMode>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut commands: Commands,
    mut cameras: Query<
        (
            Entity,
            &mut Tonemapping,
            &mut Bloom,
            Has<AtmosphereSettings>,
        ),
        With<Camera3d>,
    >,
    mut atmosphere_query: Query<(&mut Atmosphere, &mut Visibility), With<DemoAtmosphere>>,
    demo_earth_medium: Res<DemoEarthMedium>,
) {
    let new_mode = if keys.just_pressed(KeyCode::Digit1) {
        Some(PreviewMode::Pbr)
    } else if keys.just_pressed(KeyCode::Digit2) {
        Some(PreviewMode::Flow)
    } else if keys.just_pressed(KeyCode::Digit3) {
        Some(PreviewMode::Sediment)
    } else if keys.just_pressed(KeyCode::Digit4) {
        Some(PreviewMode::Erosion)
    } else if keys.just_pressed(KeyCode::Digit5) {
        Some(PreviewMode::Height)
    } else if keys.just_pressed(KeyCode::Digit6) {
        Some(PreviewMode::Normals)
    } else if keys.just_pressed(KeyCode::Digit7) {
        Some(PreviewMode::Curvature)
    } else if keys.just_pressed(KeyCode::Digit8) {
        Some(PreviewMode::Age)
    } else if keys.just_pressed(KeyCode::Digit9) {
        Some(PreviewMode::Simple)
    } else {
        None
    };

    if let Some(mode) = new_mode {
        if preview_mode.mode != mode {
            preview_mode.mode = mode;
            println!("Preview mode: {:?}", mode);

            // Update all terrain materials
            for (_, mat) in materials.iter_mut() {
                mat.extension.preview_mode = match mode {
                    PreviewMode::Pbr => 0,
                    PreviewMode::Flow => 1,
                    PreviewMode::Sediment => 2,
                    PreviewMode::Erosion => 3,
                    PreviewMode::Height => 4,
                    PreviewMode::Normals => 5,
                    PreviewMode::Curvature => 6,
                    PreviewMode::Age => 7,
                    PreviewMode::Simple => 8,
                };
            }

            // Toggle atmosphere, tonemapping, and bloom based on mode
            for (camera, mut tonemapping, mut bloom, has_atmosphere_settings) in cameras.iter_mut()
            {
                if mode == PreviewMode::Pbr {
                    let mut atmosphere = Atmosphere::earth(demo_earth_medium.0.clone());
                    atmosphere.ground_albedo = Vec3::splat(0.3);
                    for (mut atmosphere_component, mut visibility) in atmosphere_query.iter_mut() {
                        *atmosphere_component = atmosphere.clone();
                        *visibility = Visibility::Visible;
                    }
                    if !has_atmosphere_settings {
                        commands
                            .entity(camera)
                            .insert(AtmosphereSettings::default());
                    }
                    *tonemapping = Tonemapping::AcesFitted;
                    *bloom = Bloom::NATURAL;
                } else {
                    for (_, mut visibility) in atmosphere_query.iter_mut() {
                        *visibility = Visibility::Hidden;
                    }
                    if has_atmosphere_settings {
                        commands.entity(camera).remove::<AtmosphereSettings>();
                    }
                    *tonemapping = Tonemapping::None;
                    bloom.intensity = 0.0;
                }
            }
        }
    }
}
