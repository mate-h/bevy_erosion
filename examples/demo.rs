//! Erosion demo.
//!
//! The water is drawn by bevy_fft's water surface from a `WaterHeightfield` over the erosion's
//! water (see `bevy_erosion::water_heightfield`).
//!
//! `--simple` (or `EROSION_SIMPLE=1`) renders Lambert-shaded terrain with a flat water tint and
//! no water surface, atmosphere, environment map, bloom, auto exposure or shadows, with vsync off, and logs the
//! mean frame time every few seconds, so the frame is mostly the simulations.
//! `EROSION_BENCH_SECONDS=<n>` logs the mean frame time after the cascade settles and quits after
//! n seconds; `EROSION_BENCH_SPRING=1` places a spring at the map centre so the water solver runs,
//! `EROSION_BENCH_PIPES=1` switches it to the pipe model, and `EROSION_NO_ENV_FREEZE=1` keeps
//! re-filtering the environment map every frame. `EROSION_BENCH_POUR=x,y` places a spring at
//! texel (x, y) and moves the camera close to it or to the river's source (or to `EROSION_BENCH_TARGET=x,y`;
//! `EROSION_BENCH_DISTANCE`, `_PITCH` and `_YAW`,
//! in degrees, frame it), and `EROSION_BENCH_RAIN=<‰/s>` rains on the whole map, `EROSION_BENCH_RIVER=x,y` places a river from texel (x, y). `EROSION_VIEW=<view>` starts in a debug view (see the View tab). `EROSION_BENCH_SCREENSHOT=<path>` hides the settings panel (kept with `EROSION_BENCH_PANEL=1`) and saves
//! the window just before the bench quits.

use bevy::camera::visibility::NoFrustumCulling;
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
use bevy_erosion::{
    ErosionConfig,
    brush::{BrushMode, TerrainSurface, WaterBrush, WaterBrushPlugin},
    camera::{OrbitCameraPlugin, OrbitController},
    sun::{Sun, SunController, SunPlugin},
    water_heightfield, *,
};
use bevy_fft::water::{
    JerlovWaterType, WaterLodClear, WaterLodSettings, WaterOpticsUniform, WaterPlugin,
    WaterSurfaceExtension, WaterSurfaceMaterial, WaterSurfaceTag, WaterSurfaceUniform,
    WaterTransmission,
};
use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering},
};

/// Map size (width and height). Customize this to change erosion resolution.
const MAP_SIZE: UVec2 = UVec2::new(512, 512);
// Controls how much to under-expose (in F-stops). Negative values = under-expose, positive = over-expose
const UNDER_EXPOSURE_AMOUNT: f32 = -2.0;
/// Water shallower than this (world units) is not drawn, so the terrain shows on dry ground and
/// under thin films: `WET_DEPTH` in `water.wgsl` (0.0005 height units) at the terrain's height scale.
const DRY_DEPTH: f32 = 0.025;
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
    if let Some(seed) = std::env::var("EROSION_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        params.noise_seed = seed;
    }
}

/// `EROSION_VIEW=<view>` starts in another view, by its label in the View tab or its
/// [`PreviewMode`] name (any case, spaces optional), e.g. `EROSION_VIEW=bedload`.
fn view_from_env(mut view: ResMut<TerrainView>) {
    let Ok(name) = std::env::var("EROSION_VIEW") else {
        return;
    };
    let key = |s: &str| s.to_lowercase().replace(' ', "");
    let found = PreviewMode::ALL
        .iter()
        .find(|m| [key(m.label()), key(&format!("{m:?}"))].contains(&key(&name)));
    match found {
        Some(&mode) => view.mode = mode,
        None => warn!("EROSION_VIEW: no view named {name}"),
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
        .add_systems(PostStartup, (spawn_terrain, seed_from_env, view_from_env))
        .add_systems(
            Update,
            (
                handle_reset_input,
                handle_sim_input,
                handle_preview_mode_input,
                apply_terrain_view.after(handle_preview_mode_input),
                handle_brush_input,
                sync_sea_level,
                freeze_atmosphere_env_map,
                log_frame_times,
            ),
        );
    if !simple {
        // No FFT ocean underneath, so the water LOD data starts each frame empty. Five slices of
        // 1/8 m texels reach 512 m, the whole map from anywhere on it; the foam sim costs per slice.
        app.add_plugins(WaterPlugin)
            .insert_resource(WaterLodClear)
            .insert_resource(WaterLodSettings {
                slice_count: 5,
                resolution: 256,
                base_texel_size: 1.0 / 8.0,
            });
    }
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
    mut brush: ResMut<WaterBrush>,
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
        let texel = |name: &str| {
            let v = std::env::var(name).ok()?;
            let (x, y) = v.split_once(',')?;
            Some(Vec2::new(x.parse().ok()?, y.parse().ok()?))
        };
        // Place a river from there, as the river brush does.
        let river = texel("EROSION_BENCH_RIVER");
        if let Some(at) = river {
            brush.enabled = true;
            brush.mode = BrushMode::River;
            brush.place_river_at = Some(at);
        }
        let pour = texel("EROSION_BENCH_POUR");
        if let Some(at) = pour {
            water.springs.push(Spring {
                center: at,
                radius: 8.0,
                rate: 0.05,
            });
        }
        // Look at the pour or the river's source from close by.
        if let Some(focus) = texel("EROSION_BENCH_TARGET").or(pour).or(river) {
            let size = progress.get().size.as_vec2();
            let half = size.x * 0.25;
            let env = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<f32>().ok());
            for mut orbit in &mut cameras {
                let xz = focus / size * 2.0 * half - half;
                orbit.target = Vec3::new(xz.x, 20.0, xz.y);
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
            // The settings panel would cover part of the shot, unless `EROSION_BENCH_PANEL=1`.
            if !std::env::var("EROSION_BENCH_PANEL").is_ok_and(|v| v != "0") {
                for mut visibility in &mut panels {
                    *visibility = Visibility::Hidden;
                }
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

#[derive(Component)]
struct DemoAtmosphere;

fn setup(mut commands: Commands) {
    commands.insert_resource(GlobalAmbientLight::NONE);
    if simple_mode() {
        commands.insert_resource(TerrainView {
            mode: PreviewMode::Simple,
            ..default()
        });
    }
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
    println!("  0 - River momentum (Do Rivers)");
    println!("  (the View tab of the settings panel has every view)");
    println!("  B - toggle the brush (left drag paints, Shift + drag erases, right drag orbits)");
    println!("  V - cycle brush: rain / terrain / water body");
    println!("  T - cycle terrain tool: raise / lower / flatten / smooth");
    println!("  [ / ] - brush radius");
    println!("  Alt + click (water body) - place a spring, Alt + Shift + click removes springs");
    println!("  C - clear water and springs   X - reset terrain edits");
    println!("  G - shape lake and river beds under the current water");
    println!("  M - refine beds: fast-forward bed load under the current flow");
}

#[allow(clippy::too_many_arguments)]
fn spawn_terrain(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut water_materials: ResMut<Assets<WaterSurfaceMaterial>>,
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
        // The view (PBR, or Simple with --simple) follows `TerrainView`.
        extension: TerrainExtension::new(&erosion_images, height_scale),
    });

    if !simple_mode() {
        commands.spawn(water_heightfield(
            &erosion_images,
            config.map_size,
            half_size,
            height_scale,
        ));
        // The surface samples the water LOD data by world position, so it reuses the terrain's
        // grid. Its bounds are flat, so it is never frustum culled.
        commands.spawn((
            Mesh3d(mesh_handle.clone()),
            MeshMaterial3d(water_materials.add(WaterSurfaceMaterial {
                base: StandardMaterial {
                    base_color: Color::WHITE,
                    specular_transmission: 1.0,
                    ior: 1.333,
                    reflectance: 0.356,
                    perceptual_roughness: 0.08,
                    metallic: 0.0,
                    opaque_render_method: OpaqueRendererMethod::Forward,
                    ..default()
                },
                extension: WaterSurfaceExtension {
                    optics: WaterOpticsUniform {
                        enabled: 1,
                        ..WaterOpticsUniform::jerlov(JerlovWaterType::Coastal3)
                    },
                    settings: WaterSurfaceUniform {
                        dry_depth: DRY_DEPTH,
                        ..WaterSurfaceExtension::default().settings
                    },
                    ..default()
                },
            })),
            Transform::default(),
            WaterSurfaceTag,
            NoFrustumCulling,
            bevy::light::NotShadowCaster,
        ));
    }

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
        camera.insert((
            Tonemapping::None,
            Bloom {
                intensity: 0.0,
                ..Bloom::NATURAL
            },
        ));
    } else {
        camera.insert((
            WaterTransmission,
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

/// Digits pick the first ten views (the settings panel's View tab has them all).
fn handle_preview_mode_input(keys: Res<ButtonInput<KeyCode>>, mut view: ResMut<TerrainView>) {
    let mode = [
        (KeyCode::Digit1, PreviewMode::Pbr),
        (KeyCode::Digit2, PreviewMode::Flow),
        (KeyCode::Digit3, PreviewMode::Sediment),
        (KeyCode::Digit4, PreviewMode::Erosion),
        (KeyCode::Digit5, PreviewMode::Height),
        (KeyCode::Digit6, PreviewMode::Normals),
        (KeyCode::Digit7, PreviewMode::Curvature),
        (KeyCode::Digit8, PreviewMode::Age),
        (KeyCode::Digit9, PreviewMode::Simple),
        (KeyCode::Digit0, PreviewMode::Rivers),
    ]
    .into_iter()
    .find_map(|(key, mode)| keys.just_pressed(key).then_some(mode));
    if let Some(mode) = mode
        && view.mode != mode
    {
        view.mode = mode;
    }
}

/// Full PBR draws the atmosphere, the water surface, tonemapping, bloom and auto exposure; every
/// other view shows the terrain alone with none of them, so the colours are the data's.
#[allow(clippy::type_complexity)]
fn apply_terrain_view(
    view: Res<TerrainView>,
    mut commands: Commands,
    mut cameras: Query<
        (
            Entity,
            &mut Tonemapping,
            &mut Bloom,
            Has<AtmosphereSettings>,
            Option<&AutoExposure>,
        ),
        With<Camera3d>,
    >,
    mut atmosphere_query: Query<&mut Visibility, With<DemoAtmosphere>>,
    mut water_surfaces: Query<&mut Visibility, (With<WaterSurfaceTag>, Without<DemoAtmosphere>)>,
    mut stored_exposure: Local<Option<AutoExposure>>,
) {
    if !view.is_changed() || simple_mode() {
        return;
    }
    let pbr = view.mode == PreviewMode::Pbr;
    let visibility = if pbr {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
    for mut v in atmosphere_query.iter_mut().chain(water_surfaces.iter_mut()) {
        v.set_if_neq(visibility);
    }
    for (camera, mut tonemapping, mut bloom, has_atmosphere_settings, exposure) in &mut cameras {
        if pbr {
            if !has_atmosphere_settings {
                commands
                    .entity(camera)
                    .insert(AtmosphereSettings::default());
            }
            if let Some(exposure) = stored_exposure.take() {
                commands.entity(camera).insert(exposure);
            }
            *tonemapping = Tonemapping::AcesFitted;
            *bloom = Bloom::NATURAL;
        } else {
            if has_atmosphere_settings {
                commands.entity(camera).remove::<AtmosphereSettings>();
            }
            if let Some(exposure) = exposure {
                *stored_exposure = Some(exposure.clone());
                commands.entity(camera).remove::<AutoExposure>();
            }
            *tonemapping = Tonemapping::None;
            bloom.intensity = 0.0;
        }
    }
}
