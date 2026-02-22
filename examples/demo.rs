use bevy::{
    core_pipeline::tonemapping::Tonemapping,
    light::{AtmosphereEnvironmentMapLight, light_consts::lux},
    math::{cubic_splines::LinearSpline, vec2},
    pbr::{
        Atmosphere, AtmosphereSettings, EarthlikeAtmosphere, ExtendedMaterial, MaterialPlugin, OpaqueRendererMethod, StandardMaterial
    },
    post_process::{
        auto_exposure::{AutoExposure, AutoExposureCompensationCurve, AutoExposurePlugin},
        bloom::Bloom,
    },
    prelude::*,
};
use bevy_erosion::{
    camera::{OrbitCameraPlugin, OrbitController},
    sun::{SunPlugin, Sun, SunController},
    ErosionConfig, *,
};
#[cfg(feature = "ui")]
use bevy_erosion::ui::ErosionParamsPlugin;

/// Map size (width and height). Customize this to change erosion resolution.
const MAP_SIZE: UVec2 = UVec2::new(512, 512);
// Controls how much to under-expose (in F-stops). Negative values = under-expose, positive = over-expose
const UNDER_EXPOSURE_AMOUNT: f32 = -2.0;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(ErosionConfig {
            map_size: MAP_SIZE,
            ..Default::default()
        })
        .add_plugins((
            DefaultPlugins,
            ErosionComputePlugin,
            MaterialPlugin::<TerrainMaterial>::default(),
            OrbitCameraPlugin,
            SunPlugin,
            AutoExposurePlugin,
            #[cfg(feature = "ui")]
            ErosionParamsPlugin,
        ))
        .add_systems(Startup, (setup, print_controls))
        .add_systems(PostStartup, spawn_terrain)
        .add_systems(
            Update,
            (
                handle_reset_input,
                handle_sim_input,
                handle_preview_mode_input,
            ),
        )
        .run();
}

#[derive(Resource, Default)]
struct CurrentPreviewMode {
    mode: PreviewMode,
}

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
}

fn spawn_terrain(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut compensation_curves: ResMut<Assets<AutoExposureCompensationCurve>>,
    earth_atmosphere: Res<EarthlikeAtmosphere>,
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
            height_scale: 50.0,
            preview_mode: 0, // Default to PBR
        },
    });

    commands.spawn((
        Mesh3d(mesh_handle),
        MeshMaterial3d(mat_handle),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));

    let mut atmosphere = earth_atmosphere.get();
    atmosphere.ground_albedo = Vec3::splat(0.3);

    // Create a compensation curve that slightly under-exposes
    // Negative compensation values = darker/under-exposed
    // The curve shape applies more under-exposure in darker scenes and less in brighter scenes
    let under_expose_curve = compensation_curves.add(
        AutoExposureCompensationCurve::from_curve(LinearSpline::new([
            vec2(-4.0, UNDER_EXPOSURE_AMOUNT * 1.67), // Dark scenes: more under-exposure
            vec2(-2.0, UNDER_EXPOSURE_AMOUNT * 1.33), // Medium-dark: moderate under-exposure
            vec2(0.0, UNDER_EXPOSURE_AMOUNT),         // Middle gray: base under-exposure
            vec2(2.0, UNDER_EXPOSURE_AMOUNT * 0.67), // Medium-bright: less under-exposure
            vec2(4.0, UNDER_EXPOSURE_AMOUNT * 0.33), // Bright scenes: minimal under-exposure
        ]))
        .expect("Failed to create compensation curve"),
    );

    // 3D camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(150.0, 50.0, 150.0).looking_at(Vec3::ZERO, Vec3::Y),
        OrbitController {
            target: Vec3::ZERO,
            distance: 225.0,
            ..Default::default()
        },
        atmosphere,
        AtmosphereSettings {
            scene_units_to_m: 1.0,
            // rendering_method: AtmosphereMode::Raymarched,
            ..Default::default()
        },
        AtmosphereEnvironmentMapLight::default(),
        Tonemapping::AcesFitted,
        Bloom::NATURAL,
        AutoExposure {
            compensation_curve: under_expose_curve,
            ..Default::default()
        },
    ));

    // Directional light for PBR shading
    let initial_sun_pos = Vec3::new(1.0, 0.1, 0.0);
    let initial_sun_dir = -initial_sun_pos.normalize(); // Direction from origin to sun
    commands.spawn((
        DirectionalLight {
            shadows_enabled: true,
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

fn handle_preview_mode_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut preview_mode: ResMut<CurrentPreviewMode>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut cameras: Query<(&mut Atmosphere, &mut Tonemapping, &mut Bloom), With<Camera3d>>,
    earth_atmosphere: Res<EarthlikeAtmosphere>,
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
                };
            }

            // Toggle atmosphere, tonemapping, and bloom based on mode
            for (mut atmosphere, mut tonemapping, mut bloom) in cameras.iter_mut() {
                if mode == PreviewMode::Pbr {
                    // Enable atmosphere and bloom for PBR
                    *atmosphere = earth_atmosphere.get();
                    *tonemapping = Tonemapping::AcesFitted;
                    *bloom = Bloom::NATURAL;
                } else {
                    // Disable atmosphere and bloom for debug modes
                    // Preserve the medium handle to avoid invalid asset errors
                    let medium_handle = atmosphere.medium.clone();
                    *atmosphere = Atmosphere {
                        bottom_radius: 0.0,
                        top_radius: 0.0,
                        ground_albedo: Vec3::ZERO,
                        medium: medium_handle,
                    };
                    *tonemapping = Tonemapping::None;
                    bloom.intensity = 0.0;
                }
            }
        }
    }
}

