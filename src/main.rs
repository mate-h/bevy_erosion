use bevy::{
    asset::RenderAssetUsages,
    camera::Exposure,
    core_pipeline::tonemapping::Tonemapping,
    light::{AtmosphereEnvironmentMapLight, light_consts::lux},
    pbr::{
        Atmosphere, AtmosphereSettings, EarthlikeAtmosphere, ExtendedMaterial, MaterialExtension, MaterialPlugin, OpaqueRendererMethod, ScatteringMedium
    },
    post_process::bloom::Bloom,
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraph, RenderGraphContext, RenderLabel},
        render_resource::{
            binding_types::{
                storage_buffer, storage_buffer_read_only, texture_storage_2d, uniform_buffer,
            },
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
    },
    shader::{PipelineCacheError, ShaderRef},
};
use std::borrow::Cow;
mod camera;
use camera::{OrbitCameraPlugin, OrbitController};

const EROSION_SHADER: &str = "erosion.wgsl";
const TERRAIN_SHADER: &str = "terrain_extended.wgsl";

const SIZE: UVec2 = UVec2::new(512, 512);

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins((
            DefaultPlugins,
            ErosionComputePlugin,
            MaterialPlugin::<TerrainMaterial>::default(),
            OrbitCameraPlugin,
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

#[derive(Resource, Clone, ExtractResource)]
struct ErosionImages {
    height: Handle<Image>,
    color: Handle<Image>,
    normal: Handle<Image>,
    analysis: Handle<Image>, // Combined: R=flow_mag, G=sediment, B=erosion, A=flow_dir_encoded
}

type TerrainMaterial = ExtendedMaterial<StandardMaterial, TerrainExtension>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Reflect)]
enum PreviewMode {
    #[default]
    Pbr,
    Flow,
    Sediment,
    Erosion,
    Height,
    Normals,
}

#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
struct TerrainExtension {
    // Height map sampled in vertex shader
    #[texture(100)]
    #[sampler(101)]
    height_tex: Handle<Image>,
    // Color map sampled in fragment shader
    #[texture(102)]
    #[sampler(103)]
    color_tex: Handle<Image>,
    // Analysis texture: R=flow_mag, G=sediment, B=erosion
    #[texture(105)]
    #[sampler(106)]
    analysis_tex: Handle<Image>,
    // Displacement height scale
    #[uniform(104)]
    height_scale: f32,
    // Preview mode: 0=PBR, 1=Flow, 2=Sediment, 3=Erosion, 4=Height, 5=Normals
    #[uniform(107)]
    preview_mode: u32,
}

impl Default for TerrainExtension {
    fn default() -> Self {
        Self {
            height_tex: Default::default(),
            color_tex: Default::default(),
            analysis_tex: Default::default(),
            height_scale: 50.0,
            preview_mode: 0,
        }
    }
}

impl MaterialExtension for TerrainExtension {
    fn vertex_shader() -> ShaderRef {
        TERRAIN_SHADER.into()
    }
    fn fragment_shader() -> ShaderRef {
        TERRAIN_SHADER.into()
    }
}

#[derive(Resource, Clone, ExtractResource, ShaderType)]
struct ErodeParams {
    map_size: UVec2,
    max_lifetime: u32,
    border_size: u32,
    inertia: f32,
    sediment_capacity_factor: f32,
    min_sediment_capacity: f32,
    deposit_speed: f32,
    erode_speed: f32,
    evaporate_speed: f32,
    gravity: f32,
    start_speed: f32,
    start_water: f32,
    brush_length: u32,
    num_particles: u32,
}

#[derive(Resource, Clone, ExtractResource)]
struct ErosionCpuBuffers {
    brush_indices: Vec<i32>,
    brush_weights: Vec<f32>,
}

#[derive(Resource, Clone, ExtractResource, Default)]
struct ResetSim {
    generation: u32,
}

#[derive(Resource, Clone, ExtractResource, Default)]
struct SimControl {
    paused: bool,
    step_counter: u64,
}

#[derive(Resource, Default)]
struct CurrentPreviewMode {
    mode: PreviewMode,
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    // Height texture (rgba16float) written by compute blit
    let mut height = Image::new_target_texture(SIZE.x, SIZE.y, TextureFormat::Rgba16Float);
    height.asset_usage = RenderAssetUsages::RENDER_WORLD;
    height.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let height_handle = images.add(height);

    // Color texture (Rgba16Float) used as storage for color advection and sampled in terrain shader
    let mut color = Image::new_target_texture(SIZE.x, SIZE.y, TextureFormat::Rgba16Float);
    color.asset_usage = RenderAssetUsages::RENDER_WORLD;
    color.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let color_handle = images.add(color);

    // Normal texture (Rgba16Float) written by compute and optionally sampled later
    let mut normal = Image::new_target_texture(SIZE.x, SIZE.y, TextureFormat::Rgba16Float);
    normal.asset_usage = RenderAssetUsages::RENDER_WORLD;
    normal.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let normal_handle = images.add(normal);

    // Analysis texture (Rgba16Float) - combined flow/sediment/erosion data
    // R channel: flow magnitude, G channel: sediment, B channel: erosion, A channel: flow direction encoded
    let mut analysis = Image::new_target_texture(SIZE.x, SIZE.y, TextureFormat::Rgba16Float);
    analysis.asset_usage = RenderAssetUsages::RENDER_WORLD;
    analysis.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let analysis_handle = images.add(analysis);

    commands.insert_resource(ErosionImages {
        height: height_handle,
        color: color_handle,
        normal: normal_handle,
        analysis: analysis_handle,
    });

    // Parameters
    let brush_radius: i32 = 16;
    let (brush_indices, brush_weights) = build_brush(SIZE.x as i32, brush_radius);
    let num_particles: u32 = 1024 * 8;

    commands.insert_resource(ErodeParams {
        map_size: SIZE,
        max_lifetime: 32,
        border_size: 0,
        inertia: 0.05,
        sediment_capacity_factor: 4.0,
        min_sediment_capacity: 0.01,
        deposit_speed: 0.3,
        erode_speed: 0.3,
        evaporate_speed: 0.01,
        gravity: 4.0,
        start_speed: 1.0,
        start_water: 1.0,
        brush_length: brush_indices.len() as u32,
        num_particles,
    });
    commands.insert_resource(ErosionCpuBuffers {
        brush_indices,
        brush_weights,
    });
    commands.insert_resource(ResetSim::default());
    commands.insert_resource(SimControl::default());
    commands.insert_resource(CurrentPreviewMode::default());

    // commands.insert_resource(AmbientLight::default());
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
}

fn spawn_terrain(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    earth_atmosphere: Res<EarthlikeAtmosphere>,
    erosion_images: Res<ErosionImages>,
) {
    // Spawn a subdivided plane with UVs
    let size = 512.0;
    let resolution = 511; // subdivisions
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
            perceptual_roughness: 0.8,
            metallic: 0.0,
            opaque_render_method: OpaqueRendererMethod::Auto,
            ..Default::default()
        },
        extension: TerrainExtension {
            height_tex: erosion_images.height.clone(),
            color_tex: erosion_images.color.clone(),
            analysis_tex: erosion_images.analysis.clone(),
            height_scale: 50.0,
            preview_mode: 0, // Default to PBR
        },
    });

    commands.spawn((
        Mesh3d(mesh_handle),
        MeshMaterial3d(mat_handle),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));

    // 3D camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(150.0, 50.0, 150.0).looking_at(Vec3::ZERO, Vec3::Y),
        OrbitController {
            target: Vec3::ZERO,
            distance: 225.0,
            ..Default::default()
        },
        earth_atmosphere.get(),
        AtmosphereSettings {
            scene_units_to_m: 50.0,
            // rendering_method: AtmosphereMode::Raymarched,
            ..Default::default()
        },
        AtmosphereEnvironmentMapLight::default(),
        Exposure { ev100: 12.0 },
        Tonemapping::AcesFitted,
        Bloom::NATURAL,
    ));

    // Directional light for PBR shading
    commands.spawn((
        DirectionalLight {
            shadows_enabled: true,
            illuminance: lux::RAW_SUNLIGHT,
            ..default()
        },
        Transform::from_xyz(1.0, 0.1, 0.0).looking_at(Vec3::ZERO, Vec3::Y),
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

fn generate_random_indices_seeded(count: u32, max_index: u32, mut state: u64) -> Vec<u32> {
    let mut v = Vec::with_capacity(count as usize);
    for _ in 0..count {
        // xorshift64*
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let r = (state.wrapping_mul(2685821657736338717) >> 32) as u32;
        v.push(if max_index == 0 { 0 } else { r % max_index });
    }
    v
}

fn build_brush(map_size: i32, radius: i32) -> (Vec<i32>, Vec<f32>) {
    let mut idx = Vec::new();
    let mut w = Vec::new();
    let r2 = (radius * radius) as f32;
    let r = radius as f32;
    for y in -radius..=radius {
        for x in -radius..=radius {
            let d2 = (x * x + y * y) as f32;
            if d2 <= r2 {
                // Match Sebastian Lague's weighting: 1 - sqrt(distance) / radius
                let weight = 1.0 - (d2.sqrt() / r);
                idx.push(y * map_size + x);
                w.push(weight.max(0.0));
            }
        }
    }
    let sum: f32 = w.iter().sum::<f32>().max(1e-6);
    for ww in &mut w {
        *ww /= sum;
    }
    (idx, w)
}

struct ErosionComputePlugin;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct ErosionLabel;

impl Plugin for ErosionComputePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractResourcePlugin::<ErosionImages>::default(),
            ExtractResourcePlugin::<ErodeParams>::default(),
            ExtractResourcePlugin::<ErosionCpuBuffers>::default(),
            ExtractResourcePlugin::<ResetSim>::default(),
            ExtractResourcePlugin::<SimControl>::default(),
        ));

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .add_systems(RenderStartup, init_erosion_pipeline)
            .add_systems(
                Render,
                prepare_erosion_bind_groups.in_set(RenderSystems::PrepareBindGroups),
            );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(ErosionLabel, ErosionNode::default());
        render_graph.add_node_edge(ErosionLabel, bevy::render::graph::CameraDriverLabel);
    }
}

#[derive(Resource)]
struct ErosionPipeline {
    layout_erosion: BindGroupLayoutDescriptor,
    layout_color: BindGroupLayoutDescriptor,
    layout_blit: BindGroupLayoutDescriptor,
    pipeline_init: CachedComputePipelineId,
    pipeline_init_color: CachedComputePipelineId,
    pipeline_erode: CachedComputePipelineId,
    pipeline_blit: CachedComputePipelineId,
}

#[derive(Resource)]
struct ErosionBindGroups {
    erosion: BindGroup,
    color: BindGroup,
    blit: BindGroup,
}

#[derive(Resource)]
struct ErosionBuffers {
    uniform: UniformBuffer<ErodeParams>,
    height: StorageBuffer<Vec<f32>>,
    random: StorageBuffer<Vec<u32>>,
    brush_idx: StorageBuffer<Vec<i32>>,
    brush_w: StorageBuffer<Vec<f32>>,
    flow: StorageBuffer<Vec<u32>>,     // atomic<u32> in shader
    sediment: StorageBuffer<Vec<u32>>, // atomic<u32> in shader
    erosion: StorageBuffer<Vec<u32>>,  // atomic<u32> in shader
}

#[derive(Resource, Clone)]
struct ErosionRngState {
    state: u64,
}

fn init_erosion_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    // Layout for erosion (group 0)
    let layout_erosion = BindGroupLayoutDescriptor::new(
        "ErosionGroup0",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer::<f32>(false), // height
                uniform_buffer::<ErodeParams>(false),
                storage_buffer_read_only::<u32>(false), // random_indices
                storage_buffer_read_only::<i32>(false), // brush_indices
                storage_buffer_read_only::<f32>(false), // brush_weights
                storage_buffer::<u32>(false),           // flow (atomic<u32> in shader)
                storage_buffer::<u32>(false),           // sediment (atomic<u32> in shader)
                storage_buffer::<u32>(false),           // erosion (atomic<u32> in shader)
            ),
        ),
    );

    // Layout for blit (group 1)
    let layout_blit = BindGroupLayoutDescriptor::new(
        "ErosionBlitGroup1",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly), // height display
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly), // normal
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly), // analysis (flow+sediment+erosion combined)
            ),
        ),
    );

    // Layout for color (group 1 for erosion/color init)
    let layout_color = BindGroupLayoutDescriptor::new(
        "ErosionColorGroup1",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (texture_storage_2d(
                TextureFormat::Rgba16Float,
                StorageTextureAccess::ReadWrite,
            ),),
        ),
    );

    let shader = asset_server.load(EROSION_SHADER);
    let pipeline_init = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![layout_erosion.clone()],
        shader: shader.clone(),
        entry_point: Some(Cow::from("init_fbm")),
        ..default()
    });
    let pipeline_init_color = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![layout_erosion.clone(), layout_color.clone()],
        shader: shader.clone(),
        entry_point: Some(Cow::from("init_color_bands")),
        ..default()
    });
    let pipeline_erode = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![layout_erosion.clone(), layout_color.clone()],
        shader: shader.clone(),
        entry_point: Some(Cow::from("erode")),
        ..default()
    });
    let pipeline_blit = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![layout_erosion.clone(), layout_blit.clone()],
        shader,
        entry_point: Some(Cow::from("blit_to_texture")),
        ..default()
    });

    commands.insert_resource(ErosionPipeline {
        layout_erosion,
        layout_color,
        layout_blit,
        pipeline_init,
        pipeline_init_color,
        pipeline_erode,
        pipeline_blit,
    });
}

fn prepare_erosion_bind_groups(
    mut commands: Commands,
    pipeline: Res<ErosionPipeline>,
    pipeline_cache: Res<PipelineCache>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    images: Res<ErosionImages>,
    params: Res<ErodeParams>,
    cpu_buffers: Res<ErosionCpuBuffers>,
    render_device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut existing: Option<ResMut<ErosionBuffers>>,
    rng_state: Option<ResMut<ErosionRngState>>,
) {
    let view_height = gpu_images.get(&images.height).unwrap();
    let view_color = gpu_images.get(&images.color).unwrap();
    let view_normal = gpu_images.get(&images.normal).unwrap();
    let view_analysis = gpu_images.get(&images.analysis).unwrap();

    // Check if we need to verify buffer structure
    let texels = (params.map_size.x * params.map_size.y) as usize;
    let needs_recreation = if let Some(ref buffers) = existing {
        // Check if buffers have correct size for new fields
        // If any of the analysis buffers are empty/wrong size, recreate
        buffers.flow.buffer().is_none()
            || buffers.sediment.buffer().is_none()
            || buffers.erosion.buffer().is_none()
            // Check if all buffers have correct size (should all be texels, 1 u32 per pixel)
            || buffers.flow.get().len() != texels
            || buffers.sediment.get().len() != texels
            || buffers.erosion.get().len() != texels
    } else {
        false
    };

    if needs_recreation {
        // Force recreation by removing the resource
        existing = None;
    }

    match existing {
        Some(mut buffers) => {
            // Ensure we have RNG state
            let mut seed = if let Some(ref rs) = rng_state {
                rs.state
            } else {
                0xCAFEBABE_1234_5678
            };
            // Regenerate start positions every frame using evolving RNG state
            let new_random = generate_random_indices_seeded(
                params.num_particles,
                params.map_size.x * params.map_size.y,
                seed,
            );
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            if let Some(mut rs) = rng_state {
                rs.state = seed;
            } else {
                commands.insert_resource(ErosionRngState { state: seed });
            }
            buffers.random = StorageBuffer::from(new_random);
            buffers.random.write_buffer(&render_device, &queue);

            // Reuse existing buffers, just rebuild bind groups (safe across frames)
            let erosion = render_device.create_bind_group(
                None,
                &pipeline_cache.get_bind_group_layout(&pipeline.layout_erosion),
                &BindGroupEntries::sequential((
                    &buffers.height,
                    &buffers.uniform,
                    &buffers.random,
                    &buffers.brush_idx,
                    &buffers.brush_w,
                    &buffers.flow,
                    &buffers.sediment,
                    &buffers.erosion,
                )),
            );
            let blit = render_device.create_bind_group(
                None,
                &pipeline_cache.get_bind_group_layout(&pipeline.layout_blit),
                &BindGroupEntries::sequential((
                    &view_height.texture_view,
                    &view_normal.texture_view,
                    &view_analysis.texture_view,
                )),
            );
            let color = render_device.create_bind_group(
                None,
                &pipeline_cache.get_bind_group_layout(&pipeline.layout_color),
                &BindGroupEntries::sequential((&view_color.texture_view,)),
            );
            commands.insert_resource(ErosionBindGroups {
                erosion,
                color,
                blit,
            });
        }
        None => {
            // Create buffers once
            let mut uniform = UniformBuffer::from(params.clone());
            uniform.write_buffer(&render_device, &queue);

            let texels = (params.map_size.x * params.map_size.y) as usize;
            let mut height = StorageBuffer::from(vec![0.0f32; texels]);
            height.write_buffer(&render_device, &queue);
            // Create and fill initial height once using FBM on CPU for now (optional: let GPU init and copy back)

            // Initialize RNG state
            let mut seed = if let Some(ref rs) = rng_state {
                rs.state
            } else {
                0xCAFEBABE_1234_5678
            };
            let first_random = generate_random_indices_seeded(
                params.num_particles,
                params.map_size.x * params.map_size.y,
                seed,
            );
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            if let Some(mut rs) = rng_state {
                rs.state = seed;
            } else {
                commands.insert_resource(ErosionRngState { state: seed });
            }
            let mut random = StorageBuffer::from(first_random);
            random.write_buffer(&render_device, &queue);

            let mut brush_idx = StorageBuffer::from(cpu_buffers.brush_indices.clone());
            brush_idx.write_buffer(&render_device, &queue);

            let mut brush_w = StorageBuffer::from(cpu_buffers.brush_weights.clone());
            brush_w.write_buffer(&render_device, &queue);

            // Initialize buffers for atomic operations
            // All buffers store single scalar per pixel (1 u32 per pixel)
            let mut flow = StorageBuffer::from(vec![0u32; texels]);
            flow.write_buffer(&render_device, &queue);

            let mut sediment = StorageBuffer::from(vec![0u32; texels]);
            sediment.write_buffer(&render_device, &queue);

            let mut erosion = StorageBuffer::from(vec![0u32; texels]);
            erosion.write_buffer(&render_device, &queue);

            // Build bind groups using these buffers
            let erosion_bg = render_device.create_bind_group(
                None,
                &pipeline_cache.get_bind_group_layout(&pipeline.layout_erosion),
                &BindGroupEntries::sequential((
                    &height, &uniform, &random, &brush_idx, &brush_w, &flow, &sediment, &erosion,
                )),
            );
            let blit = render_device.create_bind_group(
                None,
                &pipeline_cache.get_bind_group_layout(&pipeline.layout_blit),
                &BindGroupEntries::sequential((
                    &view_height.texture_view,
                    &view_normal.texture_view,
                    &view_analysis.texture_view,
                )),
            );
            let color = render_device.create_bind_group(
                None,
                &pipeline_cache.get_bind_group_layout(&pipeline.layout_color),
                &BindGroupEntries::sequential((&view_color.texture_view,)),
            );

            // Insert buffers and groups as resources
            commands.insert_resource(ErosionBuffers {
                uniform,
                height,
                random,
                brush_idx,
                brush_w,
                flow,
                sediment,
                erosion,
            });
            commands.insert_resource(ErosionBindGroups {
                erosion: erosion_bg,
                color,
                blit,
            });
        }
    }
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

enum ErosionState {
    Loading,
    Init,
    Erode,
}

struct ErosionNode {
    state: ErosionState,
    last_reset_gen: u32,
    allow_erode_this_frame: bool,
    last_step_seen: u64,
}

impl Default for ErosionNode {
    fn default() -> Self {
        Self {
            state: ErosionState::Loading,
            last_reset_gen: 0,
            allow_erode_this_frame: true,
            last_step_seen: 0,
        }
    }
}

impl Node for ErosionNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<ErosionPipeline>();
        let cache = world.resource::<PipelineCache>();
        let reset = world.resource::<ResetSim>();
        let sim = world.resource::<SimControl>();
        if reset.generation != self.last_reset_gen {
            self.last_reset_gen = reset.generation;
            self.state = ErosionState::Loading;
        }
        if sim.paused {
            if sim.step_counter > self.last_step_seen {
                self.allow_erode_this_frame = true;
                self.last_step_seen = sim.step_counter;
            } else {
                self.allow_erode_this_frame = false;
            }
        } else {
            self.allow_erode_this_frame = true;
        }
        match self.state {
            ErosionState::Loading => match cache.get_compute_pipeline_state(pipeline.pipeline_init)
            {
                CachedPipelineState::Ok(_) => {
                    self.state = ErosionState::Init;
                }
                CachedPipelineState::Err(PipelineCacheError::ShaderNotLoaded(_)) => {}
                CachedPipelineState::Err(err) => {
                    panic!("Initializing assets/{EROSION_SHADER}:\n{err}")
                }
                _ => {}
            },
            ErosionState::Init => {
                // Wait one frame after color pipeline is ready before transitioning,
                // guaranteeing that `init_color_bands` runs at least once.
                let color_ready = matches!(
                    cache.get_compute_pipeline_state(pipeline.pipeline_init_color),
                    CachedPipelineState::Ok(_)
                );
                let erode_ready = matches!(
                    cache.get_compute_pipeline_state(pipeline.pipeline_erode),
                    CachedPipelineState::Ok(_)
                );
                if color_ready && erode_ready {
                    self.state = ErosionState::Erode;
                }
            }
            ErosionState::Erode => {}
        }
    }

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let cache = world.resource::<PipelineCache>();
        let pipes = world.resource::<ErosionPipeline>();
        let groups = world.resource::<ErosionBindGroups>();
        let params = world.resource::<ErodeParams>();
        let buffers = world.resource::<ErosionBuffers>();

        match self.state {
            ErosionState::Loading => {}
            ErosionState::Init => {
                // Pass 1: init FBM (writes buffer)
                {
                    let mut pass = render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor::default());
                    let p_init = cache.get_compute_pipeline(pipes.pipeline_init).unwrap();
                    pass.set_pipeline(p_init);
                    pass.set_bind_group(0, &groups.erosion, &[]);
                    let gx = (params.map_size.x + 7) / 8;
                    let gy = (params.map_size.y + 7) / 8;
                    pass.dispatch_workgroups(gx, gy, 1);
                }
                // Pass 1b: init color bands (uses height as input, writes color storage texture)
                if let Some(p_color_ok) =
                    match cache.get_compute_pipeline_state(pipes.pipeline_init_color) {
                        CachedPipelineState::Ok(_) => {
                            cache.get_compute_pipeline(pipes.pipeline_init_color)
                        }
                        _ => None,
                    }
                {
                    let mut pass = render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor::default());
                    pass.set_pipeline(p_color_ok);
                    pass.set_bind_group(0, &groups.erosion, &[]);
                    pass.set_bind_group(1, &groups.color, &[]);
                    let gx = (params.map_size.x + 7) / 8;
                    let gy = (params.map_size.y + 7) / 8;
                    pass.dispatch_workgroups(gx, gy, 1);
                }
                // Pass 2: blit (samples)
                {
                    let mut pass = render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor::default());
                    let p_blit = cache.get_compute_pipeline(pipes.pipeline_blit).unwrap();
                    pass.set_pipeline(p_blit);
                    pass.set_bind_group(0, &groups.erosion, &[]);
                    pass.set_bind_group(1, &groups.blit, &[]);
                    let gx = (params.map_size.x + 7) / 8;
                    let gy = (params.map_size.y + 7) / 8;
                    pass.dispatch_workgroups(gx, gy, 1);
                }
            }
            ErosionState::Erode => {
                // If reset happened recently, restore from initial before eroding
                // (Handled via state transition to Loading/Init above)
                // Pass 1: erode (writes storage) — gated by pause/step
                if self.allow_erode_this_frame {
                    if let Some(p_erode) = cache.get_compute_pipeline(pipes.pipeline_erode) {
                        let mut pass = render_context
                            .command_encoder()
                            .begin_compute_pass(&ComputePassDescriptor::default());
                        pass.set_pipeline(p_erode);
                        pass.set_bind_group(0, &groups.erosion, &[]);
                        pass.set_bind_group(1, &groups.color, &[]);
                        let workgroups = (params.num_particles + 1023) / 1024;
                        pass.dispatch_workgroups(workgroups, 1, 1);
                    } else {
                        error!("Erode pipeline not ready!");
                    }
                }
                // Pass 2: blit (samples)
                {
                    let mut pass = render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor::default());
                    let p_blit = cache.get_compute_pipeline(pipes.pipeline_blit).unwrap();
                    pass.set_pipeline(p_blit);
                    pass.set_bind_group(0, &groups.erosion, &[]);
                    pass.set_bind_group(1, &groups.blit, &[]);
                    let gx = (params.map_size.x + 7) / 8;
                    let gy = (params.map_size.y + 7) / 8;
                    pass.dispatch_workgroups(gx, gy, 1);
                }
            }
        }

        // Keep buffers alive (not strictly necessary, but avoids drop before usage on some backends)
        let _keep = (
            &buffers.uniform,
            &buffers.random,
            &buffers.brush_idx,
            &buffers.brush_w,
        );

        Ok(())
    }
}
