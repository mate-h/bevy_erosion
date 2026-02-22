use bevy::{
    asset::RenderAssetUsages,
    pbr::{
        ExtendedMaterial, MaterialExtension, StandardMaterial,
    },
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
            TextureFormat, TextureUsages, *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
    },
    shader::ShaderRef,
};
use std::borrow::Cow;

pub mod camera;
pub mod sun;

#[cfg(feature = "ui")]
pub mod ui;

const EROSION_SHADER: &str = "erosion.wgsl";
const TERRAIN_SHADER: &str = "terrain_extended.wgsl";

/// Resource containing handles to all erosion-related images
#[derive(Resource, Clone, ExtractResource)]
pub struct ErosionImages {
    pub height: Handle<Image>,
    pub color: Handle<Image>,
    pub normal: Handle<Image>,
    pub analysis: Handle<Image>, // Combined: R=flow_mag, G=sediment, B=erosion, A=flow_dir_encoded
    pub ao: Handle<Image>,        // Ambient occlusion texture
    pub ao_temp: Handle<Image>,   // Temporary texture for AO blur
}

/// Creates all erosion render-target images for the given map size and adds them to `images`.
/// Called automatically by `ErosionComputePlugin` from `ErosionConfig`; exposed for custom setups.
pub fn create_erosion_images(images: &mut Assets<Image>, size: UVec2) -> ErosionImages {
    let add = |images: &mut Assets<Image>, format: TextureFormat| {
        let mut image = Image::new_target_texture(size.x, size.y, format);
        image.asset_usage = RenderAssetUsages::RENDER_WORLD;
        image.texture_descriptor.usage =
            TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
        images.add(image)
    };
    ErosionImages {
        height: add(images, TextureFormat::Rgba16Float),
        color: add(images, TextureFormat::Rgba16Float),
        normal: add(images, TextureFormat::Rgba16Float),
        analysis: add(images, TextureFormat::Rgba16Float),
        ao: add(images, TextureFormat::R16Float),
        ao_temp: add(images, TextureFormat::R16Float),
    }
}

/// Extended material type for terrain rendering
pub type TerrainMaterial = ExtendedMaterial<StandardMaterial, TerrainExtension>;

/// Preview mode for terrain visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Reflect)]
pub enum PreviewMode {
    #[default]
    Pbr,
    Flow,
    Sediment,
    Erosion,
    Height,
    Normals,
    Curvature,
}

/// Parameters for ambient occlusion computation
#[derive(Debug, Clone, Copy, ShaderType, Reflect)]
pub struct AOParams {
    pub sample_count: u32,
    pub sample_radius: f32,
    pub strength: f32,
    pub bias: f32,
}

impl Default for AOParams {
    fn default() -> Self {
        Self {
            sample_count: 48, // Increase for smoother result
            sample_radius: 0.08,
            strength: 1.5,
            bias: 0.0,
        }
    }
}

/// Material extension for terrain rendering with height maps, color maps, and analysis textures
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
pub struct TerrainExtension {
    // Height map with sampler (used in vertex shader)
    #[texture(100)]
    #[sampler(101)]
    pub height_tex: Handle<Image>,
    // Color map (no sampler, use texelFetch)
    #[texture(102)]
    pub color_tex: Handle<Image>,
    // Analysis texture (no sampler, use texelFetch)
    #[texture(103)]
    pub analysis_tex: Handle<Image>,
    // AO texture with sampler for smooth interpolation
    #[texture(104)]
    #[sampler(105)]
    pub ao_tex: Handle<Image>,
    // Displacement height scale
    #[uniform(106)]
    pub height_scale: f32,
    // Preview mode: 0=PBR, 1=Flow, 2=Sediment, 3=Erosion, 4=Height, 5=Normals, 6=Curvature
    #[uniform(107)]
    pub preview_mode: u32,
}

impl Default for TerrainExtension {
    fn default() -> Self {
        Self {
            height_tex: Default::default(),
            color_tex: Default::default(),
            analysis_tex: Default::default(),
            ao_tex: Default::default(),
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

/// Erosion parameters
#[derive(Resource, Clone, ExtractResource, ShaderType)]
#[repr(C)]
pub struct ErodeParams {
    pub map_size: UVec2,
    pub num_particles: u32,
    pub iteration: u32,
    pub num_iterations: u32,
    pub detail_scale: f32,
    pub compute_ridge_erosion: u32,
    pub erosion_strength: f32,
    pub rock_softness: f32,
    pub sediment_compaction: f32,
    pub compaction_threshold: f32,
    pub channeling: f32,
    pub channeling_character: f32,
    pub sediment_removal: f32,
    pub removal_character: f32,
    pub wear_angle: f32,
    pub talus_angle: f32,
    pub max_deposit_angle: f32,
    pub flow_length: f32,
    pub ridge_erosion_steps: u32,
    pub ridge_softening_amount: f32,
    pub trail_density: f32,
    pub ridge_erosion_amount: f32,
    pub friction: f32,
    pub rock_friction: f32,
    pub flow_volume: f32,
    pub velocity_randomness: f32,
    pub velocity_randomness_refinement: f32,
    pub suspended_load: f32,
    pub river_scarring: f32,
    pub river_scarring_character: f32,
    pub river_friction_reduction: f32,
    pub river_volume: f32,
    pub uplift: f32,
    pub noise_frequency: f32,
    pub noise_scale: f32,
}

impl Default for ErodeParams {
    fn default() -> Self {
        Self {
            map_size: UVec2::new(512, 512),
            num_particles: 256 * 256,
            iteration: 0,
            num_iterations: 2,
            detail_scale: 1.0,
            compute_ridge_erosion: 1,
            erosion_strength: 0.5,
            rock_softness: 0.5,
            sediment_compaction: 0.0,
            compaction_threshold: 0.0,
            channeling: 0.0,
            channeling_character: 1.0,
            sediment_removal: 0.0,
            removal_character: 1.0,
            wear_angle: 0.0,
            talus_angle: 0.0,
            max_deposit_angle: 45.0,
            flow_length: 256.0,
            ridge_erosion_steps: 25,
            ridge_softening_amount: 0.0,
            trail_density: 0.1,
            ridge_erosion_amount: 1.0,
            friction: 1.0,
            rock_friction: 1.0,
            flow_volume: 0.0,
            velocity_randomness: 0.0,
            velocity_randomness_refinement: 0.0,
            suspended_load: 0.0,
            river_scarring: 0.0,
            river_scarring_character: 1.0,
            river_friction_reduction: 1.0,
            river_volume: 0.0,
            uplift: 0.0,
            noise_frequency: 0.008,
            noise_scale: 1.0,
        }
    }
}

/// Resource to trigger simulation reset
#[derive(Resource, Clone, ExtractResource, Default)]
pub struct ResetSim {
    pub generation: u32,
}

/// Resource to control simulation state
#[derive(Resource, Clone, ExtractResource, Default)]
pub struct SimControl {
    pub paused: bool,
    pub step_counter: u64,
}


/// User-facing configuration for the erosion simulation.
#[derive(Resource, Clone)]
pub struct ErosionConfig {
    /// Map width and height (e.g. 512×512).
    pub map_size: UVec2,
    /// Erosion amount / trail density (0–1). Higher = more particles, slower.
    pub trail_density: f32,
    /// Feature scale for terrain detail.
    pub feature_size: f32,
}

impl Default for ErosionConfig {
    fn default() -> Self {
        Self {
            map_size: UVec2::new(512, 512),
            trail_density: 0.1,
            feature_size: 8.0,
        }
    }
}

impl ErosionConfig {
    /// Produces `ErodeParams` for the GPU.
    pub fn to_erode_params(&self, iteration: u32) -> ErodeParams {
        let num_particles = (self.map_size.x * self.map_size.y) as u32;
        ErodeParams {
            map_size: self.map_size,
            num_particles,
            iteration,
            num_iterations: 2,
            detail_scale: self.feature_size / 8.0,
            trail_density: self.trail_density,
            ..Default::default()
        }
    }
}

/// Plugin for erosion compute simulation
pub struct ErosionComputePlugin;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct ErosionLabel;

fn setup_erosion_resources(
    config: Res<ErosionConfig>,
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
) {
    let size = config.map_size;
    commands.insert_resource(create_erosion_images(&mut images, size));
    commands.insert_resource(config.to_erode_params(0));
    commands.insert_resource(ResetSim::default());
    commands.insert_resource(SimControl::default());
}


impl Plugin for ErosionComputePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ErosionConfig>()
            .add_plugins((
                ExtractResourcePlugin::<ErosionImages>::default(),
                ExtractResourcePlugin::<ErodeParams>::default(),
                ExtractResourcePlugin::<ResetSim>::default(),
                ExtractResourcePlugin::<SimControl>::default(),
            ))
            .add_systems(Startup, setup_erosion_resources);

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
    layout_blit_blur: BindGroupLayoutDescriptor,
    pipeline_init: CachedComputePipelineId,
    pipeline_init_color: CachedComputePipelineId,
    pipeline_erode: CachedComputePipelineId,
    pipeline_write_back: CachedComputePipelineId,
    pipeline_blit: CachedComputePipelineId,
    pipeline_ao: CachedComputePipelineId,
    pipeline_blur_h: CachedComputePipelineId,
    pipeline_blur_v: CachedComputePipelineId,
}

#[derive(Resource)]
struct ErosionBindGroups {
    erosion: BindGroup,
    color: BindGroup,
    blit: BindGroup,
    blit_blur: BindGroup,
}

#[derive(Resource)]
struct ErosionBuffers {
    uniform: UniformBuffer<ErodeParams>,
    height: StorageBuffer<Vec<u32>>,
    random: StorageBuffer<Vec<u32>>,
    flow: StorageBuffer<Vec<u32>>,
    deposition: StorageBuffer<Vec<i32>>, // loose sediment per cell (atomic<i32> in shader)
    erosion: StorageBuffer<Vec<u32>>,
}

#[derive(Resource, Clone)]
struct ErosionRngState {
    state: u64,
}

/// Generate random indices using a seeded RNG
pub fn generate_random_indices_seeded(count: u32, max_index: u32, mut state: u64) -> Vec<u32> {
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
                storage_buffer::<u32>(false),           // height (atomic<u32> in shader, fixed-point)
                uniform_buffer::<ErodeParams>(false),
                storage_buffer_read_only::<u32>(false), // random_indices
                storage_buffer::<u32>(false),           // flow (atomic<u32> in shader)
                storage_buffer::<i32>(false),           // deposition (atomic<i32> in shader)
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
                texture_storage_2d(TextureFormat::R16Float, StorageTextureAccess::ReadWrite), // AO (ReadWrite for blur)
            ),
        ),
    );

    // Layout for blit with blur (group 1) - includes temp texture
    let layout_blit_blur = BindGroupLayoutDescriptor::new(
        "ErosionBlitBlurGroup1",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly), // height display
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly), // normal
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly), // analysis (flow+sediment+erosion combined)
                texture_storage_2d(TextureFormat::R16Float, StorageTextureAccess::ReadWrite), // AO
                texture_storage_2d(TextureFormat::R16Float, StorageTextureAccess::ReadWrite), // AO temp
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
    let pipeline_write_back = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![layout_erosion.clone()],
        shader: shader.clone(),
        entry_point: Some(Cow::from("write_back")),
        ..default()
    });
    let pipeline_blit = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![layout_erosion.clone(), layout_blit.clone()],
        shader: shader.clone(),
        entry_point: Some(Cow::from("blit_to_texture")),
        ..default()
    });
    let pipeline_ao = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![layout_erosion.clone(), layout_blit.clone()],
        shader: shader.clone(),
        entry_point: Some(Cow::from("compute_ao")),
        ..default()
    });
    let pipeline_blur_h = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![layout_erosion.clone(), layout_blit_blur.clone()],
        shader: shader.clone(),
        entry_point: Some(Cow::from("blur_ao_horizontal")),
        ..default()
    });
    let pipeline_blur_v = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![layout_erosion.clone(), layout_blit_blur.clone()],
        shader,
        entry_point: Some(Cow::from("blur_ao_vertical")),
        ..default()
    });

    commands.insert_resource(ErosionPipeline {
        layout_erosion,
        layout_color,
        layout_blit,
        layout_blit_blur,
        pipeline_init,
        pipeline_init_color,
        pipeline_erode,
        pipeline_write_back,
        pipeline_blit,
        pipeline_ao,
        pipeline_blur_h,
        pipeline_blur_v,
    });
}

fn prepare_erosion_bind_groups(
    mut commands: Commands,
    pipeline: Res<ErosionPipeline>,
    pipeline_cache: Res<PipelineCache>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    images: Res<ErosionImages>,
    params: Res<ErodeParams>,
    render_device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut existing: Option<ResMut<ErosionBuffers>>,
    rng_state: Option<ResMut<ErosionRngState>>,
) {
    let view_height = gpu_images.get(&images.height).unwrap();
    let view_color = gpu_images.get(&images.color).unwrap();
    let view_normal = gpu_images.get(&images.normal).unwrap();
    let view_analysis = gpu_images.get(&images.analysis).unwrap();
    let view_ao = gpu_images.get(&images.ao).unwrap();
    let view_ao_temp = gpu_images.get(&images.ao_temp).unwrap();

    // Check if we need to verify buffer structure
    let texels = (params.map_size.x * params.map_size.y) as usize;
    let needs_recreation = if let Some(ref buffers) = existing {
        buffers.flow.buffer().is_none()
            || buffers.deposition.buffer().is_none()
            || buffers.erosion.buffer().is_none()
            || buffers.flow.get().len() != texels
            || buffers.deposition.get().len() != texels
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
            // Update uniform with current params each frame
            buffers.uniform = UniformBuffer::from(params.clone());
            buffers.uniform.write_buffer(&render_device, &queue);

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
                    &buffers.flow,
                    &buffers.deposition,
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
                    &view_ao.texture_view,
                )),
            );
            let color = render_device.create_bind_group(
                None,
                &pipeline_cache.get_bind_group_layout(&pipeline.layout_color),
                &BindGroupEntries::sequential((&view_color.texture_view,)),
            );
            let blit_blur = render_device.create_bind_group(
                None,
                &pipeline_cache.get_bind_group_layout(&pipeline.layout_blit_blur),
                &BindGroupEntries::sequential((
                    &view_height.texture_view,
                    &view_normal.texture_view,
                    &view_analysis.texture_view,
                    &view_ao.texture_view,
                    &view_ao_temp.texture_view,
                )),
            );
            commands.insert_resource(ErosionBindGroups {
                erosion,
                color,
                blit,
                blit_blur,
            });
        }
        None => {
            // Create buffers once
            let mut uniform = UniformBuffer::from(params.clone());
            uniform.write_buffer(&render_device, &queue);

            let texels = (params.map_size.x * params.map_size.y) as usize;
            // Height buffer now uses u32 for atomic operations (fixed-point representation in shader)
            let mut height = StorageBuffer::from(vec![0u32; texels]);
            height.write_buffer(&render_device, &queue);

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

            // Initialize buffers for atomic operations
            // All buffers store single scalar per pixel (1 u32 per pixel)
            let mut flow = StorageBuffer::from(vec![0u32; texels]);
            flow.write_buffer(&render_device, &queue);

            let mut deposition = StorageBuffer::from(vec![0i32; texels]);
            deposition.write_buffer(&render_device, &queue);

            let mut erosion = StorageBuffer::from(vec![0u32; texels]);
            erosion.write_buffer(&render_device, &queue);

            // Build bind groups using these buffers
            let erosion_bg = render_device.create_bind_group(
                None,
                &pipeline_cache.get_bind_group_layout(&pipeline.layout_erosion),
                &BindGroupEntries::sequential((
                    &height, &uniform, &random, &flow, &deposition, &erosion,
                )),
            );
            let blit = render_device.create_bind_group(
                None,
                &pipeline_cache.get_bind_group_layout(&pipeline.layout_blit),
                &BindGroupEntries::sequential((
                    &view_height.texture_view,
                    &view_normal.texture_view,
                    &view_analysis.texture_view,
                    &view_ao.texture_view,
                )),
            );
            let color = render_device.create_bind_group(
                None,
                &pipeline_cache.get_bind_group_layout(&pipeline.layout_color),
                &BindGroupEntries::sequential((&view_color.texture_view,)),
            );
            let blit_blur = render_device.create_bind_group(
                None,
                &pipeline_cache.get_bind_group_layout(&pipeline.layout_blit_blur),
                &BindGroupEntries::sequential((
                    &view_height.texture_view,
                    &view_normal.texture_view,
                    &view_analysis.texture_view,
                    &view_ao.texture_view,
                    &view_ao_temp.texture_view,
                )),
            );

            // Insert buffers and groups as resources
            commands.insert_resource(ErosionBuffers {
                uniform,
                height,
                random,
                flow,
                deposition,
                erosion,
            });
            commands.insert_resource(ErosionBindGroups {
                erosion: erosion_bg,
                color,
                blit,
                blit_blur,
            });
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
                CachedPipelineState::Err(bevy::shader::PipelineCacheError::ShaderNotLoaded(_)) => {}
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
                // Pass 3: compute AO
                if let Some(p_ao) = cache.get_compute_pipeline(pipes.pipeline_ao) {
                    let mut pass = render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor::default());
                    pass.set_pipeline(p_ao);
                    pass.set_bind_group(0, &groups.erosion, &[]);
                    pass.set_bind_group(1, &groups.blit, &[]);
                    let gx = (params.map_size.x + 7) / 8;
                    let gy = (params.map_size.y + 7) / 8;
                    pass.dispatch_workgroups(gx, gy, 1);
                }
                // Pass 4: blur AO horizontally
                if let Some(p_blur_h) = cache.get_compute_pipeline(pipes.pipeline_blur_h) {
                    let mut pass = render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor::default());
                    pass.set_pipeline(p_blur_h);
                    pass.set_bind_group(0, &groups.erosion, &[]);
                    pass.set_bind_group(1, &groups.blit_blur, &[]);
                    let gx = (params.map_size.x + 7) / 8;
                    let gy = (params.map_size.y + 7) / 8;
                    pass.dispatch_workgroups(gx, gy, 1);
                }
                // Pass 5: blur AO vertically
                if let Some(p_blur_v) = cache.get_compute_pipeline(pipes.pipeline_blur_v) {
                    let mut pass = render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor::default());
                    pass.set_pipeline(p_blur_v);
                    pass.set_bind_group(0, &groups.erosion, &[]);
                    pass.set_bind_group(1, &groups.blit_blur, &[]);
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
                // Pass 1b: write_back (uplift, sediment compaction)
                if let Some(p_wb) = cache.get_compute_pipeline(pipes.pipeline_write_back) {
                    let mut pass = render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor::default());
                    pass.set_pipeline(p_wb);
                    pass.set_bind_group(0, &groups.erosion, &[]);
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
                // Pass 3: compute AO
                if let Some(p_ao) = cache.get_compute_pipeline(pipes.pipeline_ao) {
                    let mut pass = render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor::default());
                    pass.set_pipeline(p_ao);
                    pass.set_bind_group(0, &groups.erosion, &[]);
                    pass.set_bind_group(1, &groups.blit, &[]);
                    let gx = (params.map_size.x + 7) / 8;
                    let gy = (params.map_size.y + 7) / 8;
                    pass.dispatch_workgroups(gx, gy, 1);
                }
                // Pass 4: blur AO horizontally
                if let Some(p_blur_h) = cache.get_compute_pipeline(pipes.pipeline_blur_h) {
                    let mut pass = render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor::default());
                    pass.set_pipeline(p_blur_h);
                    pass.set_bind_group(0, &groups.erosion, &[]);
                    pass.set_bind_group(1, &groups.blit_blur, &[]);
                    let gx = (params.map_size.x + 7) / 8;
                    let gy = (params.map_size.y + 7) / 8;
                    pass.dispatch_workgroups(gx, gy, 1);
                }
                // Pass 5: blur AO vertically
                if let Some(p_blur_v) = cache.get_compute_pipeline(pipes.pipeline_blur_v) {
                    let mut pass = render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor::default());
                    pass.set_pipeline(p_blur_v);
                    pass.set_bind_group(0, &groups.erosion, &[]);
                    pass.set_bind_group(1, &groups.blit_blur, &[]);
                    let gx = (params.map_size.x + 7) / 8;
                    let gy = (params.map_size.y + 7) / 8;
                    pass.dispatch_workgroups(gx, gy, 1);
                }
            }
        }

        // Keep buffers alive (not strictly necessary, but avoids drop before usage on some backends)
        let _keep = (&buffers.uniform, &buffers.random);

        Ok(())
    }
}

