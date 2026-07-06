use bevy::{
    asset::RenderAssetUsages,
    core_pipeline::schedule::camera_driver,
    pbr::{ExtendedMaterial, MaterialExtension, StandardMaterial},
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_resource::{
            TextureFormat, TextureUsages,
            binding_types::{storage_buffer, storage_buffer_read_only, texture_storage_2d, uniform_buffer},
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue},
        texture::GpuImage,
    },
    shader::{ShaderCacheError, ShaderRef},
};
use std::borrow::Cow;

pub mod camera;
pub mod sun;

#[cfg(feature = "ui")]
pub mod ui;

const EROSION_SHADER: &str = "erosion.wgsl";
const PYRAMID_RESAMPLE_SHADER: &str = "pyramid_resample.wgsl";
/// Erode dispatches per `writeBack` at geological age 0 (all modes).
const ERODE_ITERS_PER_WRITEBACK: u32 = 200;
const TERRAIN_SHADER: &str = "terrain_extended.wgsl";
const PYRAMID_MIN_SIZE: u32 = 8;

/// KTT `Iterations` before deduplicating clamped resolutions (`Compute_Padding`).
fn compute_pyramid_raw_level_count(map_size: UVec2, detail_scale: f32) -> u32 {
    let min_res = map_size.x.min(map_size.y) as f32;
    let scale = detail_scale.max(0.001);
    let raw = (scale * min_res).log2() + 1.0;
    (raw.max(1.0).min(8.0)).ceil() as u32
}

/// Coarsest pyramid resolution (`heightfield_resample1` with `resscale = pow(0.5, ceil(Iterations))`).
pub fn pyramid_coarsest_size(base: UVec2, detail_scale: f32) -> UVec2 {
    let steps = compute_pyramid_raw_level_count(base, detail_scale);
    let div = 1u32 << steps.min(31);
    UVec2::new(
        (base.x / div).max(PYRAMID_MIN_SIZE),
        (base.y / div).max(PYRAMID_MIN_SIZE),
    )
}

/// KTT pyramid level count (`Iterations`), after merging repeat slots that share a resolution.
pub fn compute_pyramid_level_count(map_size: UVec2, detail_scale: f32) -> u32 {
    pyramid_level_sizes(map_size, detail_scale).len() as u32
}

/// Distinct pyramid resolutions, coarsest first.
///
/// Matches KTT's repeat block: initial downscale to [`pyramid_coarsest_size`], then each repeat
/// iteration doubles resolution (`heightfield_resample` with `resscale = 2`) before eroding.
pub fn pyramid_level_sizes(base: UVec2, detail_scale: f32) -> Vec<UVec2> {
    let raw_count = compute_pyramid_raw_level_count(base, detail_scale);
    let coarsest = pyramid_coarsest_size(base, detail_scale);
    let mut sizes = Vec::with_capacity(raw_count as usize);
    for level in 0..raw_count {
        let mult = 1u32 << level.min(31);
        let size = UVec2::new(
            (coarsest.x * mult).min(base.x),
            (coarsest.y * mult).min(base.y),
        );
        if sizes.last().copied() != Some(size) {
            sizes.push(size);
        }
    }
    if sizes.is_empty() {
        sizes.push(base);
    }
    sizes
}

/// Level 0 = coarsest, finest = full base resolution.
pub fn pyramid_level_size(base: UVec2, level: u32, detail_scale: f32) -> UVec2 {
    let sizes = pyramid_level_sizes(base, detail_scale);
    let idx = (level as usize).min(sizes.len().saturating_sub(1));
    sizes[idx]
}

/// Effective `Detail_Scale` for a pyramid level (larger voxels at coarser levels).
pub fn pyramid_effective_detail_scale(base: UVec2, level: UVec2, detail_scale: f32) -> f32 {
    detail_scale * (level.x as f32 / base.x as f32)
}

pub fn is_finest_pyramid_level(params: &ErodeParams) -> bool {
    params.active_pyramid_level >= params.pyramid_level_count.saturating_sub(1)
}

pub fn active_pyramid_level_size(params: &ErodeParams) -> UVec2 {
    pyramid_level_size(
        params.map_size,
        params.active_pyramid_level,
        params.detail_scale,
    )
}

pub fn level_erode_params(base: &ErodeParams, iteration: u32) -> ErodeParams {
    let level_size = active_pyramid_level_size(base);
    ErodeParams {
        map_size: level_size,
        num_particles: (level_size.x * level_size.y) as u32,
        detail_scale: pyramid_effective_detail_scale(base.map_size, level_size, base.detail_scale),
        iteration,
        ..base.clone()
    }
}

#[derive(Clone, Copy, ShaderType)]
pub struct PyramidResampleParams {
    pub src_size: UVec2,
    pub dst_size: UVec2,
    pub _pad: UVec2,
}

#[derive(Clone, Copy, ShaderType)]
pub struct BlitDisplayParams {
    pub display_size: UVec2,
    pub _pad: UVec2,
}

/// Resource containing handles to all erosion-related images
#[derive(Resource, Clone, ExtractResource)]
pub struct ErosionImages {
    pub height: Handle<Image>,
    pub color: Handle<Image>,
    pub normal: Handle<Image>,
    pub analysis: Handle<Image>, // Combined: R=flow_mag, G=sediment, B=erosion, A=flow_dir_encoded
    pub ao: Handle<Image>,       // Ambient occlusion texture
    pub ao_temp: Handle<Image>,  // Temporary texture for AO blur
}

/// Creates all erosion render-target images for the given map size and adds them to `images`.
/// Called automatically by `ErosionComputePlugin` from `ErosionConfig`; exposed for custom setups.
pub fn create_erosion_images(images: &mut Assets<Image>, size: UVec2) -> ErosionImages {
    let add = |images: &mut Assets<Image>, format: TextureFormat| {
        let mut image = Image::new_target_texture(size.x, size.y, format, None);
        image.asset_usage = RenderAssetUsages::RENDER_WORLD;
        image.texture_descriptor.usage = TextureUsages::COPY_DST
            | TextureUsages::STORAGE_BINDING
            | TextureUsages::TEXTURE_BINDING;
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

/// Erosion parameters — fields store reference UI parm values from `ref/node_tree.json`.
#[derive(Resource, Clone, ExtractResource, ShaderType)]
#[repr(C)]
pub struct ErodeParams {
    pub map_size: UVec2,
    pub num_particles: u32,
    pub iteration: u32,
    /// Batch size passed to writeBack (reference parm `NumIterations`).
    pub num_iterations: u32,
    /// Reference parm `Reference_Detail_Scale` → kernel `Detail_Scale`.
    pub detail_scale: f32,
    /// Reference parm `Do_Ridge_Erosion`.
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
    /// Reference parm `Ridge_Erosion_Steps` → kernel `Ridge_Erosion_Samples`.
    pub ridge_erosion_steps: u32,
    pub ridge_softening_amount: f32,
    /// Reference parm `Erosion_Amount` → kernel `Trail_Density` via `× 0.1`.
    pub erosion_amount: f32,
    pub ridge_erosion_amount: f32,
    /// Reference parm `Friction` (sediment) → kernel `Friction`.
    pub friction: f32,
    pub rock_friction: f32,
    pub flow_volume: f32,
    pub velocity_randomness: f32,
    pub velocity_randomness_refinement: f32,
    /// Reference parm `Suspended_Sediment` → kernel `Suspended_Load`.
    pub suspended_load: f32,
    /// Reference parm `River_Scale` → kernel `River_Scarring` via `× 0.25`.
    pub river_scale: f32,
    pub river_character: f32,
    pub river_friction_reduction: f32,
    pub river_volume: f32,
    pub do_rivers: u32,
    pub river_channeling: f32,
    pub momentum_coherence: f32,
    pub meander_longevity: f32,
    pub uplift_river_carving: f32,
    /// Reference parm `Uplift_Amount`.
    pub uplift: f32,
    /// Bevy-only: initial height FBM (not an erosion simulation parm).
    pub noise_frequency: f32,
    pub noise_scale: f32,
    /// UI-selected pyramid level (0 = coarsest).
    pub active_pyramid_level: u32,
    /// Computed from map size and detail scale (KTT `Iterations`).
    pub pyramid_level_count: u32,
}

impl Default for ErodeParams {
    fn default() -> Self {
        // --- Root node UI defaults (`ref/node_tree.json`) ---
        // Shader applies binding scales where Houdini expressions do (must match `erosion.wgsl`):
        //   Trail_Density = Erosion_Amount * 0.1
        //   River_Scarring = River_Scale * 0.25
        Self {
            map_size: UVec2::new(512, 512),
            num_particles: 256 * 256,
            iteration: 0,
            num_iterations: ERODE_ITERS_PER_WRITEBACK,
            detail_scale: 1.0,        // Reference_Detail_Scale
            compute_ridge_erosion: 1, // Do_Ridge_Erosion
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
            ridge_erosion_steps: 25, // Ridge_Erosion_Steps
            ridge_softening_amount: 0.0,
            erosion_amount: 0.1, // Erosion_Amount
            ridge_erosion_amount: 1.0,
            friction: 1.0, // UI "Friction" → kernel Sediment Friction
            rock_friction: 1.0,
            flow_volume: 0.0,
            velocity_randomness: 0.0,
            velocity_randomness_refinement: 0.0,
            suspended_load: 0.0, // Suspended_Sediment
            river_scale: 0.25,   // River_Scale
            river_character: 1.0,
            river_friction_reduction: 1.0,
            river_volume: 0.5,
            do_rivers: 0,
            river_channeling: 0.5,
            momentum_coherence: 0.0,
            meander_longevity: 0.9,
            uplift_river_carving: 1.0,
            uplift: 0.0, // Uplift_Amount
            noise_frequency: 0.005,
            noise_scale: 1.5,
            active_pyramid_level: 0,
            pyramid_level_count: compute_pyramid_level_count(UVec2::new(512, 512), 1.0),
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
    /// Reference parm `Erosion_Amount`.
    pub erosion_amount: f32,
    /// Reference parm `Reference_Detail_Scale` → kernel `Detail_Scale`.
    pub reference_detail_scale: f32,
}

impl Default for ErosionConfig {
    fn default() -> Self {
        Self {
            map_size: UVec2::new(512, 512),
            erosion_amount: 0.1,         // Erosion_Amount
            reference_detail_scale: 1.0, // Reference_Detail_Scale
        }
    }
}

impl ErosionConfig {
    /// Produces `ErodeParams` for the GPU.
    pub fn to_erode_params(&self, iteration: u32) -> ErodeParams {
        let num_particles = (self.map_size.x * self.map_size.y) as u32;
        let pyramid_level_count =
            compute_pyramid_level_count(self.map_size, self.reference_detail_scale);
        ErodeParams {
            map_size: self.map_size,
            num_particles,
            iteration,
            num_iterations: ERODE_ITERS_PER_WRITEBACK,
            detail_scale: self.reference_detail_scale,
            erosion_amount: self.erosion_amount,
            pyramid_level_count,
            ..Default::default()
        }
    }
}

fn sync_pyramid_levels(mut params: ResMut<ErodeParams>) {
    let count = compute_pyramid_level_count(params.map_size, params.detail_scale);
    params.pyramid_level_count = count;
    if params.active_pyramid_level >= count {
        params.active_pyramid_level = count.saturating_sub(1);
    }
}

/// Plugin for erosion compute simulation
pub struct ErosionComputePlugin;

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
            .add_systems(Startup, setup_erosion_resources)
            .add_systems(PostUpdate, sync_pyramid_levels);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<ErosionState>()
            .add_systems(RenderStartup, init_erosion_pipeline)
            .add_systems(
                Render,
                (
                    prepare_erosion_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                    update_erosion_state.in_set(RenderSystems::Prepare),
                ),
            )
            .add_systems(RenderGraph, erosion_compute.before(camera_driver));
    }
}

#[derive(Resource)]
struct ErosionPipeline {
    layout_erosion: BindGroupLayoutDescriptor,
    layout_color: BindGroupLayoutDescriptor,
    layout_blit: BindGroupLayoutDescriptor,
    layout_blit_blur: BindGroupLayoutDescriptor,
    layout_blit_level: BindGroupLayoutDescriptor,
    layout_resample: BindGroupLayoutDescriptor,
    pipeline_init: CachedComputePipelineId,
    pipeline_init_color: CachedComputePipelineId,
    pipeline_erode: CachedComputePipelineId,
    pipeline_write_back: CachedComputePipelineId,
    pipeline_blit: CachedComputePipelineId,
    pipeline_blit_level: CachedComputePipelineId,
    pipeline_clear_level: CachedComputePipelineId,
    pipeline_pyramid_downscale: CachedComputePipelineId,
    pipeline_ao: CachedComputePipelineId,
    pipeline_blur_h: CachedComputePipelineId,
    pipeline_blur_v: CachedComputePipelineId,
}

#[derive(Resource)]
struct ErosionBindGroups {
    erosion: BindGroup,
    level_erosion: Option<BindGroup>,
    downscale: Option<BindGroup>,
    level_blit: Option<BindGroup>,
    color: BindGroup,
    blit: BindGroup,
    blit_blur: BindGroup,
}

#[derive(Resource)]
struct ErosionBuffers {
    uniform: UniformBuffer<ErodeParams>,
    level_uniform: UniformBuffer<ErodeParams>,
    blit_display: UniformBuffer<BlitDisplayParams>,
    downscale_params: UniformBuffer<PyramidResampleParams>,
    height: StorageBuffer<Vec<u32>>,
    macc: StorageBuffer<Vec<u32>>,
    flow: StorageBuffer<Vec<u32>>,
    deposition: StorageBuffer<Vec<i32>>,
    erosion: StorageBuffer<Vec<u32>>,
    mx: StorageBuffer<Vec<i32>>,
    my: StorageBuffer<Vec<i32>>,
    height_work: Option<StorageBuffer<Vec<u32>>>,
    macc_work: Option<StorageBuffer<Vec<u32>>>,
    flow_work: Option<StorageBuffer<Vec<u32>>>,
    deposition_work: Option<StorageBuffer<Vec<i32>>>,
    erosion_work: Option<StorageBuffer<Vec<u32>>>,
    mx_work: Option<StorageBuffer<Vec<i32>>>,
    my_work: Option<StorageBuffer<Vec<i32>>>,
    active_pyramid_level: u32,
    work_texels: usize,
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
                storage_buffer::<u32>(false), // height (atomic<u32> in shader, fixed-point)
                uniform_buffer::<ErodeParams>(false),
                storage_buffer::<u32>(false), // macc (atomic<u32> in shader)
                storage_buffer::<u32>(false), // flow (atomic<u32> in shader)
                storage_buffer::<i32>(false), // deposition (atomic<i32> in shader)
                storage_buffer::<u32>(false), // erosion (atomic<u32> in shader)
                storage_buffer::<i32>(false), // mx (atomic<i32> in shader)
                storage_buffer::<i32>(false), // my (atomic<i32> in shader)
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

    let layout_blit_level = BindGroupLayoutDescriptor::new(
        "ErosionBlitLevelGroup2",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (uniform_buffer::<BlitDisplayParams>(false),),
        ),
    );

    let layout_resample = BindGroupLayoutDescriptor::new(
        "PyramidResampleLayout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only::<u32>(false),
                storage_buffer::<u32>(false),
                uniform_buffer::<PyramidResampleParams>(false),
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
    let pyramid_shader = asset_server.load(PYRAMID_RESAMPLE_SHADER);
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
    let pipeline_blit_level = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![
            layout_erosion.clone(),
            layout_blit.clone(),
            layout_blit_level.clone(),
        ],
        shader: shader.clone(),
        entry_point: Some(Cow::from("blit_level_to_display")),
        ..default()
    });
    let pipeline_clear_level = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![layout_erosion.clone()],
        shader: shader.clone(),
        entry_point: Some(Cow::from("clear_level_buffers")),
        ..default()
    });
    let pipeline_pyramid_downscale =
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![layout_resample.clone()],
            shader: pyramid_shader,
            entry_point: Some(Cow::from("pyramid_downscale")),
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
        layout_blit_level,
        layout_resample,
        pipeline_init,
        pipeline_init_color,
        pipeline_erode,
        pipeline_write_back,
        pipeline_blit,
        pipeline_blit_level,
        pipeline_clear_level,
        pipeline_pyramid_downscale,
        pipeline_ao,
        pipeline_blur_h,
        pipeline_blur_v,
    });
}

fn create_work_buffers(
    work_texels: usize,
    render_device: &RenderDevice,
    queue: &RenderQueue,
) -> (
    StorageBuffer<Vec<u32>>,
    StorageBuffer<Vec<u32>>,
    StorageBuffer<Vec<u32>>,
    StorageBuffer<Vec<i32>>,
    StorageBuffer<Vec<u32>>,
    StorageBuffer<Vec<i32>>,
    StorageBuffer<Vec<i32>>,
) {
    let mut height_work = StorageBuffer::from(vec![0u32; work_texels]);
    height_work.write_buffer(render_device, queue);
    let mut macc_work = StorageBuffer::from(vec![0u32; work_texels]);
    macc_work.write_buffer(render_device, queue);
    let mut flow_work = StorageBuffer::from(vec![0u32; work_texels]);
    flow_work.write_buffer(render_device, queue);
    let mut deposition_work = StorageBuffer::from(vec![0i32; work_texels]);
    deposition_work.write_buffer(render_device, queue);
    let mut erosion_work = StorageBuffer::from(vec![0u32; work_texels]);
    erosion_work.write_buffer(render_device, queue);
    let mut mx_work = StorageBuffer::from(vec![0i32; work_texels]);
    mx_work.write_buffer(render_device, queue);
    let mut my_work = StorageBuffer::from(vec![0i32; work_texels]);
    my_work.write_buffer(render_device, queue);
    (
        height_work,
        macc_work,
        flow_work,
        deposition_work,
        erosion_work,
        mx_work,
        my_work,
    )
}

fn build_pyramid_bind_groups(
    pipeline: &ErosionPipeline,
    pipeline_cache: &PipelineCache,
    render_device: &RenderDevice,
    height: &StorageBuffer<Vec<u32>>,
    height_work: &StorageBuffer<Vec<u32>>,
    level_uniform: &UniformBuffer<ErodeParams>,
    macc_work: &StorageBuffer<Vec<u32>>,
    flow_work: &StorageBuffer<Vec<u32>>,
    deposition_work: &StorageBuffer<Vec<i32>>,
    erosion_work: &StorageBuffer<Vec<u32>>,
    mx_work: &StorageBuffer<Vec<i32>>,
    my_work: &StorageBuffer<Vec<i32>>,
    downscale_params: &UniformBuffer<PyramidResampleParams>,
    blit_display: &UniformBuffer<BlitDisplayParams>,
) -> (BindGroup, BindGroup, BindGroup) {
    let erosion_layout = pipeline_cache.get_bind_group_layout(&pipeline.layout_erosion);
    let resample_layout = pipeline_cache.get_bind_group_layout(&pipeline.layout_resample);
    let blit_level_layout = pipeline_cache.get_bind_group_layout(&pipeline.layout_blit_level);

    let level_erosion = render_device.create_bind_group(
        None,
        &erosion_layout,
        &BindGroupEntries::sequential((
            height_work,
            level_uniform,
            macc_work,
            flow_work,
            deposition_work,
            erosion_work,
            mx_work,
            my_work,
        )),
    );
    let downscale = render_device.create_bind_group(
        None,
        &resample_layout,
        &BindGroupEntries::sequential((height, height_work, downscale_params)),
    );
    let level_blit = render_device.create_bind_group(
        None,
        &blit_level_layout,
        &BindGroupEntries::sequential((blit_display,)),
    );
    (level_erosion, downscale, level_blit)
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
) {
    let view_height = gpu_images.get(&images.height).unwrap();
    let view_color = gpu_images.get(&images.color).unwrap();
    let view_normal = gpu_images.get(&images.normal).unwrap();
    let view_analysis = gpu_images.get(&images.analysis).unwrap();
    let view_ao = gpu_images.get(&images.ao).unwrap();
    let view_ao_temp = gpu_images.get(&images.ao_temp).unwrap();

    let texels = (params.map_size.x * params.map_size.y) as usize;
    let level_size = active_pyramid_level_size(&params);
    let work_texels = (level_size.x * level_size.y) as usize;
    let use_pyramid = !is_finest_pyramid_level(&params);

    let needs_recreation = if let Some(ref buffers) = existing {
        buffers.macc.buffer().is_none()
            || buffers.mx.buffer().is_none()
            || buffers.my.buffer().is_none()
            || buffers.flow.buffer().is_none()
            || buffers.deposition.buffer().is_none()
            || buffers.erosion.buffer().is_none()
            || buffers.macc.get().len() != texels
            || buffers.flow.get().len() != texels
            || buffers.deposition.get().len() != texels
            || buffers.erosion.get().len() != texels
            || buffers.active_pyramid_level != params.active_pyramid_level
            || (use_pyramid && buffers.work_texels != work_texels)
            || (use_pyramid && buffers.height_work.is_none())
    } else {
        false
    };

    if needs_recreation {
        existing = None;
    }

    let blit = |device: &RenderDevice| {
        device.create_bind_group(
            None,
            &pipeline_cache.get_bind_group_layout(&pipeline.layout_blit),
            &BindGroupEntries::sequential((
                &view_height.texture_view,
                &view_normal.texture_view,
                &view_analysis.texture_view,
                &view_ao.texture_view,
            )),
        )
    };
    let color = |device: &RenderDevice| {
        device.create_bind_group(
            None,
            &pipeline_cache.get_bind_group_layout(&pipeline.layout_color),
            &BindGroupEntries::sequential((&view_color.texture_view,)),
        )
    };
    let blit_blur = |device: &RenderDevice| {
        device.create_bind_group(
            None,
            &pipeline_cache.get_bind_group_layout(&pipeline.layout_blit_blur),
            &BindGroupEntries::sequential((
                &view_height.texture_view,
                &view_normal.texture_view,
                &view_analysis.texture_view,
                &view_ao.texture_view,
                &view_ao_temp.texture_view,
            )),
        )
    };

    match existing {
        Some(mut buffers) => {
            *buffers.uniform.get_mut() = params.clone();
            buffers.uniform.write_buffer(&render_device, &queue);
            *buffers.level_uniform.get_mut() = level_erode_params(&params, 0);
            buffers.level_uniform.write_buffer(&render_device, &queue);

            let erosion = render_device.create_bind_group(
                None,
                &pipeline_cache.get_bind_group_layout(&pipeline.layout_erosion),
                &BindGroupEntries::sequential((
                    &buffers.height,
                    &buffers.uniform,
                    &buffers.macc,
                    &buffers.flow,
                    &buffers.deposition,
                    &buffers.erosion,
                    &buffers.mx,
                    &buffers.my,
                )),
            );

            let (level_erosion, downscale, level_blit) =
                if use_pyramid {
                    let height_work = buffers.height_work.as_ref().unwrap();
                    let macc_work = buffers.macc_work.as_ref().unwrap();
                    let flow_work = buffers.flow_work.as_ref().unwrap();
                    let deposition_work = buffers.deposition_work.as_ref().unwrap();
                    let erosion_work = buffers.erosion_work.as_ref().unwrap();
                    let mx_work = buffers.mx_work.as_ref().unwrap();
                    let my_work = buffers.my_work.as_ref().unwrap();
                    let groups = build_pyramid_bind_groups(
                        &pipeline,
                        &pipeline_cache,
                        &render_device,
                        &buffers.height,
                        height_work,
                        &buffers.level_uniform,
                        macc_work,
                        flow_work,
                        deposition_work,
                        erosion_work,
                        mx_work,
                        my_work,
                        &buffers.downscale_params,
                        &buffers.blit_display,
                    );
                    (
                        Some(groups.0),
                        Some(groups.1),
                        Some(groups.2),
                    )
                } else {
                    (None, None, None)
                };

            commands.insert_resource(ErosionBindGroups {
                erosion,
                level_erosion,
                downscale,
                level_blit,
                color: color(&render_device),
                blit: blit(&render_device),
                blit_blur: blit_blur(&render_device),
            });
        }
        None => {
            let mut uniform = UniformBuffer::from(params.clone());
            uniform.write_buffer(&render_device, &queue);
            let mut level_uniform = UniformBuffer::from(level_erode_params(&params, 0));
            level_uniform.write_buffer(&render_device, &queue);

            let level_size = active_pyramid_level_size(&params);
            let mut downscale_params = UniformBuffer::from(PyramidResampleParams {
                src_size: params.map_size,
                dst_size: level_size,
                _pad: UVec2::ZERO,
            });
            downscale_params.write_buffer(&render_device, &queue);

            let mut blit_display = UniformBuffer::from(BlitDisplayParams {
                display_size: params.map_size,
                _pad: UVec2::ZERO,
            });
            blit_display.write_buffer(&render_device, &queue);

            let mut height = StorageBuffer::from(vec![0u32; texels]);
            height.write_buffer(&render_device, &queue);
            let mut macc = StorageBuffer::from(vec![0u32; texels]);
            macc.write_buffer(&render_device, &queue);
            let mut flow = StorageBuffer::from(vec![0u32; texels]);
            flow.write_buffer(&render_device, &queue);
            let mut deposition = StorageBuffer::from(vec![0i32; texels]);
            deposition.write_buffer(&render_device, &queue);
            let mut erosion_buf = StorageBuffer::from(vec![0u32; texels]);
            erosion_buf.write_buffer(&render_device, &queue);
            let mut mx = StorageBuffer::from(vec![0i32; texels]);
            mx.write_buffer(&render_device, &queue);
            let mut my = StorageBuffer::from(vec![0i32; texels]);
            my.write_buffer(&render_device, &queue);

            let work_buffers = if use_pyramid {
                Some(create_work_buffers(work_texels, &render_device, &queue))
            } else {
                None
            };

            let erosion_bg = render_device.create_bind_group(
                None,
                &pipeline_cache.get_bind_group_layout(&pipeline.layout_erosion),
                &BindGroupEntries::sequential((
                    &height,
                    &uniform,
                    &macc,
                    &flow,
                    &deposition,
                    &erosion_buf,
                    &mx,
                    &my,
                )),
            );

            let (level_erosion, downscale, level_blit) = if let Some((
                ref height_work,
                ref macc_work,
                ref flow_work,
                ref deposition_work,
                ref erosion_work,
                ref mx_work,
                ref my_work,
            )) = work_buffers
            {
                let groups = build_pyramid_bind_groups(
                    &pipeline,
                    &pipeline_cache,
                    &render_device,
                    &height,
                    height_work,
                    &level_uniform,
                    macc_work,
                    flow_work,
                    deposition_work,
                    erosion_work,
                    mx_work,
                    my_work,
                    &downscale_params,
                    &blit_display,
                );
                (Some(groups.0), Some(groups.1), Some(groups.2))
            } else {
                (None, None, None)
            };

            let (height_work, macc_work, flow_work, deposition_work, erosion_work, mx_work, my_work) =
                if let Some(work) = work_buffers {
                    work
                } else {
                    (
                        StorageBuffer::from(vec![0u32; 0]),
                        StorageBuffer::from(vec![0u32; 0]),
                        StorageBuffer::from(vec![0u32; 0]),
                        StorageBuffer::from(vec![0i32; 0]),
                        StorageBuffer::from(vec![0u32; 0]),
                        StorageBuffer::from(vec![0i32; 0]),
                        StorageBuffer::from(vec![0i32; 0]),
                    )
                };

            commands.insert_resource(ErosionBuffers {
                uniform,
                level_uniform,
                blit_display,
                downscale_params,
                height,
                macc,
                flow,
                deposition,
                erosion: erosion_buf,
                mx,
                my,
                height_work: use_pyramid.then_some(height_work),
                macc_work: use_pyramid.then_some(macc_work),
                flow_work: use_pyramid.then_some(flow_work),
                deposition_work: use_pyramid.then_some(deposition_work),
                erosion_work: use_pyramid.then_some(erosion_work),
                mx_work: use_pyramid.then_some(mx_work),
                my_work: use_pyramid.then_some(my_work),
                active_pyramid_level: params.active_pyramid_level,
                work_texels,
            });
            commands.insert_resource(ErosionBindGroups {
                erosion: erosion_bg,
                level_erosion,
                downscale,
                level_blit,
                color: color(&render_device),
                blit: blit(&render_device),
                blit_blur: blit_blur(&render_device),
            });
        }
    }
}

#[derive(Resource, Default)]
struct ErosionState {
    phase: ErosionPhase,
    last_reset_gen: u32,
    allow_erode_this_frame: bool,
    last_step_seen: u64,
    opencl_iteration: u32,
    erode_since_writeback: u32,
}

fn writeback_interval(_params: &ErodeParams) -> u32 {
    // ~200 erosion dispatches per writeBack at geological age 0 regardless of rivers.
    ERODE_ITERS_PER_WRITEBACK
}

/// `Detail_Bias = iteration % 2`: 0 = ridge-biased, 1 = flow-biased.
/// See `ref/KTT_SMOOTH_FLUVIAL_EROSION.md`.
fn detail_bias(iteration: u32) -> u32 {
    iteration % 2
}

/// Whether `Erosion` does work for this iteration (`ref/compute_erosion.cl`).
///
/// ```c
/// if (!Compute_Ridge_Erosion && !Detail_Bias) return;
/// ```
///
/// When ridge erosion is off, even iterations are no-ops; the host may skip
/// dispatch but must still advance `opencl_iteration` so batch timing matches the reference impl.
fn erosion_pass_active(params: &ErodeParams, iteration: u32) -> bool {
    params.compute_ridge_erosion != 0 || detail_bias(iteration) != 0
}

#[derive(Default)]
enum ErosionPhase {
    #[default]
    Loading,
    Init,
    Erode,
}

fn update_erosion_state(
    pipeline: Res<ErosionPipeline>,
    cache: Res<PipelineCache>,
    reset: Res<ResetSim>,
    sim: Res<SimControl>,
    mut state: ResMut<ErosionState>,
) {
    if reset.generation != state.last_reset_gen {
        state.last_reset_gen = reset.generation;
        state.opencl_iteration = 0;
        state.erode_since_writeback = 0;
        state.phase = ErosionPhase::Loading;
    }
    if sim.paused {
        if sim.step_counter > state.last_step_seen {
            state.allow_erode_this_frame = true;
            state.last_step_seen = sim.step_counter;
        } else {
            state.allow_erode_this_frame = false;
        }
    } else {
        state.allow_erode_this_frame = true;
    }
    match state.phase {
        ErosionPhase::Loading => match cache.get_compute_pipeline_state(pipeline.pipeline_init) {
            CachedPipelineState::Ok(_) => {
                state.phase = ErosionPhase::Init;
            }
            CachedPipelineState::Err(ShaderCacheError::ShaderNotLoaded(_)) => {}
            CachedPipelineState::Err(err) => {
                panic!("Initializing assets/{EROSION_SHADER}:\n{err}")
            }
            _ => {}
        },
        ErosionPhase::Init => {
            let color_ready = matches!(
                cache.get_compute_pipeline_state(pipeline.pipeline_init_color),
                CachedPipelineState::Ok(_)
            );
            let erode_ready = matches!(
                cache.get_compute_pipeline_state(pipeline.pipeline_erode),
                CachedPipelineState::Ok(_)
            );
            if color_ready && erode_ready {
                state.phase = ErosionPhase::Erode;
            }
        }
        ErosionPhase::Erode => {}
    }
}

fn dispatch_grid(size: UVec2) -> (u32, u32) {
    ((size.x + 7) / 8, (size.y + 7) / 8)
}

fn seed_pyramid_level(
    render_context: &mut RenderContext,
    cache: &PipelineCache,
    pipes: &ErosionPipeline,
    groups: &ErosionBindGroups,
    params: &ErodeParams,
) {
    if is_finest_pyramid_level(params) {
        return;
    }
    let level_size = active_pyramid_level_size(params);
    let (gx, gy) = dispatch_grid(level_size);
    if let (Some(p_down), Some(downscale)) = (
        cache.get_compute_pipeline(pipes.pipeline_pyramid_downscale),
        groups.downscale.as_ref(),
    ) {
        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(p_down);
        pass.set_bind_group(0, downscale, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
    }
    if let (Some(p_clear), Some(level_erosion)) = (
        cache.get_compute_pipeline(pipes.pipeline_clear_level),
        groups.level_erosion.as_ref(),
    ) {
        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(p_clear);
        pass.set_bind_group(0, level_erosion, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
    }
}

fn run_display_and_ao(
    render_context: &mut RenderContext,
    cache: &PipelineCache,
    pipes: &ErosionPipeline,
    groups: &ErosionBindGroups,
    params: &ErodeParams,
) {
    let (gx, gy) = dispatch_grid(params.map_size);
    if is_finest_pyramid_level(params) {
        {
            let mut pass = render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor::default());
            let p_blit = cache.get_compute_pipeline(pipes.pipeline_blit).unwrap();
            pass.set_pipeline(p_blit);
            pass.set_bind_group(0, &groups.erosion, &[]);
            pass.set_bind_group(1, &groups.blit, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        if let Some(p_ao) = cache.get_compute_pipeline(pipes.pipeline_ao) {
            let mut pass = render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor::default());
            pass.set_pipeline(p_ao);
            pass.set_bind_group(0, &groups.erosion, &[]);
            pass.set_bind_group(1, &groups.blit, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        if let Some(p_blur_h) = cache.get_compute_pipeline(pipes.pipeline_blur_h) {
            let mut pass = render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor::default());
            pass.set_pipeline(p_blur_h);
            pass.set_bind_group(0, &groups.erosion, &[]);
            pass.set_bind_group(1, &groups.blit_blur, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        if let Some(p_blur_v) = cache.get_compute_pipeline(pipes.pipeline_blur_v) {
            let mut pass = render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor::default());
            pass.set_pipeline(p_blur_v);
            pass.set_bind_group(0, &groups.erosion, &[]);
            pass.set_bind_group(1, &groups.blit_blur, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
    } else if let (Some(p_blit_level), Some(level_erosion), Some(level_blit)) = (
        cache.get_compute_pipeline(pipes.pipeline_blit_level),
        groups.level_erosion.as_ref(),
        groups.level_blit.as_ref(),
    ) {
        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(p_blit_level);
        pass.set_bind_group(0, level_erosion, &[]);
        pass.set_bind_group(1, &groups.blit, &[]);
        pass.set_bind_group(2, level_blit, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
    }
}

fn erosion_compute(
    mut render_context: RenderContext,
    cache: Res<PipelineCache>,
    pipes: Res<ErosionPipeline>,
    groups: Res<ErosionBindGroups>,
    params: Res<ErodeParams>,
    mut buffers: ResMut<ErosionBuffers>,
    mut state: ResMut<ErosionState>,
    render_device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    match state.phase {
        ErosionPhase::Loading => {}
        ErosionPhase::Init => {
            let (gx, gy) = dispatch_grid(params.map_size);
            {
                let mut pass = render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor::default());
                let p_init = cache.get_compute_pipeline(pipes.pipeline_init).unwrap();
                pass.set_pipeline(p_init);
                pass.set_bind_group(0, &groups.erosion, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
            if let Some(p_color_ok) = match cache
                .get_compute_pipeline_state(pipes.pipeline_init_color)
            {
                CachedPipelineState::Ok(_) => cache.get_compute_pipeline(pipes.pipeline_init_color),
                _ => None,
            } {
                let mut pass = render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor::default());
                pass.set_pipeline(p_color_ok);
                pass.set_bind_group(0, &groups.erosion, &[]);
                pass.set_bind_group(1, &groups.color, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
            seed_pyramid_level(&mut render_context, &cache, &pipes, &groups, &params);
            run_display_and_ao(&mut render_context, &cache, &pipes, &groups, &params);
        }
        ErosionPhase::Erode => {
            let finest = is_finest_pyramid_level(&params);
            let level_size = active_pyramid_level_size(&params);
            let level_particles = (level_size.x * level_size.y) as u32;
            let erosion_bg = if finest {
                &groups.erosion
            } else {
                groups.level_erosion.as_ref().unwrap()
            };

            if state.allow_erode_this_frame {
                let iter = state.opencl_iteration;
                if erosion_pass_active(&params, iter) {
                    if let Some(p_erode) = cache.get_compute_pipeline(pipes.pipeline_erode) {
                        let workgroups = (level_particles + 255) / 256;
                        if finest {
                            *buffers.uniform.get_mut() = ErodeParams {
                                iteration: iter,
                                ..params.clone()
                            };
                            buffers.uniform.write_buffer(&render_device, &queue);
                        } else {
                            *buffers.level_uniform.get_mut() = level_erode_params(&params, iter);
                            buffers.level_uniform.write_buffer(&render_device, &queue);
                        }

                        let mut pass = render_context
                            .command_encoder()
                            .begin_compute_pass(&ComputePassDescriptor::default());
                        pass.set_pipeline(p_erode);
                        pass.set_bind_group(0, erosion_bg, &[]);
                        pass.set_bind_group(1, &groups.color, &[]);
                        pass.dispatch_workgroups(workgroups, 1, 1);
                    } else {
                        error!("Erode pipeline not ready!");
                    }
                }
                state.opencl_iteration += 1;
                state.erode_since_writeback += 1;
            }

            let do_writeback = state.erode_since_writeback >= writeback_interval(&params);
            if do_writeback {
                state.erode_since_writeback = 0;
                if let Some(p_wb) = cache.get_compute_pipeline(pipes.pipeline_write_back) {
                    if finest {
                        *buffers.uniform.get_mut() = ErodeParams {
                            iteration: state.opencl_iteration,
                            ..params.clone()
                        };
                        buffers.uniform.write_buffer(&render_device, &queue);
                    } else {
                        *buffers.level_uniform.get_mut() =
                            level_erode_params(&params, state.opencl_iteration);
                        buffers.level_uniform.write_buffer(&render_device, &queue);
                    }

                    let (wgx, wgy) = dispatch_grid(level_size);
                    let mut pass = render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor::default());
                    pass.set_pipeline(p_wb);
                    pass.set_bind_group(0, erosion_bg, &[]);
                    pass.dispatch_workgroups(wgx, wgy, 1);
                }
            }

            if state.allow_erode_this_frame {
                run_display_and_ao(&mut render_context, &cache, &pipes, &groups, &params);
            }
        }
    }

    let _keep = (
        &buffers.uniform,
        &buffers.level_uniform,
        &buffers.macc,
        &buffers.mx,
        &buffers.my,
    );
}
