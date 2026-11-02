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
            binding_types::{
                storage_buffer, storage_buffer_read_only, texture_2d, texture_storage_2d,
                uniform_buffer,
            },
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue},
        storage::{GpuShaderBuffer, ShaderBuffer},
        texture::GpuImage,
    },
    shader::{ShaderCacheError, ShaderRef},
};
use bevy_fft::shallow_water::{
    ShallowWaterInputs, ShallowWaterPipelines, ShallowWaterSettings, ShallowWaterSolver,
    ShallowWaterSolverDescriptor, ShallowWaterSolverPlugin, ShallowWaterTextures,
};
use std::{
    borrow::Cow,
    sync::{Arc, Mutex},
};

pub mod brush;
pub mod camera;
pub mod sun;

#[cfg(feature = "ui")]
pub mod ui;

const EROSION_SHADER: &str = "erosion.wgsl";
const PYRAMID_RESAMPLE_SHADER: &str = "pyramid_resample.wgsl";
const PYRAMID_RIVERS_SHADER: &str = "pyramid_rivers.wgsl";
const PYRAMID_MASKS_SHADER: &str = "pyramid_masks.wgsl";
const WATER_SHADER: &str = "water.wgsl";
const WATER_BED_SHADER: &str = "water_bed.wgsl";
const WATER_MORPHO_SHADER: &str = "water_morpho.wgsl";
const WATER_SPLAT_SHADER: &str = "water_splat.wgsl";
/// Erode dispatches per `writeBack` at geological age 0 (`Compute_Erosion.iterations`).
const ERODE_ITERS_PER_WRITEBACK: u32 = 200;
/// Erode dispatches between our `writeBack`s. KTT runs one per level, after every batch of
/// [`ERODE_ITERS_PER_WRITEBACK`]; we split it so uplift and river channeling show and feed back
/// while a level runs, each `writeBack` scaled by its share of the level.
const WRITEBACK_INTERVAL: u32 = 20;
/// Least cost of one erode pass, in finest-level passes: small dispatches still pay their fixed
/// overhead, so a coarse pass is never counted as free.
const MIN_PASS_COST: f32 = 1.0 / 16.0;

impl PyramidLevel {
    /// Work of one erode pass on this level, in passes of the finest level. A level at voxel size
    /// v has 1/v² as many trails, each 1/v as long, but a small dispatch keeps less of the GPU
    /// busy: measured, a pass at v = 2 takes about 0.4 of a finest pass, near 1/v^1.5.
    fn pass_cost(&self) -> f32 {
        self.voxel_size.max(1.0).powf(-1.5).max(MIN_PASS_COST)
    }
}
const TERRAIN_SHADER: &str = "terrain_extended.wesl";
/// Smallest grid the cascade will seed from (caps the level count on small maps).
const PYRAMID_MIN_SIZE: u32 = 4;

/// One eroded resolution of the KTT coarse-to-fine cascade (see `ref/KTT_PYRAMID.md`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PyramidLevel {
    pub size: UVec2,
    /// Voxel size in base-resolution voxels (1 at the finest level).
    pub voxel_size: f32,
    /// Erosion dispatches before this level's `writeBack`.
    pub iterations: u32,
    /// `Trail_Density` multiplier (fades in a fractional coarsest level).
    pub trail_weight: f32,
}

/// KTT `Iterations` (`Compute_Padding`): `log2(Feature_Size * Detail_Scale / voxel) + 1`, voxel = 1.
pub fn pyramid_level_fraction(feature_size: f32, detail_scale: f32) -> f32 {
    let detail_size = (feature_size * detail_scale).max(1e-3);
    (detail_size.log2() + 1.0).max(1.0)
}

/// Eroded pyramid levels, coarsest first; the last one is always `base`.
///
/// `repeat_begin1` starts from `base / 2^ceil(Iterations)` and doubles the resolution before each
/// of its `ceil(Iterations)` erosion passes. Like `Compute_Padding`, the base is first padded to a
/// multiple of `2^steps`, so every level is an exact halving and the finest level is the padded
/// base (see [`pyramid_padding`]).
pub fn pyramid_levels(
    base: UVec2,
    feature_size: f32,
    detail_scale: f32,
    geological_age: f32,
) -> Vec<PyramidLevel> {
    let levels = pyramid_level_fraction(feature_size, detail_scale);
    let max_steps = (base.min_element().max(PYRAMID_MIN_SIZE) / PYRAMID_MIN_SIZE).ilog2() + 1;
    let wanted_steps = levels.ceil() as u32;
    let steps = wanted_steps.min(max_steps).max(1);
    // Compute_Erosion "Trail Density": frac(Iterations)^2 + (integral ? 1 : 0) on the first pass.
    let frac = levels - levels.floor();
    let first_weight = if steps < wanted_steps || frac == 0.0 {
        1.0
    } else {
        frac * frac
    };
    let scale = detail_scale.max(1e-3);
    let grid = 1u32 << steps;
    let padded = UVec2::new(base.x.next_multiple_of(grid), base.y.next_multiple_of(grid));
    (0..steps)
        .rev()
        .enumerate()
        .map(|(pass, k)| {
            let size = padded / (1u32 << k);
            let voxel_size = padded.x as f32 / size.x as f32;
            let voxel_scale = (voxel_size / scale).max(1.0);
            let iterations = (ERODE_ITERS_PER_WRITEBACK as f32
                * voxel_scale.powf(geological_age / 10.0))
            .round()
            .max(1.0) as u32;
            PyramidLevel {
                size,
                voxel_size,
                iterations,
                trail_weight: if pass == 0 { first_weight } else { 1.0 },
            }
        })
        .collect()
}

/// `Compute_Padding` offset of the base map inside the finest (padded) level: half the padding,
/// rounded down.
pub fn pyramid_padding(base: UVec2, levels: &[PyramidLevel]) -> UVec2 {
    levels.last().map_or(UVec2::ZERO, |l| (l.size - base) / 2)
}

/// Largest padded base any level count can produce, so the work sets never need reallocating.
fn pyramid_capacity(base: UVec2) -> UVec2 {
    let max_steps = (base.min_element().max(PYRAMID_MIN_SIZE) / PYRAMID_MIN_SIZE).ilog2() + 1;
    let grid = 1u32 << max_steps;
    UVec2::new(base.x.next_multiple_of(grid), base.y.next_multiple_of(grid))
}

/// `ErodeParams` for one dispatch at `level`.
pub fn level_erode_params(base: &ErodeParams, level: &PyramidLevel, iteration: u32) -> ErodeParams {
    ErodeParams {
        map_size: level.size,
        num_particles: level.size.x * level.size.y,
        iteration,
        num_iterations: level.iterations,
        voxel_size: level.voxel_size,
        trail_weight: level.trail_weight,
        ..base.clone()
    }
}

/// Uniform for the level-transition passes (`pyramid_resample.wgsl`, `pyramid_rivers.wgsl`).
#[derive(Clone, Copy, ShaderType)]
pub struct PyramidResampleParams {
    pub src_size: UVec2,
    pub dst_size: UVec2,
    pub orig_size: UVec2,
    /// 1 when seeding the first level (no previous level to upsample).
    pub first_level: u32,
    pub dst_voxel_size: f32,
    /// KTT `Source_Terrain_Detail_Smoothing`.
    pub detail_smoothing: f32,
    /// KTT `Outflow_Integration_Method` (0 = Sharp, 1 = Accumulate).
    pub outflow_accumulate: u32,
    /// KTT `Use_Bedrock`: upscaled heights are clamped to the bedrock.
    pub use_bedrock: u32,
    /// KTT `Detail_Scale`, `Granularity_Amount`, `Granularity_Scale`, `Granularity_Strength`.
    pub detail_scale: f32,
    pub granularity_amount: f32,
    pub granularity_scale: f32,
    pub granularity_strength: f32,
    /// Finest-level (padded base) size; level texel `p` covers base texel `p * padded / size - pad_offset`.
    pub padded_size: UVec2,
    pub pad_offset: UVec2,
}

/// One unit of GPU work produced by [`Cascade::advance`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CascadeStep {
    /// Generate the source terrain.
    Init,
    /// Build work set `dst` for `level` from the source terrain and work set `src`.
    Seed {
        src: usize,
        dst: usize,
        level: usize,
    },
    Erode {
        set: usize,
        level: usize,
        iteration: u32,
    },
    WriteBack {
        set: usize,
        level: usize,
        iteration: u32,
        /// Erode dispatches since the previous `writeBack`.
        iterations: u32,
    },
}

/// Host-side state of the coarse-to-fine erosion cascade.
#[derive(Clone, Debug, Default)]
pub struct Cascade {
    pub levels: Vec<PyramidLevel>,
    pub level: usize,
    /// Erode dispatches issued in the current `writeBack` batch.
    pub batch_iter: u32,
    /// Iterations of earlier batches at this level (only when continuing past the cascade).
    pub iteration_base: u32,
    /// Work set holding the current level.
    pub current: usize,
    pub finished: bool,
    pub needs_init: bool,
}

impl Cascade {
    pub fn restart(&mut self, levels: Vec<PyramidLevel>) {
        *self = Self {
            levels,
            needs_init: true,
            ..default()
        };
    }

    pub fn current_level(&self) -> Option<&PyramidLevel> {
        self.levels.get(self.level)
    }

    /// Emits the next erosion iterations (plus the init / seed / writeBack work around them).
    ///
    /// `budget` is the work of that many finest-level passes, and coarse passes count their share
    /// of it ([`PyramidLevel::pass_cost`]), so a restart spreads the coarse levels over a few frames
    /// instead of stalling one. A `budget` of 0 (paused) emits nothing past init.
    pub fn advance(
        &mut self,
        budget: u32,
        ridge_enabled: bool,
        continue_after_cascade: bool,
        out: &mut Vec<CascadeStep>,
    ) {
        if self.levels.is_empty() {
            return;
        }
        if self.needs_init {
            self.needs_init = false;
            out.push(CascadeStep::Init);
            out.push(CascadeStep::Seed {
                src: 1,
                dst: 0,
                level: 0,
            });
        }
        let mut work = budget as f32;
        while work > 0.0 && !self.finished {
            let level = self.levels[self.level];
            let iteration = self.iteration_base + self.batch_iter;
            if ridge_enabled || iteration % 2 == 1 {
                out.push(CascadeStep::Erode {
                    set: self.current,
                    level: self.level,
                    iteration,
                });
            }
            self.batch_iter += 1;
            work -= level.pass_cost();
            if self.batch_iter < level.iterations {
                if self.batch_iter % WRITEBACK_INTERVAL == 0 {
                    out.push(CascadeStep::WriteBack {
                        set: self.current,
                        level: self.level,
                        iteration,
                        iterations: WRITEBACK_INTERVAL,
                    });
                }
                continue;
            }
            out.push(CascadeStep::WriteBack {
                set: self.current,
                level: self.level,
                iteration,
                iterations: (level.iterations - 1) % WRITEBACK_INTERVAL + 1,
            });
            self.batch_iter = 0;
            if self.level + 1 < self.levels.len() {
                self.level += 1;
                self.iteration_base = 0;
                let dst = 1 - self.current;
                out.push(CascadeStep::Seed {
                    src: self.current,
                    dst,
                    level: self.level,
                });
                self.current = dst;
            } else if continue_after_cascade {
                self.iteration_base += level.iterations;
            } else {
                self.finished = true;
            }
        }
    }
}

/// Cascade progress shared with the main world (for UI / logging).
#[derive(Clone, Copy, Debug, Default)]
pub struct PyramidProgressData {
    pub level: usize,
    pub level_count: usize,
    pub size: UVec2,
    pub batch_iter: u32,
    pub batch_len: u32,
    pub finished: bool,
}

#[derive(Resource, Clone, Default)]
pub struct PyramidProgress(pub Arc<Mutex<PyramidProgressData>>);

impl PyramidProgress {
    pub fn get(&self) -> PyramidProgressData {
        *self.0.lock().unwrap()
    }
}

/// Resource containing handles to all erosion-related images
#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
pub struct ErosionImages {
    pub height: Handle<Image>,
    pub color: Handle<Image>,
    pub normal: Handle<Image>,
    pub analysis: Handle<Image>, // Combined: R=flow_mag, G=sediment, B=erosion, A=flow_dir_encoded
    pub ao: Handle<Image>,       // Ambient occlusion texture
    pub ao_temp: Handle<Image>,  // Temporary texture for AO blur
    /// Water solver output (Rgba16Float, map size): R = depth, G = speed, B = carried sediment.
    pub water: Handle<Image>,
    /// Terrain brush edits (Rgba32Float, map size), applied to the source terrain at init:
    /// R = height offset, G = flatten weight, B = flatten target height, A = smooth weight.
    /// Edit it on the CPU and bump [`ResetSim`] to re-erode (see [`brush`]).
    pub terrain_edit: Handle<Image>,
}

/// Creates all erosion render-target images for the given map size and adds them to `images`.
/// Called automatically by `ErosionComputePlugin` from `ErosionConfig`; exposed for custom setups.
pub fn create_erosion_images(images: &mut Assets<Image>, size: UVec2) -> ErosionImages {
    let add = |images: &mut Assets<Image>, format: TextureFormat| {
        let mut image = Image::new_target_texture(size.x, size.y, format, None);
        image.asset_usage = RenderAssetUsages::RENDER_WORLD;
        image.texture_descriptor.usage = TextureUsages::COPY_DST
            | TextureUsages::COPY_SRC
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
        water: add(images, TextureFormat::Rgba16Float),
        terrain_edit: images.add(empty_terrain_edit_image(size)),
    }
}

/// Water fields other systems can read (or read back to the CPU with
/// `bevy::render::gpu_readback::Readback::buffer`).
#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
pub struct WaterFields {
    /// Shared discharge field, one `vec4<f32>` per map texel: xy = unit discharge h·u averaged
    /// over a few seconds (height units × texels per second), z = averaged depth, w = KTT's river
    /// momentum scaled to the same units as xy (0 unless Do Rivers is on).
    pub discharge: Handle<ShaderBuffer>,
}

impl WaterFields {
    /// Zeroed fields for a map of `size` texels.
    pub fn new(buffers: &mut Assets<ShaderBuffer>, size: UVec2) -> Self {
        let n = (size.x * size.y) as usize;
        let mut discharge = ShaderBuffer::new(vec![Vec4::ZERO; n], RenderAssetUsages::RENDER_WORLD);
        discharge.label = Cow::Borrowed("water_discharge");
        Self {
            discharge: buffers.add(discharge),
        }
    }
}

/// A cleared [`ErosionImages::terrain_edit`] layer.
pub fn empty_terrain_edit_image(size: UVec2) -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0u8; 16],
        TextureFormat::Rgba32Float,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    image
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
    Age,
    /// Lambert-shaded terrain colour with flat water tint: no PBR or water optics, for timing the
    /// simulations.
    Simple,
    /// KTT river momentum, with Do Rivers on.
    Rivers,
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
    // Preview mode: 0=PBR, 1=Flow, 2=Sediment, 3=Erosion, 4=Height, 5=Normals, 6=Curvature, 7=Age
    #[uniform(107)]
    pub preview_mode: u32,
    /// Water solver output ([`ErosionImages::water`]): the surface rises to terrain + depth.
    #[texture(108)]
    pub water_tex: Handle<Image>,
    /// Ocean level (0 = no ocean): terrain below it renders as a flat sea.
    #[uniform(109)]
    pub sea_level: f32,
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
            water_tex: Default::default(),
            sea_level: 0.0,
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

/// Erosion parameters — fields store reference UI parm values from `ref/houdini22/node_tree.json`.
#[derive(Resource, Clone, ExtractResource, ShaderType)]
#[extract_app(RenderApp)]
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
    /// Reference parm `Feature_Size` (in base voxels); sets the pyramid level count.
    pub feature_size: f32,
    /// Reference parm `Source_Terrain_Detail_Smoothing` (`Smooth_Flows`).
    pub source_detail_smoothing: f32,
    /// Reference parm `Geological_Age`; scales per-level batch size.
    pub geological_age: f32,
    /// Per-dispatch: level voxel size in base voxels (kernel `dx = voxel_size / Detail_Scale`).
    pub voxel_size: f32,
    /// Per-dispatch: `Trail_Density` multiplier for the current level.
    pub trail_weight: f32,
    /// Reference parm `Erosion_Cutoff_Angle` (degrees); trails stop on steeper drops.
    pub erosion_cutoff_angle: f32,
    /// Reference parm `Clamp_River_Channeling`.
    pub clamp_river_channeling: u32,
    /// Reference parm `Use_Hydro_Conditioning`.
    pub use_hydro_conditioning: u32,
    /// Reference parm `Hydro_Conditioning_Search_Distance` (voxels).
    pub hydro_conditioning_search_distance: f32,
    /// Reference parm `Use_Softness_Mask`; scales `Rock_Softness` per cell.
    pub use_softness_mask: u32,
    /// Bevy-only: frequency of the procedural mask noise (softness, erosion, uplift and bedrock maps).
    pub mask_frequency: f32,
    /// Reference parm `Use_Outflow_Map`; outflow cells neither erode nor uplift.
    pub use_outflow: u32,
    /// Reference parm `Outflow_Integration_Method`: 0 = Sharp, 1 = Accumulate across levels.
    pub outflow_integration: u32,
    /// Bevy-only: source terrain below this height becomes outflow (sea). 0 disables.
    pub outflow_sea_level: f32,
    /// Bevy-only: map borders within this many base voxels become outflow. 0 disables.
    pub outflow_border: f32,
    /// Reference parm `Use_Erosion_Mask`: trails only start where the mask is > 0, scaled by it.
    pub use_erosion_mask: u32,
    /// Reference parm `Use_Uplift_Map`: scales `Uplift_Amount` per cell.
    pub use_uplift_map: u32,
    /// Reference parm `Use_Bedrock`: erosion never cuts below the bedrock layer.
    pub use_bedrock: u32,
    /// Reference parm `Use_Bedrock_Map` (Depth mode): the bedrock depth is scaled by mask noise.
    pub use_bedrock_map: u32,
    /// Reference parm `Bedrock_Depth`: bedrock sits this far below the source terrain.
    pub bedrock_depth: f32,
    /// Reference parm `Additive_Sediment`: bedrock is the source surface and a sediment layer of
    /// `Bedrock_Depth` is added on top instead.
    pub additive_sediment: u32,
    /// Reference parm `Granularity_Amount` (`Smooth_Flows` noise on deep deposits; 0 = off).
    pub granularity_amount: f32,
    /// Reference parm `Granularity_Scale`.
    pub granularity_scale: f32,
    /// Reference parm `Granularity_Strength`.
    pub granularity_strength: f32,
    /// Reference parm `Age_Strength`: the Age view shows `age * 10^Age_Strength`.
    pub age_strength: f32,
    /// Reference parm `Normalize_Age_Field`: map the Age view through `v / (1 + v)`.
    pub normalize_age: u32,
    /// Reference parm `Bedrock_Mode` (with `Use_Bedrock_Map`): 0 = Depth, the map scales the depth
    /// below the source surface; 1 = Height, the map times `Bedrock_Depth` is the bedrock height.
    pub bedrock_mode: u32,
    /// Set per dispatch: where the base map sits inside the padded pyramid grid (`Compute_Padding`).
    pub pad_offset: UVec2,
    /// Set per dispatch: padded base size (the finest level's grid).
    pub padded_size: UVec2,
    /// Set per dispatch: rain brush disc center in level texels (see [`BrushStroke`]).
    pub spawn_center: Vec2,
    /// Set per dispatch: rain brush disc radius in level texels.
    pub spawn_radius: f32,
    /// Set per dispatch: trails to start in the brush disc; 0 = KTT's map-wide trail density.
    pub spawn_count: u32,
    /// Bevy-only: water (height units × texels²) each rain brush trail leaves per step on the
    /// finest level (set from [`WaterControl::rain_water`] for brush dispatches only).
    pub rain_water: f32,
    /// Bevy-only, set per frame: the water solver runs, so trails on the finest level that reach
    /// water (`masks.y` > 0.5) hand it their sediment and end.
    pub water_handoff: u32,
    /// Bevy-only: shifts the source terrain noise, so each seed is a different landscape.
    pub noise_seed: u32,
    /// Bevy-only: river momentum below this is hillslope, so River Scale and the other river
    /// terms only act in the channels.
    pub river_threshold: f32,
    /// Bevy-only: steepest-descent trails in Do Rivers, 0 to 1. At Friction 1 the trails keep
    /// their momentum (as without rivers); this damps it everywhere but in the rivers, so trails
    /// run straight downhill and carve a dense dendritic network.
    pub trail_damping: f32,
    /// Bevy-only: this `writeBack`'s share of the level's erosion iterations (uplift, compaction,
    /// river channeling and momentum decay are scaled by it).
    pub writeback_fraction: f32,
}

impl Default for ErodeParams {
    fn default() -> Self {
        // --- Root node UI defaults (`ref/houdini22/node_tree.json`) ---
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
            feature_size: 64.0,           // Feature_Size
            source_detail_smoothing: 1.0, // Source_Terrain_Detail_Smoothing
            geological_age: 0.0,          // Geological_Age
            voxel_size: 1.0,
            trail_weight: 1.0,
            erosion_cutoff_angle: 90.0, // Erosion_Cutoff_Angle
            clamp_river_channeling: 0,  // Clamp_River_Channeling
            use_hydro_conditioning: 0,  // Use_Hydro_Conditioning
            hydro_conditioning_search_distance: 16.0,
            use_softness_mask: 0, // Use_Softness_Mask
            mask_frequency: 0.01,
            use_outflow: 0,         // Use_Outflow_Map
            outflow_integration: 0, // Outflow_Integration_Method (Sharp)
            outflow_sea_level: 0.6,
            outflow_border: 0.0,
            use_erosion_mask: 0, // Use_Erosion_Mask
            use_uplift_map: 0,   // Use_Uplift_Map
            use_bedrock: 0,      // Use_Bedrock
            use_bedrock_map: 0,  // Use_Bedrock_Map
            // Bedrock_Depth is 1 world unit in KTT; this map's terrain is only ~1 unit tall.
            bedrock_depth: 0.05,
            additive_sediment: 0,    // Additive_Sediment
            granularity_amount: 0.0, // Granularity_Amount
            granularity_scale: 0.5,  // Granularity_Scale
            // Granularity_Strength is 5 in KTT; its noise scales with voxel size, which is large
            // relative to this map's ~1 unit terrain height, so 5 shreds the surface.
            granularity_strength: 0.1,
            age_strength: -1.0, // Age_Strength
            normalize_age: 0,   // Normalize_Age_Field
            bedrock_mode: 0,    // Bedrock_Mode (Depth)
            pad_offset: UVec2::ZERO,
            padded_size: UVec2::new(512, 512),
            spawn_center: Vec2::ZERO,
            spawn_radius: 0.0,
            spawn_count: 0,
            rain_water: 0.0,
            water_handoff: 0,
            noise_seed: 0,
            river_threshold: 0.0,
            trail_damping: 0.3,
            writeback_fraction: 1.0,
        }
    }
}

impl ErodeParams {
    /// Pyramid levels for the current settings.
    pub fn pyramid_levels(&self) -> Vec<PyramidLevel> {
        pyramid_levels(
            self.map_size,
            self.feature_size,
            self.detail_scale,
            self.geological_age,
        )
    }
}

/// Rain brush input for this frame: while `active`, extra erode dispatches start their trails in
/// a disc on the finest level, carving the current terrain without restarting the cascade.
#[derive(Resource, Clone, ExtractResource, Default, Debug)]
#[extract_app(RenderApp)]
pub struct BrushStroke {
    pub active: bool,
    /// Disc center in map texels.
    pub center: Vec2,
    /// Disc radius in map texels.
    pub radius: f32,
    /// Trails per dispatch.
    pub particles: u32,
    /// Erode dispatches per frame (alternating ridge / flow like the cascade).
    pub passes: u32,
}

/// Water bodies (lakes and rivers): a shallow-water solver on the finest level (`water.wgsl`).
/// Water fills depressions and spills downhill along the current terrain; flowing water carries
/// sediment, still water drops it, and wet cells act as KTT outflow so trails drain into them.
/// The optional ocean (`ErodeParams::outflow_sea_level`) is a sink for this water.
///
/// The solver advances in fixed steps of [`WATER_STEP`] seconds of the frame time, so all rates
/// here are per second and do not depend on the frame rate.
#[derive(Resource, Clone, ExtractResource, Debug)]
#[extract_app(RenderApp)]
pub struct WaterControl {
    /// Add (rate > 0) or remove (rate < 0) water in a disc this frame.
    pub stroke_active: bool,
    /// Disc center in map texels.
    pub center: Vec2,
    /// Disc radius in map texels.
    pub radius: f32,
    /// Water depth added per second at the disc center (height units).
    pub rate: f32,
    /// Persistent water sources (at most [`MAX_SPRINGS`]); they keep rivers flowing.
    pub springs: Vec<Spring>,
    /// Water depth added per second everywhere (height units).
    pub rainfall: f32,
    /// Water each rain brush trail leaves per step along its path (height units × texels²), so
    /// the rain brush wets the channels it carves (0 = it only carves).
    pub rain_water: f32,
    /// Bump to remove all water.
    pub clear_generation: u32,
    /// Fraction of water lost per second.
    pub evaporation: f32,
    /// Sediment carrying capacity of flowing water (0 = water never erodes).
    pub sediment_capacity: f32,
    /// Which flow model moves the water.
    pub solver: WaterSolver,
    /// Manning roughness for [`WaterSolver::Swe`] (0 = no bed friction).
    pub manning_n: f32,
    /// Carries depth with a second-order flux in [`WaterSolver::Swe`] instead of first-order
    /// upwind, whose numerical diffusion damps roll waves.
    pub second_order_depth: bool,
    /// Gravity for [`WaterSolver::Swe`] in cells per second², with heights in cells (×100).
    /// Higher levels lakes faster; the solver's depth limiter caps the depth that carries flux at
    /// about 2 / (g · step) cells, so lower lets deep water flow.
    pub gravity: f32,
    /// Seconds since the last frame (set from [`Time`]); the solver catches up by this much.
    pub dt: f32,
    /// Bed reconstruction settings (see [`BedShaping`]).
    pub bed: BedShaping,
    /// Bump to reshape the beds of the current lakes and rivers once.
    pub shape_generation: u32,
    /// Bed load moved by the flow (see [`BedLoad`]).
    pub bed_load: BedLoad,
    /// Bump to fast-forward the beds by [`BedLoad::refine_steps`] steps.
    pub refine_generation: u32,
    /// With KTT's Do Rivers on, raise its river momentum to the water's discharge, so trails on
    /// the finest level follow the rivers the water has made.
    pub steer_rivers: bool,
}

/// Bed load and Exner (`water_morpho.wgsl`): on water deeper than a thin sheet, the flow moves
/// bed material along the bed (Shields and Meyer-Peter Müller), toward the inner banks of bends
/// (helical flow) and down cross-slopes (Talmon), and the bed changes by its divergence. Units
/// are the solver's: cells (heights × 100) and seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BedLoad {
    /// Run bed load with every water step (morphological factor 1).
    pub live: bool,
    /// Submerged grain weight R·D (cells) in the Shields number θ = n²|u|² / (grain · h^(1/3)):
    /// coarser grains need faster flow to move.
    pub grain: f32,
    /// Meyer-Peter Müller coefficient: bed load K · (θ − θ_c)^1.5 in cells² per second.
    pub k: f32,
    /// Shields number θ_c below which the bed does not move.
    pub critical_shields: f32,
    /// Talmon's A_s: how weakly bed load slides down cross-slopes (larger is weaker). It is 1/r
    /// in tan γ = −(r/√θ) ∂η/∂n. Flume calibrations put r at 0.3–0.6 (Redolfi et al. 2020: 0.40);
    /// the default r = 1 keeps bends scouring their outer banks reliably at this resolution (see
    /// docs/VALIDATION.md).
    pub slope_effect: f32,
    /// A_h in the helical deflection δ = atan(A_h · I_s) toward the inner bank. Rivers measure
    /// 7–11; the default is larger because these channels are shallow against their bends.
    pub helical_strength: f32,
    /// Morphological factor while refining (bed change per water step).
    pub morph_factor: f32,
    /// Water steps one refine runs.
    pub refine_steps: u32,
    /// How much harder the hardest rock of KTT's softness mask is to move: its critical Shields
    /// number is (1 + this) times Critical Shields. Bed load also never cuts below KTT's
    /// bedrock layer (with Use Bedrock on).
    pub rock_hardness: f32,
}

impl Default for BedLoad {
    fn default() -> Self {
        Self {
            live: true,
            grain: 0.25,
            k: 4.0,
            critical_shields: 0.047,
            slope_effect: 1.0,
            helical_strength: 20.0,
            morph_factor: 20.0,
            refine_steps: 400,
            rock_hardness: 4.0,
        }
    }
}

/// Bed reconstruction (`water_bed.wgsl`): cuts a channel (rivers) or bowl (lakes) under the
/// current water and grades the banks next to it. Rivers are found and sized from the averaged
/// discharge, and cut below their banks rather than below the water surface, so one press shapes
/// a stream and a second press changes nothing. Depths in height units, widths in texels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BedShaping {
    /// Depth the bed levels off toward.
    pub max_depth: f32,
    /// Lakes: depth gained per texel of distance from the shore.
    pub shore_slope: f32,
    /// River cross-section exponent of (1 − n²)^profile: below 1 is boxy with steep banks, above
    /// 1 narrows toward a V.
    pub profile: f32,
    /// Blend from the current terrain (0) to the reconstructed bed (1).
    pub strength: f32,
    /// Bank grading reach outside the water (texels, at most [`MAX_BANK_WIDTH`]).
    pub bank_width: f32,
    /// Bank rise per texel away from the first bank cell.
    pub bank_slope: f32,
    /// River thalweg depth below the banks at a channel discharge of about a default demo
    /// spring; it grows with the discharge to the power 0.4.
    pub river_depth: f32,
    /// River half-width at the same discharge; it grows with the square root of the discharge,
    /// up to the width of the flowing water.
    pub river_width: f32,
    /// Averaged unit discharge (height units × texels per second) above which flowing water
    /// belongs to a channel.
    pub channel_discharge: f32,
}

impl Default for BedShaping {
    fn default() -> Self {
        Self {
            max_depth: 0.025,
            shore_slope: 0.0015,
            profile: 0.8,
            strength: 1.0,
            bank_width: 8.0,
            bank_slope: 0.0015,
            river_depth: 0.008,
            river_width: 5.0,
            channel_discharge: 0.02,
        }
    }
}

/// Widest bank grading (dilation steps of the shore distance).
pub const MAX_BANK_WIDTH: f32 = 32.0;
/// Red-black SOR sweeps for the bank distance τ and the depth smoothing.
const BED_TAU_SWEEPS: u32 = 400;
/// Reach of the bank reference inside rivers (texels); wider rivers fall back to their surface.
const BED_BANK_REACH: f32 = 32.0;
const BED_SMOOTH_SWEEPS: u32 = 200;
const BED_OMEGA: f32 = 1.9;

/// Uniform for `water_bed.wgsl` (one entry per red-black parity).
#[derive(Clone, Copy, ShaderType)]
struct BedParams {
    size: UVec2,
    pad_offset: UVec2,
    grid_size: UVec2,
    parity: u32,
    omega: f32,
    max_depth: f32,
    shore_slope: f32,
    profile: f32,
    strength: f32,
    bank_width: f32,
    bank_slope: f32,
    river_depth: f32,
    river_width: f32,
    channel_discharge: f32,
}

impl Default for WaterControl {
    fn default() -> Self {
        Self {
            stroke_active: false,
            center: Vec2::ZERO,
            radius: 0.0,
            rate: 0.0,
            springs: Vec::new(),
            rainfall: 0.0,
            rain_water: 0.0002,
            clear_generation: 0,
            evaporation: 0.01,
            sediment_capacity: 0.0005,
            solver: WaterSolver::default(),
            manning_n: 0.01,
            second_order_depth: false,
            gravity: 2000.0,
            dt: 0.0,
            bed: BedShaping::default(),
            shape_generation: 0,
            bed_load: BedLoad::default(),
            refine_generation: 0,
            steer_rivers: true,
        }
    }
}

/// Flow model for [`WaterControl`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WaterSolver {
    /// Virtual pipes (Mei et al. 2007) in `water.wgsl`.
    Pipes,
    /// bevy_fft's staggered shallow-water solver (Chentanez and Müller 2010): face velocities
    /// carry momentum, so jets run on into lakes and around bends, with Manning bed friction.
    #[default]
    Swe,
}

/// A persistent water source: a disc that adds water like a held brush.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spring {
    /// Disc center in map texels.
    pub center: Vec2,
    /// Disc radius in map texels.
    pub radius: f32,
    /// Water depth added per second at the center (height units).
    pub rate: f32,
}

/// Most springs [`WaterControl`] uploads.
pub const MAX_SPRINGS: usize = 16;
/// Springs plus the brush (must match `MAX_SOURCES` in `water.wgsl`).
const MAX_WATER_SOURCES: usize = MAX_SPRINGS + 1;
/// Simulated seconds per water solver step (8 steps per frame at 60 fps).
pub const WATER_STEP: f32 = 1.0 / 480.0;
/// Most solver steps per frame; below 15 fps the water slows down instead of stalling the GPU.
const MAX_WATER_STEPS: u32 = 32;

/// Uniform for `water.wgsl`.
#[derive(Clone, Copy, ShaderType)]
pub struct WaterParams {
    /// Base map size (the water grid).
    pub size: UVec2,
    /// Base map offset inside the padded finest level (height and masks grid).
    pub pad_offset: UVec2,
    pub grid_size: UVec2,
    pub clear: u32,
    /// Fraction of the depth lost per step.
    pub evaporation: f32,
    pub capacity: f32,
    /// Ocean level (0 = no ocean): terrain below it drains water away.
    pub sea_level: f32,
    /// Depth added everywhere per step.
    pub rain: f32,
    /// Seconds per step ([`WATER_STEP`]).
    pub step_seconds: f32,
    /// Bed load runs (`water_morpho.wgsl`), so the capacity model only erodes thin sheet flow.
    pub bed_load: u32,
    pub source_count: u32,
    /// Springs and the brush: center, radius, depth per step at the center.
    pub sources: [Vec4; MAX_WATER_SOURCES],
}

/// Uniform for `water_morpho.wgsl`.
#[derive(Clone, Copy, ShaderType)]
struct MorphoParams {
    size: UVec2,
    pad_offset: UVec2,
    grid_size: UVec2,
    step_seconds: f32,
    frame_seconds: f32,
    discharge_time: f32,
    ktt_scale: f32,
    ktt_vector: u32,
    sea_level: f32,
    manning_n: f32,
    grain: f32,
    bed_load_k: f32,
    critical_shields: f32,
    slope_effect: f32,
    helical_strength: f32,
    helical_length: f32,
    mor_fac: f32,
    parity: u32,
    use_bedrock: u32,
    rock_hardness: f32,
}

/// Seconds the shared discharge field averages over, so roll waves do not make it flicker.
const DISCHARGE_TIME: f32 = 3.0;
/// KTT river momentum (trail water per texel) to unit discharge (height units × texels per
/// second), fitted on a spring-fed valley (`tests/water_swe_gpu.rs`, `discharge_matches_ktt`).
const KTT_DISCHARGE_SCALE: f32 = 0.0044;
/// Helical flow adaptation length in texels (terracraft uses 42 m at 1 m per pixel).
const HELICAL_LENGTH: f32 = 20.0;
/// Most water steps one frame runs while refining beds.
const MAX_REFINE_STEPS: u32 = 64;

/// Resource to trigger simulation reset
#[derive(Resource, Clone, ExtractResource, Default)]
#[extract_app(RenderApp)]
pub struct ResetSim {
    pub generation: u32,
}

/// Resource to control simulation state
#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
pub struct SimControl {
    pub paused: bool,
    pub step_counter: u64,
    /// Finest-level erosion dispatches per frame while running. Coarser pyramid levels are cheap
    /// and always run to completion in a single frame.
    pub iterations_per_frame: u32,
    /// Keep eroding the finest level in `writeBack` batches once the cascade is done.
    pub continue_after_cascade: bool,
}

impl Default for SimControl {
    fn default() -> Self {
        Self {
            paused: false,
            step_counter: 0,
            iterations_per_frame: 4,
            continue_after_cascade: false,
        }
    }
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
        ErodeParams {
            map_size: self.map_size,
            num_particles: self.map_size.x * self.map_size.y,
            iteration,
            num_iterations: ERODE_ITERS_PER_WRITEBACK,
            detail_scale: self.reference_detail_scale,
            erosion_amount: self.erosion_amount,
            ..Default::default()
        }
    }
}

/// Plugin for erosion compute simulation
pub struct ErosionComputePlugin;

fn setup_erosion_resources(
    config: Res<ErosionConfig>,
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut shader_buffers: ResMut<Assets<ShaderBuffer>>,
) {
    let size = config.map_size;
    commands.insert_resource(create_erosion_images(&mut images, size));
    commands.insert_resource(WaterFields::new(&mut shader_buffers, size));
    commands.insert_resource(config.to_erode_params(0));
    commands.insert_resource(ResetSim::default());
    commands.insert_resource(SimControl::default());
    commands.insert_resource(BrushStroke::default());
    commands.insert_resource(WaterControl::default());
}

fn update_water_time(time: Res<Time>, water: Option<ResMut<WaterControl>>) {
    if let Some(mut water) = water {
        water.dt = time.delta_secs();
    }
}

impl Plugin for ErosionComputePlugin {
    fn build(&self, app: &mut App) {
        let progress = PyramidProgress::default();
        app.init_resource::<ErosionConfig>()
            .insert_resource(progress.clone())
            .add_plugins((
                ExtractResourcePlugin::<ErosionImages>::default(),
                ExtractResourcePlugin::<WaterFields>::default(),
                ExtractResourcePlugin::<ErodeParams>::default(),
                ExtractResourcePlugin::<ResetSim>::default(),
                ExtractResourcePlugin::<SimControl>::default(),
                ExtractResourcePlugin::<BrushStroke>::default(),
                ExtractResourcePlugin::<WaterControl>::default(),
            ))
            .add_systems(Startup, setup_erosion_resources)
            .add_systems(PostUpdate, update_water_time);
        if !app.is_plugin_added::<ShallowWaterSolverPlugin>() {
            app.add_plugins(ShallowWaterSolverPlugin);
        }

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(progress)
            .init_resource::<ErosionState>()
            .add_systems(RenderStartup, init_erosion_pipeline)
            .add_systems(
                Render,
                (
                    (prepare_erosion_buffers, plan_erosion_frame)
                        .chain()
                        .in_set(RenderSystems::PrepareResources),
                    prepare_erosion_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(RenderGraph, erosion_compute.before(camera_driver));
    }
}

#[derive(Resource)]
struct ErosionPipeline {
    layout_erosion: BindGroupLayoutDescriptor,
    layout_color: BindGroupLayoutDescriptor,
    layout_terrain_edit: BindGroupLayoutDescriptor,
    layout_water: BindGroupLayoutDescriptor,
    layout_water_out: BindGroupLayoutDescriptor,
    layout_water_bed: BindGroupLayoutDescriptor,
    layout_blit: BindGroupLayoutDescriptor,
    layout_blit_blur: BindGroupLayoutDescriptor,
    layout_upscale_height: BindGroupLayoutDescriptor,
    layout_upscale_rivers: BindGroupLayoutDescriptor,
    layout_upscale_masks: BindGroupLayoutDescriptor,
    pipeline_init: CachedComputePipelineId,
    pipeline_init_color: CachedComputePipelineId,
    pipeline_erode: CachedComputePipelineId,
    pipeline_write_back: CachedComputePipelineId,
    pipeline_blit: CachedComputePipelineId,
    pipeline_upscale_height: CachedComputePipelineId,
    pipeline_upscale_rivers: CachedComputePipelineId,
    pipeline_upscale_masks: CachedComputePipelineId,
    pipeline_ao: CachedComputePipelineId,
    pipeline_blur_h: CachedComputePipelineId,
    pipeline_blur_v: CachedComputePipelineId,
    /// `water.wgsl`: clear, flux, depth, erode, advect, copy, the bevy_fft solver's export,
    /// gather and import, and apply (the staged erosion).
    pipelines_water: [CachedComputePipelineId; 10],
    /// `water_bed.wgsl`: init, tau, prior, smooth, write, dilate, grade.
    pipelines_water_bed: [CachedComputePipelineId; 7],
    layout_water_morpho: BindGroupLayoutDescriptor,
    layout_water_bed_load: BindGroupLayoutDescriptor,
    /// `water_morpho.wgsl`: discharge, helical, flux, exner, steer.
    pipelines_water_morpho: [CachedComputePipelineId; 5],
    layout_water_splat: BindGroupLayoutDescriptor,
    /// `water_splat.wgsl`: splat, record.
    pipelines_water_splat: [CachedComputePipelineId; 2],
}

impl ErosionPipeline {
    /// Pipelines the cascade needs before it can start.
    fn required(&self) -> [CachedComputePipelineId; 8] {
        [
            self.pipeline_init,
            self.pipeline_init_color,
            self.pipeline_erode,
            self.pipeline_write_back,
            self.pipeline_blit,
            self.pipeline_upscale_height,
            self.pipeline_upscale_rivers,
            self.pipeline_upscale_masks,
        ]
    }
}

fn init_erosion_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    // Layout for erosion (group 0); the params uniform uses a dynamic offset per dispatch.
    let layout_erosion = BindGroupLayoutDescriptor::new(
        "ErosionGroup0",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer::<u32>(false), // height (atomic<u32> in shader, fixed-point)
                uniform_buffer::<ErodeParams>(true),
                storage_buffer::<u32>(false), // flow (atomic<u32> in shader)
                storage_buffer::<i32>(false), // deposition (atomic<i32> in shader)
                storage_buffer::<u32>(false), // erosion (atomic<u32> in shader)
                storage_buffer::<f32>(false), // river momentum x, y (f32 bits in atomic<u32>)
                storage_buffer::<u32>(false), // splat to the water (atomic<u32>, fixed-point)
                storage_buffer::<Vec4>(false), // masks: softness, outflow, erode mask, uplift map
                storage_buffer::<Vec2>(false), // layers: bedrock, age
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

    // Level transition: height + deposition (Update_Upscaled_Fields + Smooth_Flows).
    let layout_upscale_height = BindGroupLayoutDescriptor::new(
        "PyramidUpscaleHeight",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only::<u32>(false),  // src height
                storage_buffer_read_only::<i32>(false),  // src deposition
                storage_buffer_read_only::<u32>(false),  // source terrain height
                storage_buffer::<u32>(false),            // dst height
                storage_buffer::<i32>(false),            // dst deposition
                storage_buffer_read_only::<Vec4>(false), // dst masks
                storage_buffer_read_only::<f32>(false),  // dst outflow accumulation
                storage_buffer_read_only::<Vec2>(false), // dst layers (bedrock)
                uniform_buffer::<PyramidResampleParams>(true),
            ),
        ),
    );

    // Level transition: river momentum fields, and clearing of per-level analysis buffers.
    let layout_upscale_rivers = BindGroupLayoutDescriptor::new(
        "PyramidUpscaleRivers",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only::<f32>(false), // src river momentum
                storage_buffer::<f32>(false),           // dst river momentum
                storage_buffer::<u32>(false),           // dst flow
                storage_buffer::<u32>(false),           // dst erosion
                uniform_buffer::<PyramidResampleParams>(true),
            ),
        ),
    );

    // Level transition: masks and bedrock resampled from the source terrain, age upsampled.
    let layout_upscale_masks = BindGroupLayoutDescriptor::new(
        "PyramidUpscaleMasks",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only::<f32>(false), // src outflow accumulation
                storage_buffer_read_only::<Vec2>(false), // src layers (age)
                storage_buffer_read_only::<Vec4>(false), // source masks
                storage_buffer_read_only::<Vec2>(false), // source layers (bedrock)
                storage_buffer::<Vec4>(false),          // dst masks
                storage_buffer::<f32>(false),           // dst outflow accumulation
                storage_buffer::<Vec2>(false),          // dst layers
                uniform_buffer::<PyramidResampleParams>(true),
                storage_buffer_read_only::<f32>(false), // water depth (lakes are outflow)
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

    // Terrain brush edits for init (group 1 of init_fbm).
    let layout_terrain_edit = BindGroupLayoutDescriptor::new(
        "ErosionTerrainEditGroup1",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (texture_2d(TextureSampleType::Float { filterable: false }),),
        ),
    );

    // Water solver on the finest work set (`water.wgsl`).
    let layout_water = BindGroupLayoutDescriptor::new(
        "WaterGroup0",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer::<u32>(false),  // work set height (atomic, fixed-point)
                storage_buffer::<f32>(false),  // water depth
                storage_buffer::<Vec4>(false), // outflow flux L, R, T, B
                storage_buffer::<f32>(false),  // suspended sediment
                storage_buffer::<f32>(false),  // advected sediment (scratch)
                storage_buffer::<Vec2>(false), // velocity
                storage_buffer::<Vec4>(false), // work set masks (outflow)
                storage_buffer_read_only::<Vec4>(false), // source masks
                uniform_buffer::<WaterParams>(false),
            ),
        ),
    );
    // Bed reconstruction on the finest work set (`water_bed.wgsl`).
    let layout_water_bed = BindGroupLayoutDescriptor::new(
        "WaterBedGroup0",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer::<u32>(false), // work set height (atomic, fixed-point)
                storage_buffer::<f32>(false), // water depth
                storage_buffer_read_only::<Vec4>(false), // discharge
                storage_buffer::<f32>(false), // bank distance tau
                storage_buffer::<f32>(false), // bed depth
                storage_buffer::<f32>(false), // depth prior
                storage_buffer::<Vec2>(false), // shore (eta, distance) ping
                storage_buffer::<Vec2>(false), // shore pong
                uniform_buffer::<BedParams>(true),
            ),
        ),
    );
    // Discharge, helical flow and bed load on the finest work set (`water_morpho.wgsl`).
    let layout_water_morpho = BindGroupLayoutDescriptor::new(
        "WaterMorphoGroup0",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer::<u32>(false), // work set height (atomic, fixed-point)
                storage_buffer_read_only::<f32>(false), // water depth
                storage_buffer_read_only::<Vec2>(false), // velocity
                storage_buffer::<Vec4>(false), // discharge
                storage_buffer::<f32>(false), // work set river momentum
                storage_buffer::<Vec2>(false), // face bed load
                storage_buffer::<f32>(false), // helical intensity and pending bed change
                uniform_buffer::<MorphoParams>(false),
            ),
        ),
    );
    // Trails and the water (`water_splat.wgsl`).
    let layout_water_splat = BindGroupLayoutDescriptor::new(
        "WaterSplatGroup0",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer::<u32>(false), // splat (atomic, fixed-point)
                storage_buffer::<f32>(false), // water depth
                storage_buffer::<f32>(false), // suspended sediment
                storage_buffer::<u32>(false), // work set height (atomic, fixed-point)
                storage_buffer::<i32>(false), // work set deposition (atomic)
                storage_buffer::<u32>(false), // work set erosion (atomic)
                storage_buffer::<f32>(false), // morpho state
                uniform_buffer::<WaterParams>(false),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
            ),
        ),
    );
    // The bed load kernels of `water_morpho.wgsl`: KTT's softness and bedrock instead of the
    // discharge and river momentum.
    let layout_water_bed_load = BindGroupLayoutDescriptor::new(
        "WaterBedLoadGroup0",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (0, storage_buffer::<u32>(false)), // work set height (atomic, fixed-point)
                (1, storage_buffer_read_only::<f32>(false)), // water depth
                (2, storage_buffer_read_only::<Vec2>(false)), // velocity
                (5, storage_buffer::<Vec2>(false)), // face bed load
                (6, storage_buffer::<f32>(false)), // helical intensity and pending bed change
                (7, uniform_buffer::<MorphoParams>(false)),
                (8, storage_buffer_read_only::<Vec4>(false)), // work set masks (softness)
                (9, storage_buffer_read_only::<Vec2>(false)), // work set layers (bedrock)
            ),
        ),
    );
    let layout_water_out = BindGroupLayoutDescriptor::new(
        "WaterGroup1",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                // bevy_fft solver bed and source inputs.
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                // bevy_fft solver bed and depth, and its cell-centred velocity.
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_2d(TextureSampleType::Float { filterable: false }),
            ),
        ),
    );

    let shader = asset_server.load(EROSION_SHADER);
    let compute =
        |layout: Vec<BindGroupLayoutDescriptor>, shader: &Handle<Shader>, entry: &'static str| {
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                layout,
                shader: shader.clone(),
                entry_point: Some(Cow::from(entry)),
                ..default()
            })
        };
    let pipeline_init = compute(
        vec![layout_erosion.clone(), layout_terrain_edit.clone()],
        &shader,
        "init_fbm",
    );
    let pipeline_init_color = compute(
        vec![layout_erosion.clone(), layout_color.clone()],
        &shader,
        "init_color_bands",
    );
    let pipeline_erode = compute(
        vec![layout_erosion.clone(), layout_color.clone()],
        &shader,
        "erode",
    );
    let pipeline_write_back = compute(vec![layout_erosion.clone()], &shader, "write_back");
    let pipeline_blit = compute(
        vec![layout_erosion.clone(), layout_blit.clone()],
        &shader,
        "blit_to_texture",
    );
    let pipeline_upscale_height = compute(
        vec![layout_upscale_height.clone()],
        &asset_server.load(PYRAMID_RESAMPLE_SHADER),
        "pyramid_upscale_height",
    );
    let pipeline_upscale_rivers = compute(
        vec![layout_upscale_rivers.clone()],
        &asset_server.load(PYRAMID_RIVERS_SHADER),
        "pyramid_upscale_rivers",
    );
    let pipeline_upscale_masks = compute(
        vec![layout_upscale_masks.clone()],
        &asset_server.load(PYRAMID_MASKS_SHADER),
        "pyramid_upscale_masks",
    );
    let pipeline_ao = compute(
        vec![layout_erosion.clone(), layout_blit.clone()],
        &shader,
        "compute_ao",
    );
    let pipeline_blur_h = compute(
        vec![layout_erosion.clone(), layout_blit_blur.clone()],
        &shader,
        "blur_ao_horizontal",
    );
    let pipeline_blur_v = compute(
        vec![layout_erosion.clone(), layout_blit_blur.clone()],
        &shader,
        "blur_ao_vertical",
    );

    let water_shader = asset_server.load(WATER_SHADER);
    let pipelines_water = [
        "water_clear",
        "water_flux",
        "water_depth",
        "water_erode",
        "water_advect",
        "water_copy",
        "swe_export",
        "swe_gather",
        "swe_import",
        "water_apply",
    ]
    .map(|entry| {
        compute(
            vec![layout_water.clone(), layout_water_out.clone()],
            &water_shader,
            entry,
        )
    });

    let water_bed_shader = asset_server.load(WATER_BED_SHADER);
    let pipelines_water_bed = [
        "bed_init",
        "bed_tau",
        "bed_prior",
        "bed_smooth",
        "bed_write",
        "bed_dilate",
        "bed_grade",
    ]
    .map(|entry| compute(vec![layout_water_bed.clone()], &water_bed_shader, entry));

    let water_morpho_shader = asset_server.load(WATER_MORPHO_SHADER);
    let pipelines_water_morpho = [
        "morpho_discharge",
        "morpho_helical",
        "morpho_flux",
        "morpho_exner",
        "morpho_steer",
    ]
    .map(|entry| {
        let layout = if matches!(entry, "morpho_flux" | "morpho_exner") {
            &layout_water_bed_load
        } else {
            &layout_water_morpho
        };
        compute(vec![layout.clone()], &water_morpho_shader, entry)
    });

    let water_splat_shader = asset_server.load(WATER_SPLAT_SHADER);
    let pipelines_water_splat = ["water_splat", "water_record"]
        .map(|entry| compute(vec![layout_water_splat.clone()], &water_splat_shader, entry));

    commands.insert_resource(ErosionPipeline {
        layout_erosion,
        layout_water_splat,
        pipelines_water_splat,
        layout_water_morpho,
        layout_water_bed_load,
        pipelines_water_morpho,
        layout_water_bed,
        pipelines_water_bed,
        layout_color,
        layout_terrain_edit,
        layout_water,
        layout_water_out,
        layout_blit,
        layout_blit_blur,
        layout_upscale_height,
        layout_upscale_rivers,
        layout_upscale_masks,
        pipeline_init,
        pipeline_init_color,
        pipeline_erode,
        pipeline_write_back,
        pipeline_blit,
        pipeline_upscale_height,
        pipeline_upscale_rivers,
        pipeline_upscale_masks,
        pipeline_ao,
        pipeline_blur_h,
        pipeline_blur_v,
        pipelines_water,
    });
}

/// One full-capacity set of simulation fields; pyramid levels index it at their own width.
struct FieldSet {
    height: StorageBuffer<Vec<u32>>,
    flow: StorageBuffer<Vec<u32>>,
    deposition: StorageBuffer<Vec<i32>>,
    erosion: StorageBuffer<Vec<u32>>,
    /// KTT `__m`: river momentum x and y per cell, interleaved.
    m: StorageBuffer<Vec<f32>>,
    /// KTT `__Softness_Mask`, `__Outflow`, `__Erode_Mask`, `__Uplift` (identity values when unused).
    masks: StorageBuffer<Vec<Vec4>>,
    /// KTT `__Bedrock` and `__Age`.
    layers: StorageBuffer<Vec<Vec2>>,
    /// KTT `__Outflow_Accumulation` (max outflow over coarser levels).
    outflow_acc: StorageBuffer<Vec<f32>>,
}

impl FieldSet {
    fn new(texels: usize, device: &RenderDevice, queue: &RenderQueue) -> Self {
        let mut set = Self {
            height: StorageBuffer::from(vec![0u32; texels]),
            flow: StorageBuffer::from(vec![0u32; texels]),
            deposition: StorageBuffer::from(vec![0i32; texels]),
            erosion: StorageBuffer::from(vec![0u32; texels]),
            m: StorageBuffer::from(vec![0f32; 2 * texels]),
            masks: StorageBuffer::from(vec![Vec4::new(1.0, 0.0, 1.0, 1.0); texels]),
            layers: StorageBuffer::from(vec![Vec2::ZERO; texels]),
            outflow_acc: StorageBuffer::from(vec![0f32; texels]),
        };
        set.height.write_buffer(device, queue);
        set.flow.write_buffer(device, queue);
        set.deposition.write_buffer(device, queue);
        set.erosion.write_buffer(device, queue);
        set.m.write_buffer(device, queue);
        set.masks.write_buffer(device, queue);
        set.layers.write_buffer(device, queue);
        set.outflow_acc.write_buffer(device, queue);
        set
    }
}

/// Indices into [`ErosionBindGroups::erosion`].
const SET_SOURCE: usize = 2;

#[derive(Resource)]
struct ErosionBuffers {
    texels: usize,
    /// Work sets 0/1 (ping-pong between levels) and the source terrain (index [`SET_SOURCE`]).
    sets: [FieldSet; 3],
    params: DynamicUniformBuffer<ErodeParams>,
    /// One entry per level transition this frame (dynamic offset per `Seed`).
    resample_params: DynamicUniformBuffer<PyramidResampleParams>,
    water: WaterBuffers,
    /// Water and sediment trails hand to the water on the finest level (`erosion.wgsl` `splat`).
    splat: StorageBuffer<Vec<u32>>,
}

/// Water solver state at base map size (`water.wgsl`).
struct WaterBuffers {
    depth: StorageBuffer<Vec<f32>>,
    flux: StorageBuffer<Vec<Vec4>>,
    sediment: StorageBuffer<Vec<f32>>,
    sediment_tmp: StorageBuffer<Vec<f32>>,
    velocity: StorageBuffer<Vec<Vec2>>,
    params: UniformBuffer<WaterParams>,
    /// Bed reconstruction scratch fields (`water_bed.wgsl`).
    tau: StorageBuffer<Vec<f32>>,
    bed_depth: StorageBuffer<Vec<f32>>,
    prior: StorageBuffer<Vec<f32>>,
    shore: [StorageBuffer<Vec<Vec2>>; 2],
    bed_params: DynamicUniformBuffer<BedParams>,
    swe: SweGrid,
    /// Bed load through each cell's +x and +y faces.
    bed_load: StorageBuffer<Vec<Vec2>>,
    /// Per cell: helical intensity (two slots) and the bed change not yet written.
    morpho_state: StorageBuffer<Vec<f32>>,
    morpho_params: UniformBuffer<MorphoParams>,
}

/// bevy_fft solver grid and its inputs at base map size ([`WaterSolver::Swe`]).
struct SweGrid {
    textures: ShallowWaterTextures,
    /// `R32Float` bed height in cells, written by `swe_export`.
    bed: TextureView,
    /// `Rgba32Float` sources and sinks, written by `swe_export`.
    source: TextureView,
    /// Built once the solver's pipelines have compiled.
    solver: Option<ShallowWaterSolver>,
}

impl SweGrid {
    fn new(size: UVec2, device: &RenderDevice) -> Self {
        let input = |label: &'static str, format: TextureFormat| {
            device
                .create_texture(&TextureDescriptor {
                    label: Some(label),
                    size: Extent3d {
                        width: size.x,
                        height: size.y,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format,
                    usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&TextureViewDescriptor::default())
        };
        Self {
            textures: ShallowWaterTextures::create(device, size),
            bed: input("water_swe_bed", TextureFormat::R32Float),
            source: input("water_swe_source", TextureFormat::Rgba32Float),
            solver: None,
        }
    }
}

impl WaterBuffers {
    fn new(size: UVec2, device: &RenderDevice, queue: &RenderQueue) -> Self {
        let n = (size.x * size.y) as usize;
        let mut b = Self {
            depth: StorageBuffer::from(vec![0f32; n]),
            flux: StorageBuffer::from(vec![Vec4::ZERO; n]),
            sediment: StorageBuffer::from(vec![0f32; n]),
            sediment_tmp: StorageBuffer::from(vec![0f32; n]),
            velocity: StorageBuffer::from(vec![Vec2::ZERO; n]),
            params: UniformBuffer::from(WaterParams {
                size,
                pad_offset: UVec2::ZERO,
                grid_size: size,
                clear: 0,
                evaporation: 0.0,
                capacity: 0.0,
                sea_level: 0.0,
                rain: 0.0,
                step_seconds: WATER_STEP,
                bed_load: 0,
                source_count: 0,
                sources: [Vec4::ZERO; MAX_WATER_SOURCES],
            }),
            tau: StorageBuffer::from(vec![0f32; n]),
            bed_depth: StorageBuffer::from(vec![0f32; n]),
            prior: StorageBuffer::from(vec![0f32; n]),
            shore: [0, 1].map(|_| StorageBuffer::from(vec![Vec2::ZERO; n])),
            bed_params: DynamicUniformBuffer::default(),
            swe: SweGrid::new(size, device),
            bed_load: StorageBuffer::from(vec![Vec2::ZERO; n]),
            morpho_state: StorageBuffer::from(vec![0f32; 4 * n]),
            morpho_params: UniformBuffer::from(MorphoParams {
                size,
                pad_offset: UVec2::ZERO,
                grid_size: size,
                step_seconds: WATER_STEP,
                frame_seconds: 0.0,
                discharge_time: DISCHARGE_TIME,
                ktt_scale: KTT_DISCHARGE_SCALE,
                ktt_vector: 0,
                sea_level: 0.0,
                manning_n: 0.0,
                grain: 1.0,
                bed_load_k: 0.0,
                critical_shields: 0.0,
                slope_effect: 1.0,
                helical_strength: 0.0,
                helical_length: HELICAL_LENGTH,
                mor_fac: 0.0,
                parity: 0,
                use_bedrock: 0,
                rock_hardness: 0.0,
            }),
        };
        b.depth.write_buffer(device, queue);
        b.flux.write_buffer(device, queue);
        b.sediment.write_buffer(device, queue);
        b.sediment_tmp.write_buffer(device, queue);
        b.velocity.write_buffer(device, queue);
        b.params.write_buffer(device, queue);
        b.tau.write_buffer(device, queue);
        b.bed_depth.write_buffer(device, queue);
        b.prior.write_buffer(device, queue);
        for shore in &mut b.shore {
            shore.write_buffer(device, queue);
        }
        b.bed_load.write_buffer(device, queue);
        b.morpho_state.write_buffer(device, queue);
        b.morpho_params.write_buffer(device, queue);
        b
    }
}

#[derive(Resource)]
struct ErosionBindGroups {
    /// Indexed by work set / [`SET_SOURCE`].
    erosion: [BindGroup; 3],
    /// Indexed by destination work set (source is the other one).
    upscale_height: [BindGroup; 2],
    upscale_rivers: [BindGroup; 2],
    upscale_masks: [BindGroup; 2],
    color: BindGroup,
    terrain_edit: BindGroup,
    /// Water solver, indexed by work set.
    water: [BindGroup; 2],
    water_out: BindGroup,
    /// Bed reconstruction, indexed by work set.
    water_bed: [BindGroup; 2],
    /// Discharge, helical flow and steering, indexed by work set.
    water_morpho: [BindGroup; 2],
    /// Bed load, indexed by work set.
    water_bed_load: [BindGroup; 2],
    /// Trails and the water, indexed by work set.
    water_splat: [BindGroup; 2],
    blit: BindGroup,
    blit_blur: BindGroup,
}

fn prepare_erosion_buffers(
    mut commands: Commands,
    params: Res<ErodeParams>,
    render_device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    existing: Option<Res<ErosionBuffers>>,
) {
    let capacity = pyramid_capacity(params.map_size);
    let texels = (capacity.x * capacity.y) as usize;
    if existing.is_some_and(|b| b.texels == texels) {
        return;
    }
    commands.insert_resource(ErosionBuffers {
        texels,
        sets: std::array::from_fn(|_| FieldSet::new(texels, &render_device, &queue)),
        params: DynamicUniformBuffer::default(),
        resample_params: DynamicUniformBuffer::default(),
        water: WaterBuffers::new(params.map_size, &render_device, &queue),
        splat: {
            let mut splat = StorageBuffer::from(vec![0u32; 2 * texels]);
            splat.write_buffer(&render_device, &queue);
            splat
        },
    });
}

fn prepare_erosion_bind_groups(
    mut commands: Commands,
    pipeline: Res<ErosionPipeline>,
    pipeline_cache: Res<PipelineCache>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    images: Res<ErosionImages>,
    fields: Option<Res<WaterFields>>,
    gpu_buffers: Res<RenderAssets<GpuShaderBuffer>>,
    buffers: Option<Res<ErosionBuffers>>,
    render_device: Res<RenderDevice>,
) {
    let Some(buffers) = buffers else { return };
    let Some(discharge) = fields.and_then(|f| gpu_buffers.get(&f.discharge)) else {
        return;
    };
    let discharge = discharge.buffer.as_entire_binding();
    if buffers.params.buffer().is_none()
        || buffers.resample_params.buffer().is_none()
        || buffers.water.bed_params.buffer().is_none()
    {
        return;
    }
    let (
        Some(view_height),
        Some(view_color),
        Some(view_normal),
        Some(view_analysis),
        Some(view_ao),
        Some(view_ao_temp),
        Some(view_water),
        Some(view_terrain_edit),
    ) = (
        gpu_images.get(&images.height),
        gpu_images.get(&images.color),
        gpu_images.get(&images.normal),
        gpu_images.get(&images.analysis),
        gpu_images.get(&images.ao),
        gpu_images.get(&images.ao_temp),
        gpu_images.get(&images.water),
        gpu_images.get(&images.terrain_edit),
    )
    else {
        return;
    };

    let layout = |desc: &BindGroupLayoutDescriptor| pipeline_cache.get_bind_group_layout(desc);
    let erosion = |set: &FieldSet| {
        render_device.create_bind_group(
            None,
            &layout(&pipeline.layout_erosion),
            &BindGroupEntries::sequential((
                &set.height,
                &buffers.params,
                &set.flow,
                &set.deposition,
                &set.erosion,
                &set.m,
                &buffers.splat,
                &set.masks,
                &set.layers,
            )),
        )
    };
    let source = &buffers.sets[SET_SOURCE];
    let upscale_height = |dst: usize| {
        let (src, dst) = (&buffers.sets[1 - dst], &buffers.sets[dst]);
        render_device.create_bind_group(
            None,
            &layout(&pipeline.layout_upscale_height),
            &BindGroupEntries::sequential((
                &src.height,
                &src.deposition,
                &source.height,
                &dst.height,
                &dst.deposition,
                &dst.masks,
                &dst.outflow_acc,
                &dst.layers,
                &buffers.resample_params,
            )),
        )
    };
    let upscale_rivers = |dst: usize| {
        let (src, dst) = (&buffers.sets[1 - dst], &buffers.sets[dst]);
        render_device.create_bind_group(
            None,
            &layout(&pipeline.layout_upscale_rivers),
            &BindGroupEntries::sequential((
                &src.m,
                &dst.m,
                &dst.flow,
                &dst.erosion,
                &buffers.resample_params,
            )),
        )
    };

    let upscale_masks = |dst: usize| {
        let (src, dst) = (&buffers.sets[1 - dst], &buffers.sets[dst]);
        render_device.create_bind_group(
            None,
            &layout(&pipeline.layout_upscale_masks),
            &BindGroupEntries::sequential((
                &src.outflow_acc,
                &src.layers,
                &source.masks,
                &source.layers,
                &dst.masks,
                &dst.outflow_acc,
                &dst.layers,
                &buffers.resample_params,
                buffers.water.depth.binding().unwrap(),
            )),
        )
    };

    commands.insert_resource(ErosionBindGroups {
        erosion: std::array::from_fn(|i| erosion(&buffers.sets[i])),
        upscale_height: [upscale_height(0), upscale_height(1)],
        upscale_rivers: [upscale_rivers(0), upscale_rivers(1)],
        upscale_masks: [upscale_masks(0), upscale_masks(1)],
        color: render_device.create_bind_group(
            None,
            &layout(&pipeline.layout_color),
            &BindGroupEntries::sequential((&view_color.texture_view,)),
        ),
        terrain_edit: render_device.create_bind_group(
            None,
            &layout(&pipeline.layout_terrain_edit),
            &BindGroupEntries::sequential((&view_terrain_edit.texture_view,)),
        ),
        water: [0, 1].map(|set| {
            let w = &buffers.water;
            render_device.create_bind_group(
                None,
                &layout(&pipeline.layout_water),
                &BindGroupEntries::sequential((
                    buffers.sets[set].height.binding().unwrap(),
                    w.depth.binding().unwrap(),
                    w.flux.binding().unwrap(),
                    w.sediment.binding().unwrap(),
                    w.sediment_tmp.binding().unwrap(),
                    w.velocity.binding().unwrap(),
                    buffers.sets[set].masks.binding().unwrap(),
                    buffers.sets[SET_SOURCE].masks.binding().unwrap(),
                    w.params.binding().unwrap(),
                )),
            )
        }),
        water_bed: [0, 1].map(|set| {
            let w = &buffers.water;
            render_device.create_bind_group(
                None,
                &layout(&pipeline.layout_water_bed),
                &BindGroupEntries::sequential((
                    buffers.sets[set].height.binding().unwrap(),
                    w.depth.binding().unwrap(),
                    discharge.clone(),
                    w.tau.binding().unwrap(),
                    w.bed_depth.binding().unwrap(),
                    w.prior.binding().unwrap(),
                    w.shore[0].binding().unwrap(),
                    w.shore[1].binding().unwrap(),
                    w.bed_params.binding().unwrap(),
                )),
            )
        }),
        water_morpho: [0, 1].map(|set| {
            let w = &buffers.water;
            let set = &buffers.sets[set];
            render_device.create_bind_group(
                None,
                &layout(&pipeline.layout_water_morpho),
                &BindGroupEntries::sequential((
                    set.height.binding().unwrap(),
                    w.depth.binding().unwrap(),
                    w.velocity.binding().unwrap(),
                    discharge.clone(),
                    set.m.binding().unwrap(),
                    w.bed_load.binding().unwrap(),
                    w.morpho_state.binding().unwrap(),
                    w.morpho_params.binding().unwrap(),
                )),
            )
        }),
        water_bed_load: [0, 1].map(|set| {
            let w = &buffers.water;
            let set = &buffers.sets[set];
            render_device.create_bind_group(
                None,
                &layout(&pipeline.layout_water_bed_load),
                &BindGroupEntries::with_indices((
                    (0, set.height.binding().unwrap()),
                    (1, w.depth.binding().unwrap()),
                    (2, w.velocity.binding().unwrap()),
                    (5, w.bed_load.binding().unwrap()),
                    (6, w.morpho_state.binding().unwrap()),
                    (7, w.morpho_params.binding().unwrap()),
                    (8, set.masks.binding().unwrap()),
                    (9, set.layers.binding().unwrap()),
                )),
            )
        }),
        water_splat: [0, 1].map(|set| {
            let w = &buffers.water;
            let set = &buffers.sets[set];
            render_device.create_bind_group(
                None,
                &layout(&pipeline.layout_water_splat),
                &BindGroupEntries::sequential((
                    buffers.splat.binding().unwrap(),
                    w.depth.binding().unwrap(),
                    w.sediment.binding().unwrap(),
                    set.height.binding().unwrap(),
                    set.deposition.binding().unwrap(),
                    set.erosion.binding().unwrap(),
                    w.morpho_state.binding().unwrap(),
                    w.params.binding().unwrap(),
                    &w.swe.textures.bed_water,
                )),
            )
        }),
        water_out: render_device.create_bind_group(
            None,
            &layout(&pipeline.layout_water_out),
            &BindGroupEntries::sequential((
                &view_water.texture_view,
                &buffers.water.swe.bed,
                &buffers.water.swe.source,
                &buffers.water.swe.textures.bed_water,
                &buffers.water.swe.textures.velocity,
            )),
        ),
        blit: render_device.create_bind_group(
            None,
            &layout(&pipeline.layout_blit),
            &BindGroupEntries::sequential((
                &view_height.texture_view,
                &view_normal.texture_view,
                &view_analysis.texture_view,
                &view_ao.texture_view,
            )),
        ),
        blit_blur: render_device.create_bind_group(
            None,
            &layout(&pipeline.layout_blit_blur),
            &BindGroupEntries::sequential((
                &view_height.texture_view,
                &view_normal.texture_view,
                &view_analysis.texture_view,
                &view_ao.texture_view,
                &view_ao_temp.texture_view,
            )),
        ),
    });
}

/// A GPU command for this frame, with its dynamic offset into [`ErosionBuffers::params`].
#[derive(Clone, Copy, Debug)]
enum FrameCmd {
    Init {
        offset: u32,
        size: UVec2,
    },
    Seed {
        dst: usize,
        offset: u32,
        size: UVec2,
    },
    Erode {
        set: usize,
        offset: u32,
        particles: u32,
    },
    WriteBack {
        set: usize,
        offset: u32,
        size: UVec2,
    },
    /// Water solver steps on the finest level, after removing all water when `clear`. With
    /// `swe`, bevy_fft's solver moves the water; `import` first hands it this crate's depth.
    /// `bed_load` runs Exner after every step; the discharge and helical flow follow once.
    Water {
        set: usize,
        steps: u32,
        clear: bool,
        swe: bool,
        import: bool,
        bed_load: bool,
        /// Raise KTT's river momentum to the water's discharge (`morpho_steer`).
        steer: bool,
    },
    /// Rebuild the beds under the current water (`water_bed.wgsl`); `offsets` per SOR parity.
    /// With `swe`, the new depth is handed to bevy_fft's solver afterwards.
    ShapeBeds {
        set: usize,
        offsets: [u32; 2],
        dilate_steps: u32,
        swe: bool,
    },
    Display {
        set: usize,
        offset: u32,
        /// Recompute ambient occlusion (see [`AO_INTERVAL`]).
        ao: bool,
    },
}

/// While the finest level erodes, AO (64 height samples per texel, as costly as an erode
/// dispatch) is refreshed only every this many displays, plus on the first and the last one.
const AO_INTERVAL: u32 = 8;

#[derive(Resource, Default)]
struct ErosionState {
    ready: bool,
    last_reset_gen: u32,
    last_step_seen: u64,
    cascade: Cascade,
    steps: Vec<CascadeStep>,
    frame: Vec<FrameCmd>,
    /// Iteration counter for rain brush dispatches (randomness and ridge/flow alternation).
    brush_iteration: u32,
    /// Some water exists, so the water solver runs every frame.
    water_active: bool,
    last_water_clear: u32,
    /// bevy_fft's solver held the water last frame.
    last_swe: bool,
    /// Frame time not yet simulated by the water solver (below one [`WATER_STEP`]).
    water_time: f32,
    last_shape: u32,
    last_refine: u32,
    /// Water steps still to run for the current Refine Beds request.
    refine_left: u32,
    /// Helical flow slot holding the latest intensity.
    helical_parity: u32,
    /// Cascade was finished last frame (AO refresh on the frame it finishes).
    was_finished: bool,
    /// Finest-level displays since AO was last computed (`None` before the first one).
    displays_since_ao: Option<u32>,
}

/// Turns the cascade's next steps into this frame's command list and per-dispatch uniforms.
fn plan_erosion_frame(
    pipeline: Res<ErosionPipeline>,
    cache: Res<PipelineCache>,
    reset: Res<ResetSim>,
    sim: Res<SimControl>,
    params: Res<ErodeParams>,
    stroke: Res<BrushStroke>,
    water: Res<WaterControl>,
    progress: Res<PyramidProgress>,
    mut buffers: ResMut<ErosionBuffers>,
    mut state: ResMut<ErosionState>,
    render_device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    swe_pipelines: Option<Res<ShallowWaterPipelines>>,
) {
    let state = &mut *state;
    state.frame.clear();
    state.steps.clear();

    if !state.ready {
        for id in pipeline.required() {
            match cache.get_compute_pipeline_state(id) {
                CachedPipelineState::Ok(_) => {}
                CachedPipelineState::Err(ShaderCacheError::ShaderNotLoaded(_)) => return,
                CachedPipelineState::Err(err) => panic!("Initializing erosion pipelines:\n{err}"),
                _ => return,
            }
        }
        state.ready = true;
        state.cascade.restart(params.pyramid_levels());
    }
    if reset.generation != state.last_reset_gen {
        state.last_reset_gen = reset.generation;
        state.cascade.restart(params.pyramid_levels());
        state.displays_since_ao = None;
        state.was_finished = false;
    }

    let budget = if sim.paused {
        let stepped = sim.step_counter > state.last_step_seen;
        state.last_step_seen = sim.step_counter;
        u32::from(stepped)
    } else {
        sim.iterations_per_frame.max(1)
    };
    state.cascade.advance(
        budget,
        params.compute_ridge_erosion != 0,
        sim.continue_after_cascade,
        &mut state.steps,
    );

    let buffers = &mut *buffers;
    buffers.params.clear();
    buffers.resample_params.clear();
    let base_level = PyramidLevel {
        size: params.map_size,
        voxel_size: 1.0,
        iterations: ERODE_ITERS_PER_WRITEBACK,
        trail_weight: 1.0,
    };
    let levels = &state.cascade.levels;
    let level_at = |i: usize| levels.get(i).copied().unwrap_or(base_level);
    let finest_size = levels.last().map_or(params.map_size, |l| l.size);
    let pad_offset = pyramid_padding(params.map_size, levels);
    let params = &ErodeParams {
        pad_offset,
        padded_size: finest_size,
        water_handoff: u32::from(state.water_active),
        ..params.clone()
    };
    for step in &state.steps {
        let cmd = match *step {
            CascadeStep::Init => FrameCmd::Init {
                offset: buffers
                    .params
                    .push(&level_erode_params(params, &base_level, 0)),
                size: params.map_size,
            },
            CascadeStep::Seed { src, dst, level } => {
                let first = level == 0;
                let dst_level = level_at(level);
                let src_size = if first {
                    dst_level.size
                } else {
                    level_at(level - 1).size
                };
                let offset = buffers.resample_params.push(&PyramidResampleParams {
                    src_size,
                    dst_size: dst_level.size,
                    orig_size: params.map_size,
                    first_level: u32::from(first),
                    dst_voxel_size: dst_level.voxel_size,
                    detail_smoothing: params.source_detail_smoothing,
                    outflow_accumulate: params.outflow_integration,
                    use_bedrock: params.use_bedrock,
                    detail_scale: params.detail_scale,
                    granularity_amount: params.granularity_amount,
                    granularity_scale: params.granularity_scale,
                    granularity_strength: params.granularity_strength,
                    padded_size: finest_size,
                    pad_offset,
                });
                debug_assert!(first || src == 1 - dst);
                info!(
                    "Erosion pyramid level {}/{}: {}×{}, {} iterations",
                    level + 1,
                    levels.len(),
                    dst_level.size.x,
                    dst_level.size.y,
                    dst_level.iterations
                );
                FrameCmd::Seed {
                    dst,
                    offset,
                    size: dst_level.size,
                }
            }
            CascadeStep::Erode {
                set,
                level,
                iteration,
            } => {
                let lp = level_erode_params(params, &level_at(level), iteration);
                FrameCmd::Erode {
                    set,
                    particles: lp.num_particles,
                    offset: buffers.params.push(&lp),
                }
            }
            CascadeStep::WriteBack {
                set,
                level,
                iteration,
                iterations,
            } => {
                let level = level_at(level);
                let lp = ErodeParams {
                    writeback_fraction: iterations as f32 / level.iterations as f32,
                    ..level_erode_params(params, &level, iteration)
                };
                FrameCmd::WriteBack {
                    set,
                    size: lp.map_size,
                    offset: buffers.params.push(&lp),
                }
            }
        };
        state.frame.push(cmd);
    }
    // Rain brush: extra dispatches on the finest level once the coarse levels are done.
    let finest_ready = state.ready && state.cascade.level + 1 == levels.len();
    if stroke.active && stroke.particles > 0 && finest_ready {
        let finest = level_at(state.cascade.level);
        for _ in 0..stroke.passes {
            let lp = ErodeParams {
                spawn_center: stroke.center + pad_offset.as_vec2(),
                spawn_radius: stroke.radius,
                spawn_count: stroke.particles,
                rain_water: water.rain_water,
                ..level_erode_params(params, &finest, state.brush_iteration)
            };
            state.brush_iteration = state.brush_iteration.wrapping_add(1);
            state.frame.push(FrameCmd::Erode {
                set: state.cascade.current,
                particles: stroke.particles,
                offset: buffers.params.push(&lp),
            });
        }
    }
    // Water bodies: the solver runs on the finest level whenever any water exists.
    if stroke.active && stroke.particles > 0 && water.rain_water > 0.0 {
        state.water_active = true;
    }
    let clear = water.clear_generation != state.last_water_clear;
    if clear {
        state.last_water_clear = water.clear_generation;
        state.water_active = false;
    }
    let stroke = water.stroke_active && water.rate != 0.0;
    if (water.stroke_active && water.rate > 0.0)
        || !water.springs.is_empty()
        || water.rainfall > 0.0
    {
        state.water_active = true;
    }
    // bevy_fft's solver, once its kernels have compiled; until then the pipes move the water.
    let grid = &mut buffers.water.swe;
    if water.solver == WaterSolver::Swe
        && grid.solver.is_none()
        && let Some(swe_pipelines) = &swe_pipelines
    {
        grid.solver = ShallowWaterSolver::new(
            &render_device,
            swe_pipelines,
            &cache,
            ShallowWaterSolverDescriptor {
                cells: params.map_size,
                textures: grid.textures.clone(),
                inputs: ShallowWaterInputs {
                    bed: Some(grid.bed.clone()),
                    source: Some(grid.source.clone()),
                },
                particle_count: 0,
            },
        );
    }
    let swe = water.solver == WaterSolver::Swe && grid.solver.is_some();
    if water.refine_generation != state.last_refine && finest_ready {
        state.last_refine = water.refine_generation;
        state.refine_left = water.bed_load.refine_steps;
        state.water_active = true;
    }
    let refining = state.refine_left > 0;
    if finest_ready && (clear || stroke || refining || (state.water_active && !sim.paused)) {
        let steps = if refining {
            // Fast-forward: a fixed number of steps per frame, whatever the frame time.
            let steps = state.refine_left.min(MAX_REFINE_STEPS);
            state.refine_left -= steps;
            steps
        } else if sim.paused {
            // One step so a brush stroke still shows.
            1
        } else {
            state.water_time += water.dt;
            let steps = ((state.water_time / WATER_STEP) as u32).min(MAX_WATER_STEPS);
            state.water_time = (state.water_time - steps as f32 * WATER_STEP).min(WATER_STEP);
            steps
        };
        let sea_level = if params.use_outflow != 0 {
            params.outflow_sea_level
        } else {
            0.0
        };
        let brush = stroke.then(|| Spring {
            center: water.center,
            radius: water.radius,
            rate: water.rate,
        });
        let mut sources = [Vec4::ZERO; MAX_WATER_SOURCES];
        let mut source_count = 0;
        for s in water.springs.iter().take(MAX_SPRINGS).chain(&brush) {
            sources[source_count] = s
                .center
                .extend(s.radius.max(0.5))
                .extend(s.rate * WATER_STEP);
            source_count += 1;
        }
        buffers.water.params.set(WaterParams {
            size: params.map_size,
            pad_offset,
            grid_size: finest_size,
            clear: u32::from(clear),
            evaporation: 1.0 - (1.0 - water.evaporation.clamp(0.0, 0.999)).powf(WATER_STEP),
            capacity: water.sediment_capacity,
            sea_level,
            rain: water.rainfall * WATER_STEP,
            step_seconds: WATER_STEP,
            bed_load: u32::from(water.bed_load.live || refining),
            source_count: source_count as u32,
            sources,
        });
        buffers.water.params.write_buffer(&render_device, &queue);
        let bed_load = &water.bed_load;
        let mor_fac = if refining {
            bed_load.morph_factor.max(0.0)
        } else if bed_load.live {
            1.0
        } else {
            0.0
        };
        buffers.water.morpho_params.set(MorphoParams {
            size: params.map_size,
            pad_offset,
            grid_size: finest_size,
            step_seconds: WATER_STEP,
            frame_seconds: steps as f32 * WATER_STEP,
            discharge_time: DISCHARGE_TIME,
            ktt_scale: KTT_DISCHARGE_SCALE,
            ktt_vector: u32::from(params.do_rivers != 0 && params.momentum_coherence > 0.0),
            sea_level,
            manning_n: water.manning_n,
            grain: bed_load.grain,
            bed_load_k: bed_load.k,
            critical_shields: bed_load.critical_shields,
            slope_effect: bed_load.slope_effect,
            helical_strength: bed_load.helical_strength,
            helical_length: HELICAL_LENGTH,
            mor_fac,
            parity: state.helical_parity,
            use_bedrock: params.use_bedrock,
            rock_hardness: bed_load.rock_hardness,
        });
        buffers
            .water
            .morpho_params
            .write_buffer(&render_device, &queue);
        if swe && let Some(solver) = &buffers.water.swe.solver {
            // Sources, sinks and the map border come in through `swe_export`, so the borders
            // stay closed walls. The overshoot limiter is off: it does not conserve water.
            solver.write_settings(
                &queue,
                &ShallowWaterSettings {
                    dt: WATER_STEP,
                    cell_size: 1.0,
                    gravity: water.gravity.max(1.0),
                    manning_n: water.manning_n,
                    evaporation: -(1.0 - water.evaporation.clamp(0.0, 0.999)).ln(),
                    overshoot_limiter: false,
                    second_order_depth: water.second_order_depth,
                    ..default()
                },
            );
        }
        if clear || steps > 0 {
            state.frame.push(FrameCmd::Water {
                set: state.cascade.current,
                steps,
                clear,
                swe,
                import: swe && !state.last_swe && !clear,
                bed_load: mor_fac > 0.0,
                steer: water.steer_rivers && params.do_rivers != 0,
            });
            state.last_swe = swe;
            if steps > 0 {
                state.helical_parity ^= 1;
            }
        }
    } else {
        state.water_time = 0.0;
    }
    // Bed reconstruction: one shot per request, once the finest level holds the terrain.
    buffers.water.bed_params.clear();
    let bed = &water.bed;
    let bed_params = |parity| BedParams {
        size: params.map_size,
        pad_offset,
        grid_size: finest_size,
        parity,
        omega: BED_OMEGA,
        max_depth: bed.max_depth,
        shore_slope: bed.shore_slope,
        profile: bed.profile,
        strength: bed.strength,
        bank_width: bed.bank_width.clamp(1.0, MAX_BANK_WIDTH),
        bank_slope: bed.bank_slope,
        river_depth: bed.river_depth,
        river_width: bed.river_width,
        channel_discharge: bed.channel_discharge,
    };
    let offsets = [0, 1].map(|parity| buffers.water.bed_params.push(&bed_params(parity)));
    if water.shape_generation != state.last_shape && finest_ready {
        state.last_shape = water.shape_generation;
        state.water_active = true;
        state.frame.push(FrameCmd::ShapeBeds {
            set: state.cascade.current,
            offsets,
            // Even, so the result ends up back in the first shore buffer.
            dilate_steps: 2 * (bed.bank_width.max(BED_BANK_REACH) / 2.0).ceil() as u32,
            swe,
        });
        // The pipes' depth now differs from the solver's: hand it over when it takes back over.
        state.last_swe &= swe;
    }
    buffers
        .water
        .bed_params
        .write_buffer(&render_device, &queue);
    let level = level_at(state.cascade.level);
    let finest = level.size == finest_size;
    // Only the finest level is drawn: while a restart runs the coarse levels, the screen keeps the
    // last finest result.
    if !state.frame.is_empty() && finest {
        // Paused steps and the frame the cascade finishes are the last display for a while:
        // refresh AO. (Later frames still display while rain or water keeps changing heights.)
        let just_finished = state.cascade.finished && !state.was_finished;
        let last = just_finished || sim.paused;
        let ao = last || state.displays_since_ao.is_none_or(|n| n + 1 >= AO_INTERVAL);
        state.displays_since_ao = if ao {
            Some(0)
        } else {
            state.displays_since_ao.map(|n| n + 1)
        };
        state.was_finished = state.cascade.finished;
        state.frame.push(FrameCmd::Display {
            set: state.cascade.current,
            offset: buffers.params.push(&level_erode_params(params, &level, 0)),
            ao,
        });
    } else if buffers.params.is_empty() {
        // Keep a valid binding around for bind group creation.
        buffers.params.push(params);
    }
    if buffers.resample_params.is_empty() {
        buffers.resample_params.push(&PyramidResampleParams {
            src_size: UVec2::ONE,
            dst_size: UVec2::ONE,
            orig_size: UVec2::ONE,
            first_level: 1,
            dst_voxel_size: 1.0,
            detail_smoothing: 0.0,
            outflow_accumulate: 0,
            use_bedrock: 0,
            detail_scale: 1.0,
            granularity_amount: 0.0,
            granularity_scale: 0.0,
            granularity_strength: 0.0,
            padded_size: UVec2::ONE,
            pad_offset: UVec2::ZERO,
        });
    }
    buffers.params.write_buffer(&render_device, &queue);
    buffers.resample_params.write_buffer(&render_device, &queue);

    let cascade = &state.cascade;
    if let Some(level) = cascade.current_level() {
        *progress.0.lock().unwrap() = PyramidProgressData {
            level: cascade.level,
            level_count: cascade.levels.len(),
            size: level.size,
            batch_iter: cascade.batch_iter,
            batch_len: level.iterations,
            finished: cascade.finished,
        };
    }
}

fn dispatch_grid(size: UVec2) -> (u32, u32) {
    (size.x.div_ceil(8), size.y.div_ceil(8))
}

fn erosion_compute(
    mut render_context: RenderContext,
    cache: Res<PipelineCache>,
    pipes: Res<ErosionPipeline>,
    groups: Option<Res<ErosionBindGroups>>,
    params: Res<ErodeParams>,
    state: Res<ErosionState>,
    buffers: Option<Res<ErosionBuffers>>,
) {
    let Some(groups) = groups else { return };
    let solver = buffers.as_ref().and_then(|b| b.water.swe.solver.as_ref());
    if state.frame.is_empty() {
        return;
    }
    let get = |id| cache.get_compute_pipeline(id);
    let mut pass = render_context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("erosion"),
            ..default()
        });

    for cmd in &state.frame {
        match *cmd {
            FrameCmd::Init { offset, size } => {
                let (gx, gy) = dispatch_grid(size);
                let source = &groups.erosion[SET_SOURCE];
                if let Some(p) = get(pipes.pipeline_init) {
                    pass.set_pipeline(p);
                    pass.set_bind_group(0, source, &[offset]);
                    pass.set_bind_group(1, &groups.terrain_edit, &[]);
                    pass.dispatch_workgroups(gx, gy, 1);
                }
                if let Some(p) = get(pipes.pipeline_init_color) {
                    pass.set_pipeline(p);
                    pass.set_bind_group(0, source, &[offset]);
                    pass.set_bind_group(1, &groups.color, &[]);
                    pass.dispatch_workgroups(gx, gy, 1);
                }
            }
            FrameCmd::Seed { dst, offset, size } => {
                let (gx, gy) = dispatch_grid(size);
                // Masks first: the height pass reads this level's outflow and bedrock.
                if let Some(p) = get(pipes.pipeline_upscale_masks) {
                    pass.set_pipeline(p);
                    pass.set_bind_group(0, &groups.upscale_masks[dst], &[offset]);
                    pass.dispatch_workgroups(gx, gy, 1);
                }
                if let Some(p) = get(pipes.pipeline_upscale_height) {
                    pass.set_pipeline(p);
                    pass.set_bind_group(0, &groups.upscale_height[dst], &[offset]);
                    pass.dispatch_workgroups(gx, gy, 1);
                }
                if let Some(p) = get(pipes.pipeline_upscale_rivers) {
                    pass.set_pipeline(p);
                    pass.set_bind_group(0, &groups.upscale_rivers[dst], &[offset]);
                    pass.dispatch_workgroups(gx, gy, 1);
                }
            }
            FrameCmd::Erode {
                set,
                offset,
                particles,
            } => {
                if let Some(p) = get(pipes.pipeline_erode) {
                    pass.set_pipeline(p);
                    pass.set_bind_group(0, &groups.erosion[set], &[offset]);
                    pass.set_bind_group(1, &groups.color, &[]);
                    pass.dispatch_workgroups(particles.div_ceil(256), 1, 1);
                }
            }
            FrameCmd::WriteBack { set, offset, size } => {
                if let Some(p) = get(pipes.pipeline_write_back) {
                    let (gx, gy) = dispatch_grid(size);
                    pass.set_pipeline(p);
                    pass.set_bind_group(0, &groups.erosion[set], &[offset]);
                    pass.dispatch_workgroups(gx, gy, 1);
                }
            }
            FrameCmd::Water {
                set,
                steps,
                clear,
                swe,
                import,
                bed_load,
                steer,
            } => {
                let (gx, gy) = dispatch_grid(params.map_size);
                let [p_discharge, p_helical, p_bed_flux, p_exner, p_steer] =
                    pipes.pipelines_water_morpho.map(get);
                let [p_splat, p_record] = pipes.pipelines_water_splat.map(get);
                if let Some(p) = p_splat {
                    pass.set_bind_group(0, &groups.water_splat[set], &[]);
                    pass.set_pipeline(p);
                    pass.dispatch_workgroups(gx, gy, 1);
                }
                let [
                    p_clear,
                    p_flux,
                    p_depth,
                    p_erode,
                    p_advect,
                    p_copy,
                    p_export,
                    p_gather,
                    p_import,
                    p_apply,
                ] = pipes.pipelines_water.map(get);
                let (
                    Some(p_flux),
                    Some(p_depth),
                    Some(p_erode),
                    Some(p_apply),
                    Some(p_advect),
                    Some(p_copy),
                ) = (p_flux, p_depth, p_erode, p_apply, p_advect, p_copy)
                else {
                    continue;
                };
                // The solver sets its own bind groups, so ours are set again after it.
                let run = |pass: &mut ComputePass, p| {
                    pass.set_bind_group(0, &groups.water[set], &[]);
                    pass.set_bind_group(1, &groups.water_out, &[]);
                    pass.set_pipeline(p);
                    pass.dispatch_workgroups(gx, gy, 1);
                };
                let swe = match (swe, solver, p_export, p_gather, p_import) {
                    (true, Some(solver), Some(e), Some(g), Some(i)) => Some((solver, e, g, i)),
                    _ => None,
                };
                if clear && let Some(p) = p_clear {
                    run(&mut pass, p);
                    if let Some((solver, ..)) = swe {
                        solver.encode_clear(&mut pass);
                    }
                }
                if import && let Some((_, _, _, p_import)) = swe {
                    run(&mut pass, p_import);
                }
                for _ in 0..steps {
                    if let Some((solver, p_export, p_gather, _)) = swe {
                        run(&mut pass, p_export);
                        solver.encode_sync_bed(&mut pass);
                        solver.encode_step(&mut pass);
                        solver.encode_reconstruct_velocity(&mut pass);
                        run(&mut pass, p_gather);
                    } else {
                        run(&mut pass, p_flux);
                        run(&mut pass, p_depth);
                    }
                    for p in [p_erode, p_apply, p_advect, p_copy] {
                        run(&mut pass, p);
                    }
                    if bed_load && let (Some(f), Some(e)) = (p_bed_flux, p_exner) {
                        pass.set_bind_group(0, &groups.water_bed_load[set], &[]);
                        for p in [f, e] {
                            pass.set_pipeline(p);
                            pass.dispatch_workgroups(gx, gy, 1);
                        }
                    }
                }
                if steps > 0 && let (Some(d), Some(h)) = (p_discharge, p_helical) {
                    pass.set_bind_group(0, &groups.water_morpho[set], &[]);
                    for p in [d, h] {
                        pass.set_pipeline(p);
                        pass.dispatch_workgroups(gx, gy, 1);
                    }
                    if steer && let Some(p) = p_steer {
                        pass.set_pipeline(p);
                        pass.dispatch_workgroups(gx, gy, 1);
                    }
                }
                if let Some(p) = p_record {
                    pass.set_bind_group(0, &groups.water_splat[set], &[]);
                    pass.set_pipeline(p);
                    pass.dispatch_workgroups(gx, gy, 1);
                }
            }
            FrameCmd::ShapeBeds {
                set,
                offsets,
                dilate_steps,
                swe,
            } => {
                let (gx, gy) = dispatch_grid(params.map_size);
                let pipes_bed = pipes.pipelines_water_bed.map(get);
                let [
                    Some(init),
                    Some(tau),
                    Some(prior),
                    Some(smooth),
                    Some(write),
                    Some(dilate),
                    Some(grade),
                ] = pipes_bed
                else {
                    continue;
                };
                let group = &groups.water_bed[set];
                let mut run = |p, parity: usize| {
                    pass.set_pipeline(p);
                    pass.set_bind_group(0, group, &[offsets[parity]]);
                    pass.dispatch_workgroups(gx, gy, 1);
                };
                run(init, 0);
                for _ in 0..BED_TAU_SWEEPS {
                    run(tau, 0);
                    run(tau, 1);
                }
                for k in 0..dilate_steps {
                    run(dilate, (k % 2) as usize);
                }
                run(prior, 0);
                for _ in 0..BED_SMOOTH_SWEEPS {
                    run(smooth, 0);
                    run(smooth, 1);
                }
                run(write, 0);
                run(grade, 0);
                // The shaped depth keeps the surface in place; the solver takes it over.
                if swe && let Some(p) = get(pipes.pipelines_water[8]) {
                    pass.set_pipeline(p);
                    pass.set_bind_group(0, &groups.water[set], &[]);
                    pass.set_bind_group(1, &groups.water_out, &[]);
                    pass.dispatch_workgroups(gx, gy, 1);
                }
            }
            FrameCmd::Display { set, offset, ao } => {
                let (gx, gy) = dispatch_grid(params.map_size);
                let erosion = &groups.erosion[set];
                let passes = [
                    (pipes.pipeline_blit, &groups.blit),
                    (pipes.pipeline_ao, &groups.blit),
                    (pipes.pipeline_blur_h, &groups.blit_blur),
                    (pipes.pipeline_blur_v, &groups.blit_blur),
                ];
                for &(id, group1) in &passes[..if ao { 4 } else { 1 }] {
                    if let Some(p) = get(id) {
                        pass.set_pipeline(p);
                        pass.set_bind_group(0, erosion, &[offset]);
                        pass.set_bind_group(1, group1, &[]);
                        pass.dispatch_workgroups(gx, gy, 1);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_levels_match_ktt() {
        // Feature_Size 8, voxel 1 → Iterations = log2(8) + 1 = 4 → coarsest 512/16, erode 64..512.
        let levels = pyramid_levels(UVec2::splat(512), 8.0, 1.0, 0.0);
        let sizes: Vec<u32> = levels.iter().map(|l| l.size.x).collect();
        assert_eq!(sizes, vec![64, 128, 256, 512]);
        assert_eq!(levels.last().unwrap().voxel_size, 1.0);
        assert_eq!(levels[0].voxel_size, 8.0);
        assert!(
            levels
                .iter()
                .all(|l| l.iterations == 200 && l.trail_weight == 1.0)
        );
    }

    #[test]
    fn fractional_level_count_fades_first_pass() {
        // Feature_Size 12 → Iterations = log2(12) + 1 ≈ 4.585 → 5 passes, first weighted frac².
        let levels = pyramid_levels(UVec2::splat(512), 12.0, 1.0, 0.0);
        assert_eq!(levels.len(), 5);
        assert_eq!(levels[0].size, UVec2::splat(32));
        let frac = 12f32.log2() + 1.0 - 4.0;
        assert!((levels[0].trail_weight - frac * frac).abs() < 1e-5);
        assert!(levels[1..].iter().all(|l| l.trail_weight == 1.0));
    }

    #[test]
    fn geological_age_scales_coarse_batches() {
        let levels = pyramid_levels(UVec2::splat(512), 8.0, 1.0, 10.0);
        let iters: Vec<u32> = levels.iter().map(|l| l.iterations).collect();
        assert_eq!(iters, vec![1600, 800, 400, 200]);
    }

    #[test]
    fn level_count_is_capped_on_small_maps() {
        let levels = pyramid_levels(UVec2::splat(16), 1024.0, 1.0, 0.0);
        assert_eq!(levels.len(), 3);
        assert_eq!(levels[0].size, UVec2::splat(PYRAMID_MIN_SIZE));
        assert_eq!(levels.last().unwrap().size, UVec2::splat(16));
        assert_eq!(levels[0].trail_weight, 1.0);
    }

    #[test]
    fn non_power_of_two_sizes_are_padded() {
        // 4 steps: padded to a multiple of 16, so each level is an exact halving.
        let levels = pyramid_levels(UVec2::new(500, 300), 8.0, 1.0, 0.0);
        let sizes: Vec<UVec2> = levels.iter().map(|l| l.size).collect();
        assert_eq!(
            sizes,
            [
                UVec2::new(64, 38),
                UVec2::new(128, 76),
                UVec2::new(256, 152),
                UVec2::new(512, 304)
            ]
        );
        assert_eq!(levels[0].voxel_size, 8.0);
        assert_eq!(levels[3].voxel_size, 1.0);
        assert_eq!(
            pyramid_padding(UVec2::new(500, 300), &levels),
            UVec2::new(6, 2)
        );
        assert!(
            pyramid_capacity(UVec2::new(500, 300))
                .cmpge(UVec2::new(512, 304))
                .all()
        );
    }

    #[test]
    fn power_of_two_sizes_are_not_padded() {
        let levels = pyramid_levels(UVec2::splat(512), 8.0, 1.0, 0.0);
        assert_eq!(levels.last().unwrap().size, UVec2::splat(512));
        assert_eq!(pyramid_padding(UVec2::splat(512), &levels), UVec2::ZERO);
        assert_eq!(pyramid_capacity(UVec2::splat(512)), UVec2::splat(512));
    }

    #[test]
    fn single_level_when_feature_size_is_tiny() {
        let levels = pyramid_levels(UVec2::splat(256), 0.5, 1.0, 0.0);
        assert_eq!(levels.len(), 1);
        assert_eq!(levels[0].size, UVec2::splat(256));
    }

    fn run_to_end(cascade: &mut Cascade, budget: u32, ridge: bool) -> Vec<CascadeStep> {
        let mut all = Vec::new();
        for _ in 0..100_000 {
            if cascade.finished {
                break;
            }
            cascade.advance(budget, ridge, false, &mut all);
        }
        assert!(cascade.finished);
        all
    }

    #[test]
    fn cascade_runs_every_level_then_stops() {
        let mut cascade = Cascade::default();
        cascade.restart(pyramid_levels(UVec2::splat(512), 8.0, 1.0, 0.0));
        let steps = run_to_end(&mut cascade, 7, true);

        assert_eq!(steps[0], CascadeStep::Init);
        assert_eq!(
            steps[1],
            CascadeStep::Seed {
                src: 1,
                dst: 0,
                level: 0
            }
        );
        let erodes = steps
            .iter()
            .filter(|s| matches!(s, CascadeStep::Erode { .. }))
            .count();
        assert_eq!(erodes, 4 * 200);
        let writebacks: Vec<_> = steps
            .iter()
            .filter_map(|s| match s {
                CascadeStep::WriteBack {
                    level, iteration, ..
                } => Some((*level, *iteration)),
                _ => None,
            })
            .collect();
        // Ten writeBacks per level, each after a flow-biased (odd) pass.
        let expected: Vec<_> = (0..4)
            .flat_map(|level| (1..=10).map(move |k| (level, k * WRITEBACK_INTERVAL - 1)))
            .collect();
        assert_eq!(writebacks, expected);
        let seeds: Vec<_> = steps
            .iter()
            .filter_map(|s| match s {
                CascadeStep::Seed { src, dst, level } => Some((*src, *dst, *level)),
                _ => None,
            })
            .collect();
        assert_eq!(seeds, vec![(1, 0, 0), (0, 1, 1), (1, 0, 2), (0, 1, 3)]);
        assert_eq!(cascade.current, 1);

        // Each level's dispatches restart at iteration 0 and target the level's work set.
        let level1: Vec<_> = steps
            .iter()
            .filter_map(|s| match s {
                CascadeStep::Erode {
                    set,
                    level: 1,
                    iteration,
                } => Some((*set, *iteration)),
                _ => None,
            })
            .collect();
        assert_eq!(level1.first(), Some(&(1, 0)));
        assert_eq!(level1.last(), Some(&(1, 199)));
        assert_eq!(level1.len(), 200);

        let mut more = Vec::new();
        cascade.advance(10, true, false, &mut more);
        assert!(more.is_empty());
    }

    #[test]
    fn budget_spreads_every_level() {
        let mut cascade = Cascade::default();
        let levels = pyramid_levels(UVec2::splat(512), 8.0, 1.0, 0.0);
        cascade.restart(levels.clone());
        let work = |steps: &[CascadeStep]| -> f32 {
            steps
                .iter()
                .filter_map(|s| match s {
                    CascadeStep::Erode { level, .. } => Some(levels[*level].pass_cost()),
                    _ => None,
                })
                .sum()
        };

        // Paused: only init and the first seed.
        let mut paused = Vec::new();
        cascade.advance(0, true, false, &mut paused);
        assert_eq!(paused.len(), 2);

        // Every frame does about `budget` finest passes of work, coarse levels included, until
        // the finest level runs exactly `budget` passes a frame.
        let mut frames = 0;
        while cascade.level + 1 < levels.len() {
            let mut steps = Vec::new();
            cascade.advance(7, true, false, &mut steps);
            assert!(work(&steps) < 7.0 + 1.0, "frame {frames} did {} passes of work", work(&steps));
            frames += 1;
        }
        assert!(frames > 1, "the coarse levels ran in one frame");
        let mut steps = Vec::new();
        cascade.advance(7, true, false, &mut steps);
        assert_eq!(work(&steps), 7.0);
    }

    #[test]
    fn ridge_off_skips_even_iterations() {
        let mut cascade = Cascade::default();
        cascade.restart(pyramid_levels(UVec2::splat(64), 1.0, 1.0, 0.0));
        let steps = run_to_end(&mut cascade, 16, false);
        assert!(steps.iter().all(|s| match s {
            CascadeStep::Erode { iteration, .. } => iteration % 2 == 1,
            _ => true,
        }));
        let erodes = steps
            .iter()
            .filter(|s| matches!(s, CascadeStep::Erode { .. }))
            .count();
        assert_eq!(erodes, 100);
    }

    #[test]
    fn continue_after_cascade_keeps_batching_finest_level() {
        let mut cascade = Cascade::default();
        cascade.restart(pyramid_levels(UVec2::splat(64), 1.0, 1.0, 0.0));
        let mut steps = Vec::new();
        for _ in 0..100 {
            cascade.advance(10, true, true, &mut steps);
        }
        assert!(!cascade.finished);
        let writebacks: Vec<u32> = steps
            .iter()
            .filter_map(|s| match s {
                CascadeStep::WriteBack { iteration, .. } => Some(*iteration),
                _ => None,
            })
            .collect();
        let expected: Vec<u32> = (1..=50).map(|k| k * WRITEBACK_INTERVAL - 1).collect();
        assert_eq!(writebacks, expected);
    }
}
