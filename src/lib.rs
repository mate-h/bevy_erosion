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
        texture::GpuImage,
    },
    shader::{ShaderCacheError, ShaderRef},
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
/// Erode dispatches per `writeBack` at geological age 0 (`Compute_Erosion.iterations`).
const ERODE_ITERS_PER_WRITEBACK: u32 = 200;
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

#[derive(Clone, Copy, ShaderType)]
pub struct BlitDisplayParams {
    pub display_size: UVec2,
    pub _pad: UVec2,
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
    /// Coarse levels are cheap and run to completion in one call; `budget` only limits the erosion
    /// iterations of the finest level. A `budget` of 0 (paused) emits nothing past init.
    pub fn advance(
        &mut self,
        mut budget: u32,
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
        // Only finest-level iterations consume the budget, so it reaches 0 there (or starts at 0).
        while budget > 0 && !self.finished {
            let finest = self.level + 1 == self.levels.len();
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
            if finest {
                budget -= 1;
            }
            if self.batch_iter < level.iterations {
                continue;
            }
            out.push(CascadeStep::WriteBack {
                set: self.current,
                level: self.level,
                iteration: iteration,
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
            feature_size: 8.0,            // Feature_Size
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
#[derive(Resource, Clone, ExtractResource, Debug)]
#[extract_app(RenderApp)]
pub struct WaterControl {
    /// Add (rate > 0) or remove (rate < 0) water in a disc this frame.
    pub stroke_active: bool,
    /// Disc center in map texels.
    pub center: Vec2,
    /// Disc radius in map texels.
    pub radius: f32,
    /// Water depth added per frame at the disc center (height units).
    pub rate: f32,
    /// Bump to remove all water.
    pub clear_generation: u32,
    /// Fraction of water lost per substep.
    pub evaporation: f32,
    /// Sediment carrying capacity of flowing water (0 = water never erodes).
    pub sediment_capacity: f32,
    /// Solver substeps per frame.
    pub substeps: u32,
}

impl Default for WaterControl {
    fn default() -> Self {
        Self {
            stroke_active: false,
            center: Vec2::ZERO,
            radius: 0.0,
            rate: 0.0,
            clear_generation: 0,
            evaporation: 0.00002,
            sediment_capacity: 0.0005,
            substeps: 8,
        }
    }
}

/// Uniform for `water.wgsl`.
#[derive(Clone, Copy, ShaderType)]
pub struct WaterParams {
    /// Base map size (the water grid).
    pub size: UVec2,
    /// Base map offset inside the padded finest level (height and masks grid).
    pub pad_offset: UVec2,
    pub grid_size: UVec2,
    pub brush_center: Vec2,
    pub brush_radius: f32,
    pub brush_rate: f32,
    pub clear: u32,
    pub evaporation: f32,
    pub capacity: f32,
    /// Ocean level (0 = no ocean): terrain below it drains water away.
    pub sea_level: f32,
}

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
) {
    let size = config.map_size;
    commands.insert_resource(create_erosion_images(&mut images, size));
    commands.insert_resource(config.to_erode_params(0));
    commands.insert_resource(ResetSim::default());
    commands.insert_resource(SimControl::default());
    commands.insert_resource(BrushStroke::default());
    commands.insert_resource(WaterControl::default());
}

impl Plugin for ErosionComputePlugin {
    fn build(&self, app: &mut App) {
        let progress = PyramidProgress::default();
        app.init_resource::<ErosionConfig>()
            .insert_resource(progress.clone())
            .add_plugins((
                ExtractResourcePlugin::<ErosionImages>::default(),
                ExtractResourcePlugin::<ErodeParams>::default(),
                ExtractResourcePlugin::<ResetSim>::default(),
                ExtractResourcePlugin::<SimControl>::default(),
                ExtractResourcePlugin::<BrushStroke>::default(),
                ExtractResourcePlugin::<WaterControl>::default(),
            ))
            .add_systems(Startup, setup_erosion_resources);

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
    layout_blit: BindGroupLayoutDescriptor,
    layout_blit_blur: BindGroupLayoutDescriptor,
    layout_blit_level: BindGroupLayoutDescriptor,
    layout_upscale_height: BindGroupLayoutDescriptor,
    layout_upscale_rivers: BindGroupLayoutDescriptor,
    layout_upscale_masks: BindGroupLayoutDescriptor,
    pipeline_init: CachedComputePipelineId,
    pipeline_init_color: CachedComputePipelineId,
    pipeline_erode: CachedComputePipelineId,
    pipeline_write_back: CachedComputePipelineId,
    pipeline_blit: CachedComputePipelineId,
    pipeline_blit_level: CachedComputePipelineId,
    pipeline_upscale_height: CachedComputePipelineId,
    pipeline_upscale_rivers: CachedComputePipelineId,
    pipeline_upscale_masks: CachedComputePipelineId,
    pipeline_ao: CachedComputePipelineId,
    pipeline_blur_h: CachedComputePipelineId,
    pipeline_blur_v: CachedComputePipelineId,
    /// `water.wgsl`: source/clear, flux, depth, erode, advect, copy.
    pipelines_water: [CachedComputePipelineId; 6],
}

impl ErosionPipeline {
    /// Pipelines the cascade needs before it can start.
    fn required(&self) -> [CachedComputePipelineId; 9] {
        [
            self.pipeline_init,
            self.pipeline_init_color,
            self.pipeline_erode,
            self.pipeline_write_back,
            self.pipeline_blit,
            self.pipeline_blit_level,
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
                storage_buffer::<f32>(false), // mx (f32 bits in atomic<u32>)
                storage_buffer::<f32>(false), // my (f32 bits in atomic<u32>)
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

    let layout_blit_level = BindGroupLayoutDescriptor::new(
        "ErosionBlitLevelGroup2",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (uniform_buffer::<BlitDisplayParams>(false),),
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
                storage_buffer_read_only::<f32>(false), // src mx
                storage_buffer_read_only::<f32>(false), // src my
                storage_buffer::<f32>(false),           // dst mx
                storage_buffer::<f32>(false),           // dst my
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
    let layout_water_out = BindGroupLayoutDescriptor::new(
        "WaterGroup1",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (texture_storage_2d(
                TextureFormat::Rgba16Float,
                StorageTextureAccess::WriteOnly,
            ),),
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
    let pipeline_blit_level = compute(
        vec![
            layout_erosion.clone(),
            layout_blit.clone(),
            layout_blit_level.clone(),
        ],
        &shader,
        "blit_level_to_display",
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
        "water_source",
        "water_flux",
        "water_depth",
        "water_erode",
        "water_advect",
        "water_copy",
    ]
    .map(|entry| {
        compute(
            vec![layout_water.clone(), layout_water_out.clone()],
            &water_shader,
            entry,
        )
    });

    commands.insert_resource(ErosionPipeline {
        layout_erosion,
        layout_color,
        layout_terrain_edit,
        layout_water,
        layout_water_out,
        layout_blit,
        layout_blit_blur,
        layout_blit_level,
        layout_upscale_height,
        layout_upscale_rivers,
        layout_upscale_masks,
        pipeline_init,
        pipeline_init_color,
        pipeline_erode,
        pipeline_write_back,
        pipeline_blit,
        pipeline_blit_level,
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
    mx: StorageBuffer<Vec<f32>>,
    my: StorageBuffer<Vec<f32>>,
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
            mx: StorageBuffer::from(vec![0f32; texels]),
            my: StorageBuffer::from(vec![0f32; texels]),
            masks: StorageBuffer::from(vec![Vec4::new(1.0, 0.0, 1.0, 1.0); texels]),
            layers: StorageBuffer::from(vec![Vec2::ZERO; texels]),
            outflow_acc: StorageBuffer::from(vec![0f32; texels]),
        };
        set.height.write_buffer(device, queue);
        set.flow.write_buffer(device, queue);
        set.deposition.write_buffer(device, queue);
        set.erosion.write_buffer(device, queue);
        set.mx.write_buffer(device, queue);
        set.my.write_buffer(device, queue);
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
    blit_display: UniformBuffer<BlitDisplayParams>,
    water: WaterBuffers,
}

/// Water solver state at base map size (`water.wgsl`).
struct WaterBuffers {
    depth: StorageBuffer<Vec<f32>>,
    flux: StorageBuffer<Vec<Vec4>>,
    sediment: StorageBuffer<Vec<f32>>,
    sediment_tmp: StorageBuffer<Vec<f32>>,
    velocity: StorageBuffer<Vec<Vec2>>,
    params: UniformBuffer<WaterParams>,
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
                brush_center: Vec2::ZERO,
                brush_radius: 0.0,
                brush_rate: 0.0,
                clear: 0,
                evaporation: 0.0,
                capacity: 0.0,
                sea_level: 0.0,
            }),
        };
        b.depth.write_buffer(device, queue);
        b.flux.write_buffer(device, queue);
        b.sediment.write_buffer(device, queue);
        b.sediment_tmp.write_buffer(device, queue);
        b.velocity.write_buffer(device, queue);
        b.params.write_buffer(device, queue);
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
    level_blit: BindGroup,
    color: BindGroup,
    terrain_edit: BindGroup,
    /// Water solver, indexed by work set.
    water: [BindGroup; 2],
    water_out: BindGroup,
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
    let mut blit_display = UniformBuffer::from(BlitDisplayParams {
        display_size: params.map_size,
        _pad: UVec2::ZERO,
    });
    blit_display.write_buffer(&render_device, &queue);
    commands.insert_resource(ErosionBuffers {
        texels,
        sets: std::array::from_fn(|_| FieldSet::new(texels, &render_device, &queue)),
        params: DynamicUniformBuffer::default(),
        resample_params: DynamicUniformBuffer::default(),
        blit_display,
        water: WaterBuffers::new(params.map_size, &render_device, &queue),
    });
}

fn prepare_erosion_bind_groups(
    mut commands: Commands,
    pipeline: Res<ErosionPipeline>,
    pipeline_cache: Res<PipelineCache>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    images: Res<ErosionImages>,
    buffers: Option<Res<ErosionBuffers>>,
    render_device: Res<RenderDevice>,
) {
    let Some(buffers) = buffers else { return };
    if buffers.params.buffer().is_none() || buffers.resample_params.buffer().is_none() {
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
                &set.mx,
                &set.my,
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
                &src.mx,
                &src.my,
                &dst.mx,
                &dst.my,
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
            )),
        )
    };

    commands.insert_resource(ErosionBindGroups {
        erosion: std::array::from_fn(|i| erosion(&buffers.sets[i])),
        upscale_height: [upscale_height(0), upscale_height(1)],
        upscale_rivers: [upscale_rivers(0), upscale_rivers(1)],
        upscale_masks: [upscale_masks(0), upscale_masks(1)],
        level_blit: render_device.create_bind_group(
            None,
            &layout(&pipeline.layout_blit_level),
            &BindGroupEntries::sequential((&buffers.blit_display,)),
        ),
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
        water_out: render_device.create_bind_group(
            None,
            &layout(&pipeline.layout_water_out),
            &BindGroupEntries::sequential((&view_water.texture_view,)),
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
    /// Water solver substeps on the finest level (plus the brush / clear pass when `source`).
    Water {
        set: usize,
        substeps: u32,
        source: bool,
    },
    Display {
        set: usize,
        offset: u32,
        finest: bool,
        /// Recompute ambient occlusion (finest level only; see [`AO_INTERVAL`]).
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
            } => {
                let lp = level_erode_params(params, &level_at(level), iteration);
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
    let clear = water.clear_generation != state.last_water_clear;
    if clear {
        state.last_water_clear = water.clear_generation;
        state.water_active = false;
    }
    let source = clear || (water.stroke_active && water.rate != 0.0);
    if water.stroke_active && water.rate > 0.0 {
        state.water_active = true;
    }
    if finest_ready && (source || (state.water_active && !sim.paused)) {
        let sea_level = if params.use_outflow != 0 {
            params.outflow_sea_level
        } else {
            0.0
        };
        buffers.water.params.set(WaterParams {
            size: params.map_size,
            pad_offset,
            grid_size: finest_size,
            brush_center: water.center,
            brush_radius: water.radius,
            brush_rate: if water.stroke_active { water.rate } else { 0.0 },
            clear: u32::from(clear),
            evaporation: water.evaporation,
            capacity: water.sediment_capacity,
            sea_level,
        });
        buffers.water.params.write_buffer(&render_device, &queue);
        state.frame.push(FrameCmd::Water {
            set: state.cascade.current,
            // Paused: one substep so a brush stroke still shows.
            substeps: if sim.paused { 1 } else { water.substeps },
            source,
        });
    }
    if !state.frame.is_empty() {
        let level = level_at(state.cascade.level);
        let finest = level.size == finest_size;
        // Paused steps and the frame the cascade finishes are the last display for a while:
        // refresh AO. (Later frames still display while rain or water keeps changing heights.)
        let just_finished = state.cascade.finished && !state.was_finished;
        let last = just_finished || sim.paused;
        let ao = finest && (last || state.displays_since_ao.is_none_or(|n| n + 1 >= AO_INTERVAL));
        state.displays_since_ao = match (finest, ao) {
            (false, _) => None,
            (true, true) => Some(0),
            (true, false) => state.displays_since_ao.map(|n| n + 1),
        };
        state.was_finished = state.cascade.finished;
        state.frame.push(FrameCmd::Display {
            set: state.cascade.current,
            offset: buffers.params.push(&level_erode_params(params, &level, 0)),
            finest,
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
) {
    let Some(groups) = groups else { return };
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
                substeps,
                source,
            } => {
                let (gx, gy) = dispatch_grid(params.map_size);
                let [p_source, p_flux, p_depth, p_erode, p_advect, p_copy] =
                    pipes.pipelines_water.map(get);
                let (Some(p_flux), Some(p_depth), Some(p_erode), Some(p_advect), Some(p_copy)) =
                    (p_flux, p_depth, p_erode, p_advect, p_copy)
                else {
                    continue;
                };
                pass.set_bind_group(0, &groups.water[set], &[]);
                pass.set_bind_group(1, &groups.water_out, &[]);
                if source && let Some(p) = p_source {
                    pass.set_pipeline(p);
                    pass.dispatch_workgroups(gx, gy, 1);
                }
                for _ in 0..substeps {
                    for p in [p_flux, p_depth, p_erode, p_advect, p_copy] {
                        pass.set_pipeline(p);
                        pass.dispatch_workgroups(gx, gy, 1);
                    }
                }
            }
            FrameCmd::Display {
                set,
                offset,
                finest,
                ao,
            } => {
                let (gx, gy) = dispatch_grid(params.map_size);
                let erosion = &groups.erosion[set];
                if !finest {
                    if let Some(p) = get(pipes.pipeline_blit_level) {
                        pass.set_pipeline(p);
                        pass.set_bind_group(0, erosion, &[offset]);
                        pass.set_bind_group(1, &groups.blit, &[]);
                        pass.set_bind_group(2, &groups.level_blit, &[]);
                        pass.dispatch_workgroups(gx, gy, 1);
                    }
                    continue;
                }
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
        // One writeBack per level, after the last (flow-biased) pass of its batch.
        assert_eq!(writebacks, vec![(0, 199), (1, 199), (2, 199), (3, 199)]);
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
    fn budget_only_limits_finest_level() {
        let mut cascade = Cascade::default();
        cascade.restart(pyramid_levels(UVec2::splat(512), 8.0, 1.0, 0.0));
        let erodes_at = |steps: &[CascadeStep], lvl: usize| {
            steps
                .iter()
                .filter(|s| matches!(s, CascadeStep::Erode { level, .. } if *level == lvl))
                .count()
        };

        // Paused: only init and the first seed.
        let mut paused = Vec::new();
        cascade.advance(0, true, false, &mut paused);
        assert_eq!(paused.len(), 2);

        // One call runs every coarse level, then `budget` finest iterations.
        let mut steps = Vec::new();
        cascade.advance(7, true, false, &mut steps);
        assert_eq!(
            (0..4).map(|l| erodes_at(&steps, l)).collect::<Vec<_>>(),
            vec![200, 200, 200, 7]
        );
        assert_eq!(cascade.level, 3);
        assert_eq!(cascade.batch_iter, 7);
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
        assert_eq!(writebacks, vec![199, 399, 599, 799, 999]);
    }
}
