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

pub mod boulders;
pub mod brush;
pub mod camera;
pub mod drainage;
pub mod river;
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
const RIVER_SHADER: &str = "river.wgsl";
const BOULDER_SHADER: &str = "boulders.wgsl";
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
const TERRAIN_PREPASS_SHADER: &str = "terrain_prepass.wesl";
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
    /// [`WET_DEPTH`] and [`FULL_DEPTH`] in height units: water bodies count as outflow.
    pub wet_depth: f32,
    pub full_depth: f32,
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
    /// Water surface for bevy_fft's [`WaterHeightfield`](bevy_fft::water::WaterHeightfield)
    /// (Rgba32Float, map size): R = bed (dry cells sunk below the terrain), G = water depth including the ocean, both in
    /// height units. See [`water_heightfield`].
    pub water_surface: Handle<Image>,
    /// Water velocity at texel centres for the same heightfield (Rgba32Float, map size): RG in
    /// texels per second.
    pub water_velocity: Handle<Image>,
    /// Terrain brush edits (Rgba32Float, map size), applied to the source terrain at init:
    /// R = height offset, G = flatten weight, B = flatten target height, A = smooth weight.
    /// Edit it on the CPU and bump [`ResetSim`] to re-erode (see [`brush`]).
    pub terrain_edit: Handle<Image>,
    /// Debug views of KTT's masks (Rgba16Float, map size): softness, outflow, erode mask, uplift.
    pub debug_masks: Handle<Image>,
    /// Debug views (Rgba16Float, map size): R = sediment above the bedrock, G = raw age.
    pub debug_layers: Handle<Image>,
    /// Debug views of [`WaterFields::discharge`] (Rgba16Float, map size).
    pub debug_discharge: Handle<Image>,
    /// Debug views of the bed load (Rgba16Float, map size): R = bed load through the cell,
    /// G = helical intensity, B = bed change not yet written to the heights.
    pub debug_bed: Handle<Image>,
    /// Drainage area in m² (R32Float, map size), written on the CPU by [`drainage`].
    pub drainage: Handle<Image>,
    /// Height the boulders add to the shallow-water solver's bed (height units;
    /// `boulders.wgsl`).
    pub obstacle: Handle<Image>,
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
        water_surface: add(images, TextureFormat::Rgba32Float),
        water_velocity: add(images, TextureFormat::Rgba32Float),
        terrain_edit: images.add(empty_terrain_edit_image(size)),
        debug_masks: add(images, TextureFormat::Rgba16Float),
        debug_layers: add(images, TextureFormat::Rgba16Float),
        debug_discharge: add(images, TextureFormat::Rgba16Float),
        debug_bed: add(images, TextureFormat::Rgba16Float),
        drainage: images.add(cpu_image(size, TextureFormat::R32Float, 4)),
        obstacle: add(images, TextureFormat::R32Float),
    }
}

/// A bevy_fft [`WaterHeightfield`](bevy_fft::water::WaterHeightfield) over the erosion's water,
/// for terrain drawn on a plane of `size` world units centred on the origin, displaced by
/// `height_scale` world units per height unit (see [`TerrainExtension`]). bevy_fft's
/// `WaterSurfaceMaterial` then draws the lakes, rivers and ocean. Add `bevy_fft::water::WaterPlugin`
/// and, with no FFT ocean, insert `bevy_fft::water::WaterLodClear`.
pub fn water_heightfield(
    images: &ErosionImages,
    map_size: UVec2,
    size: f32,
    height_scale: f32,
) -> bevy_fft::water::WaterHeightfield {
    let cell_size = size / map_size.x as f32;
    bevy_fft::water::WaterHeightfield {
        cell_size,
        blend_cells: 1.0,
        height_scale,
        velocity_scale: cell_size,
        ..bevy_fft::water::WaterHeightfield::new(
            images.water_surface.clone(),
            images.water_velocity.clone(),
            map_size,
        )
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
    /// [`boulders::MAX_BOULDERS`] [`boulders::Boulder`] slots, owned by the GPU.
    pub boulders: Handle<ShaderBuffer>,
}

impl WaterFields {
    /// Zeroed fields for a map of `size` texels.
    pub fn new(buffers: &mut Assets<ShaderBuffer>, size: UVec2) -> Self {
        let n = (size.x * size.y) as usize;
        let mut discharge = ShaderBuffer::new(vec![Vec4::ZERO; n], RenderAssetUsages::RENDER_WORLD);
        discharge.label = Cow::Borrowed("water_discharge");
        let mut boulders = ShaderBuffer::new(
            vec![boulders::Boulder::default(); boulders::MAX_BOULDERS],
            RenderAssetUsages::RENDER_WORLD,
        );
        boulders.label = Cow::Borrowed("boulders");
        Self {
            discharge: buffers.add(discharge),
            boulders: buffers.add(boulders),
        }
    }
}

/// A cleared [`ErosionImages::terrain_edit`] layer.
pub fn empty_terrain_edit_image(size: UVec2) -> Image {
    cpu_image(size, TextureFormat::Rgba32Float, 16)
}

/// A zeroed image the CPU writes and the GPU samples, `texel_bytes` per texel.
fn cpu_image(size: UVec2, format: TextureFormat, texel_bytes: usize) -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &vec![0u8; texel_bytes],
        format,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    image
}

/// Extended material type for terrain rendering
pub type TerrainMaterial = ExtendedMaterial<StandardMaterial, TerrainExtension>;

/// How the terrain is drawn: shaded, or one simulation field in false colour. The discriminant
/// is the `mode` the terrain shader switches on (`terrain_extended.wesl`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Reflect)]
#[repr(u32)]
pub enum PreviewMode {
    /// Full PBR with the atmosphere and bevy_fft's water surface.
    #[default]
    Pbr = 0,
    /// Lambert-shaded terrain colour with flat water tint: no PBR or water optics, for timing the
    /// simulations.
    Simple = 1,
    // Terrain
    Albedo = 2,
    Height = 3,
    Slope = 4,
    Normals = 5,
    Curvature = 6,
    AmbientOcclusion = 7,
    TerrainEdits = 8,
    // Erosion
    Flow = 9,
    Sediment = 10,
    Erosion = 11,
    /// KTT river momentum, with Do Rivers on.
    Rivers = 12,
    Age = 13,
    // KTT masks and layers
    Softness = 14,
    Outflow = 15,
    ErodeMask = 16,
    Uplift = 17,
    Bedrock = 18,
    // Water
    WaterDepth = 19,
    WaterVelocity = 20,
    SuspendedSediment = 21,
    Discharge = 22,
    MeanDepth = 23,
    RiverDischarge = 24,
    BedLoad = 25,
    Helical = 26,
    BedChange = 27,
    /// Drainage area, log scale.
    Drainage = 28,
    /// Boulder fraction of the bed surface.
    Boulders = 29,
}

impl PreviewMode {
    pub const ALL: [Self; 30] = [
        Self::Pbr,
        Self::Simple,
        Self::Albedo,
        Self::Height,
        Self::Slope,
        Self::Normals,
        Self::Curvature,
        Self::AmbientOcclusion,
        Self::TerrainEdits,
        Self::Flow,
        Self::Sediment,
        Self::Erosion,
        Self::Rivers,
        Self::Age,
        Self::Softness,
        Self::Outflow,
        Self::ErodeMask,
        Self::Uplift,
        Self::Bedrock,
        Self::WaterDepth,
        Self::WaterVelocity,
        Self::SuspendedSediment,
        Self::Discharge,
        Self::MeanDepth,
        Self::RiverDischarge,
        Self::BedLoad,
        Self::Helical,
        Self::BedChange,
        Self::Drainage,
        Self::Boulders,
    ];

    /// Shaded views draw the atmosphere and water; the rest show the bare field.
    pub fn is_shaded(self) -> bool {
        matches!(self, Self::Pbr | Self::Simple)
    }

    pub fn group(self) -> &'static str {
        match self as u32 {
            0..=1 => "Shading",
            2..=8 => "Terrain",
            9..=13 => "Erosion",
            14..=18 => "Masks & Layers",
            28 => "Erosion",
            _ => "Water",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Pbr => "Full PBR",
            Self::Simple => "Simple",
            Self::Albedo => "Albedo",
            Self::Height => "Height",
            Self::Slope => "Slope",
            Self::Normals => "Normals",
            Self::Curvature => "Curvature",
            Self::AmbientOcclusion => "AO",
            Self::TerrainEdits => "Edits",
            Self::Flow => "Flow",
            Self::Sediment => "Sediment",
            Self::Erosion => "Erosion",
            Self::Rivers => "Rivers",
            Self::Age => "Age",
            Self::Softness => "Softness",
            Self::Outflow => "Outflow",
            Self::ErodeMask => "Erode Mask",
            Self::Uplift => "Uplift",
            Self::Bedrock => "Bedrock",
            Self::WaterDepth => "Depth",
            Self::WaterVelocity => "Velocity",
            Self::SuspendedSediment => "Suspended",
            Self::Discharge => "Discharge",
            Self::MeanDepth => "Mean Depth",
            Self::RiverDischarge => "KTT Rivers",
            Self::BedLoad => "Bed Load",
            Self::Helical => "Helical",
            Self::BedChange => "Bed Change",
            Self::Drainage => "Drainage",
            Self::Boulders => "Boulders",
        }
    }

    /// What the view shows and how it is coloured.
    pub fn description(self) -> &'static str {
        match self {
            Self::Pbr => "Lit terrain with atmosphere, shadows and the water surface.",
            Self::Simple => {
                "Lambert-shaded terrain colour with a flat water tint; no atmosphere or water optics."
            }
            Self::Albedo => "Unlit terrain colour from the erosion's colour bands.",
            Self::Height => "Height with contour bands every 0.01.",
            Self::Slope => "Slope angle: black flat, white 60°+.",
            Self::Normals => "View-space normals.",
            Self::Curvature => "Laplacian of height: blue concave (valleys), red convex (ridges).",
            Self::AmbientOcclusion => "Baked ambient occlusion.",
            Self::TerrainEdits => {
                "Brush height offset: blue lowered, red raised; green flatten, white smooth weight."
            }
            Self::Flow => "KTT trail flow magnitude (heat).",
            Self::Sediment => "Loose deposited sediment (heat).",
            Self::Erosion => "Material removed by trails (heat).",
            Self::Rivers => "KTT river momentum, with Do Rivers on (heat).",
            Self::Age => "Time since a trail last changed the cell (Age Strength scales it).",
            Self::Softness => "Softness mask: black hard rock, white soft.",
            Self::Outflow => "Outflow mask: white drains away (ocean, border).",
            Self::ErodeMask => "Erode mask: trails start only where white.",
            Self::Uplift => "Uplift map multiplier.",
            Self::Bedrock => {
                "Sediment above the bedrock (heat; black at bedrock). Needs Use Bedrock."
            }
            Self::WaterDepth => "Water depth from the solver (heat), ocean in grey.",
            Self::WaterVelocity => "Water velocity: hue is direction, brightness is speed.",
            Self::SuspendedSediment => "Sediment carried by the water (heat).",
            Self::Discharge => {
                "Averaged unit discharge h·u: hue is direction, brightness magnitude."
            }
            Self::MeanDepth => "Depth averaged over a few seconds (heat).",
            Self::RiverDischarge => "KTT river discharge in water units (heat).",
            Self::BedLoad => "Bed load through each cell (heat).",
            Self::Helical => "Helical (secondary) flow intensity: blue one way, red the other.",
            Self::BedChange => "Bed change not yet written: blue scour, red deposit.",
            Self::Boulders => {
                "Boulder fraction of the bed surface: black none, white all. Flow carries the fines \
                 off first and armours the bed."
            }
            Self::Drainage => {
                "Drainage area (heat, log scale: 100 m² to 100 km²), after the cascade finishes."
            }
        }
    }
}

/// The terrain view the demo's terrain material draws. [`ErosionComputePlugin`] copies it to
/// every [`TerrainMaterial`] when it changes.
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct TerrainView {
    pub mode: PreviewMode,
    /// Multiplies the field before colouring, as a power of ten.
    pub gain: f32,
    /// How strongly the terrain colour shows the bed's grain size (0 = off): boulder beds and
    /// scree turn a lighter grey.
    pub grain_colour: f32,
}

impl Default for TerrainView {
    fn default() -> Self {
        Self {
            mode: PreviewMode::Pbr,
            gain: 0.0,
            grain_colour: 1.0,
        }
    }
}

/// [`TerrainExtension`] view uniform.
#[derive(Debug, Clone, Copy, ShaderType, Reflect)]
pub struct TerrainViewUniform {
    pub mode: u32,
    /// Linear gain (10^[`TerrainView::gain`]).
    pub gain: f32,
    /// [`TerrainView::grain_colour`].
    pub grain_colour: f32,
}

impl From<TerrainView> for TerrainViewUniform {
    fn from(view: TerrainView) -> Self {
        Self {
            mode: view.mode as u32,
            gain: 10f32.powf(view.gain),
            grain_colour: view.grain_colour,
        }
    }
}

fn sync_terrain_view(view: Res<TerrainView>, materials: Option<ResMut<Assets<TerrainMaterial>>>) {
    let Some(mut materials) = materials else {
        return;
    };
    if !view.is_changed() {
        return;
    }
    for (_, mat) in materials.iter_mut() {
        mat.extension.view = (*view).into();
    }
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
    /// [`PreviewMode`] and gain; set from [`TerrainView`].
    #[uniform(107)]
    pub view: TerrainViewUniform,
    /// Water solver output ([`ErosionImages::water`]): the surface rises to terrain + depth.
    #[texture(108)]
    pub water_tex: Handle<Image>,
    /// Ocean level (0 = no ocean): terrain below it renders as a flat sea.
    #[uniform(109)]
    pub sea_level: f32,
    /// Debug views: [`ErosionImages::terrain_edit`] and the `debug_*` images.
    #[texture(110, sample_type = "float", filterable = false)]
    pub terrain_edit_tex: Handle<Image>,
    #[texture(111)]
    pub debug_masks_tex: Handle<Image>,
    #[texture(112)]
    pub debug_layers_tex: Handle<Image>,
    #[texture(113)]
    pub debug_discharge_tex: Handle<Image>,
    #[texture(114)]
    pub debug_bed_tex: Handle<Image>,
    #[texture(115, sample_type = "float", filterable = false)]
    pub drainage_tex: Handle<Image>,
}

impl TerrainExtension {
    /// A terrain material over `images`, displaced by `height_scale` world units per height unit.
    pub fn new(images: &ErosionImages, height_scale: f32) -> Self {
        Self {
            height_tex: images.height.clone(),
            color_tex: images.color.clone(),
            analysis_tex: images.analysis.clone(),
            ao_tex: images.ao.clone(),
            height_scale,
            water_tex: images.water.clone(),
            terrain_edit_tex: images.terrain_edit.clone(),
            debug_masks_tex: images.debug_masks.clone(),
            debug_layers_tex: images.debug_layers.clone(),
            debug_discharge_tex: images.debug_discharge.clone(),
            debug_bed_tex: images.debug_bed.clone(),
            drainage_tex: images.drainage.clone(),
            ..default()
        }
    }
}

impl Default for TerrainExtension {
    fn default() -> Self {
        Self {
            height_tex: Default::default(),
            color_tex: Default::default(),
            analysis_tex: Default::default(),
            ao_tex: Default::default(),
            height_scale: 50.0,
            view: TerrainView::default().into(),
            water_tex: Default::default(),
            sea_level: 0.0,
            terrain_edit_tex: Default::default(),
            debug_masks_tex: Default::default(),
            debug_layers_tex: Default::default(),
            debug_discharge_tex: Default::default(),
            debug_bed_tex: Default::default(),
            drainage_tex: Default::default(),
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
    fn prepass_vertex_shader() -> ShaderRef {
        TERRAIN_PREPASS_SHADER.into()
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
/// The water runs in seconds and metres: the solver advances in steps of
/// [`WaterControl::step_seconds`], sized by the CFL condition for the map's texel size, and
/// [`WaterControl::time_scale`] simulated seconds pass per second of frame time. Rates are per
/// simulated second and do not depend on the frame rate.
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
    /// Rain on the whole map (mm per hour).
    pub rainfall: f32,
    /// Water each rain brush trail leaves per step along its path (height units × texels²), so
    /// the rain brush wets the channels it carves (0 = it only carves).
    pub rain_water: f32,
    /// Bump to remove all water.
    pub clear_generation: u32,
    /// Water soaking into the ground wherever there is any (mm per hour). Rain below this runs
    /// off nowhere (Horton), and thin films dry out.
    pub infiltration: f32,
    /// Sediment carrying capacity of flowing water (0 = water never erodes).
    pub sediment_capacity: f32,
    /// Which flow model moves the water.
    pub solver: WaterSolver,
    /// Manning roughness for [`WaterSolver::Swe`] in s/m^(1/3) (0 = no bed friction): about
    /// 0.03 for a clean channel, 0.04–0.07 for mountain streams.
    pub manning_n: f32,
    /// Carries depth with a second-order flux in [`WaterSolver::Swe`] instead of first-order
    /// upwind, whose numerical diffusion damps roll waves.
    pub second_order_depth: bool,
    /// Gravity (m/s²).
    pub gravity: f32,
    /// Simulated seconds per second of frame time.
    pub time_scale: f32,
    /// Seconds since the last frame (set from [`Time`]); the solver catches up by this much
    /// times [`Self::time_scale`].
    pub dt: f32,
    /// Bed reconstruction settings (see [`BedShaping`]).
    pub bed: BedShaping,
    /// River channel sizes, for Shape Beds and the river brush.
    pub channel: ChannelModel,
    /// Bump to reshape the beds of the current lakes and rivers once.
    pub shape_generation: u32,
    /// Bed load moved by the flow (see [`BedLoad`]).
    pub bed_load: BedLoad,
    /// Bump to fast-forward the beds by [`BedLoad::refine_steps`] steps.
    pub refine_generation: u32,
    /// With KTT's Do Rivers on, raise its river momentum to the water's discharge, so trails on
    /// the finest level follow the rivers the water has made.
    pub steer_rivers: bool,
    /// Boulders as bodies ([`boulders`]).
    pub boulders: boulders::BoulderPhysics,
}

/// Standard gravity (m/s²).
pub const GRAVITY: f32 = 9.81;
/// SI seconds per second of the solver before it ran in metres (g = 2000 cells/s²), at 10 m per
/// texel: the two are Froude-similar at this time stretch (`docs/UNIFIED_PLAN.md`). Defaults
/// fitted in the old solver seconds are converted with it.
pub const LEGACY_SECONDS: f32 = 45.0;
/// Deepest water the time step is sized for, in texels of depth (200 m at 10 m texels: lakes
/// can be as deep as a fifth of a height unit). Deeper water is held back by the solver's depth
/// limiter.
const CFL_DEPTH: f32 = 20.0;
/// bevy_fft holds back water above β·dx/(g·dt) from moving (β = 2). The formula is not a length,
/// so it moves with the time unit: the solver runs in a time unit of
/// [`WaterControl::solver_seconds`] that puts the cap at this depth (texels), the old solver's.
const LIMITER_DEPTH: f32 = 2.0 / (2000.0 / 480.0);
/// bevy_fft's depth limiter β.
const SWE_LIMITER_BETA: f32 = 2.0;
/// Courant number of a gravity wave over [`CFL_DEPTH`].
const COURANT: f32 = 0.4;
/// The capacity model's rates (`water_erode`) are per this many seconds.
const CAPACITY_STEP: f32 = 0.1;
/// Water shallower than this (texels of depth) is dry: it is not drawn, and is not KTT outflow.
/// Depths are relative to the grid: on 10 m texels this is 0.5 m. A thin film over a wide cell
/// is sheet flow on the grid, whatever its depth in metres.
pub const WET_DEPTH: f32 = 0.05;
/// Water this deep (texels of depth) is fully wet: KTT outflow, bed load instead of sheet-flow
/// erosion, and banks that slump (4 m on 10 m texels).
pub const FULL_DEPTH: f32 = 0.4;
/// Slowest flow (m/s) with a direction for bed load and curvature.
const MIN_SPEED: f32 = 0.01;

impl WaterControl {
    /// Seconds per solver step: a Courant number of [`COURANT`] for a gravity wave over
    /// [`CFL_DEPTH`] texels of water.
    pub fn step_seconds(&self, scale: &WorldScale) -> f32 {
        let depth = CFL_DEPTH * scale.metres_per_texel;
        COURANT * scale.metres_per_texel / (self.gravity.max(0.1) * depth).sqrt()
    }

    /// Seconds per time unit of bevy_fft's solver (see [`LIMITER_DEPTH`]). Gravity, the step,
    /// Manning n, sources and velocities are handed to it in this unit; the flow is the same in
    /// any unit except for the depth limiter.
    pub fn solver_seconds(&self, scale: &WorldScale) -> f32 {
        SWE_LIMITER_BETA / (LIMITER_DEPTH * self.gravity_cells(scale) * self.step_seconds(scale))
    }

    /// Gravity in the solver's cells (texels) per second².
    pub fn gravity_cells(&self, scale: &WorldScale) -> f32 {
        self.gravity / scale.metres_per_texel
    }

    /// Manning n in the solver's cells: u = h^(2/3) S^(1/2) / n holds in metres with `n`, and in
    /// cells with n · dx^(1/3).
    pub fn manning_cells(&self, scale: &WorldScale) -> f32 {
        self.manning_n * scale.metres_per_texel.cbrt()
    }

    /// Net water added everywhere per second (height units): rain less infiltration.
    pub fn net_rain(&self, scale: &WorldScale) -> f32 {
        scale.heights((self.rainfall - self.infiltration) / 3.6e6)
    }
}

/// Bed load and Exner (`water_morpho.wgsl`): on water deeper than a thin sheet, the flow moves
/// bed material along the bed (Shields and Wong & Parker), toward the inner banks of bends
/// (helical flow) and down cross-slopes (Talmon), and the bed changes by its divergence
/// ∂η/∂t = −∇·q / (1 − porosity). Settings are in metres and seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BedLoad {
    /// Run bed load with every water step (morphological factor 1).
    pub live: bool,
    /// Grain diameter D (m) in the Shields number θ = n²|u|² / (R · D · h^(1/3)): coarser grains
    /// need faster flow to move. 0.02 is gravel, 0.1 cobbles, 1 boulders.
    pub diameter: f32,
    /// Wong & Parker (2006) coefficient: bed load q = K (θ − θ_c)^1.5 √(R g D³) in m² per second
    /// (3.97 in flumes).
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
    /// Hours of the bed changing under the current flow that one refine runs (water time ×
    /// [`Self::morph_factor`]).
    pub refine_hours: f32,
    /// How much harder the hardest rock of KTT's softness mask is to move: its critical Shields
    /// number is (1 + this) times Critical Shields. Bed load also never cuts below KTT's
    /// bedrock layer (with Use Bedrock on).
    pub rock_hardness: f32,
    /// Boulders: a second grain size in the bed (m). [`Self::diameter`] is the fines.
    pub boulder_diameter: f32,
    /// Boulder fraction of the bed everywhere (0 to 1).
    pub boulder_base: f32,
    /// Boulder fraction added on the hardest rock of KTT's softness mask.
    pub boulder_rock: f32,
    /// Boulder fraction added on slopes past the talus (scree), from tan 0.6 to 1.
    pub boulder_scree: f32,
    /// γ in θ_c,i = θ_c (D_i / D_m)^(−γ): 0 lets each size move on its own; toward 1 all sizes
    /// start moving at the same flow (Parker 1990: about 0.9).
    pub hiding: f32,
    /// Sternberg abrasion coefficient α (per km): moving boulders wear into fines as
    /// e^(−α · distance travelled). Hard rock measures 0.01–0.1 per km, weak rock up to 1.
    pub abrasion: f32,
}

impl Default for BedLoad {
    fn default() -> Self {
        Self {
            live: true,
            diameter: 1.5,
            k: WONG_PARKER_K,
            critical_shields: 0.0495,
            slope_effect: 1.0,
            helical_strength: 20.0,
            morph_factor: 20.0,
            refine_hours: 0.2,
            rock_hardness: 4.0,
            boulder_diameter: 4.0,
            boulder_base: 0.0,
            boulder_rock: 0.3,
            boulder_scree: 0.3,
            hiding: 0.85,
            abrasion: 0.05,
        }
    }
}

/// Wong & Parker's (2006) bed load coefficient.
pub const WONG_PARKER_K: f32 = 3.97;
/// Submerged specific gravity of quartz grains, (ρ_s − ρ) / ρ.
pub const SUBMERGED_DENSITY: f32 = 1.65;
/// Pore space of the bed: deposited load builds 1 / (1 − this) times its volume of bed.
const POROSITY: f32 = 0.4;
/// Active surface layer of the bed (Hirano 1971), in mean grain diameters.
const ACTIVE_LAYER: f32 = 2.0;

impl BedLoad {
    /// Submerged grain weight R·D in cells, for θ = n²|u|² / (R D h^(1/3)) in cells.
    pub fn grain_cells(&self, scale: &WorldScale) -> f32 {
        SUBMERGED_DENSITY * self.diameter / scale.metres_per_texel
    }

    /// The bed load coefficient in cells² per second of bed: K √(R g D³) / (1 − porosity).
    pub fn k_cells(&self, gravity: f32, scale: &WorldScale) -> f32 {
        let d = self.diameter.max(1e-5);
        self.k * (SUBMERGED_DENSITY * gravity * d * d * d).sqrt()
            / (1.0 - POROSITY)
            / (scale.metres_per_texel * scale.metres_per_texel)
    }

    /// The boulders, as a bed load of their own size.
    pub fn boulders(&self) -> Self {
        Self {
            diameter: self.boulder_diameter,
            ..*self
        }
    }

    /// Fraction of the boulder load that wears into fines crossing one texel.
    pub fn abrasion_per_texel(&self, scale: &WorldScale) -> f32 {
        1.0 - (-self.abrasion.max(0.0) * scale.metres_per_texel / 1000.0).exp()
    }

    /// Water steps one refine runs, at `step` seconds a step.
    pub fn refine_steps(&self, step: f32) -> u32 {
        (self.refine_hours.max(0.0) * 3600.0 / (step * self.morph_factor.max(1.0))).ceil() as u32
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
            channel_discharge: 0.02 / LEGACY_SECONDS,
        }
    }
}

/// River channels from their discharge, shared by Shape Beds and the river brush (see
/// `docs/UNIFIED_PLAN.md`): a catchment of A km² carries a bankfull discharge
/// Q = k · A^0.8 m³/s, and a channel for Q is a_W · Q^0.5 m wide and a_H · Q^0.4 m deep
/// (Leopold & Maddock 1953; Parker et al. 2007).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChannelModel {
    /// k in Q = k · A^0.8: about 1–3 in humid regions, much less in arid ones.
    pub bankfull_k: f32,
    /// a_W in width = a_W · Q^0.5 (m).
    pub width_coef: f32,
    /// a_H in depth = a_H · Q^0.4 (m).
    pub depth_coef: f32,
    /// Narrowest channel (texels) the water can run in. Narrower ones are widened to it and made
    /// shallower so they carry the same flow (Manning conveyance W · H^(5/3)).
    pub min_width: f32,
}

impl Default for ChannelModel {
    fn default() -> Self {
        Self {
            bankfull_k: 1.5,
            width_coef: 4.4,
            depth_coef: 0.3,
            min_width: 3.0,
        }
    }
}

impl ChannelModel {
    /// Bankfull discharge (m³/s) of a catchment of `area` m².
    pub fn discharge(&self, area: f32) -> f32 {
        self.bankfull_k * (area.max(0.0) / 1.0e6).powf(0.8)
    }

    /// Width and depth (m) of the channel for `q` m³/s, widened to [`Self::min_width`] texels.
    pub fn section(&self, q: f32, scale: &WorldScale) -> (f32, f32) {
        let q = q.max(1e-6);
        let (w, h) = (self.width_coef * q.sqrt(), self.depth_coef * q.powf(0.4));
        let min = self.min_width * scale.metres_per_texel;
        if w < min {
            (min, h * (w / min).powf(0.6))
        } else {
            (w, h)
        }
    }

    /// Half-width (texels) and depth (height units) of the channel for `q` m³/s.
    pub fn half_width_and_depth(&self, q: f32, scale: &WorldScale) -> (f32, f32) {
        let (w, h) = self.section(q, scale);
        (scale.texels(0.5 * w), scale.heights(h))
    }
}

/// Widest bank grading (dilation steps of the shore distance).
pub const MAX_BANK_WIDTH: f32 = 32.0;
/// Red-black SOR sweeps for the bank distance τ and the depth smoothing.
const BED_TAU_SWEEPS: u32 = 400;
/// Reach of the bank reference inside rivers (texels); wider rivers fall back to their surface.
const BED_BANK_REACH: f32 = 32.0;
const BED_SMOOTH_SWEEPS: u32 = 200;
/// Still water deeper than this (m) is a lake for Shape Beds. Shape Beds was tuned on the old
/// solver; this and the speeds below are its thresholds at 10 m texels.
const DOMAIN_DEPTH: f32 = 2.0;
/// Averaged speeds (m/s) below which water is still (a lake) and above which it is a river.
const STILL_SPEED: f32 = 0.44;
const RIVER_SPEED: f32 = 1.8;

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
    channel_discharge: f32,
    domain_depth: f32,
    wet_depth: f32,
    still_speed: f32,
    river_speed: f32,
    q_to_si: f32,
    width_coef: f32,
    depth_coef: f32,
    min_half_width: f32,
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
            infiltration: 5.0,
            sediment_capacity: 0.0005,
            solver: WaterSolver::default(),
            manning_n: 0.2,
            second_order_depth: false,
            gravity: GRAVITY,
            time_scale: 50.0,
            dt: 0.0,
            bed: BedShaping::default(),
            channel: ChannelModel::default(),
            shape_generation: 0,
            bed_load: BedLoad::default(),
            refine_generation: 0,
            steer_rivers: true,
            boulders: boulders::BoulderPhysics::default(),
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
/// Most solver steps per frame; past this the water runs slower than its time scale instead of
/// stalling the GPU.
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
    pub capacity: f32,
    /// Ocean level (0 = no ocean): terrain below it drains water away.
    pub sea_level: f32,
    /// Depth added everywhere per step (negative: infiltration exceeds rain).
    pub rain: f32,
    /// Seconds per step ([`WaterControl::step_seconds`]).
    pub step_seconds: f32,
    /// Bed load runs (`water_morpho.wgsl`), so the capacity model only erodes thin sheet flow.
    pub bed_load: u32,
    /// [`WET_DEPTH`] and [`FULL_DEPTH`] in height units.
    pub wet_depth: f32,
    pub full_depth: f32,
    /// Steps of the capacity model ([`CAPACITY_STEP`]) per solver step.
    pub capacity_steps: f32,
    /// Seconds per time unit of bevy_fft's solver ([`WaterControl::solver_seconds`]).
    pub solver_seconds: f32,
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
    wet_depth: f32,
    full_depth: f32,
    /// [`MIN_SPEED`] in cells per second.
    min_speed: f32,
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
    boulder_grain: f32,
    boulder_k: f32,
    boulder_base: f32,
    boulder_rock: f32,
    boulder_scree: f32,
    hiding: f32,
    active_layer: f32,
    abrasion: f32,
    reset_boulders: u32,
}

/// `BoulderParams` in `boulders.wgsl`.
#[derive(Clone, Copy, Debug, Default, ShaderType)]
struct BoulderParams {
    size: UVec2,
    pad_offset: UVec2,
    grid_size: UVec2,
    slots: u32,
    seconds: f32,
    step_seconds: f32,
    metres_per_texel: f32,
    metres_per_height: f32,
    gravity: f32,
    density: f32,
    drag: f32,
    lift: f32,
    friction_static: f32,
    friction_kinetic: f32,
    sea_level: f32,
}

/// Floats per cell in the morpho state (`STATE_STRIDE` in `water_morpho.wgsl`).
const MORPHO_STATE_STRIDE: usize = 8;

/// Seconds the shared discharge field averages over, so roll waves do not make it flicker.
const DISCHARGE_TIME: f32 = 3.0 * LEGACY_SECONDS;
/// KTT river momentum (trail water per texel) to unit discharge (height units × texels per
/// second), fitted on a spring-fed valley (`tests/water_swe_gpu.rs`, `discharge_matches_ktt`) in
/// the old solver seconds.
const KTT_DISCHARGE_SCALE: f32 = 0.0044 / LEGACY_SECONDS;
/// Helical flow adaptation length in metres (terracraft uses 42 m at 1 m per pixel; 20 texels here).
const HELICAL_LENGTH: f32 = 200.0;
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

/// Map texels per height unit: heights are isotropic with the grid, so a slope of 1 height unit
/// per 100 texels is 45°. The shaders count water depths and the bed in cells of this size.
pub const CELLS_PER_HEIGHT: f32 = 100.0;

/// The physical size of the map (see `docs/UNIFIED_PLAN.md`). The simulation works in texels and
/// height units; settings, the scene and the physics constants are converted with this.
///
/// Set it before the app starts: the scene and the settings panel are built from it once.
#[derive(Resource, Clone, Copy, Debug, PartialEq, ExtractResource)]
#[extract_app(RenderApp)]
pub struct WorldScale {
    /// Texel size (dx) in metres.
    pub metres_per_texel: f32,
}

impl Default for WorldScale {
    fn default() -> Self {
        Self {
            metres_per_texel: 10.0,
        }
    }
}

impl WorldScale {
    /// Metres per height unit ([`CELLS_PER_HEIGHT`] texels).
    pub fn metres_per_height(&self) -> f32 {
        CELLS_PER_HEIGHT * self.metres_per_texel
    }

    /// Metres to texels.
    pub fn texels(&self, metres: f32) -> f32 {
        metres / self.metres_per_texel
    }

    /// Metres to height units.
    pub fn heights(&self, metres: f32) -> f32 {
        metres / self.metres_per_height()
    }

    /// Width and depth of the map in metres.
    pub fn map_metres(&self, size: UVec2) -> Vec2 {
        size.as_vec2() * self.metres_per_texel
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
    commands.insert_resource(river::RiverBed::default());
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
            .init_resource::<WorldScale>()
            .insert_resource(progress.clone())
            .add_plugins((
                ExtractResourcePlugin::<WorldScale>::default(),
                ExtractResourcePlugin::<ErosionImages>::default(),
                ExtractResourcePlugin::<WaterFields>::default(),
                ExtractResourcePlugin::<ErodeParams>::default(),
                ExtractResourcePlugin::<ResetSim>::default(),
                ExtractResourcePlugin::<SimControl>::default(),
                ExtractResourcePlugin::<BrushStroke>::default(),
                ExtractResourcePlugin::<WaterControl>::default(),
                ExtractResourcePlugin::<river::RiverBed>::default(),
            ))
            .add_plugins((drainage::DrainagePlugin, boulders::BoulderPlugin))
            .init_resource::<TerrainView>()
            .add_systems(Startup, setup_erosion_resources)
            .add_systems(PostUpdate, (update_water_time, sync_terrain_view));
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
    /// Two debug view outputs (group 1 of `blit_fields` and `morpho_debug`).
    layout_debug_out: BindGroupLayoutDescriptor,
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
    /// Debug views: `blit_fields` (`erosion.wgsl`) and `morpho_debug` (`water_morpho.wgsl`).
    pipeline_blit_fields: CachedComputePipelineId,
    pipeline_morpho_debug: CachedComputePipelineId,
    /// `water.wgsl`: clear, flux, depth, erode, advect, copy, the bevy_fft solver's export,
    /// gather and import, apply (the staged erosion), and the surface for bevy_fft's renderer.
    pipelines_water: [CachedComputePipelineId; 11],
    /// `water_bed.wgsl`: init, tau, prior, smooth, write, dilate, grade.
    pipelines_water_bed: [CachedComputePipelineId; 7],
    layout_water_morpho: BindGroupLayoutDescriptor,
    layout_water_bed_load: BindGroupLayoutDescriptor,
    /// `water_morpho.wgsl`: discharge, helical, flux, exner, steer.
    pipelines_water_morpho: [CachedComputePipelineId; 5],
    layout_water_splat: BindGroupLayoutDescriptor,
    /// `water_splat.wgsl`: splat, record.
    pipelines_water_splat: [CachedComputePipelineId; 2],
    layout_river: BindGroupLayoutDescriptor,
    /// `river.wgsl`: carve, armour.
    pipelines_river: [CachedComputePipelineId; 2],
    layout_boulders: BindGroupLayoutDescriptor,
    /// `boulders.wgsl`: move, obstacle clear, obstacle stamp.
    pipelines_boulders: [CachedComputePipelineId; 3],
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

    let layout_debug_out = BindGroupLayoutDescriptor::new(
        "ErosionDebugGroup1",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
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
                storage_buffer::<Vec4>(false), // face bed load, fines and boulders
                storage_buffer::<f32>(false), // helical, pending bed change, boulder fraction
                uniform_buffer::<MorphoParams>(false),
                storage_buffer_read_only::<Vec4>(false), // work set masks (softness)
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
                (5, storage_buffer::<Vec4>(false)), // face bed load, fines and boulders
                (6, storage_buffer::<f32>(false)), // helical, pending bed change, boulder fraction
                (7, uniform_buffer::<MorphoParams>(false)),
                (8, storage_buffer_read_only::<Vec4>(false)), // work set masks (softness)
                (9, storage_buffer_read_only::<Vec2>(false)), // work set layers (bedrock)
                (10, storage_buffer_read_only::<Vec2>(false)), // river armour band
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
                // Water surface and velocity for bevy_fft's heightfield.
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                // Boulders on the solver's bed (`boulders.wgsl`).
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

    let pipeline_blit_fields = compute(
        vec![layout_erosion.clone(), layout_debug_out.clone()],
        &shader,
        "blit_fields",
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
        "water_surface",
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
    let pipeline_morpho_debug = compute(
        vec![layout_water_morpho.clone(), layout_debug_out.clone()],
        &water_morpho_shader,
        "morpho_debug",
    );

    let water_splat_shader = asset_server.load(WATER_SPLAT_SHADER);
    let pipelines_water_splat = ["water_splat", "water_record"]
        .map(|entry| compute(vec![layout_water_splat.clone()], &water_splat_shader, entry));

    // River brush beds and their armour (`river.wgsl`).
    let layout_river = BindGroupLayoutDescriptor::new(
        "RiverGroup0",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer::<u32>(false), // work set height (atomic, fixed-point)
                storage_buffer_read_only::<Vec4>(false), // river raster
                storage_buffer::<Vec2>(false), // armour band
                storage_buffer::<f32>(false), // suspended sediment
                storage_buffer::<f32>(false), // staged erosion (`water_erode`)
                uniform_buffer::<WaterParams>(false),
            ),
        ),
    );
    let river_shader = asset_server.load(RIVER_SHADER);
    let pipelines_river = ["river_carve", "river_armour"]
        .map(|entry| compute(vec![layout_river.clone()], &river_shader, entry));

    let layout_boulders = BindGroupLayoutDescriptor::new(
        "BoulderGroup0",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only::<u32>(false), // work set height (fixed-point)
                storage_buffer_read_only::<f32>(false), // water depth
                storage_buffer_read_only::<Vec2>(false), // water velocity
                storage_buffer::<boulders::Boulder>(false),
                uniform_buffer::<BoulderParams>(false),
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadWrite),
            ),
        ),
    );
    let boulder_shader = asset_server.load(BOULDER_SHADER);
    let pipelines_boulders = ["boulder_move", "obstacle_clear", "obstacle_stamp"]
        .map(|entry| compute(vec![layout_boulders.clone()], &boulder_shader, entry));

    commands.insert_resource(ErosionPipeline {
        layout_river,
        pipelines_river,
        layout_boulders,
        pipelines_boulders,
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
        layout_debug_out,
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
        pipeline_blit_fields,
        pipeline_morpho_debug,
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
    bed_load: StorageBuffer<Vec<Vec4>>,
    /// Per cell: helical intensity (two slots) and the bed change not yet written.
    morpho_state: StorageBuffer<Vec<f32>>,
    morpho_params: UniformBuffer<MorphoParams>,
    /// River brush raster ([`river::RiverCell`] per texel) and the armour band (floor, ceiling)
    /// it leaves.
    river: StorageBuffer<Vec<Vec4>>,
    river_band: StorageBuffer<Vec<Vec2>>,
    boulder_params: UniformBuffer<BoulderParams>,
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
                capacity: 0.0,
                sea_level: 0.0,
                rain: 0.0,
                step_seconds: 1.0,
                bed_load: 0,
                wet_depth: 0.0,
                full_depth: 1.0,
                capacity_steps: 1.0,
                solver_seconds: 1.0,
                source_count: 0,
                sources: [Vec4::ZERO; MAX_WATER_SOURCES],
            }),
            tau: StorageBuffer::from(vec![0f32; n]),
            bed_depth: StorageBuffer::from(vec![0f32; n]),
            prior: StorageBuffer::from(vec![0f32; n]),
            shore: [0, 1].map(|_| StorageBuffer::from(vec![Vec2::ZERO; n])),
            bed_params: DynamicUniformBuffer::default(),
            swe: SweGrid::new(size, device),
            bed_load: StorageBuffer::from(vec![Vec4::ZERO; n]),
            morpho_state: StorageBuffer::from(vec![0f32; MORPHO_STATE_STRIDE * n]),
            morpho_params: UniformBuffer::from(MorphoParams {
                size,
                pad_offset: UVec2::ZERO,
                grid_size: size,
                step_seconds: 1.0,
                frame_seconds: 0.0,
                discharge_time: DISCHARGE_TIME,
                ktt_scale: KTT_DISCHARGE_SCALE,
                ktt_vector: 0,
                sea_level: 0.0,
                manning_n: 0.0,
                wet_depth: 0.0,
                full_depth: 1.0,
                min_speed: 0.0,
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
                boulder_grain: 1.0,
                boulder_k: 0.0,
                boulder_base: 0.0,
                boulder_rock: 0.0,
                boulder_scree: 0.0,
                hiding: 0.0,
                active_layer: 2.0,
                abrasion: 0.0,
                reset_boulders: 0,
            }),
            river: StorageBuffer::from(vec![Vec4::new(river::NO_RIVER, 1.0, 0.0, 0.0); n]),
            river_band: StorageBuffer::from(vec![Vec2::ZERO; n]),
            boulder_params: UniformBuffer::from(BoulderParams::default()),
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
        b.river.write_buffer(device, queue);
        b.river_band.write_buffer(device, queue);
        b.boulder_params.write_buffer(device, queue);
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
    /// River brush beds, indexed by work set.
    river: [BindGroup; 2],
    /// Boulders, indexed by work set.
    boulders: [BindGroup; 2],
    blit: BindGroup,
    blit_blur: BindGroup,
    /// Debug view outputs of `blit_fields` and `morpho_debug`.
    debug_fields: BindGroup,
    debug_water: BindGroup,
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
    let Some(fields) = fields else { return };
    let (Some(discharge), Some(boulder_buffer)) = (
        gpu_buffers.get(&fields.discharge),
        gpu_buffers.get(&fields.boulders),
    ) else {
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
        Some(view_water_surface),
        Some(view_water_velocity),
        Some(view_terrain_edit),
        Some(view_debug_masks),
        Some(view_debug_layers),
        Some(view_debug_discharge),
        Some(view_debug_bed),
        Some(view_obstacle),
    ) = (
        gpu_images.get(&images.height),
        gpu_images.get(&images.color),
        gpu_images.get(&images.normal),
        gpu_images.get(&images.analysis),
        gpu_images.get(&images.ao),
        gpu_images.get(&images.ao_temp),
        gpu_images.get(&images.water),
        gpu_images.get(&images.water_surface),
        gpu_images.get(&images.water_velocity),
        gpu_images.get(&images.terrain_edit),
        gpu_images.get(&images.debug_masks),
        gpu_images.get(&images.debug_layers),
        gpu_images.get(&images.debug_discharge),
        gpu_images.get(&images.debug_bed),
        gpu_images.get(&images.obstacle),
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
                    set.masks.binding().unwrap(),
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
                    (10, w.river_band.binding().unwrap()),
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
        boulders: [0, 1].map(|set| {
            let w = &buffers.water;
            render_device.create_bind_group(
                None,
                &layout(&pipeline.layout_boulders),
                &BindGroupEntries::sequential((
                    buffers.sets[set].height.binding().unwrap(),
                    w.depth.binding().unwrap(),
                    w.velocity.binding().unwrap(),
                    boulder_buffer.buffer.as_entire_binding(),
                    w.boulder_params.binding().unwrap(),
                    &view_obstacle.texture_view,
                )),
            )
        }),
        river: [0, 1].map(|set| {
            let w = &buffers.water;
            render_device.create_bind_group(
                None,
                &layout(&pipeline.layout_river),
                &BindGroupEntries::sequential((
                    buffers.sets[set].height.binding().unwrap(),
                    w.river.binding().unwrap(),
                    w.river_band.binding().unwrap(),
                    w.sediment.binding().unwrap(),
                    w.sediment_tmp.binding().unwrap(),
                    w.params.binding().unwrap(),
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
                &view_water_surface.texture_view,
                &view_water_velocity.texture_view,
                &view_obstacle.texture_view,
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
        debug_fields: render_device.create_bind_group(
            None,
            &layout(&pipeline.layout_debug_out),
            &BindGroupEntries::sequential((
                &view_debug_masks.texture_view,
                &view_debug_layers.texture_view,
            )),
        ),
        debug_water: render_device.create_bind_group(
            None,
            &layout(&pipeline.layout_debug_out),
            &BindGroupEntries::sequential((
                &view_debug_discharge.texture_view,
                &view_debug_bed.texture_view,
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
        /// Boulder slots in use: they move after the steps and stamp the solver's bed.
        boulders: u32,
    },
    /// Rebuild the beds under the current water (`water_bed.wgsl`); `offsets` per SOR parity.
    /// With `swe`, the new depth is handed to bevy_fft's solver afterwards.
    ShapeBeds {
        set: usize,
        offsets: [u32; 2],
        dilate_steps: u32,
        swe: bool,
    },
    /// Carve the river brush's beds into the work set (`river_carve`).
    RiverCarve {
        set: usize,
    },
    Display {
        set: usize,
        offset: u32,
        /// Recompute ambient occlusion (see [`AO_INTERVAL`]).
        ao: bool,
        /// Also refresh the water surface for bevy_fft's renderer (`water_surface`).
        water: bool,
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
    /// Simulated time not yet run by the water solver (below one step).
    water_time: f32,
    last_shape: u32,
    last_refine: u32,
    /// Water steps still to run for the current Refine Beds request.
    refine_left: u32,
    /// Helical flow slot holding the latest intensity.
    helical_parity: u32,
    /// Boulder settings the bed's surface layer was set from (base, rock, scree).
    boulder_settings: [f32; 3],
    /// [`boulders::BoulderEdits::generation`] last written to the GPU.
    boulder_generation: u64,
    /// Cascade was finished last frame (AO refresh on the frame it finishes).
    was_finished: bool,
    /// Finest-level displays since AO was last computed (`None` before the first one).
    displays_since_ao: Option<u32>,
    /// [`river::RiverBed::generation`] last uploaded and carved.
    last_river: u32,
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
    river_bed: Option<Res<river::RiverBed>>,
    scale: Res<WorldScale>,
    (boulder_edits, fields, gpu_buffers): (
        Option<Res<boulders::BoulderEdits>>,
        Option<Res<WaterFields>>,
        Res<RenderAssets<GpuShaderBuffer>>,
    ),
) {
    let state = &mut *state;
    // Boulders placed or removed on the CPU go into their slots of the GPU buffer once.
    let boulder_buffer = fields.and_then(|f| gpu_buffers.get(&f.boulders));
    let boulder_slots = boulder_edits.as_ref().map_or(0, |e| e.slots);
    if let (Some(edits), Some(gpu)) = (&boulder_edits, boulder_buffer)
        && edits.generation != state.boulder_generation
    {
        state.boulder_generation = edits.generation;
        let size = std::mem::size_of::<boulders::Boulder>() as u64;
        if edits.clear {
            let zeros = vec![boulders::Boulder::default(); boulders::MAX_BOULDERS];
            queue.write_buffer(&gpu.buffer, 0, bytemuck::cast_slice(&zeros));
        }
        for (slot, boulder) in &edits.writes {
            queue.write_buffer(&gpu.buffer, *slot as u64 * size, bytemuck::bytes_of(boulder));
        }
    }
    let step = water.step_seconds(&scale);
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
        // A new terrain: the bed's surface layer starts again from its substrate.
        let n = (params.map_size.x * params.map_size.y) as usize;
        buffers
            .water
            .morpho_state
            .set(vec![0.0; MORPHO_STATE_STRIDE * n]);
        buffers
            .water
            .morpho_state
            .write_buffer(&render_device, &queue);
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
                    wet_depth: WET_DEPTH / CELLS_PER_HEIGHT,
                    full_depth: FULL_DEPTH / CELLS_PER_HEIGHT,
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
        || water.rainfall > water.infiltration
        || boulder_slots > 0
    {
        // (Boulders roll on dry slopes too, and the solver steps move them.)
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
        state.refine_left = water.bed_load.refine_steps(step);
        state.water_active = true;
    }
    let refining = state.refine_left > 0;
    let sea_level = if params.use_outflow != 0 {
        params.outflow_sea_level
    } else {
        0.0
    };
    // Written on every finest-level frame: the display's `water_surface` reads it too.
    if finest_ready {
        let brush = stroke.then(|| Spring {
            center: water.center,
            radius: water.radius,
            rate: water.rate,
        });
        let mut sources = [Vec4::ZERO; MAX_WATER_SOURCES];
        let mut source_count = 0;
        for s in water.springs.iter().take(MAX_SPRINGS).chain(&brush) {
            sources[source_count] = s.center.extend(s.radius.max(0.5)).extend(s.rate * step);
            source_count += 1;
        }
        buffers.water.params.set(WaterParams {
            size: params.map_size,
            pad_offset,
            grid_size: finest_size,
            clear: u32::from(clear),
            capacity: water.sediment_capacity,
            sea_level,
            rain: water.net_rain(&scale) * step,
            step_seconds: step,
            bed_load: u32::from(water.bed_load.live || refining),
            wet_depth: WET_DEPTH / CELLS_PER_HEIGHT,
            full_depth: FULL_DEPTH / CELLS_PER_HEIGHT,
            capacity_steps: step / CAPACITY_STEP,
            solver_seconds: water.solver_seconds(&scale),
            source_count: source_count as u32,
            sources,
        });
        buffers.water.params.write_buffer(&render_device, &queue);
    }
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
            state.water_time += water.dt * water.time_scale.max(0.0);
            let steps = ((state.water_time / step) as u32).min(MAX_WATER_STEPS);
            state.water_time = (state.water_time - steps as f32 * step).min(step);
            steps
        };
        let bed_load = &water.bed_load;
        // New boulder settings: the bed's surface layer starts again from its substrate.
        let boulders = [bed_load.boulder_base, bed_load.boulder_rock, bed_load.boulder_scree];
        // (The discharge pass that applies it runs only on frames with steps.)
        let reset_boulders = steps > 0 && boulders != state.boulder_settings;
        if steps > 0 {
            state.boulder_settings = boulders;
        }
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
            step_seconds: step,
            frame_seconds: steps as f32 * step,
            discharge_time: DISCHARGE_TIME,
            ktt_scale: KTT_DISCHARGE_SCALE,
            ktt_vector: u32::from(params.do_rivers != 0 && params.momentum_coherence > 0.0),
            sea_level,
            manning_n: water.manning_cells(&scale),
            wet_depth: WET_DEPTH / CELLS_PER_HEIGHT,
            full_depth: FULL_DEPTH / CELLS_PER_HEIGHT,
            min_speed: scale.texels(MIN_SPEED),
            grain: bed_load.grain_cells(&scale),
            bed_load_k: bed_load.k_cells(water.gravity, &scale),
            critical_shields: bed_load.critical_shields,
            slope_effect: bed_load.slope_effect,
            helical_strength: bed_load.helical_strength,
            helical_length: scale.texels(HELICAL_LENGTH),
            mor_fac,
            parity: state.helical_parity,
            use_bedrock: params.use_bedrock,
            rock_hardness: bed_load.rock_hardness,
            boulder_grain: bed_load.boulders().grain_cells(&scale),
            boulder_k: bed_load.boulders().k_cells(water.gravity, &scale),
            boulder_base: bed_load.boulder_base,
            boulder_rock: bed_load.boulder_rock,
            boulder_scree: bed_load.boulder_scree,
            hiding: bed_load.hiding,
            active_layer: ACTIVE_LAYER,
            abrasion: bed_load.abrasion_per_texel(&scale),
            reset_boulders: u32::from(reset_boulders),
        });
        buffers
            .water
            .morpho_params
            .write_buffer(&render_device, &queue);
        let physics = &water.boulders;
        buffers.water.boulder_params.set(BoulderParams {
            size: params.map_size,
            pad_offset,
            grid_size: finest_size,
            slots: boulder_slots,
            seconds: steps as f32 * step,
            step_seconds: step,
            metres_per_texel: scale.metres_per_texel,
            metres_per_height: scale.metres_per_height(),
            gravity: water.gravity,
            density: physics.density,
            drag: physics.drag,
            lift: physics.lift,
            friction_static: physics.friction_static,
            friction_kinetic: physics.friction_kinetic,
            sea_level,
        });
        buffers
            .water
            .boulder_params
            .write_buffer(&render_device, &queue);
        if swe && let Some(solver) = &buffers.water.swe.solver {
            let unit = water.solver_seconds(&scale);
            // Sources, sinks and the map border come in through `swe_export`, so the borders
            // stay closed walls. The overshoot limiter is off: it does not conserve water.
            solver.write_settings(
                &queue,
                &ShallowWaterSettings {
                    // In the solver's time unit (see `LIMITER_DEPTH`).
                    dt: step / unit,
                    cell_size: 1.0,
                    gravity: water.gravity_cells(&scale) * unit * unit,
                    manning_n: water.manning_cells(&scale) / unit,
                    evaporation: 0.0,
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
                boulders: boulder_slots,
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
        channel_discharge: bed.channel_discharge,
        domain_depth: scale.heights(DOMAIN_DEPTH),
        wet_depth: WET_DEPTH / CELLS_PER_HEIGHT,
        still_speed: scale.texels(STILL_SPEED),
        river_speed: scale.texels(RIVER_SPEED),
        q_to_si: scale.metres_per_height() * scale.metres_per_texel * scale.metres_per_texel,
        width_coef: water.channel.width_coef / (2.0 * scale.metres_per_texel),
        depth_coef: water.channel.depth_coef / scale.metres_per_height(),
        min_half_width: 0.5 * water.channel.min_width,
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
    // River brush: carve once the cascade has finished (or while it keeps eroding the finest
    // level), so the trails do not wear the beds away.
    if let Some(bed) = river_bed
        && bed.generation != state.last_river
        && bed.cells.len() == buffers.water.river.get().len()
        && finest_ready
        && (state.cascade.finished || sim.continue_after_cascade)
    {
        state.last_river = bed.generation;
        buffers.water.river.set(bed.cells.as_ref().clone());
        buffers.water.river.write_buffer(&render_device, &queue);
        state.frame.push(FrameCmd::RiverCarve {
            set: state.cascade.current,
        });
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
            water: finest_ready,
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
            wet_depth: 0.0,
            full_depth: 1.0,
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
                boulders,
            } => {
                let (gx, gy) = dispatch_grid(params.map_size);
                let [p_discharge, p_helical, p_bed_flux, p_exner, p_steer] =
                    pipes.pipelines_water_morpho.map(get);
                let [p_splat, p_record] = pipes.pipelines_water_splat.map(get);
                let p_armour = get(pipes.pipelines_river[1]);
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
                    _,
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
                    run(&mut pass, p_erode);
                    if let Some(p) = p_armour {
                        pass.set_bind_group(0, &groups.river[set], &[]);
                        pass.set_pipeline(p);
                        pass.dispatch_workgroups(gx, gy, 1);
                    }
                    for p in [p_apply, p_advect, p_copy] {
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
                if steps > 0
                    && let (Some(d), Some(h)) = (p_discharge, p_helical)
                {
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
                // Boulders move under this frame's flow, then stamp the solver's bed for the
                // next frame (cleared every frame, so removed boulders leave no bump).
                let [p_move, p_obstacle_clear, p_stamp] = pipes.pipelines_boulders.map(get);
                if steps > 0
                    && let (Some(m), Some(c), Some(st)) = (p_move, p_obstacle_clear, p_stamp)
                {
                    pass.set_bind_group(0, &groups.boulders[set], &[]);
                    if boulders > 0 {
                        pass.set_pipeline(m);
                        pass.dispatch_workgroups(boulders.div_ceil(64), 1, 1);
                    }
                    pass.set_pipeline(c);
                    pass.dispatch_workgroups(gx, gy, 1);
                    if boulders > 0 {
                        pass.set_pipeline(st);
                        pass.dispatch_workgroups(boulders, 1, 1);
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
            FrameCmd::RiverCarve { set } => {
                if let Some(p) = get(pipes.pipelines_river[0]) {
                    let (gx, gy) = dispatch_grid(params.map_size);
                    pass.set_pipeline(p);
                    pass.set_bind_group(0, &groups.river[set], &[]);
                    pass.dispatch_workgroups(gx, gy, 1);
                }
            }
            FrameCmd::Display {
                set,
                offset,
                ao,
                water,
            } => {
                let (gx, gy) = dispatch_grid(params.map_size);
                let erosion = &groups.erosion[set];
                let passes = [
                    (pipes.pipeline_blit, &groups.blit),
                    (pipes.pipeline_ao, &groups.blit),
                    (pipes.pipeline_blur_h, &groups.blit_blur),
                    (pipes.pipeline_blur_v, &groups.blit_blur),
                ];
                let fields = (pipes.pipeline_blit_fields, &groups.debug_fields);
                for &(id, group1) in passes[..if ao { 4 } else { 1 }]
                    .iter()
                    .chain(std::iter::once(&fields))
                {
                    if let Some(p) = get(id) {
                        pass.set_pipeline(p);
                        pass.set_bind_group(0, erosion, &[offset]);
                        pass.set_bind_group(1, group1, &[]);
                        pass.dispatch_workgroups(gx, gy, 1);
                    }
                }
                if water && let Some(p) = get(pipes.pipeline_morpho_debug) {
                    pass.set_pipeline(p);
                    pass.set_bind_group(0, &groups.water_morpho[set], &[]);
                    pass.set_bind_group(1, &groups.debug_water, &[]);
                    pass.dispatch_workgroups(gx, gy, 1);
                }
                // The surface follows the terrain and the ocean even while no water moves.
                if water && let Some(p) = get(pipes.pipelines_water[10]) {
                    pass.set_pipeline(p);
                    pass.set_bind_group(0, &groups.water[set], &[]);
                    pass.set_bind_group(1, &groups.water_out, &[]);
                    pass.dispatch_workgroups(gx, gy, 1);
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
            assert!(
                work(&steps) < 7.0 + 1.0,
                "frame {frames} did {} passes of work",
                work(&steps)
            );
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
