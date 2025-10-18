use bevy::{
    asset::RenderAssetUsages,
    camera::Exposure,
    core_pipeline::tonemapping::Tonemapping,
    light::{AtmosphereEnvironmentMapLight, light_consts::lux},
    pbr::{
        Atmosphere, AtmosphereSettings, ExtendedMaterial, MaterialExtension, MaterialPlugin,
        OpaqueRendererMethod,
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

const BASE_TILE_SIZE: UVec2 = UVec2::new(512, 512);
const OVERLAP_SIZE: u32 = 4; // Halo region for neighbor data exchange
const TILE_COUNT: UVec2 = UVec2::new(2, 2); // 2x2 grid of tiles

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

#[derive(Resource, Clone, ExtractResource, ShaderType)]
struct TileConfig {
    base_tile_size: UVec2,
    overlap_size: u32,
    tile_count: UVec2,
    // Derived values (computed in setup)
    tile_size_with_overlap: UVec2,
    total_world_size: UVec2,
}

impl TileConfig {
    fn new(base_tile_size: UVec2, overlap_size: u32, tile_count: UVec2) -> Self {
        let tile_size_with_overlap = base_tile_size + UVec2::splat(overlap_size * 2);
        let total_world_size = base_tile_size * tile_count;
        Self {
            base_tile_size,
            overlap_size,
            tile_count,
            tile_size_with_overlap,
            total_world_size,
        }
    }

    fn tile_index_to_coord(&self, tile_idx: usize) -> UVec2 {
        let tx = (tile_idx as u32) % self.tile_count.x;
        let ty = (tile_idx as u32) / self.tile_count.x;
        UVec2::new(tx, ty)
    }

    fn coord_to_tile_index(&self, coord: UVec2) -> Option<usize> {
        if coord.x >= self.tile_count.x || coord.y >= self.tile_count.y {
            None
        } else {
            Some((coord.y * self.tile_count.x + coord.x) as usize)
        }
    }

    fn total_tiles(&self) -> usize {
        (self.tile_count.x * self.tile_count.y) as usize
    }

    // Get neighbor tile indices: [left, right, top, bottom]
    // Returns None for edges where there's no neighbor
    fn get_neighbor_indices(&self, tile_idx: usize) -> [Option<usize>; 4] {
        let coord = self.tile_index_to_coord(tile_idx);
        let tx = coord.x as i32;
        let ty = coord.y as i32;

        [
            // Left
            if tx > 0 {
                self.coord_to_tile_index(UVec2::new((tx - 1) as u32, ty as u32))
            } else {
                None
            },
            // Right
            if tx < (self.tile_count.x as i32 - 1) {
                self.coord_to_tile_index(UVec2::new((tx + 1) as u32, ty as u32))
            } else {
                None
            },
            // Top
            if ty > 0 {
                self.coord_to_tile_index(UVec2::new(tx as u32, (ty - 1) as u32))
            } else {
                None
            },
            // Bottom
            if ty < (self.tile_count.y as i32 - 1) {
                self.coord_to_tile_index(UVec2::new(tx as u32, (ty + 1) as u32))
            } else {
                None
            },
        ]
    }
}

#[derive(Resource, Clone, ExtractResource)]
struct ErosionImages {
    tiles: Vec<TileImages>,
}

#[derive(Clone)]
struct TileImages {
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
    // UV crop region to exclude overlap borders: min UV and max UV
    #[uniform(108)]
    uv_min: Vec2,
    #[uniform(109)]
    uv_max: Vec2,
}

impl Default for TerrainExtension {
    fn default() -> Self {
        Self {
            height_tex: Default::default(),
            color_tex: Default::default(),
            analysis_tex: Default::default(),
            height_scale: 50.0,
            preview_mode: 0,
            uv_min: Vec2::ZERO,
            uv_max: Vec2::ONE,
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
    tile_world_offset: Vec2, // World-space offset for noise sampling (per-tile)
    _padding: Vec2,          // Align to 16 bytes
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

fn create_tile_texture(
    size: UVec2,
    format: TextureFormat,
    images: &mut ResMut<Assets<Image>>,
) -> Handle<Image> {
    let mut texture = Image::new_target_texture(size.x, size.y, format);
    texture.asset_usage = RenderAssetUsages::RENDER_WORLD;
    texture.texture_descriptor.usage = TextureUsages::COPY_DST
        | TextureUsages::COPY_SRC
        | TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING;
    images.add(texture)
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    // Create tile configuration
    let tile_config = TileConfig::new(BASE_TILE_SIZE, OVERLAP_SIZE, TILE_COUNT);
    let tile_size = tile_config.tile_size_with_overlap;

    println!("Tile configuration:");
    println!(
        "  Base tile size: {}x{}",
        tile_config.base_tile_size.x, tile_config.base_tile_size.y
    );
    println!("  Overlap size: {}", tile_config.overlap_size);
    println!(
        "  Tile count: {}x{}",
        tile_config.tile_count.x, tile_config.tile_count.y
    );
    println!("  Tile size with overlap: {}x{}", tile_size.x, tile_size.y);
    println!(
        "  Total world size: {}x{}",
        tile_config.total_world_size.x, tile_config.total_world_size.y
    );

    // Create textures for each tile
    let mut tiles = Vec::new();
    for tile_idx in 0..tile_config.total_tiles() {
        let tile_coord = tile_config.tile_index_to_coord(tile_idx);
        println!(
            "Creating textures for tile {} at ({}, {})",
            tile_idx, tile_coord.x, tile_coord.y
        );

        tiles.push(TileImages {
            height: create_tile_texture(tile_size, TextureFormat::Rgba16Float, &mut images),
            color: create_tile_texture(tile_size, TextureFormat::Rgba16Float, &mut images),
            normal: create_tile_texture(tile_size, TextureFormat::Rgba16Float, &mut images),
            analysis: create_tile_texture(tile_size, TextureFormat::Rgba16Float, &mut images),
        });
    }

    commands.insert_resource(ErosionImages { tiles });

    // Parameters - per tile
    let brush_radius: i32 = 16;
    let (brush_indices, brush_weights) = build_brush(tile_size.x as i32, brush_radius);
    let num_particles_per_tile: u32 = 1024 * 8;
    let overlap_size = tile_config.overlap_size; // Store before moving

    commands.insert_resource(tile_config);

    commands.insert_resource(ErodeParams {
        map_size: tile_size, // Each tile size including overlap
        max_lifetime: 32,
        border_size: overlap_size, // Prevent droplets from writing to halo region
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
        num_particles: num_particles_per_tile,
        tile_world_offset: Vec2::ZERO, // Will be set per-tile in prepare_erosion_bind_groups
        _padding: Vec2::ZERO,
    });
    commands.insert_resource(ErosionCpuBuffers {
        brush_indices,
        brush_weights,
    });
    commands.insert_resource(ResetSim::default());
    commands.insert_resource(SimControl::default());
    commands.insert_resource(CurrentPreviewMode::default());

    commands.insert_resource(AmbientLight::NONE);
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
    erosion_images: Res<ErosionImages>,
    tile_config: Res<TileConfig>,
) {
    let tile_world_size = tile_config.base_tile_size.x as f32;
    let resolution = 511; // subdivisions per tile (reduced for performance with multiple tiles)
    let half_size = tile_world_size * 0.5;

    // Create mesh once and reuse for all tiles
    let plane = Mesh::from(
        Plane3d::default()
            .mesh()
            .subdivisions(resolution)
            .size(half_size, half_size),
    );
    let mesh_handle = meshes.add(plane);

    // Calculate UV crop region to exclude halo/overlap borders
    let tile_size_with_overlap = tile_config.tile_size_with_overlap.x as f32;
    let overlap = tile_config.overlap_size as f32;
    let uv_min = Vec2::splat(overlap / tile_size_with_overlap);
    let uv_max = Vec2::splat((tile_size_with_overlap - overlap) / tile_size_with_overlap);

    println!("UV crop region: min={:?}, max={:?}", uv_min, uv_max);

    // Spawn one terrain mesh per tile
    for tile_idx in 0..tile_config.total_tiles() {
        let tile_coord = tile_config.tile_index_to_coord(tile_idx);
        let tile_images = &erosion_images.tiles[tile_idx];

        // Position tiles so they touch at base_tile_size intervals (no gaps)
        let world_x = tile_coord.x as f32 * tile_world_size * 0.5 - half_size * 0.5;
        let world_z = tile_coord.y as f32 * tile_world_size * 0.5 - half_size * 0.5;

        let mat_handle = materials.add(ExtendedMaterial {
            base: StandardMaterial {
                base_color: Color::srgb(1.0, 1.0, 1.0),
                perceptual_roughness: 0.8,
                metallic: 0.0,
                opaque_render_method: OpaqueRendererMethod::Auto,
                ..Default::default()
            },
            extension: TerrainExtension {
                height_tex: tile_images.height.clone(),
                color_tex: tile_images.color.clone(),
                analysis_tex: tile_images.analysis.clone(),
                height_scale: 50.0,
                preview_mode: 0, // Default to PBR
                uv_min,          // Crop to exclude overlap borders
                uv_max,
            },
        });

        commands.spawn((
            Mesh3d(mesh_handle.clone()),
            MeshMaterial3d(mat_handle),
            Transform::from_xyz(world_x, 0.0, world_z),
        ));
    }

    // 3D camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(150.0, 50.0, 150.0).looking_at(Vec3::ZERO, Vec3::Y),
        OrbitController {
            target: Vec3::ZERO,
            distance: 225.0,
            ..Default::default()
        },
        Atmosphere::EARTH,
        AtmosphereSettings {
            scene_units_to_m: 50.0,
            // rendering_method: AtmosphereMode::Raymarched,
            ..Default::default()
        },
        AtmosphereEnvironmentMapLight::default(),
        Exposure { ev100: 13.0 },
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
            ExtractResourcePlugin::<TileConfig>::default(),
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
    layout_erosion: BindGroupLayout,
    layout_color: BindGroupLayout,
    layout_blit: BindGroupLayout,
    pipeline_init: CachedComputePipelineId,
    pipeline_init_color: CachedComputePipelineId,
    pipeline_erode: CachedComputePipelineId,
    pipeline_halo_exchange: CachedComputePipelineId,
    pipeline_blit: CachedComputePipelineId,
}

#[derive(Resource)]
struct ErosionBindGroups {
    tiles: Vec<TileBindGroups>,
}

struct TileBindGroups {
    erosion: BindGroup,
    color: BindGroup,
    blit: BindGroup,
}

#[derive(Resource)]
struct ErosionBuffers {
    tiles: Vec<TileBuffers>,
    // Shared across all tiles (brush data only)
    brush_idx: StorageBuffer<Vec<i32>>,
    brush_w: StorageBuffer<Vec<f32>>,
    // Dummy buffer for edge tiles without neighbors (to avoid binding conflicts)
    dummy_height: StorageBuffer<Vec<f32>>,
}

struct TileBuffers {
    uniform: UniformBuffer<ErodeParams>, // Per-tile params (contains tile_world_offset)
    height: StorageBuffer<Vec<f32>>,
    random: StorageBuffer<Vec<u32>>,
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
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    // Layout for erosion (group 0)
    let layout_erosion = render_device.create_bind_group_layout(
        "ErosionGroup0",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer::<f32>(false),           // height (read_write)
                uniform_buffer::<ErodeParams>(false),   // params
                storage_buffer_read_only::<u32>(false), // random_indices
                storage_buffer_read_only::<i32>(false), // brush_indices
                storage_buffer_read_only::<f32>(false), // brush_weights
                storage_buffer::<u32>(false),           // flow (atomic<u32>)
                storage_buffer::<u32>(false),           // sediment (atomic<u32>)
                storage_buffer::<u32>(false),           // erosion (atomic<u32>)
                storage_buffer_read_only::<f32>(false), // height_left neighbor
                storage_buffer_read_only::<f32>(false), // height_right neighbor
                storage_buffer_read_only::<f32>(false), // height_top neighbor
                storage_buffer_read_only::<f32>(false), // height_bottom neighbor
            ),
        ),
    );

    // Layout for blit (group 1)
    let layout_blit = render_device.create_bind_group_layout(
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
    let layout_color = render_device.create_bind_group_layout(
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
    let pipeline_halo_exchange = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![layout_erosion.clone()],
        shader: shader.clone(),
        entry_point: Some(Cow::from("halo_exchange")),
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
        pipeline_halo_exchange,
        pipeline_blit,
    });
}

fn prepare_erosion_bind_groups(
    mut commands: Commands,
    pipeline: Res<ErosionPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    images: Res<ErosionImages>,
    tile_config: Res<TileConfig>,
    params: Res<ErodeParams>,
    cpu_buffers: Res<ErosionCpuBuffers>,
    render_device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut existing: Option<ResMut<ErosionBuffers>>,
    rng_state: Option<ResMut<ErosionRngState>>,
) {
    let texels_per_tile = (params.map_size.x * params.map_size.y) as usize;
    let num_tiles = tile_config.total_tiles();

    // Check if we need to recreate buffers (e.g., tile count changed)
    let needs_recreation = if let Some(ref buffers) = existing {
        buffers.tiles.len() != num_tiles
            || buffers.tiles.iter().any(|tile| {
                tile.flow.buffer().is_none() || tile.flow.get().len() != texels_per_tile
            })
    } else {
        false
    };

    if needs_recreation {
        existing = None;
    }

    match existing {
        Some(mut buffers) => {
            // Update RNG state and regenerate random indices for each tile
            let mut seed = if let Some(ref rs) = rng_state {
                rs.state
            } else {
                0xCAFEBABE_1234_5678
            };

            let mut tile_bind_groups = Vec::new();

            // Update random indices for all tiles first
            for tile_idx in 0..num_tiles {
                let tile_buffers = &mut buffers.tiles[tile_idx];

                // Generate new random indices for this tile
                let new_random = generate_random_indices_seeded(
                    params.num_particles,
                    params.map_size.x * params.map_size.y,
                    seed,
                );
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);

                tile_buffers.random = StorageBuffer::from(new_random);
                tile_buffers.random.write_buffer(&render_device, &queue);
            }

            // Now create bind groups (with immutable borrows only)
            for tile_idx in 0..num_tiles {
                let tile_images = &images.tiles[tile_idx];
                let tile_buffers = &buffers.tiles[tile_idx];

                // Get neighbor tile indices [left, right, top, bottom]
                let neighbors = tile_config.get_neighbor_indices(tile_idx);

                // Get neighbor height buffers (use dummy if neighbor doesn't exist)
                let height_left = neighbors[0]
                    .map(|idx| &buffers.tiles[idx].height)
                    .unwrap_or(&buffers.dummy_height);
                let height_right = neighbors[1]
                    .map(|idx| &buffers.tiles[idx].height)
                    .unwrap_or(&buffers.dummy_height);
                let height_top = neighbors[2]
                    .map(|idx| &buffers.tiles[idx].height)
                    .unwrap_or(&buffers.dummy_height);
                let height_bottom = neighbors[3]
                    .map(|idx| &buffers.tiles[idx].height)
                    .unwrap_or(&buffers.dummy_height);

                // Get GPU texture views for this tile
                let view_height = gpu_images.get(&tile_images.height).unwrap();
                let view_color = gpu_images.get(&tile_images.color).unwrap();
                let view_normal = gpu_images.get(&tile_images.normal).unwrap();
                let view_analysis = gpu_images.get(&tile_images.analysis).unwrap();

                // Create bind groups for this tile
                let erosion = render_device.create_bind_group(
                    None,
                    &pipeline.layout_erosion,
                    &BindGroupEntries::sequential((
                        &tile_buffers.height,
                        &tile_buffers.uniform,
                        &tile_buffers.random,
                        &buffers.brush_idx,
                        &buffers.brush_w,
                        &tile_buffers.flow,
                        &tile_buffers.sediment,
                        &tile_buffers.erosion,
                        height_left,
                        height_right,
                        height_top,
                        height_bottom,
                    )),
                );

                let blit = render_device.create_bind_group(
                    None,
                    &pipeline.layout_blit,
                    &BindGroupEntries::sequential((
                        &view_height.texture_view,
                        &view_normal.texture_view,
                        &view_analysis.texture_view,
                    )),
                );

                let color = render_device.create_bind_group(
                    None,
                    &pipeline.layout_color,
                    &BindGroupEntries::sequential((&view_color.texture_view,)),
                );

                tile_bind_groups.push(TileBindGroups {
                    erosion,
                    color,
                    blit,
                });
            }

            // Update RNG state
            if let Some(mut rs) = rng_state {
                rs.state = seed;
            } else {
                commands.insert_resource(ErosionRngState { state: seed });
            }

            commands.insert_resource(ErosionBindGroups {
                tiles: tile_bind_groups,
            });
        }
        None => {
            // Create shared buffers (brush data only)
            let mut brush_idx = StorageBuffer::from(cpu_buffers.brush_indices.clone());
            brush_idx.write_buffer(&render_device, &queue);

            let mut brush_w = StorageBuffer::from(cpu_buffers.brush_weights.clone());
            brush_w.write_buffer(&render_device, &queue);

            // Create dummy buffer for edge tiles (same size as a height buffer)
            let mut dummy_height = StorageBuffer::from(vec![0.0f32; texels_per_tile]);
            dummy_height.write_buffer(&render_device, &queue);

            // Initialize RNG state
            let mut seed = if let Some(ref rs) = rng_state {
                rs.state
            } else {
                0xCAFEBABE_1234_5678
            };

            // Create per-tile buffers first (we need all buffers to exist before creating bind groups)
            let mut tile_buffers = Vec::new();

            for tile_idx in 0..num_tiles {
                let tile_coord = tile_config.tile_index_to_coord(tile_idx);

                // Calculate world offset for this tile (for noise sampling)
                let tile_world_offset = Vec2::new(
                    tile_coord.x as f32 * tile_config.base_tile_size.x as f32,
                    tile_coord.y as f32 * tile_config.base_tile_size.y as f32,
                );

                // Create per-tile uniform with world offset
                let mut tile_params = params.clone();
                tile_params.tile_world_offset = tile_world_offset;
                let mut uniform = UniformBuffer::from(tile_params);
                uniform.write_buffer(&render_device, &queue);

                // Create height buffer for this tile
                let mut height = StorageBuffer::from(vec![0.0f32; texels_per_tile]);
                height.write_buffer(&render_device, &queue);

                // Generate random indices for this tile
                let first_random = generate_random_indices_seeded(
                    params.num_particles,
                    params.map_size.x * params.map_size.y,
                    seed,
                );
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let mut random = StorageBuffer::from(first_random);
                random.write_buffer(&render_device, &queue);

                // Create analysis buffers (flow, sediment, erosion)
                let mut flow = StorageBuffer::from(vec![0u32; texels_per_tile]);
                flow.write_buffer(&render_device, &queue);

                let mut sediment = StorageBuffer::from(vec![0u32; texels_per_tile]);
                sediment.write_buffer(&render_device, &queue);

                let mut erosion_buf = StorageBuffer::from(vec![0u32; texels_per_tile]);
                erosion_buf.write_buffer(&render_device, &queue);

                tile_buffers.push(TileBuffers {
                    uniform,
                    height,
                    random,
                    flow,
                    sediment,
                    erosion: erosion_buf,
                });
            }

            // Now create bind groups with neighbor references
            let mut tile_bind_groups = Vec::new();

            for tile_idx in 0..num_tiles {
                let tile_images = &images.tiles[tile_idx];
                let tile_buffers_ref = &tile_buffers[tile_idx];

                // Get neighbor tile indices [left, right, top, bottom]
                let neighbors = tile_config.get_neighbor_indices(tile_idx);

                // Get neighbor height buffers (use dummy if neighbor doesn't exist)
                let height_left = neighbors[0]
                    .map(|idx| &tile_buffers[idx].height)
                    .unwrap_or(&dummy_height);
                let height_right = neighbors[1]
                    .map(|idx| &tile_buffers[idx].height)
                    .unwrap_or(&dummy_height);
                let height_top = neighbors[2]
                    .map(|idx| &tile_buffers[idx].height)
                    .unwrap_or(&dummy_height);
                let height_bottom = neighbors[3]
                    .map(|idx| &tile_buffers[idx].height)
                    .unwrap_or(&dummy_height);

                // Get GPU texture views for this tile
                let view_height = gpu_images.get(&tile_images.height).unwrap();
                let view_color = gpu_images.get(&tile_images.color).unwrap();
                let view_normal = gpu_images.get(&tile_images.normal).unwrap();
                let view_analysis = gpu_images.get(&tile_images.analysis).unwrap();

                // Build bind groups for this tile
                let erosion_bg = render_device.create_bind_group(
                    None,
                    &pipeline.layout_erosion,
                    &BindGroupEntries::sequential((
                        &tile_buffers_ref.height,
                        &tile_buffers_ref.uniform,
                        &tile_buffers_ref.random,
                        &brush_idx,
                        &brush_w,
                        &tile_buffers_ref.flow,
                        &tile_buffers_ref.sediment,
                        &tile_buffers_ref.erosion,
                        height_left,
                        height_right,
                        height_top,
                        height_bottom,
                    )),
                );

                let blit = render_device.create_bind_group(
                    None,
                    &pipeline.layout_blit,
                    &BindGroupEntries::sequential((
                        &view_height.texture_view,
                        &view_normal.texture_view,
                        &view_analysis.texture_view,
                    )),
                );

                let color = render_device.create_bind_group(
                    None,
                    &pipeline.layout_color,
                    &BindGroupEntries::sequential((&view_color.texture_view,)),
                );

                tile_bind_groups.push(TileBindGroups {
                    erosion: erosion_bg,
                    color,
                    blit,
                });
            }

            // Update RNG state
            if let Some(mut rs) = rng_state {
                rs.state = seed;
            } else {
                commands.insert_resource(ErosionRngState { state: seed });
            }

            // Insert buffers and bind groups as resources
            commands.insert_resource(ErosionBuffers {
                tiles: tile_buffers,
                brush_idx,
                brush_w,
                dummy_height,
            });
            commands.insert_resource(ErosionBindGroups {
                tiles: tile_bind_groups,
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
                    *atmosphere = Atmosphere::EARTH;
                    *tonemapping = Tonemapping::AcesFitted;
                    *bloom = Bloom::NATURAL;
                } else {
                    // Disable atmosphere and bloom for debug modes
                    *atmosphere = Atmosphere {
                        rayleigh_scattering: Vec3::ZERO,
                        mie_scattering: 0.0,
                        ..default()
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
        let tile_config = world.resource::<TileConfig>();

        let gx = (params.map_size.x + 7) / 8;
        let gy = (params.map_size.y + 7) / 8;

        match self.state {
            ErosionState::Loading => {}
            ErosionState::Init => {
                // Initialize all tiles
                for tile_idx in 0..tile_config.total_tiles() {
                    let tile_groups = &groups.tiles[tile_idx];

                    // Pass 1: init FBM (writes buffer)
                    {
                        let mut pass = render_context
                            .command_encoder()
                            .begin_compute_pass(&ComputePassDescriptor::default());
                        let p_init = cache.get_compute_pipeline(pipes.pipeline_init).unwrap();
                        pass.set_pipeline(p_init);
                        pass.set_bind_group(0, &tile_groups.erosion, &[]);
                        pass.dispatch_workgroups(gx, gy, 1);
                    }

                    // Pass 1b: init color bands
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
                        pass.set_bind_group(0, &tile_groups.erosion, &[]);
                        pass.set_bind_group(1, &tile_groups.color, &[]);
                        pass.dispatch_workgroups(gx, gy, 1);
                    }

                    // Pass 2: blit
                    {
                        let mut pass = render_context
                            .command_encoder()
                            .begin_compute_pass(&ComputePassDescriptor::default());
                        let p_blit = cache.get_compute_pipeline(pipes.pipeline_blit).unwrap();
                        pass.set_pipeline(p_blit);
                        pass.set_bind_group(0, &tile_groups.erosion, &[]);
                        pass.set_bind_group(1, &tile_groups.blit, &[]);
                        pass.dispatch_workgroups(gx, gy, 1);
                    }
                }
            }
            ErosionState::Erode => {
                // Erode all tiles
                for tile_idx in 0..tile_config.total_tiles() {
                    let tile_groups = &groups.tiles[tile_idx];

                    // Pass 1: erode
                    if self.allow_erode_this_frame {
                        if let Some(p_erode) = cache.get_compute_pipeline(pipes.pipeline_erode) {
                            let mut pass = render_context
                                .command_encoder()
                                .begin_compute_pass(&ComputePassDescriptor::default());
                            pass.set_pipeline(p_erode);
                            pass.set_bind_group(0, &tile_groups.erosion, &[]);
                            pass.set_bind_group(1, &tile_groups.color, &[]);
                            let workgroups = (params.num_particles + 1023) / 1024;
                            pass.dispatch_workgroups(workgroups, 1, 1);
                        } else {
                            error!("Erode pipeline not ready!");
                        }

                        // Pass 1.5: halo exchange (sync neighbor data into halo regions)
                        if let Some(p_halo) =
                            cache.get_compute_pipeline(pipes.pipeline_halo_exchange)
                        {
                            let mut pass = render_context
                                .command_encoder()
                                .begin_compute_pass(&ComputePassDescriptor::default());
                            pass.set_pipeline(p_halo);
                            pass.set_bind_group(0, &tile_groups.erosion, &[]);
                            pass.dispatch_workgroups(gx, gy, 1);
                        }
                    }

                    // Pass 2: blit
                    {
                        let mut pass = render_context
                            .command_encoder()
                            .begin_compute_pass(&ComputePassDescriptor::default());
                        let p_blit = cache.get_compute_pipeline(pipes.pipeline_blit).unwrap();
                        pass.set_pipeline(p_blit);
                        pass.set_bind_group(0, &tile_groups.erosion, &[]);
                        pass.set_bind_group(1, &tile_groups.blit, &[]);
                        pass.dispatch_workgroups(gx, gy, 1);
                    }
                }
            }
        }

        // Keep buffers alive
        let _keep = (&buffers.brush_idx, &buffers.brush_w);

        Ok(())
    }
}
