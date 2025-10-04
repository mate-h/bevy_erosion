use bevy::{
    asset::RenderAssetUsages,
    pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin, OpaqueRendererMethod},
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
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

const EROSION_SHADER: &str = "erosion.wgsl";
const TERRAIN_SHADER: &str = "terrain_extended.wgsl";

const SIZE: UVec2 = UVec2::new(512, 512);

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins((
            DefaultPlugins,
            ErosionComputePlugin,
            PanOrbitCameraPlugin,
            MaterialPlugin::<TerrainMaterial>::default(),
        ))
        .add_systems(Startup, setup)
        .add_systems(PostStartup, spawn_terrain)
        .add_systems(Update, (handle_reset_input, handle_sim_input))
        .run();
}

#[derive(Resource, Clone, ExtractResource)]
struct ErosionImages {
    display: Handle<Image>,
    color: Handle<Image>,
}

type TerrainMaterial = ExtendedMaterial<StandardMaterial, TerrainExtension>;

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
    // Displacement height scale
    #[uniform(104)]
    height_scale: f32,
}

impl Default for TerrainExtension {
    fn default() -> Self {
        Self {
            height_tex: Default::default(),
            color_tex: Default::default(),
            height_scale: 50.0,
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
    random_indices: Vec<u32>,
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

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    // Display texture (rgba8unorm) written by compute blit
    let mut display = Image::new_target_texture(SIZE.x, SIZE.y, TextureFormat::Rgba8Unorm);
    display.asset_usage = RenderAssetUsages::RENDER_WORLD;
    display.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let display_handle = images.add(display);

    // Color texture (rgba8unorm) used as storage for color advection and sampled in terrain shader
    let mut color = Image::new_target_texture(SIZE.x, SIZE.y, TextureFormat::Rgba8Unorm);
    color.asset_usage = RenderAssetUsages::RENDER_WORLD;
    color.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let color_handle = images.add(color);

    commands.insert_resource(ErosionImages { display: display_handle, color: color_handle });

    // Parameters
    let brush_radius: i32 = 3;
    let (brush_indices, brush_weights) = build_brush(SIZE.x as i32, brush_radius);
    let num_particles: u32 = 10_000;
    let random_indices = generate_random_indices(num_particles, (SIZE.x * SIZE.y) as u32);

    commands.insert_resource(ErodeParams {
        map_size: SIZE,
        max_lifetime: 30,
        border_size: 0,
        inertia: 0.5,
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
        random_indices,
        brush_indices,
        brush_weights,
    });
    commands.insert_resource(ResetSim::default());
    commands.insert_resource(SimControl::default());
}

fn spawn_terrain(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>, 
    mut materials: ResMut<Assets<TerrainMaterial>>,
    erosion_images: Res<ErosionImages>,
) {
    // Spawn a subdivided plane with UVs
    let size = 256.0;
    let resolution = 255; // subdivisions
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
            height_tex: erosion_images.display.clone(),
            color_tex: erosion_images.color.clone(),
            height_scale: 50.0,
        },
    });

    commands.spawn((Mesh3d(mesh_handle), MeshMaterial3d(mat_handle), Transform::from_xyz(0.0, 0.0, 0.0)));

    // 3D camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(150.0, 150.0, 150.0).looking_at(Vec3::ZERO, Vec3::Y),
        PanOrbitCamera::default(),
    ));

    // Directional light for PBR shading
    commands.spawn((
        DirectionalLight::default(),
        Transform::from_xyz(0.0, 100.0, 0.0)
            .looking_at(Vec3::new(-0.5, -1.0, -0.3), Vec3::Y),
    ));
}

fn generate_random_indices(count: u32, max_index: u32) -> Vec<u32> {
    let mut v = Vec::with_capacity(count as usize);
    let mut state: u64 = 0x1234_5678_ABCD_EFFF;
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
    layout_erosion: BindGroupLayout,
    layout_color: BindGroupLayout,
    layout_blit: BindGroupLayout,
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
                storage_buffer::<f32>(false),
                uniform_buffer::<ErodeParams>(false),
                storage_buffer_read_only::<u32>(false),
                storage_buffer_read_only::<i32>(false),
                storage_buffer_read_only::<f32>(false),
            ),
        ),
    );

    // Layout for blit (group 1)
    let layout_blit = render_device.create_bind_group_layout(
        "ErosionBlitGroup1",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rgba8Unorm, StorageTextureAccess::WriteOnly),
            ),
        ),
    );

    // Layout for color (group 1 for erosion/color init)
    let layout_color = render_device.create_bind_group_layout(
        "ErosionColorGroup1",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(
                    TextureFormat::Rgba8Unorm,
                    StorageTextureAccess::ReadWrite,
                ),
            ),
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
    gpu_images: Res<RenderAssets<GpuImage>>,
    images: Res<ErosionImages>,
    params: Res<ErodeParams>,
    cpu_buffers: Res<ErosionCpuBuffers>,
    render_device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    existing: Option<Res<ErosionBuffers>>,
) {
    let view_display = gpu_images.get(&images.display).unwrap();
    let view_color = gpu_images.get(&images.color).unwrap();

    match existing {
        Some(buffers) => {
            // Reuse existing buffers, just rebuild bind groups (safe across frames)
            let erosion = render_device.create_bind_group(
                None,
                &pipeline.layout_erosion,
                &BindGroupEntries::sequential((
                    &buffers.height,
                    &buffers.uniform,
                    &buffers.random,
                    &buffers.brush_idx,
                    &buffers.brush_w,
                )),
            );
            let blit = render_device.create_bind_group(
                None,
                &pipeline.layout_blit,
                &BindGroupEntries::sequential((&view_display.texture_view,)),
            );
            let color = render_device.create_bind_group(
                None,
                &pipeline.layout_color,
                &BindGroupEntries::sequential((&view_color.texture_view,)),
            );
            commands.insert_resource(ErosionBindGroups { erosion, color, blit });
        }
        None => {
            // Create buffers once
            let mut uniform = UniformBuffer::from(params.clone());
            uniform.write_buffer(&render_device, &queue);

            let texels = (params.map_size.x * params.map_size.y) as usize;
            let mut height = StorageBuffer::from(vec![0.0f32; texels]);
            height.write_buffer(&render_device, &queue);
            // Create and fill initial height once using FBM on CPU for now (optional: let GPU init and copy back)

            let mut random = StorageBuffer::from(cpu_buffers.random_indices.clone());
            random.write_buffer(&render_device, &queue);

            let mut brush_idx = StorageBuffer::from(cpu_buffers.brush_indices.clone());
            brush_idx.write_buffer(&render_device, &queue);

            let mut brush_w = StorageBuffer::from(cpu_buffers.brush_weights.clone());
            brush_w.write_buffer(&render_device, &queue);

            // Build bind groups using these buffers
            let erosion = render_device.create_bind_group(
                None,
                &pipeline.layout_erosion,
                &BindGroupEntries::sequential((
                    &height,
                    &uniform,
                    &random,
                    &brush_idx,
                    &brush_w,
                )),
            );
            let blit = render_device.create_bind_group(
                None,
                &pipeline.layout_blit,
                &BindGroupEntries::sequential((&view_display.texture_view,)),
            );
            let color = render_device.create_bind_group(
                None,
                &pipeline.layout_color,
                &BindGroupEntries::sequential((&view_color.texture_view,)),
            );

            // Insert buffers and groups as resources
            commands.insert_resource(ErosionBuffers { uniform, height, random, brush_idx, brush_w });
            commands.insert_resource(ErosionBindGroups { erosion, color, blit });
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

enum ErosionState {
    Loading,
    Init,
    Erode(usize),
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
                if let CachedPipelineState::Ok(_) = cache.get_compute_pipeline_state(pipeline.pipeline_erode) {
                    self.state = ErosionState::Erode(0);
                }
            }
            ErosionState::Erode(0) => {
                self.state = ErosionState::Erode(1);
            }
            ErosionState::Erode(1) => {
                self.state = ErosionState::Erode(0);
            }
            ErosionState::Erode(_) => {}
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
                if let Some(p_color_ok) = match cache.get_compute_pipeline_state(pipes.pipeline_init_color) {
                    CachedPipelineState::Ok(_) => cache.get_compute_pipeline(pipes.pipeline_init_color),
                    _ => None,
                } {
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
                // No intermediate copy; initial FBM is in-place in height
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
            ErosionState::Erode(_index) => {
                // If reset happened recently, restore from initial before eroding
                // (Handled via state transition to Loading/Init above)
                // Pass 1: erode (writes storage) — gated by pause/step
                if self.allow_erode_this_frame {
                    let mut pass = render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor::default());
                    let p_erode = cache.get_compute_pipeline(pipes.pipeline_erode).unwrap();
                    pass.set_pipeline(p_erode);
                    pass.set_bind_group(0, &groups.erosion, &[]);
                    pass.set_bind_group(1, &groups.color, &[]);
                    let workgroups = (params.num_particles + 1023) / 1024;
                    pass.dispatch_workgroups(workgroups, 1, 1);
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
