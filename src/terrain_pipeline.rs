use bevy::{
    core_pipeline::core_3d::{Opaque3d, Opaque3dBatchSetKey, Opaque3dBinKey, CORE_3D_DEPTH_FORMAT}, ecs::component::Tick, pbr::{
        DrawMesh, MeshPipeline, MeshPipelineKey, MeshPipelineViewLayoutKey, RenderMeshInstances,
        SetMeshBindGroup, SetMeshViewBindGroup,
    }, prelude::*, render::{
        batching::gpu_preprocessing::GpuPreprocessingSupport, extract_component::{ExtractComponent, ExtractComponentPlugin}, mesh::allocator::MeshAllocator, render_asset::RenderAssets, render_phase::{
            AddRenderCommand, BinnedRenderPhaseType, DrawFunctions, PhaseItem, RenderCommand, RenderCommandResult, SetItemPipeline, TrackedRenderPass,
            ViewBinnedRenderPhases,
        }, render_resource::{
            binding_types::{sampler, texture_2d}, BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, ColorTargetState, ColorWrites, CompareFunction, DepthStencilState, Face, FragmentState, FrontFace, MultisampleState, PipelineCache, PolygonMode, PrimitiveState, RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages, SpecializedMeshPipeline, SpecializedMeshPipelineError, SpecializedMeshPipelines, TextureFormat, TextureSampleType, VertexState
        }, texture::GpuImage, view::{ExtractedView, RenderVisibleEntities, ViewTarget}, Render, RenderApp, RenderStartup, RenderSystems
    }
};
use bevy::camera::visibility::{self, VisibilityClass};
use bevy::ecs::system::lifetimeless::SRes;
use bevy::ecs::system::SystemParamItem;
use bevy::ecs::query::ROQueryItem;

const TERRAIN_SHADER: &str = "terrain.wgsl";

#[derive(Resource)]
pub struct TerrainPipeline {
    mesh_pipeline: MeshPipeline,
    shader: Handle<Shader>,
    layout_terrain: BindGroupLayout,
}

#[derive(Component, Clone, ExtractComponent)]
#[require(VisibilityClass)]
#[component(on_add = visibility::add_visibility_class::<TerrainMarker>)]
pub struct TerrainMarker;

#[derive(Resource)]
pub struct TerrainBindGroup {
    pub bind_group: BindGroup,
    pub sampler: Sampler,
}

struct SetTerrainBindGroup<const I: usize>;

impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetTerrainBindGroup<I> {
    type Param = SRes<TerrainBindGroup>;
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        _phase_item: &P,
        _view: ROQueryItem<'w, '_, Self::ViewQuery>,
        _item: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        terrain_bg: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let terrain_bg = terrain_bg.into_inner();
        pass.set_bind_group(I, &terrain_bg.bind_group, &[]);
        RenderCommandResult::Success
    }
}

type DrawTerrain = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetTerrainBindGroup<1>,
    SetMeshBindGroup<2>,
    DrawMesh,
);

pub struct TerrainPlugin;
impl Plugin for TerrainPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<TerrainMarker>::default());
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<SpecializedMeshPipelines<TerrainPipeline>>()
            .add_render_command::<Opaque3d, DrawTerrain>()
            .add_systems(RenderStartup, init_terrain_pipeline)
            .add_systems(
                Render,
                (
                    prepare_terrain_bind_group.in_set(RenderSystems::PrepareBindGroups),
                    queue_terrain.in_set(RenderSystems::Queue),
                ),
            );
    }
}

fn init_terrain_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mesh_pipeline: Res<MeshPipeline>,
    render_device: Res<bevy::render::renderer::RenderDevice>,
) {
    let layout_terrain = render_device.create_bind_group_layout(
        "TerrainLayout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                texture_2d(TextureSampleType::Float {
                    filterable: true,
                }),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    );
    let shader: Handle<Shader> = asset_server.load(TERRAIN_SHADER);
    commands.insert_resource(TerrainPipeline {
        mesh_pipeline: mesh_pipeline.clone(),
        shader,
        layout_terrain,
    });
}

impl SpecializedMeshPipeline for TerrainPipeline {
    type Key = MeshPipelineKey;
    fn specialize(
        &self,
        mesh_key: Self::Key,
        layout: &bevy::mesh::MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut attributes = Vec::new();
        if layout.0.contains(Mesh::ATTRIBUTE_POSITION) {
            attributes.push(Mesh::ATTRIBUTE_POSITION.at_shader_location(0));
        }
        if layout.0.contains(Mesh::ATTRIBUTE_NORMAL) {
            attributes.push(Mesh::ATTRIBUTE_NORMAL.at_shader_location(1));
        }
        if layout.0.contains(Mesh::ATTRIBUTE_UV_0) {
            attributes.push(Mesh::ATTRIBUTE_UV_0.at_shader_location(2));
        }
        let vb = layout.0.get_layout(&attributes)?;
        let view_layout = self
            .mesh_pipeline
            .get_view_layout(MeshPipelineViewLayoutKey::from(mesh_key));
        Ok(RenderPipelineDescriptor {
            label: Some("TerrainPipeline".into()),
            layout: vec![
                view_layout.main_layout.clone(),
                self.layout_terrain.clone(),
                self.mesh_pipeline.mesh_layouts.model_only.clone(),
            ],
            vertex: VertexState {
                shader: self.shader.clone(),
                buffers: vec![vb],
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                targets: vec![Some(ColorTargetState {
                    format: if mesh_key.contains(MeshPipelineKey::HDR) {
                        ViewTarget::TEXTURE_FORMAT_HDR
                    } else {
                        TextureFormat::bevy_default()
                    },
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            primitive: PrimitiveState {
                topology: mesh_key.primitive_topology(),
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                polygon_mode: PolygonMode::Fill,
                ..default()
            },
            // Note that if your view has no depth buffer this will need to be
            // changed.
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: CompareFunction::GreaterEqual,
                stencil: default(),
                bias: default(),
            }),
            // It's generally recommended to specialize your pipeline for MSAA,
            // but it's not always possible
            multisample: MultisampleState {
                count: mesh_key.msaa_samples(),
                ..default()
            },
            ..default()
        })
    }
}

fn prepare_terrain_bind_group(
    mut commands: Commands,
    terrain_pipeline: Res<TerrainPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    erosion_images: Res<crate::ErosionImages>,
    render_device: Res<bevy::render::renderer::RenderDevice>,
) {
    if let Some(view) = gpu_images.get(&erosion_images.display) {
        let sampler = render_device.create_sampler(&SamplerDescriptor::default());
        let bind_group = render_device.create_bind_group(
            None,
            &terrain_pipeline.layout_terrain,
            &BindGroupEntries::sequential((&view.texture_view, &sampler)),
        );
        commands.insert_resource(TerrainBindGroup {
            bind_group,
            sampler,
        });
    }
}

fn queue_terrain(
    pipeline_cache: Res<PipelineCache>,
    terrain_pipeline: Res<TerrainPipeline>,
    mut specialized: ResMut<SpecializedMeshPipelines<TerrainPipeline>>,
    (mut opaque_render_phases, draw_functions): (
        ResMut<ViewBinnedRenderPhases<Opaque3d>>,
        Res<DrawFunctions<Opaque3d>>,
    ),
    views: Query<(&RenderVisibleEntities, &ExtractedView, &Msaa)>,
    (render_meshes, render_mesh_instances): (
        Res<RenderAssets<bevy::render::mesh::RenderMesh>>,
        Res<RenderMeshInstances>,
    ),
    mesh_allocator: Res<MeshAllocator>,
    gpu_preprocessing_support: Res<GpuPreprocessingSupport>,
    maybe_bg: Option<Res<TerrainBindGroup>>,
) {
    if maybe_bg.is_none() {
        return;
    }
    let draw_function = draw_functions.read().id::<DrawTerrain>();
    for (visible, view, msaa) in &views {
        let Some(phase) = opaque_render_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        let view_key = MeshPipelineKey::from_msaa_samples(msaa.samples())
            | MeshPipelineKey::from_hdr(view.hdr);
        for &(render_entity, visible_entity) in visible.get::<TerrainMarker>().iter() {
            let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(visible_entity)
            else {
                continue;
            };
            let Some(mesh) = render_meshes.get(mesh_instance.mesh_asset_id) else {
                continue;
            };
            let (vertex_slab, index_slab) = mesh_allocator.mesh_slabs(&mesh_instance.mesh_asset_id);
            let mut key = view_key;
            key |= MeshPipelineKey::from_primitive_topology(mesh.primitive_topology());
            let pipeline_id = specialized
                .specialize(&pipeline_cache, &terrain_pipeline, key, &mesh.layout)
                .expect("Failed to specialize terrain pipeline");
            phase.add(
                Opaque3dBatchSetKey {
                    draw_function,
                    pipeline: pipeline_id,
                    material_bind_group_index: None,
                    vertex_slab: vertex_slab.unwrap_or_default(),
                    index_slab,
                    lightmap_slab: None,
                },
                Opaque3dBinKey {
                    asset_id: mesh_instance.mesh_asset_id.into(),
                },
                (render_entity, visible_entity),
                mesh_instance.current_uniform_index,
                BinnedRenderPhaseType::mesh(
                    mesh_instance.should_batch(),
                    &gpu_preprocessing_support,
                ),
                Tick::new(0),
            );
        }
    }
}
