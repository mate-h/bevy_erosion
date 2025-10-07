#import bevy_pbr::{
    mesh_functions,
    view_transformations::{position_world_to_clip, position_world_to_view},
    forward_io::{VertexOutput, FragmentOutput},
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing, alpha_discard},
}

// Extension bind group uses MATERIAL_BIND_GROUP with slots starting at 100
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var height_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var height_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var color_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(103) var color_sampler: sampler;

struct TerrainParams {
    height_scale: f32,
}
@group(#{MATERIAL_BIND_GROUP}) @binding(104) var<uniform> terrain_params: TerrainParams;

struct VertexIn {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    // Optional attributes (tangent, color, etc.) are handled by the base pipeline; we only need core ones here
};

@vertex
fn vertex(v: VertexIn) -> VertexOutput {
    // Sample height at UV to displace along local +Y
    let h = textureSampleLevel(height_tex, height_sampler, v.uv, 0.0).x;
    let displaced_local = vec3<f32>(v.position.x, v.position.y + h * terrain_params.height_scale, v.position.z);

    var out: VertexOutput;
    var world_from_local = mesh_functions::get_world_from_local(v.instance_index);
    let world_pos4 = mesh_functions::mesh_position_local_to_world(world_from_local, vec4(displaced_local, 1.0));
    out.world_position = world_pos4;
    out.position = position_world_to_clip(world_pos4.xyz);
    // Compute normal from height gradients in UV space
    let dims = vec2<f32>(textureDimensions(height_tex, 0));
    let texel = 1.0 / max(dims, vec2(1.0, 1.0));
    let h_l = textureSampleLevel(height_tex, height_sampler, v.uv - vec2(texel.x, 0.0), 0.0).x;
    let h_r = textureSampleLevel(height_tex, height_sampler, v.uv + vec2(texel.x, 0.0), 0.0).x;
    let h_d = textureSampleLevel(height_tex, height_sampler, v.uv - vec2(0.0, texel.y), 0.0).x;
    let h_u = textureSampleLevel(height_tex, height_sampler, v.uv + vec2(0.0, texel.y), 0.0).x;
    let d_hx = (h_r - h_l) * terrain_params.height_scale;
    let d_hz = (h_u - h_d) * terrain_params.height_scale;
    let n_local = normalize(vec3<f32>(-d_hx, 1.0, -d_hz));
    let nmat = mat3x3<f32>(
        world_from_local[0].xyz,
        world_from_local[1].xyz,
        world_from_local[2].xyz,
    );
    let n_world = normalize(nmat * n_local);
    out.world_normal = n_world;
    out.uv = v.uv;
    return out;
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // Use color texture from compute as base color
    let albedo = textureSample(color_tex, color_sampler, in.uv);
    pbr_input.material.base_color = alpha_discard(pbr_input.material, albedo);

    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);

    // Transform to view space by differencing two nearby points in view space
    let n_world = in.world_normal;
    let p0_view = position_world_to_view(in.world_position.xyz).xyz;
    let p1_view = position_world_to_view(in.world_position.xyz + n_world * 0.01).xyz;
    let n_view = normalize(p1_view - p0_view);

    // color by the view normal
    // let color = n_view * 0.5 + 0.5;
    // out.color = vec4(pow(color, vec3(2.2)), 1.0);

    // out.color = albedo;
    return out;
}


