#import bevy_pbr::{
    mesh_functions,
    view_transformations::position_world_to_clip,
    view_transformations::position_world_to_view
}

// Material bind group (group 1) to align with Bevy's Material pipeline
@group(1) @binding(0) var height_tex: texture_2d<f32>;
@group(1) @binding(1) var height_sampler: sampler;

struct VertexIn {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vertex(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    let height_scale = 50.0;
    let h = textureSampleLevel(height_tex, height_sampler, v.uv, 0.0).x;
    let displaced = vec3<f32>(v.position.x, v.position.y + h * height_scale, v.position.z);
    var world_from_local = mesh_functions::get_world_from_local(v.instance_index);
    let world = mesh_functions::mesh_position_local_to_world(world_from_local, vec4(displaced, 1.0));
    out.world_pos = world.xyz;
    out.clip_position = position_world_to_clip(world.xyz);
    // Compute local-space height gradients from the heightmap
    let dims = vec2<f32>(textureDimensions(height_tex, 0));
    let texel = 1.0 / dims;
    let h_l = textureSampleLevel(height_tex, height_sampler, v.uv - vec2(texel.x, 0.0), 0.0).x;
    let h_r = textureSampleLevel(height_tex, height_sampler, v.uv + vec2(texel.x, 0.0), 0.0).x;
    let h_d = textureSampleLevel(height_tex, height_sampler, v.uv - vec2(0.0, texel.y), 0.0).x;
    let h_u = textureSampleLevel(height_tex, height_sampler, v.uv + vec2(0.0, texel.y), 0.0).x;
    let d_hx = (h_r - h_l);
    let d_hz = (h_u - h_d);
    let n_local = normalize(vec3<f32>(-d_hx * height_scale, 1.0, -d_hz * height_scale));
    // Transform normal to world space (assumes uniform scale)
    let nmat = mat3x3<f32>(
        world_from_local[0].xyz,
        world_from_local[1].xyz,
        world_from_local[2].xyz,
    );
    let n_world = normalize(nmat * n_local);
    // Transform to view space by differencing two nearby points in view space
    let p0_view = position_world_to_view(world.xyz).xyz;
    let p1_view = position_world_to_view(world.xyz + n_world * 0.01).xyz;
    let n_view = normalize(p1_view - p0_view);
    out.normal = n_view;
    return out;
}

@fragment
fn fragment(v: VertexOut) -> @location(0) vec4<f32> {
    let color = v.normal * 0.5 + 0.5;
    return vec4(pow(color, vec3(2.2)), 1.0);
}


