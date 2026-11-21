// Prepass and shadow vertex stage for the terrain: the same displacement as `terrain_extended.wesl`,
// so the depth prepass (which bevy_fft's water refraction reads) and the shadow maps see the
// eroded terrain instead of the flat plane.

import bevy_pbr::render::{mesh_functions, view_transformations::position_world_to_clip};
import bevy_pbr::prepass::io::{Vertex, VertexOutput};

@group(constants::MATERIAL_BIND_GROUP) @binding(100) var height_tex: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(101) var height_sampler: sampler;

struct TerrainParams {
    height_scale: f32,
}
// Height units per two texels of run: half of `CELLS_PER_HEIGHT` in `lib.rs`.
const NORMAL_SLOPE_SCALE: f32 = 50.0;
@group(constants::MATERIAL_BIND_GROUP) @binding(106) var<uniform> terrain_params: TerrainParams;

fn height_at(uv: vec2<f32>) -> f32 {
    return textureSampleLevel(height_tex, height_sampler, uv, 0.0).x;
}

@vertex
fn vertex(v: Vertex) -> VertexOutput {
    var out: VertexOutput;
    var uv = vec2<f32>(0.0);
    @if(VERTEX_UVS_A) {
        uv = v.uv;
    }
    let world_from_local = mesh_functions::get_world_from_local(v.instance_index);
    let local = vec3<f32>(v.position.x, v.position.y + height_at(uv) * terrain_params.height_scale, v.position.z);
    out.world_position = mesh_functions::mesh_position_local_to_world(world_from_local, vec4(local, 1.0));
    out.position = position_world_to_clip(out.world_position.xyz);
@if(UNCLIPPED_DEPTH_ORTHO_EMULATION) {
    out.unclipped_depth = out.position.z;
    out.position.z = min(out.position.z, 1.0);
}
@if(VERTEX_UVS_A)
    out.uv = uv;
@if(NORMAL_PREPASS_OR_DEFERRED_PREPASS) {
    let dims = vec2<f32>(textureDimensions(height_tex, 0));
    let texel = 1.0 / max(dims, vec2(1.0));
    // Rise over the two texels' run (see `terrain_extended.wesl`).
    let d_hx = (height_at(uv + vec2(texel.x, 0.0)) - height_at(uv - vec2(texel.x, 0.0))) * NORMAL_SLOPE_SCALE;
    let d_hz = (height_at(uv + vec2(0.0, texel.y)) - height_at(uv - vec2(0.0, texel.y))) * NORMAL_SLOPE_SCALE;
    let nmat = mat3x3<f32>(world_from_local[0].xyz, world_from_local[1].xyz, world_from_local[2].xyz);
    out.world_normal = normalize(nmat * vec3<f32>(-d_hx, 1.0, -d_hz));
}
@if(MOTION_VECTOR_PREPASS)
    out.previous_world_position = out.world_position;
@if(VERTEX_OUTPUT_INSTANCE_INDEX)
    out.instance_index = v.instance_index;
@if(VERTEX_COLORS)
    out.color = v.color;
    return out;
}
