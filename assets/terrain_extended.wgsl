#import bevy_pbr::{
    mesh_functions,
    view_transformations::{position_world_to_clip, position_world_to_view},
    forward_io::{VertexOutput, FragmentOutput},
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing, alpha_discard},
    mesh_view_bindings::view,
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

// Analysis texture: R=flow_mag, G=sediment, B=erosion, A=flow_angle
@group(#{MATERIAL_BIND_GROUP}) @binding(105) var analysis_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(106) var analysis_sampler: sampler;

struct PreviewParams {
    mode: u32,  // 0=PBR, 1=Flow, 2=Sediment, 3=Erosion, 4=Height, 5=ViewSpaceNormals
}
@group(#{MATERIAL_BIND_GROUP}) @binding(107) var<uniform> preview_params: PreviewParams;

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

// Preview mode constants
const MODE_PBR: u32 = 0u;
const MODE_FLOW: u32 = 1u;
const MODE_SEDIMENT: u32 = 2u;
const MODE_EROSION: u32 = 3u;
const MODE_HEIGHT: u32 = 4u;
const MODE_VIEW_NORMALS: u32 = 5u;

// Convert HSV to RGB (hue, saturation, value all in [0, 1])
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let x = c * (1.0 - abs((h * 6.0) % 2.0 - 1.0));
    let m = v - c;
    
    let h_segment = h * 6.0;
    var rgb = vec3<f32>(0.0);
    
    if (h_segment < 1.0) {
        rgb = vec3<f32>(c, x, 0.0);
    } else if (h_segment < 2.0) {
        rgb = vec3<f32>(x, c, 0.0);
    } else if (h_segment < 3.0) {
        rgb = vec3<f32>(0.0, c, x);
    } else if (h_segment < 4.0) {
        rgb = vec3<f32>(0.0, x, c);
    } else if (h_segment < 5.0) {
        rgb = vec3<f32>(x, 0.0, c);
    } else {
        rgb = vec3<f32>(c, 0.0, x);
    }
    
    return rgb + m;
}

// Generate a heat map color from a normalized value [0, 1]
// gradient_colors: array of colors to interpolate between
fn heat_map(value: f32, color0: vec3<f32>, color1: vec3<f32>, color2: vec3<f32>, color3: vec3<f32>, color4: vec3<f32>) -> vec3<f32> {
    let clamped = clamp(value, 0.0, 1.0);
    let scaled = clamped * 4.0;
    
    if (scaled < 1.0) {
        return mix(color0, color1, scaled);
    } else if (scaled < 2.0) {
        return mix(color1, color2, scaled - 1.0);
    } else if (scaled < 3.0) {
        return mix(color2, color3, scaled - 2.0);
    } else {
        return mix(color3, color4, scaled - 3.0);
    }
}

// Simple Lambert diffuse shading with fixed white light
fn apply_lambert_shading(base_color: vec4<f32>, world_normal: vec3<f32>) -> vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.5)); // Fixed light direction
    let light_color = vec3<f32>(1.0, 1.0, 1.0); // White light
    let diffuse = max(0.0, dot(world_normal, light_dir));
    let exaggerated = pow(diffuse, 2.5); // Exaggerate shadows with power curve
    return vec4<f32>(base_color.rgb * light_color * exaggerated, base_color.a);
}

// Get base color for the current preview mode
fn get_preview_color(mode: u32, uv: vec2<f32>, world_normal: vec3<f32>) -> vec4<f32> {
    switch (mode) {
        case MODE_FLOW: {
            // Flow map: direction as hue, magnitude as brightness
            let analysis = textureSample(analysis_tex, analysis_sampler, uv);
            let flow_mag = analysis.r;
            let flow_angle = analysis.a;
            return vec4<f32>(hsv_to_rgb(flow_angle, 1.0, flow_mag), 1.0);
        }
        case MODE_SEDIMENT: {
            // Sediment: cool to warm heat map (blue -> cyan -> yellow -> red)
            let analysis = textureSample(analysis_tex, analysis_sampler, uv);
            let sediment = analysis.g * 20.0;
            let color = heat_map(
                sediment,
                vec3<f32>(0.0, 0.0, 0.5),  // dark blue
                vec3<f32>(0.0, 0.0, 1.0),  // blue
                vec3<f32>(0.0, 1.0, 1.0),  // cyan
                vec3<f32>(1.0, 1.0, 0.0),  // yellow
                vec3<f32>(1.0, 0.0, 0.0)   // red
            );
            return vec4<f32>(color, 1.0);
        }
        case MODE_EROSION: {
            // Erosion: dark to bright heat map (black -> purple -> red -> orange -> yellow)
            let analysis = textureSample(analysis_tex, analysis_sampler, uv);
            let erosion = analysis.b;
            let color = heat_map(
                erosion,
                vec3<f32>(0.0, 0.0, 0.0),  // black
                vec3<f32>(0.5, 0.0, 0.5),  // purple
                vec3<f32>(1.0, 0.0, 0.0),  // red
                vec3<f32>(1.0, 0.5, 0.0),  // orange
                vec3<f32>(1.0, 1.0, 0.0)   // yellow
            );
            return vec4<f32>(color, 1.0);
        }
        case MODE_HEIGHT: {
            // Height: repeating black/white bands for high contrast visualization
            let height = textureSample(height_tex, height_sampler, uv).x;
            let bands = fract(height * 100.0); // 10 repeating bands
            let color = step(0.5, bands); // Black (0) or white (1)
            return vec4(vec3(height + (color * .1)), 1.0);
        }
        default: {
            // PBR mode: use computed color texture
            return textureSample(color_tex, color_sampler, uv);
        }
    }
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var out: FragmentOutput;
    
    // Create PBR input for lighting
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    
    // Get base color based on preview mode
    let base_color = get_preview_color(preview_params.mode, in.uv, in.world_normal);
    
    // Apply base color and lighting
    if (preview_params.mode == MODE_PBR) {
        // Full PBR lighting for default mode
        pbr_input.material.base_color = alpha_discard(pbr_input.material, base_color);
        out.color = apply_pbr_lighting(pbr_input);
        out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    } 
    else if (preview_params.mode == MODE_VIEW_NORMALS) {
        // Transform world normal to view space
        let view_normal = normalize((view.view_from_world * vec4<f32>(in.world_normal, 0.0)).xyz);
        let color = vec4<f32>(view_normal * 0.5 + 0.5, 1.0);
        out.color = pow(color, vec4(vec3(4.0), 1.0));
    }
    else {
        // Simple Lambert shading for preview modes
        out.color = apply_lambert_shading(base_color, in.world_normal);
    }
    
    return out;
}


