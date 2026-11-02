import bevy_pbr::render::{
    mesh_functions,
    view_transformations::{position_world_to_clip, position_world_to_view},
    forward_io::{VertexOutput, FragmentOutput},
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing, alpha_discard},
    mesh_view_bindings::{view, globals},
};

// Extension bind group uses MATERIAL_BIND_GROUP with slots starting at 100
// Height has sampler for smooth interpolation in vertex shader
@group(constants::MATERIAL_BIND_GROUP) @binding(100) var height_tex: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(101) var height_sampler: sampler;
// Color and analysis textures use texelFetch (no samplers) to reduce sampler count
@group(constants::MATERIAL_BIND_GROUP) @binding(102) var color_tex: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(103) var analysis_tex: texture_2d<f32>;
// AO texture with sampler for smooth interpolation
@group(constants::MATERIAL_BIND_GROUP) @binding(104) var ao_tex: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(105) var ao_sampler: sampler;

struct TerrainParams {
    height_scale: f32,
}
@group(constants::MATERIAL_BIND_GROUP) @binding(106) var<uniform> terrain_params: TerrainParams;

struct PreviewParams {
    mode: u32,  // 0=PBR, 1=Flow, 2=Sediment, 3=Erosion, 4=Height, 5=ViewSpaceNormals, 6=Curvature, 7=Age
}
@group(constants::MATERIAL_BIND_GROUP) @binding(107) var<uniform> preview_params: PreviewParams;
// Water solver output: R = depth, GB = velocity (texels per solver step), A = suspended sediment.
@group(constants::MATERIAL_BIND_GROUP) @binding(108) var water_tex: texture_2d<f32>;
struct SeaParams {
    level: f32, // 0 = no ocean
}
@group(constants::MATERIAL_BIND_GROUP) @binding(109) var<uniform> sea: SeaParams;

// Water depth above the terrain at `uv`: lakes and rivers from the solver, plus the ocean.
fn water_depth_at(uv: vec2<f32>, ground: f32) -> f32 {
    var d = textureSampleLevel(water_tex, height_sampler, uv, 0.0).x;
    if (sea.level > 0.0) {
        d = max(d, sea.level - ground);
    }
    return max(d, 0.0);
}

// Displayed surface: terrain, or the water surface where there is water.
fn surface_at(uv: vec2<f32>) -> f32 {
    let ground = textureSampleLevel(height_tex, height_sampler, uv, 0.0).x;
    return ground + water_depth_at(uv, ground);
}

struct VertexIn {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    // Optional attributes (tangent, color, etc.) are handled by the base pipeline; we only need core ones here
};

@vertex
fn vertex(v: VertexIn) -> VertexOutput {
    // Sample height at UV to displace along local +Y (using sampler for smooth interpolation)
    let h = surface_at(v.uv);
    let displaced_local = vec3<f32>(v.position.x, v.position.y + h * terrain_params.height_scale, v.position.z);

    var out: VertexOutput;
    var world_from_local = mesh_functions::get_world_from_local(v.instance_index);
    let world_pos4 = mesh_functions::mesh_position_local_to_world(world_from_local, vec4(displaced_local, 1.0));
    out.world_position = world_pos4;
    out.position = position_world_to_clip(world_pos4.xyz);
    // Compute normal from height gradients in UV space
    let dims = vec2<f32>(textureDimensions(height_tex, 0));
    let texel = 1.0 / max(dims, vec2(1.0, 1.0));
    let h_l = surface_at(v.uv - vec2(texel.x, 0.0));
    let h_r = surface_at(v.uv + vec2(texel.x, 0.0));
    let h_d = surface_at(v.uv - vec2(0.0, texel.y));
    let h_u = surface_at(v.uv + vec2(0.0, texel.y));
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
const MODE_CURVATURE: u32 = 6u;
const MODE_AGE: u32 = 7u;
const MODE_SIMPLE: u32 = 8u;

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
fn heat_map(value: f32) -> vec3<f32> {
    let color0 = vec3<f32>(0.0, 0.0, 0.5);  // dark blue
    let color1 = vec3<f32>(0.0, 0.0, 1.0);  // blue
    let color2 = vec3<f32>(0.0, 1.0, 1.0);  // cyan
    let color3 = vec3<f32>(1.0, 1.0, 0.0);  // yellow
    let color4 = vec3<f32>(1.0, 0.0, 0.0);  // red
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
    let dims = textureDimensions(color_tex, 0);
    let px = vec2<i32>(vec2<f32>(dims) * uv);
    
    switch (mode) {
        case MODE_FLOW: {
            let analysis = textureSample(analysis_tex, ao_sampler, uv);
            let flow_mag = analysis.r * 2.0;
            let color = heat_map(flow_mag);
            return vec4<f32>(color, 1.0);
        }
        case MODE_SEDIMENT: {
            let analysis = textureSample(analysis_tex, ao_sampler, uv);
            let sediment = analysis.g * 20.0;
            let color = heat_map(sediment);
            return vec4<f32>(color, 1.0);
        }
        case MODE_EROSION: {
            let analysis = textureSample(analysis_tex, ao_sampler, uv);
            let erosion = analysis.b * 2.0;
            let color = heat_map(erosion);
            return vec4<f32>(color, 1.0);
        }
        case MODE_HEIGHT: {
            // Height: repeating black/white bands for high contrast visualization
            let height = textureSampleLevel(height_tex, height_sampler, uv, 0.0).x;
            let bands = fract(height * 100.0); // 100 repeating bands
            let color = step(0.5, bands); // Black (0) or white (1)
            return vec4(vec3(height + (color * .2)), 1.0);
        }
        case MODE_CURVATURE: {
            let analysis = textureSample(analysis_tex, ao_sampler, uv);
            let curvature = analysis.a; // Alpha channel contains curvature
            let color = heat_map(curvature);
            return vec4<f32>(color, 1.0);
        }
        case MODE_AGE: {
            // KTT __Age (time since a trail last changed the cell), scaled in the erosion blit
            let age = textureSampleLevel(height_tex, height_sampler, uv, 0.0).y;
            return vec4<f32>(heat_map(age), 1.0);
        }
        default: {
            // PBR mode: use computed color texture with linear sampling
            return textureSample(color_tex, ao_sampler, uv);
        }
    }
}

// --- Water surface shading -------------------------------------------------------------------

// Wet thresholds, as in `water.wgsl`.
const WET_DEPTH: f32 = 0.0005;
const FULL_DEPTH: f32 = 0.004;
// Per world unit of path length through the water: red goes first, blue last.
const ABSORPTION: vec3<f32> = vec3<f32>(2.2, 0.7, 0.45);
// In-scattered colour of deep clear water and of silty water.
const DEEP_COLOR: vec3<f32> = vec3<f32>(0.015, 0.06, 0.08);
const SILT_COLOR: vec3<f32> = vec3<f32>(0.22, 0.17, 0.10);
// Ripples per UV unit, and how far (UV per second) a ripple drifts at solver speed 1.
const RIPPLE_SCALE: f32 = 160.0;
const RIPPLE_FLOW: f32 = 0.02;

fn hash2(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2<f32>(123.34, 456.21));
    let r = q + dot(q, q + 45.32);
    return fract(r.x * r.y);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = mix(hash2(i), hash2(i + vec2<f32>(1.0, 0.0)), u.x);
    let b = mix(hash2(i + vec2<f32>(0.0, 1.0)), hash2(i + vec2<f32>(1.0, 1.0)), u.x);
    return mix(a, b, u.y);
}

fn ripple_height(p: vec2<f32>) -> f32 {
    return value_noise(p) * 0.65 + value_noise(p * 2.3 + 17.0) * 0.35;
}

// Gradient of the ripple field at `p` (ripple units).
fn ripple_gradient(p: vec2<f32>) -> vec2<f32> {
    let e = 0.15;
    let c = ripple_height(p);
    return vec2<f32>(ripple_height(p + vec2<f32>(e, 0.0)) - c, ripple_height(p + vec2<f32>(0.0, e)) - c) / e;
}

// Flow-map ripples: two copies advected along `flow` (UV per second), each restarting every
// period and faded out as it restarts, so the pattern never stretches. `x` is the ripple value
// (for foam), `yz` its gradient.
fn flow_ripples(uv: vec2<f32>, flow: vec2<f32>, time: f32) -> vec3<f32> {
    let period = 2.0;
    var out = vec3<f32>(0.0);
    for (var k = 0; k < 2; k = k + 1) {
        let phase = fract(time / period + f32(k) * 0.5);
        let w = 1.0 - abs(2.0 * phase - 1.0);
        let p = (uv - flow * phase * period) * RIPPLE_SCALE + f32(k) * 31.7;
        out += w * vec3<f32>(ripple_height(p), ripple_gradient(p));
    }
    return out;
}

struct WaterShade {
    color: vec3<f32>,
    normal: vec3<f32>,
    roughness: f32,
    wet: f32,
}

// Water over the bed colour `bed`: depth absorption along the view path, silt, ripple normals
// following the solver velocity (ocean waves drift with time) and foam on rapids and shores.
fn shade_water(uv: vec2<f32>, bed: vec3<f32>, n: vec3<f32>, v: vec3<f32>) -> WaterShade {
    let ground = textureSampleLevel(height_tex, height_sampler, uv, 0.0).x;
    let water = textureSampleLevel(water_tex, height_sampler, uv, 0.0);
    let depth = water_depth_at(uv, ground);
    var shade: WaterShade;
    shade.wet = smoothstep(WET_DEPTH, FULL_DEPTH, depth);
    shade.color = bed;
    shade.normal = n;
    shade.roughness = 1.0;
    if (shade.wet <= 0.0) {
        return shade;
    }
    let ocean = sea.level > 0.0 && sea.level - ground > water.x;
    let vel = select(water.yz, vec2<f32>(0.0), ocean);
    let speed = length(vel);
    let time = globals.time;

    var flow = vel * RIPPLE_FLOW;
    var amp = 0.04 + 0.25 * smoothstep(0.05, 0.6, speed);
    if (ocean) {
        // Open water: waves drift with the wind instead of a current.
        flow = vec2<f32>(0.004, 0.0025);
        amp = 0.18;
    }
    let ripples = flow_ripples(uv, flow, time);
    let tilt = ripples.yz * amp * shade.wet;
    shade.normal = normalize(n + vec3<f32>(-tilt.x, 0.0, -tilt.y));

    // Light crosses the water column down to the bed and back up toward the eye.
    let column = depth * terrain_params.height_scale;
    let path = column * (1.0 + 1.0 / max(abs(v.y), 0.15));
    let transmit = exp(-ABSORPTION * path);
    let silt = clamp(water.w * 400.0, 0.0, 0.8);
    let scatter = mix(DEEP_COLOR, SILT_COLOR, silt);
    var color = mix(bed * transmit + scatter * (1.0 - transmit), scatter, silt * (1.0 - transmit.b));

    // Foam: white water on fast flow, and a thin broken line where the water meets the shore.
    let shore = 1.0 - smoothstep(FULL_DEPTH, 3.0 * FULL_DEPTH, depth);
    let rapids = smoothstep(0.35, 0.9, speed);
    let breaking = select(0.0, 0.5, ocean) * shore;
    let foam = clamp(max(rapids, max(shore * 0.6, breaking)) * smoothstep(0.75, 1.05, ripples.x), 0.0, 1.0);
    color = mix(color, vec3<f32>(0.85, 0.88, 0.9), foam);

    shade.color = mix(bed, color, shade.wet);
    shade.roughness = mix(0.04, 0.5, foam);
    return shade;
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var out: FragmentOutput;
    
    // Sample precomputed ambient occlusion with linear interpolation for smooth result
    let ao_value = textureSample(ao_tex, ao_sampler, in.uv).x;
    
    // Create PBR input for lighting
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    
    // Get base color based on preview mode
    let base_color = get_preview_color(preview_params.mode, in.uv, in.world_normal);
    
    // Apply base color and lighting
    if (preview_params.mode == MODE_PBR) {
        // Water: absorption, ripples and foam over the bed; glossy, with a water-like F0 so the
        // atmosphere environment map supplies the Fresnel sky reflection.
        let water = shade_water(in.uv, base_color.rgb, pbr_input.N, pbr_input.V);
        let color = vec4<f32>(water.color, base_color.a);
        pbr_input.N = normalize(mix(pbr_input.N, water.normal, water.wet));
        pbr_input.material.perceptual_roughness = mix(pbr_input.material.perceptual_roughness, water.roughness, water.wet);
        pbr_input.material.reflectance = mix(pbr_input.material.reflectance, vec3<f32>(0.35), water.wet);
        // Full PBR lighting for default mode
        pbr_input.material.base_color = alpha_discard(pbr_input.material.flags, pbr_input.material.alpha_cutoff, color);
        // Apply AO to diffuse_occlusion
        pbr_input.diffuse_occlusion = vec3<f32>(ao_value);
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
        // Simple Lambert shading for preview modes - apply AO as simple multiplier
        var color = base_color;
        if (preview_params.mode == MODE_SIMPLE) {
            // Flat tint where there is water, so the simulations stay visible without optics.
            let ground = textureSampleLevel(height_tex, height_sampler, in.uv, 0.0).x;
            let wet = smoothstep(WET_DEPTH, 4.0 * FULL_DEPTH, water_depth_at(in.uv, ground));
            color = vec4<f32>(mix(color.rgb, vec3<f32>(0.05, 0.2, 0.3), wet), color.a);
        }
        out.color = apply_lambert_shading(color, in.world_normal);
        out.color = vec4<f32>(out.color.rgb * ao_value, out.color.a);
    }
    
    return out;
}


