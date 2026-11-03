import bevy_pbr::render::{
    mesh_functions,
    view_transformations::{position_world_to_clip, position_world_to_view},
    forward_io::{VertexOutput, FragmentOutput},
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing, alpha_discard},
    mesh_view_bindings::view,
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
// Height units per two texels of run: half of `CELLS_PER_HEIGHT` in `lib.rs`.
const NORMAL_SLOPE_SCALE: f32 = 50.0;
@group(constants::MATERIAL_BIND_GROUP) @binding(106) var<uniform> terrain_params: TerrainParams;

// `PreviewMode` and a linear gain on the debug fields.
struct PreviewParams {
    mode: u32,
    gain: f32,
}
@group(constants::MATERIAL_BIND_GROUP) @binding(107) var<uniform> preview_params: PreviewParams;
// Water solver output: R = depth, GB = velocity (texels per solver step), A = suspended sediment.
// bevy_fft's water surface draws the water itself; the terrain only tints it in the simple view.
@group(constants::MATERIAL_BIND_GROUP) @binding(108) var water_tex: texture_2d<f32>;
struct SeaParams {
    level: f32, // 0 = no ocean
}
@group(constants::MATERIAL_BIND_GROUP) @binding(109) var<uniform> sea: SeaParams;
// Debug views (see `ErosionImages`): terrain brush edits; KTT masks (softness, outflow, erode,
// uplift); layers (sediment above bedrock, raw age); discharge (q.xy, mean depth, KTT river
// discharge); bed (bed load, helical intensity, pending bed change).
@group(constants::MATERIAL_BIND_GROUP) @binding(110) var terrain_edit_tex: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(111) var debug_masks_tex: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(112) var debug_layers_tex: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(113) var debug_discharge_tex: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(114) var debug_bed_tex: texture_2d<f32>;
// Drainage area (m²), from the CPU (`drainage.rs`).
@group(constants::MATERIAL_BIND_GROUP) @binding(115) var drainage_tex: texture_2d<f32>;

// Water depth above the terrain at `uv`: lakes and rivers from the solver, plus the ocean.
fn water_depth_at(uv: vec2<f32>, ground: f32) -> f32 {
    var d = textureSampleLevel(water_tex, height_sampler, uv, 0.0).x;
    if (sea.level > 0.0) {
        d = max(d, sea.level - ground);
    }
    return max(d, 0.0);
}

fn surface_at(uv: vec2<f32>) -> f32 {
    return textureSampleLevel(height_tex, height_sampler, uv, 0.0).x;
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
    // Rise over the two texels' run: heights are isotropic with the grid (100 texels per
    // height unit), so this does not depend on the scene's scale.
    let d_hx = (h_r - h_l) * NORMAL_SLOPE_SCALE;
    let d_hz = (h_u - h_d) * NORMAL_SLOPE_SCALE;
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

// `PreviewMode` discriminants.
const MODE_PBR: u32 = 0u;
const MODE_SIMPLE: u32 = 1u;
const MODE_ALBEDO: u32 = 2u;
const MODE_HEIGHT: u32 = 3u;
const MODE_SLOPE: u32 = 4u;
const MODE_VIEW_NORMALS: u32 = 5u;
const MODE_CURVATURE: u32 = 6u;
const MODE_AO: u32 = 7u;
const MODE_EDITS: u32 = 8u;
const MODE_FLOW: u32 = 9u;
const MODE_SEDIMENT: u32 = 10u;
const MODE_EROSION: u32 = 11u;
const MODE_RIVERS: u32 = 12u;
const MODE_AGE: u32 = 13u;
const MODE_SOFTNESS: u32 = 14u;
const MODE_OUTFLOW: u32 = 15u;
const MODE_ERODE_MASK: u32 = 16u;
const MODE_UPLIFT: u32 = 17u;
const MODE_BEDROCK: u32 = 18u;
const MODE_WATER_DEPTH: u32 = 19u;
const MODE_WATER_VELOCITY: u32 = 20u;
const MODE_SUSPENDED: u32 = 21u;
const MODE_DISCHARGE: u32 = 22u;
const MODE_MEAN_DEPTH: u32 = 23u;
const MODE_RIVER_DISCHARGE: u32 = 24u;
const MODE_BED_LOAD: u32 = 25u;
const MODE_HELICAL: u32 = 26u;
const MODE_BED_CHANGE: u32 = 27u;
const MODE_DRAINAGE: u32 = 28u;
const MODE_BOULDERS: u32 = 29u;

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

fn load(tex: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(tex, 0));
    let px = clamp(vec2<i32>(vec2<f32>(dims) * uv), vec2<i32>(0), dims - 1);
    return textureLoad(tex, px, 0);
}

// Blue for negative, white at zero, red for positive; `v` in [-1, 1].
fn diverging(v: f32) -> vec3<f32> {
    let t = clamp(v, -1.0, 1.0);
    let white = vec3<f32>(0.95);
    if (t < 0.0) {
        return mix(white, vec3<f32>(0.1, 0.3, 0.9), -t);
    }
    return mix(white, vec3<f32>(0.9, 0.15, 0.1), t);
}

// Hue for direction, brightness for magnitude (`m` in [0, 1]).
fn direction_color(v: vec2<f32>, m: f32) -> vec3<f32> {
    let hue = fract(atan2(v.y, v.x) / 6.2831853 + 1.0);
    return hsv_to_rgb(hue, 0.85, clamp(m, 0.0, 1.0));
}

fn gray(v: f32) -> vec4<f32> {
    return vec4<f32>(vec3<f32>(mix(0.06, 1.0, clamp(v, 0.0, 1.0))), 1.0);
}

fn heat(v: f32) -> vec4<f32> {
    // Exact zero reads dark grey, so dry or untouched cells stand apart from small values and
    // the terrain still shows against the black sky.
    if (v <= 0.0) {
        return vec4<f32>(0.06, 0.06, 0.06, 1.0);
    }
    return vec4<f32>(heat_map(v), 1.0);
}

// Base colour of the current view; every debug field is scaled by the view's gain.
fn get_preview_color(mode: u32, uv: vec2<f32>, world_normal: vec3<f32>) -> vec4<f32> {
    let g = preview_params.gain;
    switch (mode) {
        case MODE_HEIGHT: {
            // Height with repeating contour bands.
            let height = textureSampleLevel(height_tex, height_sampler, uv, 0.0).x;
            let bands = step(0.5, fract(height * 100.0));
            return vec4(vec3(height * g + bands * 0.2), 1.0);
        }
        case MODE_SLOPE: {
            let slope = acos(clamp(world_normal.y, -1.0, 1.0));
            return gray(slope / radians(60.0) * g);
        }
        case MODE_CURVATURE: {
            // Stored as laplacian * 50 + 0.5.
            let c = load(analysis_tex, uv).a - 0.5;
            return vec4<f32>(diverging(-c * 2.0 * g), 1.0);
        }
        case MODE_AO: {
            return gray(textureSample(ao_tex, ao_sampler, uv).x);
        }
        case MODE_EDITS: {
            let e = load(terrain_edit_tex, uv);
            var c = diverging(e.r * 20.0 * g);
            c = mix(c, vec3<f32>(0.1, 0.8, 0.2), clamp(e.g, 0.0, 1.0) * 0.6);
            c = mix(c, vec3<f32>(1.0), clamp(e.a, 0.0, 1.0) * 0.6);
            return vec4<f32>(c, 1.0);
        }
        case MODE_FLOW: {
            return heat(load(analysis_tex, uv).r * 2.0 * g);
        }
        case MODE_SEDIMENT: {
            return heat(load(analysis_tex, uv).g * 20.0 * g);
        }
        case MODE_EROSION: {
            return heat(load(analysis_tex, uv).b * 2.0 * g);
        }
        case MODE_RIVERS: {
            // KTT river momentum (Do Rivers), scaled in the erosion blit
            return heat(textureSampleLevel(height_tex, height_sampler, uv, 0.0).z * g);
        }
        case MODE_AGE: {
            // KTT __Age (time since a trail last changed the cell), scaled in the erosion blit
            return heat(textureSampleLevel(height_tex, height_sampler, uv, 0.0).y * g);
        }
        case MODE_SOFTNESS: {
            return gray(load(debug_masks_tex, uv).x * g);
        }
        case MODE_OUTFLOW: {
            return gray(load(debug_masks_tex, uv).y * g);
        }
        case MODE_ERODE_MASK: {
            return gray(load(debug_masks_tex, uv).z * g);
        }
        case MODE_UPLIFT: {
            return gray(load(debug_masks_tex, uv).w * 0.5 * g);
        }
        case MODE_BEDROCK: {
            return heat(load(debug_layers_tex, uv).x * 20.0 * g);
        }
        case MODE_WATER_DEPTH: {
            let ground = textureSampleLevel(height_tex, height_sampler, uv, 0.0).x;
            let d = load(water_tex, uv).x;
            if (d <= 0.0 && sea.level > ground) {
                return vec4<f32>(0.3, 0.3, 0.35, 1.0);
            }
            return heat(d * 50.0 * g);
        }
        case MODE_WATER_VELOCITY: {
            let w = load(water_tex, uv);
            return vec4<f32>(direction_color(w.yz, length(w.yz) * 4.0 * g), 1.0);
        }
        case MODE_SUSPENDED: {
            return heat(load(water_tex, uv).w * 500.0 * g);
        }
        case MODE_DISCHARGE: {
            let q = load(debug_discharge_tex, uv).xy;
            // Log scale: discharge spans orders of magnitude between rills and rivers.
            let m = log2(1.0 + length(q) * 1000.0 * g) / 8.0;
            return vec4<f32>(direction_color(q, m), 1.0);
        }
        case MODE_MEAN_DEPTH: {
            return heat(load(debug_discharge_tex, uv).z * 50.0 * g);
        }
        case MODE_RIVER_DISCHARGE: {
            let w = load(debug_discharge_tex, uv).w;
            return heat(log2(1.0 + w * 1000.0 * g) / 8.0);
        }
        case MODE_BED_LOAD: {
            let qb = load(debug_bed_tex, uv).x;
            return heat(log2(1.0 + qb * 10.0 * g) / 8.0);
        }
        case MODE_HELICAL: {
            return vec4<f32>(diverging(load(debug_bed_tex, uv).y * 100.0 * g), 1.0);
        }
        case MODE_BED_CHANGE: {
            return vec4<f32>(diverging(load(debug_bed_tex, uv).z * 100000.0 * g), 1.0);
        }
        case MODE_BOULDERS: {
            return gray(load(debug_bed_tex, uv).w * g);
        }
        case MODE_DRAINAGE: {
            // 100 m² (one 10 m texel) to 100 km².
            let a = load(drainage_tex, uv).x;
            return heat((log2(max(a, 1.0)) / 3.3219 - 2.0) / 6.0 * g);
        }
        default: {
            // PBR, simple and albedo: the computed colour bands with linear sampling.
            return textureSample(color_tex, ao_sampler, uv);
        }
    }
}

// Wet thresholds, as in `water.wgsl`.
const WET_DEPTH: f32 = 0.0005;
const FULL_DEPTH: f32 = 0.004;

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var out: FragmentOutput;
    let mode = preview_params.mode;

    // Sample precomputed ambient occlusion with linear interpolation for smooth result
    let ao_value = textureSample(ao_tex, ao_sampler, in.uv).x;

    // Create PBR input for lighting
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // Get base color based on preview mode
    let base_color = get_preview_color(mode, in.uv, in.world_normal);

    if (mode == MODE_PBR) {
        // Full PBR lighting for default mode
        pbr_input.material.base_color = alpha_discard(pbr_input.material.flags, pbr_input.material.alpha_cutoff, base_color);
        // Apply AO to diffuse_occlusion
        pbr_input.diffuse_occlusion = vec3<f32>(ao_value);
        out.color = apply_pbr_lighting(pbr_input);
        out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    } else if (mode == MODE_VIEW_NORMALS) {
        // Transform world normal to view space
        let view_normal = normalize((view.view_from_world * vec4<f32>(in.world_normal, 0.0)).xyz);
        let color = vec4<f32>(view_normal * 0.5 + 0.5, 1.0);
        out.color = pow(color, vec4(vec3(4.0), 1.0));
    } else if (mode == MODE_SIMPLE) {
        // Flat tint where there is water, so the simulations stay visible without optics.
        let ground = textureSampleLevel(height_tex, height_sampler, in.uv, 0.0).x;
        let wet = smoothstep(WET_DEPTH, 4.0 * FULL_DEPTH, water_depth_at(in.uv, ground));
        let color = vec4<f32>(mix(base_color.rgb, vec3<f32>(0.05, 0.2, 0.3), wet), base_color.a);
        out.color = apply_lambert_shading(color, in.world_normal);
        out.color = vec4<f32>(out.color.rgb * ao_value, out.color.a);
    } else {
        // Data views: the field itself with a light relief shade, so the shape still reads
        // without hiding the values.
        let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.5));
        let relief = mix(0.65, 1.0, max(0.0, dot(in.world_normal, light_dir)));
        out.color = vec4<f32>(base_color.rgb * relief, 1.0);
    }

    return out;
}
