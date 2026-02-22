// Erosion compute for a heightmap.
// Algorithm: Smooth Fluvial Erosion
// Pass 1: init_fbm - initialize heightmap with FBM noise.
// Pass 2: erode - particle-based fluvial erosion.
// Pass 3: write_back - uplift, sediment compaction.
// Pass 4: blit_to_texture - copy height/analysis to display.

const HEIGHT_SCALE: f32 = 1000000.0;
const DEPOSITION_SCALE: f32 = 100000.0;  // for i32 fixed-point

@group(0) @binding(0) var<storage, read_write> height: array<atomic<u32>>;

// Parameters
struct ErodeParams {
    map_size: vec2<u32>,
    num_particles: u32,
    iteration: u32,
    num_iterations: u32,
    detail_scale: f32,
    compute_ridge_erosion: u32,
    erosion_strength: f32,
    rock_softness: f32,
    sediment_compaction: f32,
    compaction_threshold: f32,
    channeling: f32,
    channeling_character: f32,
    sediment_removal: f32,
    removal_character: f32,
    wear_angle: f32,
    talus_angle: f32,
    max_deposit_angle: f32,
    flow_length: f32,
    ridge_erosion_steps: u32,
    ridge_softening_amount: f32,
    trail_density: f32,
    ridge_erosion_amount: f32,
    friction: f32,
    rock_friction: f32,
    flow_volume: f32,
    velocity_randomness: f32,
    velocity_randomness_refinement: f32,
    suspended_load: f32,
    river_scarring: f32,
    river_scarring_character: f32,
    river_friction_reduction: f32,
    river_volume: f32,
    uplift: f32,
    // Padding/legacy (keep for uniform size)
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(1) var<uniform> params: ErodeParams;

@group(0) @binding(2) var<storage, read> random_indices: array<u32>;
@group(0) @binding(3) var<storage, read> brush_indices: array<i32>;
@group(0) @binding(4) var<storage, read> brush_weights: array<f32>;

// Accumulation buffers
@group(0) @binding(5) var<storage, read_write> flow_buffer: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> deposition: array<atomic<i32>>;  // loose sediment per cell (fixed-point)
@group(0) @binding(7) var<storage, read_write> erosion_buffer: array<atomic<u32>>;

// Color image as read-write storage texture
@group(1) @binding(0) var color_image: texture_storage_2d<rgba16float, read_write>;

// Hash for value noise (Inigo Quilez, https://iquilezles.org/articles/fbm/)
fn hash21(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn noise_hash(p: vec2<i32>) -> f32 {
    return hash21(vec2<f32>(f32(p.x), f32(p.y)));
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = fract(p);

    let a = noise_hash(i);
    let b = noise_hash(i + vec2<i32>(1,0));
    let c = noise_hash(i + vec2<i32>(0,1));
    let d = noise_hash(i + vec2<i32>(1,1));

    let u = f * f * (3.0 - 2.0 * f);
    let x1 = mix(a, b, u.x);
    let x2 = mix(c, d, u.x);
    return mix(x1, x2, u.y);
}

fn fbm(p: vec2<f32>) -> f32 {
    var total = 0.0;
    var amp = 0.5;
    var freq = 0.004;
    for (var i = 0; i < 6; i = i + 1) {
        total += value_noise(p * freq) * amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    return total;
}

@compute @workgroup_size(8,8,1)
fn init_fbm(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.map_size.x || gid.y >= params.map_size.y) { return; }
    let uv = vec2<f32>(gid.xy);
    let hval = fbm(uv) * 2.0;
    let idx = gid.y * params.map_size.x + gid.x;
    atomicStore(&height[idx], u32(hval * HEIGHT_SCALE));
    
    // Clear flow, deposition, and erosion buffers
    atomicStore(&flow_buffer[idx], 0u);
    atomicStore(&deposition[idx], 0i);
    atomicStore(&erosion_buffer[idx], 0u);
}

fn band_color_from_height(h: f32) -> vec3<f32> {
    // Smooth banding with feathering: water/sand/grass/rock/snow
    let o = 0.0;
    let feather = 0.05; // feather width for blending between layers
    
    let water = vec3<f32>(0.08, 0.35, 0.65);
    let sand = vec3<f32>(0.76, 0.70, 0.50);
    let grass = vec3<f32>(0.20, 0.55, 0.25) * 0.5;
    let rock = vec3<f32>(0.40, 0.40, 0.40) * 0.5;
    let snow = vec3<f32>(0.95, 0.95, 0.95);
    
    // Define thresholds
    let t1 = 0.25 + o; // water -> sand
    let t2 = 0.35 + o; // sand -> grass
    let t3 = 0.60 + o; // grass -> rock
    let t4 = 0.80 + o; // rock -> snow

    return rock;
    
    // Blend between adjacent colors
    // if (h < t1 - feather) {
    //     return water;
    // } else if (h < t1 + feather) {
    //     let t = (h - (t1 - feather)) / (2.0 * feather);
    //     return mix(water, sand, smoothstep(0.0, 1.0, t));
    // } else if (h < t2 - feather) {
    //     return sand;
    // } else if (h < t2 + feather) {
    //     let t = (h - (t2 - feather)) / (2.0 * feather);
    //     return mix(sand, grass, smoothstep(0.0, 1.0, t));
    // } else if (h < t3 - feather) {
    //     return grass;
    // } else if (h < t3 + feather) {
    //     let t = (h - (t3 - feather)) / (2.0 * feather);
    //     return mix(grass, rock, smoothstep(0.0, 1.0, t));
    // } else if (h < t4 - feather) {
    //     return rock;
    // } else if (h < t4 + feather) {
    //     let t = (h - (t4 - feather)) / (2.0 * feather);
    //     return mix(rock, snow, smoothstep(0.0, 1.0, t));
    // } else {
    //     return snow;
    // }
}

@compute @workgroup_size(8,8,1)
fn init_color_bands(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.map_size.x || gid.y >= params.map_size.y) { return; }
    let size = i32(params.map_size.x);
    let idx = u32(gid.y * params.map_size.x + gid.x);
    let h = f32(atomicLoad(&height[idx])) / HEIGHT_SCALE;
    let c = band_color_from_height(h);
    
    // Generate FBM noise for black overlay (using different scale/offset from height field)
    let uv = vec2<f32>(gid.xy);
    let noise = fbm(uv * 0.015 + vec2<f32>(100.0, 100.0)); // Different scale and offset
    let noise_factor = clamp(noise, 0.0, 1.0);
    let black = vec3<f32>(0.0, 0.0, 0.0);
    
    // Blend black FBM noise on top of color bands
    let blended = mix(c, black, noise_factor); // 30% max darkening
    
    textureStore(color_image, vec2<i32>(gid.xy), vec4<f32>(blended, 1.0));
}

fn load_height(p: vec2<i32>) -> f32 {
    let size = i32(params.map_size.x);
    let idx = u32(p.y * size + p.x);
    return f32(atomicLoad(&height[idx])) / HEIGHT_SCALE;
}

fn load_deposition(p: vec2<i32>) -> f32 {
    let size = i32(params.map_size.x);
    let idx = u32(p.y * size + p.x);
    return f32(atomicLoad(&deposition[idx])) / DEPOSITION_SCALE;
}

// 2D hash for random scatter (Inigo Quilez, https://iquilezles.org/articles/fbm/)
fn random22(p: vec2<f32>) -> vec2<f32> {
    let q = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)));
    return fract(sin(q) * 43758.5453);
}

fn random31(p: vec3<f32>) -> f32 {
    let q = dot(p, vec3<f32>(127.1, 311.7, 74.7));
    return fract(sin(q) * 43758.5453);
}

fn bilinear_sample_height(pos: vec2<f32>) -> f32 {
    let res = vec2<f32>(params.map_size.xy);
    var p = pos - 0.5;
    p = vec2<f32>(clamp(p.x, 0.0, res.x - 1.00001), clamp(p.y, 0.0, res.y - 1.00001));
    let w = vec2<f32>(p.x - floor(p.x), p.y - floor(p.y));
    let size = i32(params.map_size.x);
    let ix = i32(floor(p.x));
    let iy = i32(floor(p.y));
    let h00 = load_height(vec2<i32>(ix, iy));
    let h10 = load_height(vec2<i32>(min(ix + 1, size - 1), iy));
    let h01 = load_height(vec2<i32>(ix, min(iy + 1, size - 1)));
    let h11 = load_height(vec2<i32>(min(ix + 1, size - 1), min(iy + 1, size - 1)));
    return h00 * (1.0 - w.x) * (1.0 - w.y) + h10 * w.x * (1.0 - w.y) + h01 * (1.0 - w.x) * w.y + h11 * w.x * w.y;
}

fn sample_height_and_gradient(pos: vec2<f32>) -> vec3<f32> {
    let size = vec2<i32>(i32(params.map_size.x), i32(params.map_size.y));
    let cx = clamp(i32(floor(pos.x)), 0, size.x - 2);
    let cy = clamp(i32(floor(pos.y)), 0, size.y - 2);
    let x = pos.x - f32(cx);
    let y = pos.y - f32(cy);
    let idx = vec2<i32>(cx, cy);
    let hNW = load_height(idx);
    let hNE = load_height(idx + vec2<i32>(1,0));
    let hSW = load_height(idx + vec2<i32>(0,1));
    let hSE = load_height(idx + vec2<i32>(1,1));
    let gradX = (hNE - hNW) * (1.0 - y) + (hSE - hSW) * y;
    let gradY = (hSW - hNW) * (1.0 - x) + (hSE - hNE) * x;
    let h = hNW * (1.0 - x) * (1.0 - y) + hNE * x * (1.0 - y) + hSW * (1.0 - x) * y + hSE * x * y;
    return vec3<f32>(gradX, gradY, h);
}

// Smooth Fluvial Erosion
@compute @workgroup_size(256, 1, 1)
fn erode(@builtin(global_invocation_id) gid: vec3<u32>) {
    let particle_id = gid.x;
    let res_x = i32(params.map_size.x);
    let res_y = i32(params.map_size.y);
    let total_pixels = u32(res_x * res_y);
    let num_particles_per_pass = params.num_particles / 2u;
    let dx = 1.0;
    let detail_scale = max(params.detail_scale, 0.001);
    let dx_scaled = dx / detail_scale;

    // Run both ridge and flow passes in one dispatch: first half = ridge (iteration 0), second half = flow (iteration 1)
    let iteration = select(0u, 1u, particle_id >= num_particles_per_pass);
    let detail_bias = select(0.0, 1.0, f32(iteration) > f32(params.num_iterations) / 2.0);
    let num_trails = f32(total_pixels) * min(pow(dx_scaled, detail_bias), 1.0) * params.trail_density;
    let do_ridge = params.compute_ridge_erosion != 0u && detail_bias < 0.5;
    var adjusted_trails = num_trails;
    if (do_ridge) {
        adjusted_trails *= params.ridge_erosion_amount;
    }
    let local_particle_id = select(particle_id, particle_id - num_particles_per_pass, particle_id >= num_particles_per_pass);
    if (f32(local_particle_id) >= adjusted_trails) { return; }

    var wear_angle = params.wear_angle * 3.14159265 / 180.0;
    wear_angle = tan(wear_angle);
    var talus_angle = params.talus_angle * 3.14159265 / 180.0;
    talus_angle = tan(talus_angle) / 1.25;
    var max_deposit_angle = params.max_deposit_angle * 3.14159265 / 180.0;
    max_deposit_angle = tan(max_deposit_angle);

    var p = random22(vec2<f32>(f32(local_particle_id), f32(local_particle_id) - f32(iteration) * 5.643));
    p *= vec2<f32>(f32(res_x), f32(res_y));
    var v = vec2<f32>(0.0, 0.0);

    for (var i = 0; i < i32(params.ridge_softening_amount) + 1; i = i + 1) {
        let dirs = array<vec2<i32>, 4>(vec2<i32>(1,0), vec2<i32>(0,1), vec2<i32>(-1,0), vec2<i32>(0,-1));
        var grad = vec2<f32>(0.0, 0.0);
        for (var j = 0; j < 4; j = j + 1) {
            let pj = vec2<i32>(clamp(i32(p.x) + dirs[j].x, 0, res_x - 1), clamp(i32(p.y) + dirs[j].y, 0, res_y - 1));
            grad += vec2<f32>(f32(dirs[j].x), f32(dirs[j].y)) * load_height(pj);
        }
        let glen = length(grad);
        if (glen > 0.0001) {
            p += normalize(grad) * clamp(params.ridge_softening_amount - f32(i), 0.0, 1.0);
        }
    }

    let sample_idx = vec2<i32>(clamp(i32(p.x), 0, res_x - 1), clamp(i32(p.y), 0, res_y - 1));
    let dirs_grad = array<vec2<i32>, 4>(vec2<i32>(1,0), vec2<i32>(0,1), vec2<i32>(-1,0), vec2<i32>(0,-1));
    var slope = 0.0;
    for (var j = 0; j < 4; j = j + 1) {
        let pj = vec2<i32>(clamp(sample_idx.x + dirs_grad[j].x, 0, res_x - 1), clamp(sample_idx.y + dirs_grad[j].y, 0, res_y - 1));
        slope = max(slope, abs(load_height(pj) - load_height(sample_idx)) / dx_scaled);
    }
    if (slope <= wear_angle || slope <= talus_angle) { return; }

    var erosion_strength = params.erosion_strength;
    var flow_steps = select(i32(params.ridge_erosion_steps), i32(params.flow_length / dx_scaled), detail_bias > 0.5);
    flow_steps = i32(f32(flow_steps) * (0.5 + random31(vec3<f32>(p.x, p.y, f32(iteration)))));

    let dx_inv = 1.0 / dx_scaled;
    let friction_coef = pow(1.0 - params.friction, dx_scaled);
    let rock_friction_coef = pow(1.0 - params.rock_friction, dx_scaled);
    let channeling_coef = pow(1.0 - 0.1 * params.channeling, pow(dx_scaled, params.channeling_character));
    let removal_coef = pow(1.0 - params.sediment_removal, pow(dx_scaled, params.removal_character));
    let river_scarring = params.river_scarring * pow(dx_scaled, params.river_scarring_character);

    var carry = 0.0;
    var h_current = bilinear_sample_height(p);

    for (var step = 0; step < flow_steps; step = step + 1) {
        let cell = vec2<i32>(clamp(i32(p.x), 0, res_x - 1), clamp(i32(p.y), 0, res_y - 1));
        let idx = u32(cell.y * res_x + cell.x);

        var hmin = load_height(cell);
        var hmax = hmin;
        var grad = vec2<f32>(0.0, 0.0);
        for (var j = -1; j <= 1; j = j + 2) {
            let pxj = vec2<i32>(clamp(cell.x + j, 0, res_x - 1), cell.y);
            let dep_pxj = load_deposition(pxj);
            grad.x += f32(j) * (load_height(pxj) + params.flow_volume * dep_pxj);
            hmin = min(hmin, load_height(pxj));
            hmax = max(hmax, load_height(pxj));
        }
        for (var j = -1; j <= 1; j = j + 2) {
            let pyj = vec2<i32>(cell.x, clamp(cell.y + j, 0, res_y - 1));
            grad.y += f32(j) * (load_height(pyj) + params.flow_volume * load_deposition(pyj));
            hmin = min(hmin, load_height(pyj));
            hmax = max(hmax, load_height(pyj));
        }
        grad *= 0.5 * dx_inv;

        let dep_val = load_deposition(cell);
        if (dep_val == 0.0) {
            v *= pow(1.0 - params.rock_friction, dx_scaled);
        } else {
            v *= pow(1.0 - params.friction, dx_scaled);
        }
        v += -grad * dx_scaled;

        if (params.velocity_randomness != 0.0) {
            let d_rand = random31(vec3<f32>(p.x, p.y, f32(step))) * 6.2831853;
            p += normalize(v + vec2<f32>(cos(d_rand), sin(d_rand)) * params.velocity_randomness * pow(1.0 - params.velocity_randomness_refinement, f32(iteration)));
        } else {
            let vlen = length(v);
            if (vlen > 0.0001) {
                p += normalize(v);
            }
        }

        let h_prev = h_current;
        h_current = bilinear_sample_height(p);

        let dh_raw = erosion_strength * min(max(h_current - h_prev, -max_deposit_angle * dx_scaled) + talus_angle * dx_scaled, 0.0);
        var dh = max(dh_raw, -dep_val) + min(dh_raw + dep_val, 0.0) * params.rock_softness;
        dh += min((1.0 - params.suspended_load) * carry, max_deposit_angle * dx_scaled);
        carry -= dh;
        carry = max(carry, 0.0);

        carry *= channeling_coef;
        if (dh > 0.0) {
            dh *= removal_coef;
        }
        dh = max(dh, hmin - load_height(cell));
        dh = min(dh, hmax - load_height(cell));

        let dep_delta = max(dh, -dep_val);
        if (dh >= 0.0) {
            atomicAdd(&height[idx], u32(dh * HEIGHT_SCALE));
        } else {
            atomicSub(&height[idx], u32(-dh * HEIGHT_SCALE));
        }
        atomicAdd(&deposition[idx], i32(dep_delta * DEPOSITION_SCALE));
        if (dh < 0.0) {
            atomicAdd(&erosion_buffer[idx], u32(-dh * 100000.0));
        }
        let flow_val = max(0.0, dh) * dx_inv;
        atomicAdd(&flow_buffer[idx], u32(flow_val * 10000.0));

        let px = i32(p.x);
        let py = i32(p.y);
        if (px < 1 || px >= res_x - 1 || py < 1 || py >= res_y - 1) { break; }
    }
}

// writeBack: uplift, sediment compaction (runs after erosion passes)
// Uses @group(0) only (same as erosion pass group 0)
@compute @workgroup_size(8, 8, 1)
fn write_back(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.map_size.x || gid.y >= params.map_size.y) { return; }
    let size = i32(params.map_size.x);
    let idx = u32(gid.y * params.map_size.x + gid.x);
    let dx = 1.0 / max(params.detail_scale, 0.001);
    let uplift_scaled = params.uplift * params.trail_density * dx;

    let h_old = f32(atomicLoad(&height[idx])) / HEIGHT_SCALE;
    let h_new = h_old + uplift_scaled;
    atomicStore(&height[idx], u32(h_new * HEIGHT_SCALE));

    let dep = f32(atomicLoad(&deposition[idx])) / DEPOSITION_SCALE;
    if (dep > params.compaction_threshold * params.detail_scale) {
        let dep_sub = dep - params.compaction_threshold;
        let dep_compacted = dep_sub * pow(1.0 - params.sediment_compaction, 0.25 * dx) + params.compaction_threshold;
        atomicStore(&deposition[idx], i32(dep_compacted * DEPOSITION_SCALE));
    }
}

@group(1) @binding(0) var display_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var normal_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var analysis_out: texture_storage_2d<rgba16float, write>;  // R=flow_mag, G=sediment, B=erosion, A=curvature
@group(1) @binding(3) var ao_out: texture_storage_2d<r16float, read_write>;

@compute @workgroup_size(8,8,1)
fn blit_to_texture(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.map_size.x || gid.y >= params.map_size.y) { return; }
    let size_x = i32(params.map_size.x);
    let size_y = i32(params.map_size.y);
    let px = i32(gid.x);
    let py = i32(gid.y);
    let idx = u32(py * size_x + px);
    let h = f32(atomicLoad(&height[idx])) / HEIGHT_SCALE;

    // grayscale height to display
    let c = h;
    textureStore(display_out, vec2<i32>(px, py), vec4<f32>(c, c, c, 1.0));

    // compute normals from central differences of height field
    let xm1 = max(px - 1, 0);
    let xp1 = min(px + 1, size_x - 1);
    let ym1 = max(py - 1, 0);
    let yp1 = min(py + 1, size_y - 1);

    let hL = f32(atomicLoad(&height[u32(py * size_x + xm1)])) / HEIGHT_SCALE;
    let hR = f32(atomicLoad(&height[u32(py * size_x + xp1)])) / HEIGHT_SCALE;
    let hD = f32(atomicLoad(&height[u32(ym1 * size_x + px)])) / HEIGHT_SCALE;
    let hU = f32(atomicLoad(&height[u32(yp1 * size_x + px)])) / HEIGHT_SCALE;

    let dx = hR - hL;
    let dy = hU - hD;
    let n = normalize(vec3<f32>(-dx, 1.0, -dy));
    let enc = n * 0.5 + vec3<f32>(0.5, 0.5, 0.5);
    textureStore(normal_out, vec2<i32>(px, py), vec4<f32>(enc, 1.0));
    
    // Pack flow, deposition (sediment), and erosion into single analysis texture
    let flow_mag = f32(atomicLoad(&flow_buffer[idx])) / 10000.0;
    let sediment = max(0.0, f32(atomicLoad(&deposition[idx])) / DEPOSITION_SCALE);
    let erosion = f32(atomicLoad(&erosion_buffer[idx])) / 100000.0;
    
    // Compute curvature using Laplacian of height field (5-point stencil)
    // Positive curvature = convex (peaks/ridges), negative = concave (valleys)
    let curvature_raw = (hL + hR + hD + hU - 4.0 * h);
    // Scale and center curvature for visualization (map to [0,1] range approximately)
    let curvature = curvature_raw * 50.0 + 0.5;
    
    // Pack into analysis texture: R=flow_mag, G=sediment, B=erosion, A=curvature
    let analysis = vec4<f32>(
        flow_mag,
        sediment,
        erosion,
        curvature
    );
    textureStore(analysis_out, vec2<i32>(px, py), analysis);
}

// Hammersley low-discrepancy sequence for uniform sampling
// Based on Bevy's hammersley_2d implementation
fn radical_inverse_VdC(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10; // / 0x100000000
}

fn hammersley_2d(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(n), radical_inverse_VdC(i));
}

// PCG hash for noise generation
fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Interleaved gradient noise for dithering
fn interleaved_gradient_noise(pixel_coordinates: vec2<f32>, frame: u32) -> f32 {
    let xy = pixel_coordinates + 5.588238 * f32(frame % 64u);
    return fract(52.9829189 * fract(0.06711056 * xy.x + 0.00583715 * xy.y));
}

const PI = 3.14159265359;
const HALF_PI = 1.57079632679;
const PI_2 = 6.28318530718;

// Compute ambient occlusion from heightmap
// Sample count, radius, strength are hardcoded for compute shader
@compute @workgroup_size(8,8,1)
fn compute_ao(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.map_size.x || gid.y >= params.map_size.y) { return; }
    
    let size_x = i32(params.map_size.x);
    let size_y = i32(params.map_size.y);
    let px = i32(gid.x);
    let py = i32(gid.y);
    let idx = u32(py * size_x + px);
    
    let current_height = f32(atomicLoad(&height[idx])) / HEIGHT_SCALE;
    
    // AO parameters (matching the default values from Rust)
    let sample_count = 64u; // Increased for smoother result
    let sample_radius_px = 0.08 * f32(params.map_size.x); // Convert UV radius to pixels
    let strength = 1.5;
    let bias = 0.0;
    let height_scale = 50.0; // Match terrain height scale
    let terrain_size = 512.0; // Terrain world size
    
    var occlusion = 0.0;
    
    for (var i = 0u; i < sample_count; i = i + 1u) {
        // Use Hammersley sequence for well-distributed samples (no noise for consistency)
        let h = hammersley_2d(i, sample_count);
        
        // Angle from Hammersley sequence only
        let angle = h.x * PI_2;
        
        // Radius using sqrt for uniform disk distribution
        let radius_t = h.y;
        let radius_px = sqrt(radius_t) * sample_radius_px;
        
        // Sample offset in pixel space
        let offset = vec2<f32>(cos(angle), sin(angle)) * radius_px;
        let sample_x = clamp(px + i32(round(offset.x)), 0, size_x - 1);
        let sample_y = clamp(py + i32(round(offset.y)), 0, size_y - 1);
        let sample_idx = u32(sample_y * size_x + sample_x);
        
        // Sample height
        let sample_height = f32(atomicLoad(&height[sample_idx])) / HEIGHT_SCALE;
        
        // Height difference (scaled to world space)
        let height_diff = (sample_height - current_height) * height_scale;
        
        // Distance in pixel space
        let px_distance = length(offset);
        
        // Convert to world distance
        let world_distance = (px_distance / f32(params.map_size.x)) * terrain_size;
        
        // Skip if distance is too small
        if (world_distance < 0.1) {
            continue;
        }
        
        // Calculate occlusion: samples higher than current point block sky light
        if (height_diff > 0.001) {
            // Calculate the horizon angle
            let elevation_angle = atan(height_diff / world_distance);
            
            // Distance falloff with smoothstep
            let t = px_distance / sample_radius_px;
            let falloff = 1.0 - (t * t * (3.0 - 2.0 * t));
            
            // Accumulate occlusion
            let contribution = (elevation_angle / HALF_PI) * falloff;
            occlusion = occlusion + contribution;
        }
    }
    
    // Average and scale occlusion
    occlusion = (occlusion / f32(sample_count)) * strength * 8.0;
    
    // Convert to AO value (1.0 = bright, 0.0 = dark)
    let ao_value = 1.0 - clamp(occlusion, 0.0, 1.0);
    
    // Apply power curve for contrast
    let ao_final = pow(ao_value, 0.8);
    
    // Apply bias
    let ao = max(ao_final, bias);
    
    // Write AO value to texture (single channel R16Float)
    textureStore(ao_out, vec2<i32>(px, py), vec4<f32>(ao, 0.0, 0.0, 0.0));
}

// Gaussian blur pass for smoothing AO
// Temporary storage texture for separable blur
@group(1) @binding(4) var ao_temp: texture_storage_2d<r16float, read_write>;

@compute @workgroup_size(8,8,1)
fn blur_ao_horizontal(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.map_size.x || gid.y >= params.map_size.y) { return; }
    
    let px = i32(gid.x);
    let py = i32(gid.y);
    let size_x = i32(params.map_size.x);
    
    // 5-tap Gaussian kernel weights (approximation: 1, 4, 6, 4, 1) / 16
    let weights = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);
    
    var sum = 0.0;
    for (var i = 0; i < 5; i = i + 1) {
        let offset = i - 2;
        let sample_x = clamp(px + offset, 0, size_x - 1);
        let ao_val = textureLoad(ao_out, vec2<i32>(sample_x, py)).x;
        sum = sum + ao_val * weights[i];
    }
    
    // Write to temp texture
    textureStore(ao_temp, vec2<i32>(px, py), vec4<f32>(sum, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8,8,1)
fn blur_ao_vertical(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.map_size.x || gid.y >= params.map_size.y) { return; }
    
    let px = i32(gid.x);
    let py = i32(gid.y);
    let size_y = i32(params.map_size.y);
    
    // 5-tap Gaussian kernel weights
    let weights = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);
    
    var sum = 0.0;
    for (var i = 0; i < 5; i = i + 1) {
        let offset = i - 2;
        let sample_y = clamp(py + offset, 0, size_y - 1);
        let ao_val = textureLoad(ao_temp, vec2<i32>(px, sample_y)).x;
        sum = sum + ao_val * weights[i];
    }
    
    // Write back to AO output
    textureStore(ao_out, vec2<i32>(px, py), vec4<f32>(sum, 0.0, 0.0, 0.0));
}

