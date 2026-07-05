// Erosion compute for a heightmap.
// Algorithm: Smooth Fluvial Erosion
// Pass 1: init_fbm - initialize heightmap with FBM noise.
// Pass 2: erode - particle-based fluvial erosion.
// Pass 3: write_back - uplift, sediment compaction.
// Pass 4: blit_to_texture - copy height/analysis to display.

const HEIGHT_SCALE: f32 = 1000000.0;
const DEPOSITION_SCALE: f32 = 100000.0;  // for i32 fixed-point
const MACC_SCALE: f32 = 100000.0;
const MOMENTUM_SCALE: f32 = 100000.0;
// Compute_Erosion binding transforms (see `ref/node_tree.json`)
const TRAIL_DENSITY_SCALE: f32 = 0.1;   // Trail_Density = Erosion_Amount * 0.1
const RIVER_SCARRING_SCALE: f32 = 0.25; // River_Scarring = River_Scale * 0.25

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
    erosion_amount: f32,
    ridge_erosion_amount: f32,
    friction: f32,
    rock_friction: f32,
    flow_volume: f32,
    velocity_randomness: f32,
    velocity_randomness_refinement: f32,
    suspended_load: f32,
    river_scale: f32,
    river_character: f32,
    river_friction_reduction: f32,
    river_volume: f32,
    do_rivers: u32,
    river_channeling: f32,
    momentum_coherence: f32,
    meander_longevity: f32,
    uplift_river_carving: f32,
    uplift: f32,
    noise_frequency: f32,
    noise_scale: f32,
}

@group(0) @binding(1) var<uniform> params: ErodeParams;

// River / momentum accumulation (enabled when do_rivers != 0)
@group(0) @binding(2) var<storage, read_write> macc: array<atomic<u32>>;

// Accumulation buffers
@group(0) @binding(3) var<storage, read_write> flow_buffer: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> deposition: array<atomic<i32>>;  // loose sediment per cell (fixed-point)
@group(0) @binding(5) var<storage, read_write> erosion_buffer: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> mx: array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> my: array<atomic<i32>>;

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

const FBM_OCTAVES: u32 = 10u;
const FBM_LACUNARITY: f32 = 2.0;
const FBM_PERSISTENCE: f32 = 0.5;

fn fbm(p: vec2<f32>) -> f32 {
    var total = 0.0;
    var amp = 0.5;
    var freq = params.noise_frequency;
    for (var i = 0u; i < FBM_OCTAVES; i = i + 1u) {
        // Offset each octave so stacked layers don't align on the same grid.
        let offset = vec2<f32>(f32(i) * 17.3, f32(i) * 23.7);
        total += value_noise(p * freq + offset) * amp;
        amp *= FBM_PERSISTENCE;
        freq *= FBM_LACUNARITY;
    }
    return total;
}

@compute @workgroup_size(8,8,1)
fn init_fbm(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.map_size.x || gid.y >= params.map_size.y) { return; }
    let uv = vec2<f32>(gid.xy);
    let hval = fbm(uv) * params.noise_scale;
    let idx = gid.y * params.map_size.x + gid.x;
    atomicStore(&height[idx], u32(hval * HEIGHT_SCALE));
    
    // Clear flow, deposition, erosion, and river/momentum buffers
    atomicStore(&flow_buffer[idx], 0u);
    atomicStore(&deposition[idx], 0i);
    atomicStore(&erosion_buffer[idx], 0u);
    atomicStore(&macc[idx], 0u);
    atomicStore(&mx[idx], 0i);
    atomicStore(&my[idx], 0i);
}

@compute @workgroup_size(8,8,1)
fn init_color_bands(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.map_size.x || gid.y >= params.map_size.y) { return; }
    let gray = vec3<f32>(0.5, 0.5, 0.5);
    textureStore(color_image, vec2<i32>(gid.xy), vec4<f32>(gray, 1.0));
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

fn load_macc(p: vec2<i32>) -> f32 {
    let size = i32(params.map_size.x);
    let idx = u32(p.y * size + p.x);
    return f32(atomicLoad(&macc[idx])) / MACC_SCALE;
}

fn load_mx(p: vec2<i32>) -> f32 {
    let size = i32(params.map_size.x);
    let idx = u32(p.y * size + p.x);
    return f32(atomicLoad(&mx[idx])) / MOMENTUM_SCALE;
}

fn load_my(p: vec2<i32>) -> f32 {
    let size = i32(params.map_size.x);
    let idx = u32(p.y * size + p.x);
    return f32(atomicLoad(&my[idx])) / MOMENTUM_SCALE;
}

fn add_macc(idx: u32, val: f32) {
    atomicAdd(&macc[idx], u32(val * MACC_SCALE));
}

fn store_macc(idx: u32, val: f32) {
    atomicStore(&macc[idx], u32(val * MACC_SCALE));
}

fn add_mx(idx: u32, val: f32) {
    atomicAdd(&mx[idx], i32(val * MOMENTUM_SCALE));
}

fn add_my(idx: u32, val: f32) {
    atomicAdd(&my[idx], i32(val * MOMENTUM_SCALE));
}

fn store_mx(idx: u32, val: f32) {
    atomicStore(&mx[idx], i32(val * MOMENTUM_SCALE));
}

fn store_my(idx: u32, val: f32) {
    atomicStore(&my[idx], i32(val * MOMENTUM_SCALE));
}

// Reference hash functions (compute_erosion.cl)
fn random22(p: vec2<f32>) -> vec2<f32> {
    let out = vec2<f32>(
        sin(dot(p, vec2<f32>(127.141, 311.742))),
        sin(dot(p, vec2<f32>(269.513, 183.357))),
    ) * 621.0;
    return out - floor(out);
}

fn random31(p: vec3<f32>) -> f32 {
    let out = sin(dot(p, vec3<f32>(127.141, 311.742, 251.523))) * 621.5153;
    return out - floor(out);
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

// Smooth Fluvial Erosion (ported from ref/compute_erosion.cl)
@compute @workgroup_size(256, 1, 1)
fn erode(@builtin(global_invocation_id) gid: vec3<u32>) {
    let particle_id = gid.x;
    let res_x = i32(params.map_size.x);
    let res_y = i32(params.map_size.y);
    let total_pixels = u32(res_x * res_y);
    let dx = 1.0;
    let detail_scale = max(params.detail_scale, 0.001);
    let dx_scaled = dx / detail_scale;
    let iteration = params.iteration;

    // Detail_Bias: ridge-biased (0) vs flow-biased (1), alternating each OpenCL iteration
    let detail_bias = f32(iteration % 2u);
    if (params.compute_ridge_erosion == 0u && detail_bias < 0.5) {
        return;
    }

    var num_trails = f32(total_pixels)
        * min(pow(dx_scaled, detail_bias), 1.0)
        * params.erosion_amount
        * TRAIL_DENSITY_SCALE;
    if (params.compute_ridge_erosion != 0u && detail_bias < 0.5) {
        num_trails *= params.ridge_erosion_amount;
    }
    if (f32(particle_id) >= num_trails) {
        return;
    }

    let gidx = i32(particle_id % u32(res_x));
    let gidy = i32(particle_id / u32(res_x));

    var wear_angle = params.wear_angle * 3.14159265 / 180.0;
    wear_angle = tan(wear_angle);
    var talus_angle = params.talus_angle * 3.14159265 / 180.0;
    talus_angle = tan(talus_angle) / 1.25;
    var max_deposit_angle = params.max_deposit_angle * 3.14159265 / 180.0;
    max_deposit_angle = tan(max_deposit_angle);

    var p = random22(vec2<f32>(f32(gidx), f32(gidy) - f32(iteration) * 5.643));
    p *= vec2<f32>(f32(res_x), f32(res_y));
    var v = vec2<f32>(0.0, 0.0);

    let dirs = array<vec2<i32>, 4>(
        vec2<i32>(1, 0), vec2<i32>(0, 1), vec2<i32>(-1, 0), vec2<i32>(0, -1),
    );

    for (var i = 0; i < i32(params.ridge_softening_amount) + 1; i = i + 1) {
        var grad = vec2<f32>(0.0, 0.0);
        for (var j = 0; j < 4; j = j + 1) {
            let pj = vec2<i32>(
                clamp(i32(p.x) + dirs[j].x, 0, res_x - 1),
                clamp(i32(p.y) + dirs[j].y, 0, res_y - 1),
            );
            grad += vec2<f32>(f32(dirs[j].x), f32(dirs[j].y)) * load_height(pj);
        }
        let glen = length(grad);
        if (glen > 0.0001) {
            p += normalize(grad) * clamp(params.ridge_softening_amount - f32(i), 0.0, 1.0);
        }
    }

    let sample_idx = vec2<i32>(clamp(i32(p.x), 0, res_x - 1), clamp(i32(p.y), 0, res_y - 1));
    var slope = 0.0;
    for (var j = 0; j < 4; j = j + 1) {
        let pj = vec2<i32>(
            clamp(sample_idx.x + dirs[j].x, 0, res_x - 1),
            clamp(sample_idx.y + dirs[j].y, 0, res_y - 1),
        );
        slope = max(slope, abs(load_height(pj) - load_height(sample_idx)) / dx_scaled);
    }
    if (slope <= wear_angle || slope <= talus_angle) {
        return;
    }

    var erosion_strength = params.erosion_strength;
    var flow_steps = select(
        i32(params.ridge_erosion_steps),
        i32(params.flow_length / dx_scaled),
        detail_bias > 0.5,
    );
    flow_steps = i32(f32(flow_steps) * (0.5 + random31(vec3<f32>(p.x, p.y, f32(iteration)))));

    let dx_inv = 1.0 / dx_scaled;
    let do_rivers = params.do_rivers != 0u;
    let use_momentum = do_rivers && params.momentum_coherence > 0.0;

    // UI `River_Scale` → kernel `River_Scarring` via RIVER_SCARRING_SCALE
    var river_scarring = params.river_scale * RIVER_SCARRING_SCALE;
    var river_friction_reduction = params.river_friction_reduction;
    var river_volume = params.river_volume;
    var momentum_coherence = params.momentum_coherence;

    var particle_water = 0.0;
    if (do_rivers) {
        momentum_coherence *= dx_scaled * 0.001 * detail_bias;
        river_scarring *= max(dx_scaled, pow(dx_scaled, params.river_character)) * detail_bias;
        river_volume *= river_scarring * 0.002 * dx_scaled;
        river_friction_reduction *= river_scarring;
        let momentum_extinction = pow(
            params.meander_longevity,
            pow(dx_scaled, 1.0 - params.river_character),
        );
        particle_water = 1.0 - momentum_extinction;
    }

    var carry = 0.0;
    var h_current = bilinear_sample_height(p);

    for (var step = 0; step < flow_steps; step = step + 1) {
        let cell = vec2<i32>(clamp(i32(p.x), 0, res_x - 1), clamp(i32(p.y), 0, res_y - 1));
        let idx = u32(cell.y * res_x + cell.x);

        var hmin = load_height(cell);
        var hmax = hmin;
        var grad = vec2<f32>(0.0, 0.0);
        var mgrad = vec2<f32>(0.0, 0.0);

        for (var j = -1; j <= 1; j = j + 2) {
            let pxj = vec2<i32>(clamp(cell.x + j, 0, res_x - 1), cell.y);
            grad.x += f32(j) * (load_height(pxj) + params.flow_volume * load_deposition(pxj));
            if (do_rivers) {
                mgrad.x += f32(j) * load_macc(pxj);
            }
            hmin = min(hmin, load_height(pxj));
            hmax = max(hmax, load_height(pxj));
        }
        for (var j = -1; j <= 1; j = j + 2) {
            let pyj = vec2<i32>(cell.x, clamp(cell.y + j, 0, res_y - 1));
            grad.y += f32(j) * (load_height(pyj) + params.flow_volume * load_deposition(pyj));
            if (do_rivers) {
                mgrad.y += f32(j) * load_macc(pyj);
            }
            hmin = min(hmin, load_height(pyj));
            hmax = max(hmax, load_height(pyj));
        }

        let dep_val = load_deposition(cell);
        var dt = 1.0;

        // Scarring uses macc before this step's water deposit (ref/compute_erosion.cl HAS_macc)
        var scarring_coefficient = 1.0;
        if (do_rivers) {
            scarring_coefficient =
                1.0 / (1.0 + river_scarring * load_macc(cell));
        }

        // Surface-normal transport (ref HAS_macc). When Do_Rivers is off the OpenCL #else branch
        // sets v = -grad at friction=1, which zig-zags on our grid and spikes; keep normal-force
        // for stability while still skipping macc/scarring when rivers are disabled.
        var vlen = length(v);
        dt = min(1.0 / max(vlen, 0.0001), 1.0);

        if (do_rivers) {
            let friction_coefficient =
                1.0 / (1.0 + river_friction_reduction * min(load_macc(cell), 250.0));
            if (dep_val == 0.0) {
                v *= pow(1.0 - params.rock_friction * friction_coefficient, dx_scaled);
            } else {
                v *= pow(1.0 - params.friction * friction_coefficient, dx_scaled);
            }
        } else if (params.rock_friction != 1.0 || params.friction != 1.0) {
            if (dep_val == 0.0) {
                v *= pow(1.0 - params.rock_friction, dx_scaled);
            } else {
                v *= pow(1.0 - params.friction, dx_scaled);
            }
        }

        v -= normalize(vec3<f32>(grad.x, grad.y, dx_scaled)).xy * dx_scaled;

        if (do_rivers && use_momentum) {
            let macc_val = load_macc(cell);
            let mtmp = vec2<f32>(load_mx(cell), load_my(cell));
            let blend = momentum_coherence * min(macc_val, 250.0) * dt;
            add_mx(idx, v.x);
            add_my(idx, v.y);
            if (macc_val >= particle_water) {
                v += (mtmp * particle_water / macc_val - v) * blend;
            }
        }

        if (do_rivers) {
            v += -river_volume * mgrad;
            add_macc(idx, particle_water);
        }

        if (params.velocity_randomness != 0.0) {
            let d_rand = random31(vec3<f32>(p.x, p.y, f32(step))) * 6.2831853;
            let dp = normalize(v) + vec2<f32>(cos(d_rand), sin(d_rand))
                * params.velocity_randomness
                * pow(1.0 - params.velocity_randomness_refinement, f32(iteration));
            let dplen = length(dp);
            if (dplen > 0.0001) {
                p += dp / dplen;
            }
        } else {
            let vlen = length(v);
            if (vlen > 0.0001) {
                p += v / vlen;
            }
        }

        if (p.x < 0.0 || p.y < 0.0 || p.x > f32(res_x - 1) || p.y > f32(res_y - 1)) {
            return;
        }

        let h_prev = h_current;
        h_current = bilinear_sample_height(p);

        var dh = erosion_strength * min(
            max(h_current - h_prev, -max_deposit_angle * dx_scaled) + talus_angle * dx_scaled,
            0.0,
        );
        dh = max(dh, -dep_val) + min(dh + dep_val, 0.0) * params.rock_softness;

        dh += min((1.0 - params.suspended_load) * carry * scarring_coefficient, max_deposit_angle * dx_scaled);
        carry -= dh;
        carry = max(carry, 0.0);

        if (do_rivers) {
            carry *= pow(
                1.0 - 0.1 * params.channeling * scarring_coefficient,
                pow(dx_scaled, params.channeling_character),
            );
            if (dh > 0.0) {
                dh *= pow(
                    1.0 - params.sediment_removal * scarring_coefficient,
                    pow(dx_scaled, params.removal_character),
                );
            }
        } else {
            carry *= pow(1.0 - 0.1 * params.channeling, pow(dx_scaled, params.channeling_character));
            if (dh > 0.0) {
                dh *= pow(1.0 - params.sediment_removal, pow(dx_scaled, params.removal_character));
            }
        }

        let cell_h = load_height(cell);
        dh = max(dh, hmin - cell_h);
        dh = min(dh, hmax - cell_h);

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
    }
}

// writeBack: uplift, sediment compaction, river momentum decay (ref/compute_erosion.cl)
@compute @workgroup_size(8, 8, 1)
fn write_back(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.map_size.x || gid.y >= params.map_size.y) {
        return;
    }
    let res_x = i32(params.map_size.x);
    let res_y = i32(params.map_size.y);
    let px = i32(gid.x);
    let py = i32(gid.y);
    let idx = u32(py * res_x + px);
    let dx = 1.0 / max(params.detail_scale, 0.001);
    let iteration = params.iteration;
    let do_rivers = params.do_rivers != 0u;
    let use_momentum = do_rivers && params.momentum_coherence > 0.0;

    var uplift = params.uplift * params.erosion_amount * TRAIL_DENSITY_SCALE;
    let h_old_bits = atomicLoad(&height[idx]);
    let h_old = f32(h_old_bits) / HEIGHT_SCALE;

    if (do_rivers) {
        let river_scarring = params.river_scale * RIVER_SCARRING_SCALE;
        let macc_val = load_macc(vec2<i32>(px, py));
        var river_uplift_negation = macc_val * params.uplift_river_carving * river_scarring * dx;
        river_uplift_negation = river_uplift_negation / (1.0 + river_uplift_negation);
        uplift *= 1.0 - river_uplift_negation;
    }

    let h_new = h_old + uplift * dx;
    let h_new_bits = u32(h_new * HEIGHT_SCALE);
    atomicStore(&height[idx], h_new_bits);

    let dep = f32(atomicLoad(&deposition[idx])) / DEPOSITION_SCALE;
    if (dep > params.compaction_threshold * params.detail_scale) {
        let dep_sub = dep - params.compaction_threshold;
        let dep_compacted = dep_sub * pow(1.0 - params.sediment_compaction, 0.25 * dx)
            + params.compaction_threshold;
        atomicStore(&deposition[idx], i32(dep_compacted * DEPOSITION_SCALE));
    }

    if (do_rivers) {
        let river_scarring = params.river_scale * RIVER_SCARRING_SCALE;
        var momentum_extinction = pow(
            params.meander_longevity,
            pow(dx, 1.0 - params.river_character) * f32(iteration % 2u),
        );

        let neighbors = array<vec2<i32>, 8>(
            vec2<i32>(1, 0), vec2<i32>(0, 1), vec2<i32>(-1, 0), vec2<i32>(0, -1),
            vec2<i32>(-1, -1), vec2<i32>(1, 1), vec2<i32>(-1, 1), vec2<i32>(1, -1),
        );
        var havg = 0.0;
        var hmax_bits = h_new_bits;
        var hmin_bits = h_new_bits;
        for (var i = 0; i < 8; i = i + 1) {
            let nh = vec2<i32>(
                clamp(px + neighbors[i].x, 0, res_x - 1),
                clamp(py + neighbors[i].y, 0, res_y - 1),
            );
            let nidx = u32(nh.y * res_x + nh.x);
            let hs_bits = atomicLoad(&height[nidx]);
            let hs = f32(hs_bits) / HEIGHT_SCALE;
            havg += hs;
            hmax_bits = max(hmax_bits, hs_bits);
            hmin_bits = min(hmin_bits, hs_bits);
        }
        havg *= 0.125;
        let hmax = f32(hmax_bits) / HEIGHT_SCALE;
        let hmin = f32(hmin_bits) / HEIGHT_SCALE;

        let macc_val = load_macc(vec2<i32>(px, py));
        // Exact local-max test on fixed-point heights (ref: height[idx] == hmax)
        if (h_new_bits == hmax_bits) {
            momentum_extinction *= pow(
                0.001,
                min(pow(dx, 1.0 - params.river_character), 1.0)
                    * params.river_channeling
                    * (h_new - havg)
                    / max(0.001 * dx, hmax - hmin)
                    / max(0.025, macc_val * 0.025),
            );
        }

        store_macc(idx, macc_val * momentum_extinction);

        if (use_momentum) {
            store_mx(idx, load_mx(vec2<i32>(px, py)) * momentum_extinction);
            store_my(idx, load_my(vec2<i32>(px, py)) * momentum_extinction);
        }
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
    var erosion = f32(atomicLoad(&erosion_buffer[idx])) / 100000.0;
    if (params.do_rivers != 0u) {
        // When rivers are on, show accumulated flow visits (macc) in the erosion channel.
        erosion = load_macc(vec2<i32>(px, py)) * 0.02;
    }
    
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

