// Erosion compute for a heightmap.
// Algorithm: Smooth Fluvial Erosion
// Pass 1: init_fbm - initialize heightmap with FBM noise.
// Pass 2: erode - particle-based fluvial erosion.
// Pass 3: write_back - uplift, sediment compaction, river channeling.
// Pass 4: blit_to_texture - copy height/analysis to display.

const HEIGHT_SCALE: f32 = 1000000.0;
const DEPOSITION_SCALE: f32 = 100000.0;  // for i32 fixed-point
// Fixed point of `splat` (must match `water_splat.wgsl`).
const SPLAT_SCALE: f32 = 100000000.0;
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
    feature_size: f32,
    source_detail_smoothing: f32,
    geological_age: f32,
    voxel_size: f32,       // level voxel size in base voxels (kernel dx before / Detail_Scale)
    trail_weight: f32,     // Trail_Density multiplier for this pyramid level
    erosion_cutoff_angle: f32,
    clamp_river_channeling: u32,
    use_hydro_conditioning: u32,
    hydro_conditioning_search_distance: f32,
    use_softness_mask: u32,
    mask_frequency: f32,
    use_outflow: u32,
    outflow_integration: u32,
    outflow_sea_level: f32,
    outflow_border: f32,
    use_erosion_mask: u32,
    use_uplift_map: u32,
    use_bedrock: u32,
    use_bedrock_map: u32,
    bedrock_depth: f32,
    additive_sediment: u32,
    granularity_amount: f32,
    granularity_scale: f32,
    granularity_strength: f32,
    age_strength: f32,
    normalize_age: u32,
    bedrock_mode: u32,     // 0 = Depth (below the source surface), 1 = Height (absolute)
    // Compute_Padding: this level's grid covers the padded base; the source sits at `pad_offset`.
    pad_offset: vec2<u32>,
    padded_size: vec2<u32>,
    // Bevy-only rain brush: when spawn_count > 0, this dispatch starts exactly spawn_count trails
    // at random points of the disc (level texels) instead of KTT's map-wide trail density.
    spawn_center: vec2<f32>,
    spawn_radius: f32,
    spawn_count: u32,
    // Bevy-only: water each brush trail leaves per step on the finest level (`splat`).
    rain_water: f32,
    // Bevy-only: the water solver is running, so trails that reach water hand it their load.
    water_handoff: u32,
    // Bevy-only: shifts the source terrain noise.
    noise_seed: u32,
}

@group(0) @binding(1) var<uniform> params: ErodeParams;

// Accumulation buffers
@group(0) @binding(2) var<storage, read_write> flow_buffer: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> deposition: array<atomic<i32>>;  // loose sediment per cell (fixed-point)
@group(0) @binding(4) var<storage, read_write> erosion_buffer: array<atomic<u32>>;
// River momentum `__m` (enabled when do_rivers != 0), two floats per cell (x at 2i, y at 2i + 1),
// packed into one buffer to keep a binding free for `splat`. Without momentum coherence only x is
// used, as a scalar water accumulation (KTT compiles without HAS_my).
// Stored as f32 bits (KTT `__m` is a float field); adds go through a compare-exchange loop.
@group(0) @binding(5) var<storage, read_write> m: array<atomic<u32>>;
// Handed to the water on the finest level (`water_splat.wgsl`), two fixed-point values per cell
// (`SPLAT_SCALE`): water rain trails leave along their path (2i), and the sediment of trails that
// end in water (2i + 1).
@group(0) @binding(6) var<storage, read_write> splat: array<atomic<u32>>;
// Per-cell masks, packed to stay within 8 storage buffers per stage:
//   x = `__Softness_Mask` (scales Rock_Softness)
//   y = `__Outflow` (cells that drain away: no erosion, deposition or uplift)
//   z = `__Erode_Mask` (trails start only where > 0; scales Erosion_Strength)
//   w = `__Uplift` map (scales Uplift_Amount)
// Generated on the source set by init_fbm and resampled per pyramid level. Unused masks hold
// 1, 0, 1, 1, which matches KTT compiled without the corresponding HAS_* define.
@group(0) @binding(7) var<storage, read_write> masks: array<vec4<f32>>;
// x = `__Bedrock` (only read when use_bedrock != 0), y = `__Age` (time since the cell last changed).
@group(0) @binding(8) var<storage, read_write> layers: array<vec2<f32>>;

const MASK_SOFTNESS_SEED: vec2<f32> = vec2<f32>(91.7, 37.1);
const MASK_ERODE_SEED: vec2<f32> = vec2<f32>(13.3, 71.9);
const MASK_UPLIFT_SEED: vec2<f32> = vec2<f32>(57.1, 5.3);
const MASK_BEDROCK_SEED: vec2<f32> = vec2<f32>(29.9, 83.7);

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
    // Each seed samples a different region of the noise (0 keeps the original landscape).
    let s = f32(params.noise_seed);
    let seed_offset = select(
        vec2<f32>(0.0),
        vec2<f32>(hash21(vec2<f32>(s, 1.7)), hash21(vec2<f32>(s, 9.2))) * 1000.0,
        params.noise_seed != 0u,
    );
    for (var i = 0u; i < FBM_OCTAVES; i = i + 1u) {
        // Offset each octave so stacked layers don't align on the same grid.
        let offset = vec2<f32>(f32(i) * 17.3, f32(i) * 23.7);
        total += value_noise(p * freq + offset + seed_offset) * amp;
        amp *= FBM_PERSISTENCE;
        freq *= FBM_LACUNARITY;
    }
    return total;
}

// Bevy-only procedural mask (stands in for KTT's painted layers): patches of 0 and 1.
fn mask_noise(p: vec2<f32>, seed: vec2<f32>) -> f32 {
    var total = 0.0;
    var amp = 0.5;
    var q = p * params.mask_frequency + seed;
    for (var i = 0u; i < 4u; i = i + 1u) {
        total += value_noise(q) * amp;
        amp *= 0.5;
        q *= 2.0;
    }
    return smoothstep(0.35, 0.6, total / 0.9375);
}

// Bevy-only procedural outflow: a sea below `outflow_sea_level` plus optional draining map borders.
fn outflow_source(gid: vec2<u32>, h: f32) -> f32 {
    var o = 0.0;
    if (params.outflow_sea_level > 0.0) {
        o = 1.0 - smoothstep(params.outflow_sea_level - 0.02, params.outflow_sea_level, h);
    }
    if (params.outflow_border > 0.0) {
        let edge = min(min(gid.x, gid.y), min(params.map_size.x - 1u - gid.x, params.map_size.y - 1u - gid.y));
        o = max(o, 1.0 - smoothstep(0.0, params.outflow_border, f32(edge)));
    }
    return o;
}

// Terrain brush edits (`ErosionImages::terrain_edit`), only bound for init_fbm:
// R = height offset, G = flatten weight, B = flatten target, A = smooth weight.
@group(1) @binding(0) var terrain_edit: texture_2d<f32>;
// Smooth brush: radius (texels) of the fbm average.
const SMOOTH_RADIUS: f32 = 6.0;

fn edited_height(uv: vec2<f32>, gid: vec2<u32>) -> f32 {
    var h = fbm(uv) * params.noise_scale;
    let e = textureLoad(terrain_edit, vec2<i32>(gid), 0);
    if (e.a > 0.0) {
        var sum = h;
        for (var k = 0; k < 8; k = k + 1) {
            let a = f32(k) * 0.785398;
            sum += fbm(uv + vec2<f32>(cos(a), sin(a)) * SMOOTH_RADIUS) * params.noise_scale;
        }
        h = mix(h, sum / 9.0, clamp(e.a, 0.0, 1.0));
    }
    h += e.r;
    return mix(h, e.b, clamp(e.g, 0.0, 1.0));
}

@compute @workgroup_size(8,8,1)
fn init_fbm(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.map_size.x || gid.y >= params.map_size.y) { return; }
    let uv = vec2<f32>(gid.xy);
    var hval = edited_height(uv, gid.xy);
    let idx = gid.y * params.map_size.x + gid.x;

    // Adjust_Bedrock: bedrock = height - depth, or with a bedrock map height - map * depth (Depth
    // mode) / map * depth (Height mode, clamped to the surface). Additive_Sediment instead keeps the
    // bedrock at the source surface and adds the gap as loose sediment on top.
    var bedrock = 0.0;
    if (params.use_bedrock != 0u) {
        bedrock = hval - params.bedrock_depth;
        if (params.use_bedrock_map != 0u) {
            let depth = params.bedrock_depth * mask_noise(uv, MASK_BEDROCK_SEED);
            bedrock = select(hval - depth, min(depth, hval), params.bedrock_mode == 1u);
        }
        if (params.additive_sediment != 0u) {
            let sediment = hval - bedrock;
            bedrock = hval;
            hval += sediment;
        }
    }
    atomicStore(&height[idx], u32(max(hval, 0.0) * HEIGHT_SCALE));
    
    // Clear flow, deposition, erosion, and river/momentum buffers
    atomicStore(&flow_buffer[idx], 0u);
    atomicStore(&deposition[idx], 0i);
    atomicStore(&erosion_buffer[idx], 0u);
    atomicStore(&m[2u * idx], 0u);
    atomicStore(&m[2u * idx + 1u], 0u);

    var cell_mask = vec4<f32>(1.0, 0.0, 1.0, 1.0);
    if (params.use_softness_mask != 0u) {
        cell_mask.x = mask_noise(uv, MASK_SOFTNESS_SEED);
    }
    if (params.use_outflow != 0u) {
        cell_mask.y = outflow_source(gid.xy, hval);
    }
    if (params.use_erosion_mask != 0u) {
        cell_mask.z = mask_noise(uv, MASK_ERODE_SEED);
    }
    if (params.use_uplift_map != 0u) {
        cell_mask.w = mask_noise(uv, MASK_UPLIFT_SEED);
    }
    masks[idx] = cell_mask;
    layers[idx] = vec2<f32>(bedrock, 0.0);
}

@compute @workgroup_size(8,8,1)
fn init_color_bands(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.map_size.x || gid.y >= params.map_size.y) { return; }
    let idx = gid.y * params.map_size.x + gid.x;
    // Tint the masks so they read in the PBR view: darker hard rock, blue outflow.
    var base = vec3<f32>(0.5, 0.5, 0.5);
    if (params.use_softness_mask != 0u) {
        base = vec3<f32>(mix(0.38, 0.56, masks[idx].x));
    }
    base = mix(base, vec3<f32>(0.12, 0.26, 0.42), masks[idx].y);
    textureStore(color_image, vec2<i32>(gid.xy), vec4<f32>(base, 1.0));
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

fn load_mx(p: vec2<i32>) -> f32 {
    let size = i32(params.map_size.x);
    return bitcast<f32>(atomicLoad(&m[2u * u32(p.y * size + p.x)]));
}

fn load_my(p: vec2<i32>) -> f32 {
    let size = i32(params.map_size.x);
    return bitcast<f32>(atomicLoad(&m[2u * u32(p.y * size + p.x) + 1u]));
}

// Float atomic add via compare-exchange on the bit pattern.
fn add_mx(idx: u32, val: f32) {
    var old = atomicLoad(&m[2u * idx]);
    loop {
        let r = atomicCompareExchangeWeak(&m[2u * idx], old, bitcast<u32>(bitcast<f32>(old) + val));
        if (r.exchanged) {
            break;
        }
        old = r.old_value;
    }
}

fn add_my(idx: u32, val: f32) {
    var old = atomicLoad(&m[2u * idx + 1u]);
    loop {
        let r = atomicCompareExchangeWeak(&m[2u * idx + 1u], old, bitcast<u32>(bitcast<f32>(old) + val));
        if (r.exchanged) {
            break;
        }
        old = r.old_value;
    }
}

fn store_mx(idx: u32, val: f32) {
    atomicStore(&m[2u * idx], bitcast<u32>(val));
}

fn store_my(idx: u32, val: f32) {
    atomicStore(&m[2u * idx + 1u], bitcast<u32>(val));
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

fn bilinear_grid_weights(pos: vec2<f32>) -> vec4<f32> {
    let res = vec2<f32>(params.map_size.xy);
    var p = pos - 0.5;
    p = vec2<f32>(clamp(p.x, 0.0, res.x - 1.00001), clamp(p.y, 0.0, res.y - 1.00001));
    return vec4<f32>(p.x - floor(p.x), p.y - floor(p.y), floor(p.x), floor(p.y));
}

fn bilinear_sample_height(pos: vec2<f32>) -> f32 {
    let w = bilinear_grid_weights(pos);
    let size = i32(params.map_size.x);
    let ix = i32(w.z);
    let iy = i32(w.w);
    let h00 = load_height(vec2<i32>(ix, iy));
    let h10 = load_height(vec2<i32>(min(ix + 1, size - 1), iy));
    let h01 = load_height(vec2<i32>(ix, min(iy + 1, size - 1)));
    let h11 = load_height(vec2<i32>(min(ix + 1, size - 1), min(iy + 1, size - 1)));
    return h00 * (1.0 - w.x) * (1.0 - w.y) + h10 * w.x * (1.0 - w.y) + h01 * (1.0 - w.x) * w.y + h11 * w.x * w.y;
}

fn bilinear_sample_flow(pos: vec2<f32>) -> f32 {
    let w = bilinear_grid_weights(pos);
    let size_x = i32(params.map_size.x);
    let size_y = i32(params.map_size.y);
    let ix = i32(w.z);
    let iy = i32(w.w);
    let x1 = min(ix + 1, size_x - 1);
    let y1 = min(iy + 1, size_y - 1);
    let f00 = f32(atomicLoad(&flow_buffer[u32(iy * size_x + ix)])) / 10000.0;
    let f10 = f32(atomicLoad(&flow_buffer[u32(iy * size_x + x1)])) / 10000.0;
    let f01 = f32(atomicLoad(&flow_buffer[u32(y1 * size_x + ix)])) / 10000.0;
    let f11 = f32(atomicLoad(&flow_buffer[u32(y1 * size_x + x1)])) / 10000.0;
    return f00 * (1.0 - w.x) * (1.0 - w.y) + f10 * w.x * (1.0 - w.y) + f01 * (1.0 - w.x) * w.y + f11 * w.x * w.y;
}

fn bilinear_sample_deposition(pos: vec2<f32>) -> f32 {
    let w = bilinear_grid_weights(pos);
    let size_x = i32(params.map_size.x);
    let size_y = i32(params.map_size.y);
    let ix = i32(w.z);
    let iy = i32(w.w);
    let x1 = min(ix + 1, size_x - 1);
    let y1 = min(iy + 1, size_y - 1);
    let d00 = max(0.0, f32(atomicLoad(&deposition[u32(iy * size_x + ix)])) / DEPOSITION_SCALE);
    let d10 = max(0.0, f32(atomicLoad(&deposition[u32(iy * size_x + x1)])) / DEPOSITION_SCALE);
    let d01 = max(0.0, f32(atomicLoad(&deposition[u32(y1 * size_x + ix)])) / DEPOSITION_SCALE);
    let d11 = max(0.0, f32(atomicLoad(&deposition[u32(y1 * size_x + x1)])) / DEPOSITION_SCALE);
    return d00 * (1.0 - w.x) * (1.0 - w.y) + d10 * w.x * (1.0 - w.y) + d01 * (1.0 - w.x) * w.y + d11 * w.x * w.y;
}

fn bilinear_sample_erosion(pos: vec2<f32>) -> f32 {
    let w = bilinear_grid_weights(pos);
    let size_x = i32(params.map_size.x);
    let size_y = i32(params.map_size.y);
    let ix = i32(w.z);
    let iy = i32(w.w);
    let x1 = min(ix + 1, size_x - 1);
    let y1 = min(iy + 1, size_y - 1);
    let e00 = f32(atomicLoad(&erosion_buffer[u32(iy * size_x + ix)])) / 100000.0;
    let e10 = f32(atomicLoad(&erosion_buffer[u32(iy * size_x + x1)])) / 100000.0;
    let e01 = f32(atomicLoad(&erosion_buffer[u32(y1 * size_x + ix)])) / 100000.0;
    let e11 = f32(atomicLoad(&erosion_buffer[u32(y1 * size_x + x1)])) / 100000.0;
    return e00 * (1.0 - w.x) * (1.0 - w.y) + e10 * w.x * (1.0 - w.y) + e01 * (1.0 - w.x) * w.y + e11 * w.x * w.y;
}

fn river_momentum_at_idx(idx: u32) -> f32 {
    let x = bitcast<f32>(atomicLoad(&m[2u * idx]));
    if (params.momentum_coherence > 0.0) {
        let y = bitcast<f32>(atomicLoad(&m[2u * idx + 1u]));
        return length(vec2<f32>(x, y));
    }
    return x;
}

fn river_momentum(p: vec2<i32>) -> f32 {
    return river_momentum_at_idx(u32(p.y * i32(params.map_size.x) + p.x));
}

fn bilinear_sample_river_momentum(pos: vec2<f32>) -> f32 {
    let w = bilinear_grid_weights(pos);
    let size_x = i32(params.map_size.x);
    let size_y = i32(params.map_size.y);
    let ix = i32(w.z);
    let iy = i32(w.w);
    let x1 = min(ix + 1, size_x - 1);
    let y1 = min(iy + 1, size_y - 1);
    let m00 = river_momentum(vec2<i32>(ix, iy));
    let m10 = river_momentum(vec2<i32>(x1, iy));
    let m01 = river_momentum(vec2<i32>(ix, y1));
    let m11 = river_momentum(vec2<i32>(x1, y1));
    return m00 * (1.0 - w.x) * (1.0 - w.y) + m10 * w.x * (1.0 - w.y) + m01 * (1.0 - w.x) * w.y + m11 * w.x * w.y;
}

fn safe_normalize(v: vec2<f32>) -> vec2<f32> {
    let len = length(v);
    if (len > 0.0) {
        return v / len;
    }
    return vec2<f32>(0.0);
}

// hfGrad4: 4-neighbour central-difference gradient.
fn height_grad4(p: vec2<i32>, voxel_size: f32) -> vec2<f32> {
    let res = vec2<i32>(params.map_size);
    let xm = load_height(clamp(p - vec2<i32>(1, 0), vec2<i32>(0), res - 1));
    let xp = load_height(clamp(p + vec2<i32>(1, 0), vec2<i32>(0), res - 1));
    let ym = load_height(clamp(p - vec2<i32>(0, 1), vec2<i32>(0), res - 1));
    let yp = load_height(clamp(p + vec2<i32>(0, 1), vec2<i32>(0), res - 1));
    return vec2<f32>(xp - xm, yp - ym) * 0.5 / voxel_size;
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

// Smooth Fluvial Erosion (ported from ref/houdini22/compute_erosion.cl)
@compute @workgroup_size(256, 1, 1)
fn erode(@builtin(global_invocation_id) gid: vec3<u32>) {
    let particle_id = gid.x;
    let res_x = i32(params.map_size.x);
    let res_y = i32(params.map_size.y);
    let total_pixels = u32(res_x * res_y);
    let dx = params.voxel_size;
    let detail_scale = max(params.detail_scale, 0.001);
    let dx_scaled = dx / detail_scale;
    let iteration = params.iteration;

    let brush = params.spawn_count > 0u;
    // Age grows on every cell each dispatch; trails decay it where they change the height.
    if (!brush && particle_id < total_pixels) {
        layers[particle_id].y += dx_scaled;
    }

    // Detail_Bias: ridge-biased (0) vs flow-biased (1), alternating each OpenCL iteration
    let detail_bias = f32(iteration % 2u);
    if (params.compute_ridge_erosion == 0u && detail_bias < 0.5) {
        return;
    }

    var num_trails = f32(total_pixels)
        * min(pow(dx_scaled, detail_bias), 1.0)
        * params.erosion_amount
        * TRAIL_DENSITY_SCALE
        * params.trail_weight;
    if (params.compute_ridge_erosion != 0u && detail_bias < 0.5) {
        num_trails *= params.ridge_erosion_amount;
    }
    if (brush) {
        num_trails = f32(params.spawn_count);
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
    if (brush) {
        // Uniform point in the brush disc.
        let r = params.spawn_radius * sqrt(p.x);
        let a = p.y * 6.28318530718;
        p = clamp(
            params.spawn_center + vec2<f32>(cos(a), sin(a)) * r,
            vec2<f32>(0.0),
            vec2<f32>(f32(res_x - 1), f32(res_y - 1)),
        );
    } else {
        p *= vec2<f32>(f32(res_x), f32(res_y));
    }
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
    let start_idx = u32(sample_idx.y * res_x + sample_idx.x);
    // Erode_Mask: cull trails where the mask is 0 and scale erosion by it elsewhere.
    let erode_mask = masks[start_idx].z;
    if (erode_mask == 0.0) {
        return;
    }
    erosion_strength *= erode_mask;
    // Cull trails that start on exposed bedrock.
    if (params.use_bedrock != 0u && load_height(sample_idx) <= layers[start_idx].x) {
        return;
    }
    var flow_steps = select(
        i32(params.ridge_erosion_steps),
        i32(params.flow_length / dx_scaled),
        detail_bias > 0.5,
    );
    flow_steps = i32(f32(flow_steps) * (0.5 + random31(vec3<f32>(p.x, p.y, f32(iteration)))));

    let dx_inv = 1.0 / dx_scaled;
    let do_rivers = params.do_rivers != 0u;
    let use_my = do_rivers && params.momentum_coherence > 0.0;
    // Erosion_Cutoff_Angle: stop the trail on drops steeper than this (90° = never).
    let erosion_cutoff = tan(params.erosion_cutoff_angle * 3.14159 / 180.0) * dx_scaled;

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
    // On the finest level, trails meet the water bodies (`water_splat.wgsl`).
    let finest = all(params.map_size == params.padded_size);

    for (var step = 0; step < flow_steps; step = step + 1) {
        let cell = vec2<i32>(clamp(i32(p.x), 0, res_x - 1), clamp(i32(p.y), 0, res_y - 1));
        let idx = u32(cell.y * res_x + cell.x);

        // Issue every per-cell load up front so they overlap instead of stalling one by one
        // through the step (the trail is a long serial chain; this is its latency).
        let h_bits = atomicLoad(&height[idx]);
        let h = f32(h_bits) / HEIGHT_SCALE;
        let cell_masks = masks[idx];
        let cell_layers = layers[idx];
        if (finest) {
            // Rain trails leave water along their path, so the brush wets what it carves.
            if (brush && params.rain_water > 0.0) {
                atomicAdd(&splat[2u * idx], u32(params.rain_water * SPLAT_SCALE));
            }
            // A trail that reaches water (or the ocean) hands its load to it and ends: the water
            // carries it on and drops it where it slows, instead of a pile at the shore.
            if (params.water_handoff != 0u && cell_masks.y > 0.5) {
                if (carry > 0.0) {
                    atomicAdd(&splat[2u * idx + 1u], u32(carry * SPLAT_SCALE));
                }
                return;
            }
        }
        let dep_val = load_deposition(cell);
        var hmin = h;
        var hmax = h;
        var havg = 0.0;
        var grad = vec2<f32>(0.0, 0.0);
        var mgrad = vec2<f32>(0.0, 0.0);

        for (var j = -1; j <= 1; j = j + 2) {
            let pxj = vec2<i32>(clamp(cell.x + j, 0, res_x - 1), cell.y);
            let hs = load_height(pxj);
            var gs = hs;
            if (params.flow_volume != 0.0) {
                gs += params.flow_volume * load_deposition(pxj);
            }
            grad.x += f32(j) * gs;
            if (do_rivers) {
                mgrad.x += f32(j) * river_momentum(pxj);
                havg += hs;
            }
            hmin = min(hmin, hs);
            hmax = max(hmax, hs);
        }
        for (var j = -1; j <= 1; j = j + 2) {
            let pyj = vec2<i32>(cell.x, clamp(cell.y + j, 0, res_y - 1));
            let hs = load_height(pyj);
            var gs = hs;
            if (params.flow_volume != 0.0) {
                gs += params.flow_volume * load_deposition(pyj);
            }
            grad.y += f32(j) * gs;
            if (do_rivers) {
                mgrad.y += f32(j) * river_momentum(pyj);
                havg += hs;
            }
            hmin = min(hmin, hs);
            hmax = max(hmax, hs);
        }

        var scarring_coefficient = 1.0;

        if (do_rivers) {
            let river_mom = river_momentum_at_idx(idx);
            havg *= 0.25;

            // Prevents rivers from accumulating on local maxima
            var protrusion_coefficient = 1.0;
            if (h == hmax) {
                protrusion_coefficient = pow(
                    0.001,
                    0.5 * pow(dx_scaled, 0.05) * max(h - havg, 0.0)
                        / max(0.0001 * dx_scaled, hmax - hmin),
                );
            }
            scarring_coefficient =
                1.0 / (1.0 + protrusion_coefficient * river_scarring * river_mom);
            let friction_coefficient = 1.0
                / (1.0 + protrusion_coefficient * river_friction_reduction * min(river_mom, 250.0));

            let vlen = length(v);
            // River coherence (HAS_my): deposit flow direction, steer along the momentum field.
            if (use_my) {
                let mtmp = vec2<f32>(load_mx(cell), load_my(cell));
                let blend = momentum_coherence * min(river_mom, 250.0);
                let vnorm = safe_normalize(v);
                add_mx(idx, vnorm.x * particle_water);
                add_my(idx, vnorm.y * particle_water);
                if (length(mtmp) >= particle_water) {
                    v += (safe_normalize(mtmp) * vlen - v) * blend;
                }
            } else {
                add_mx(idx, particle_water);
            }

            if (dep_val == 0.0) {
                v *= pow(1.0 - params.rock_friction * friction_coefficient, dx_scaled);
            } else {
                v *= pow(1.0 - params.friction * friction_coefficient, dx_scaled);
            }
            v += -river_volume * mgrad;
            v -= normalize(vec3<f32>(grad.x, grad.y, dx_scaled)).xy * dx_scaled;
        } else {
            // When Do_Rivers is off the OpenCL #else branch sets v = -grad at friction=1, which
            // zig-zags on our grid and spikes; keep normal-force transport for stability.
            if (params.rock_friction != 1.0 || params.friction != 1.0) {
                if (dep_val == 0.0) {
                    v *= pow(1.0 - params.rock_friction, dx_scaled);
                } else {
                    v *= pow(1.0 - params.friction, dx_scaled);
                }
            }
            v -= normalize(vec3<f32>(grad.x, grad.y, dx_scaled)).xy * dx_scaled;
        }

        let vnorm = safe_normalize(v);
        if (params.velocity_randomness != 0.0) {
            let d_rand = random31(vec3<f32>(p.x, p.y, f32(step))) * 6.28318;
            let dp = vnorm + vec2<f32>(cos(d_rand), sin(d_rand))
                * params.velocity_randomness
                * pow(1.0 - params.velocity_randomness_refinement, f32(iteration));
            p += safe_normalize(dp);
        } else {
            p += vnorm;
        }

        // Written as an in-bounds test so a NaN position (runaway velocity) also ends the trail.
        if (!(p.x >= 0.0 && p.y >= 0.0 && p.x <= f32(res_x - 1) && p.y <= f32(res_y - 1))) {
            return;
        }

        let h_prev = h_current;
        h_current = bilinear_sample_height(p);

        var dh = erosion_strength * min(
            max(h_current - h_prev, -max_deposit_angle * dx_scaled) + talus_angle * dx_scaled,
            0.0,
        );

        // Stop erosion when encountering a slope greater than the Erosion Cutoff Angle
        if (h_prev - h_current > erosion_cutoff) {
            return;
        }

        dh = max(dh, -dep_val) + min(dh + dep_val, 0.0) * params.rock_softness * cell_masks.x;

        // Release the carried material (scarring is 1 without rivers)
        dh += min((1.0 - params.suspended_load) * carry * scarring_coefficient, max_deposit_angle * dx_scaled);

        // Never cut into the bedrock
        let bedrock = cell_layers.x;
        if (params.use_bedrock != 0u) {
            dh = max(dh, bedrock - h);
        }
        carry -= dh;
        carry = max(carry, 0.0);

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

        // Clamp to the neighbourhood, scale by outflow, keep above bedrock, and write, all against
        // the cell's current value (see `update_cell_height`).
        dh = update_cell_height(idx, h_bits, dh, hmin, hmax, 1.0 - cell_masks.y, bedrock);

        let dep_delta = max(dh, -dep_val);
        atomicAdd(&deposition[idx], i32(dep_delta * DEPOSITION_SCALE));
        layers[idx].y = cell_layers.y * pow(0.1, abs(dh) * dx_inv);
        if (dh < 0.0) {
            atomicAdd(&erosion_buffer[idx], u32(-dh * 100000.0));
        }
        let flow_val = max(0.0, dh) * dx_inv;
        atomicAdd(&flow_buffer[idx], u32(flow_val * 10000.0));
    }

    if (params.use_hydro_conditioning != 0u && iteration % 5u == 0u) {
        hydro_condition(p, dx_scaled);
    }
}

// KTT's `dh = clamp(dh, hmin - height, hmax - height); dh *= 1 - Outflow;
// height = max(Bedrock, height + dh)` as one compare-exchange, returning the applied change.
// KTT does a plain read-modify-write, so concurrent trails on a cell lose updates but the height
// stays inside [hmin, hmax]. Separate atomicAdd/atomicSub calls would instead stack every
// concurrent cut sized against the same reading; where many trails converge (strong momentum
// coherence) that drove river cells below zero and wrapped the u32 height.
// `expected_bits` is the caller's earlier read: the exchange only retries if another trail wrote
// the cell since.
fn update_cell_height(idx: u32, expected_bits: u32, dh_in: f32, hmin: f32, hmax: f32, scale: f32, bedrock: f32) -> f32 {
    var old_bits = expected_bits;
    loop {
        let old_h = f32(old_bits) / HEIGHT_SCALE;
        var dh = clamp(old_h + dh_in, hmin, max(hmin, hmax)) - old_h;
        dh *= scale;
        var new_h = old_h + dh;
        if (params.use_bedrock != 0u) {
            new_h = max(new_h, bedrock);
        }
        let new_bits = u32(max(new_h, 0.0) * HEIGHT_SCALE);
        let res = atomicCompareExchangeWeak(&height[idx], old_bits, new_bits);
        if (res.exchanged) {
            return (f32(new_bits) - f32(old_bits)) / HEIGHT_SCALE;
        }
        old_bits = res.old_value;
    }
    return 0.0;
}

// "Respect Outflows": a trail that ends in a pit searches a spiral for an equal-or-lower cell and
// carves a path to it, so closed depressions drain (Use_Hydro_Conditioning).
fn hydro_condition(p_end: vec2<f32>, dx: f32) {
    let res = vec2<i32>(params.map_size);
    var p = p_end;
    let gid = clamp(vec2<i32>(p), vec2<i32>(0), res - 1);
    let idx = u32(gid.y * res.x + gid.x);
    let h = load_height(gid);

    for (var i = -1; i <= 1; i = i + 1) {
        for (var j = -1; j <= 1; j = j + 1) {
            let s = clamp(vec2<i32>(p) + vec2<i32>(i, j), vec2<i32>(0), res - 1);
            if (load_height(s) < h) {
                return;
            }
        }
    }

    var outflow_height = h;
    var found = false;
    var outflow_gid = vec2<i32>(p);
    for (var i = 1.0; i < params.hydro_conditioning_search_distance; i = i * 1.01 + 0.2) {
        let sample_gid = vec2<i32>(p + i * vec2<f32>(cos(i), sin(i)));
        var hs = h;
        if (all(sample_gid >= vec2<i32>(0)) && all(sample_gid <= res - 1)) {
            hs = load_height(sample_gid);
        }
        if (hs <= h) {
            outflow_height = hs;
            found = true;
            outflow_gid = sample_gid;
            break;
        }
    }
    if (!found) {
        return;
    }

    let outflow_bits = u32(max(outflow_height, 0.0) * HEIGHT_SCALE);
    var sample_gid = vec2<i32>(p);
    let path_len = length(vec2<f32>(outflow_gid - gid)) * 2.0;
    for (var i = 0; f32(i) <= path_len; i = i + 1) {
        let dp = safe_normalize(vec2<f32>(outflow_gid - sample_gid))
            - safe_normalize(height_grad4(sample_gid, dx)) * 0.75;
        p += safe_normalize(dp);
        sample_gid = clamp(vec2<i32>(p), vec2<i32>(0), res - 1);
        let sample_idx = u32(sample_gid.y * res.x + sample_gid.x);
        if (sample_idx == idx) {
            continue;
        }
        if (load_height(sample_gid) < outflow_height) {
            break;
        }
        atomicMin(&height[sample_idx], outflow_bits);
    }
}

// writeBack: uplift, sediment compaction, river channeling and momentum decay (ref/compute_erosion.cl)
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
    let dx = params.voxel_size / max(params.detail_scale, 0.001);
    let iteration = params.iteration;
    let do_rivers = params.do_rivers != 0u;
    let use_my = do_rivers && params.momentum_coherence > 0.0;

    var uplift = params.uplift * params.erosion_amount * TRAIL_DENSITY_SCALE * params.trail_weight
        * masks[idx].w;
    let h_old = load_height(vec2<i32>(px, py));

    var river_scarring = 0.0;
    var river_mom = 0.0;
    if (do_rivers) {
        river_scarring = params.river_scale * RIVER_SCARRING_SCALE
            * min(1.0, pow(dx, params.river_character));
        river_mom = river_momentum_at_idx(idx) * (1.0 - masks[idx].y);
        var river_uplift_negation = river_mom * params.uplift_river_carving * river_scarring * dx;
        river_uplift_negation = river_uplift_negation / (1.0 + river_uplift_negation);
        uplift *= 1.0 - river_uplift_negation;
    }

    var h_new = h_old + uplift * dx * (1.0 - masks[idx].y);

    let dep = f32(atomicLoad(&deposition[idx])) / DEPOSITION_SCALE;
    if (dep > params.compaction_threshold * params.detail_scale) {
        let dep_sub = dep - params.compaction_threshold;
        let dep_compacted = dep_sub * pow(1.0 - params.sediment_compaction, 0.25 * dx)
            + params.compaction_threshold;
        atomicStore(&deposition[idx], i32(dep_compacted * DEPOSITION_SCALE));
    }

    if (do_rivers) {
        let momentum_extinction = pow(
            params.meander_longevity,
            pow(dx, 1.0 - params.river_character) * f32(iteration % 2u),
        );

        // River channeling, optionally clamped to the neighbourhood minimum so rivers don't dig pits
        var hmin = h_new;
        if (params.clamp_river_channeling != 0u) {
            for (var i = -1; i <= 1; i = i + 1) {
                for (var j = -1; j <= 1; j = j + 1) {
                    let s = vec2<i32>(clamp(px + i, 0, res_x - 1), clamp(py + j, 0, res_y - 1));
                    hmin = min(hmin, load_height(s));
                }
            }
        }
        h_new -= river_mom * dx * river_scarring * params.river_channeling * params.river_channeling * 0.0025;
        if (params.clamp_river_channeling != 0u) {
            h_new = max(h_new, hmin);
        }

        store_mx(idx, load_mx(vec2<i32>(px, py)) * momentum_extinction);
        if (use_my) {
            store_my(idx, load_my(vec2<i32>(px, py)) * momentum_extinction);
        }
    }

    // Final validation that the erosion doesn't eat into the bedrock
    if (params.use_bedrock != 0u) {
        h_new = max(h_new, layers[idx].x);
    }
    atomicStore(&height[idx], u32(max(h_new, 0.0) * HEIGHT_SCALE));
}

@group(1) @binding(0) var display_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var normal_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var analysis_out: texture_storage_2d<rgba16float, write>;  // R=flow_mag, G=sediment, B=erosion, A=curvature
@group(1) @binding(3) var ao_out: texture_storage_2d<r16float, read_write>;

@compute @workgroup_size(8,8,1)
fn blit_to_texture(@builtin(global_invocation_id) gid: vec3<u32>) {
    // The finest level may be padded (Compute_Padding): read at the offset, write the base map.
    let out_px = vec2<i32>(gid.xy);
    if (any(gid.xy >= textureDimensions(display_out))) { return; }
    let size_x = i32(params.map_size.x);
    let size_y = i32(params.map_size.y);
    let px = i32(gid.x + params.pad_offset.x);
    let py = i32(gid.y + params.pad_offset.y);
    let idx = u32(py * size_x + px);
    let h = f32(atomicLoad(&height[idx])) / HEIGHT_SCALE;

    // Height in R (and B); the Age view in G.
    let c = h;
    textureStore(display_out, out_px, vec4<f32>(c, age_display(layers[idx].y), c, 1.0));

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
    textureStore(normal_out, out_px, vec4<f32>(enc, 1.0));
    
    // Pack flow, deposition (sediment), and erosion into single analysis texture
    let flow_mag = f32(atomicLoad(&flow_buffer[idx])) / 10000.0;
    let sediment = max(0.0, f32(atomicLoad(&deposition[idx])) / DEPOSITION_SCALE);
    var erosion = f32(atomicLoad(&erosion_buffer[idx])) / 100000.0;
    if (params.do_rivers != 0u) {
        // When rivers are on, show river momentum `|__m|` in the erosion channel.
        erosion = river_momentum(vec2<i32>(px, py)) * 0.02;
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
    textureStore(analysis_out, out_px, analysis);
}

// KTT age visualization: Age * 10^Age_Strength, optionally normalized to v / (1 + v).
fn age_display(age: f32) -> f32 {
    var v = age * pow(10.0, params.age_strength);
    if (params.normalize_age != 0u) {
        v = v / (1.0 + v);
    }
    return v;
}

fn bilinear_sample_age(pos: vec2<f32>) -> f32 {
    let w = bilinear_grid_weights(pos);
    let size = vec2<i32>(params.map_size);
    let ix = i32(w.z);
    let iy = i32(w.w);
    let ix1 = min(ix + 1, size.x - 1);
    let iy1 = min(iy + 1, size.y - 1);
    let a00 = layers[u32(iy * size.x + ix)].y;
    let a10 = layers[u32(iy * size.x + ix1)].y;
    let a01 = layers[u32(iy1 * size.x + ix)].y;
    let a11 = layers[u32(iy1 * size.x + ix1)].y;
    return a00 * (1.0 - w.x) * (1.0 - w.y) + a10 * w.x * (1.0 - w.y) + a01 * (1.0 - w.x) * w.y + a11 * w.x * w.y;
}

struct BlitDisplayParams {
    display_size: vec2<u32>,
    _pad: vec2<u32>,
}

@group(2) @binding(0) var<uniform> blit_display: BlitDisplayParams;

// Map a full-res display pixel center into coarse-grid coordinates (same space as erosion `pos`).
fn display_to_coarse_pos(gid: vec2<u32>) -> vec2<f32> {
    return (vec2<f32>(gid + params.pad_offset) + 0.5)
        * vec2<f32>(params.map_size.xy)
        / vec2<f32>(params.padded_size.xy);
}

@compute @workgroup_size(8, 8, 1)
fn blit_level_to_display(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= blit_display.display_size.x || gid.y >= blit_display.display_size.y) {
        return;
    }
    let px = i32(gid.x);
    let py = i32(gid.y);
    let pos = display_to_coarse_pos(gid.xy);
    let dpos = vec2<f32>(params.map_size.xy) / vec2<f32>(params.padded_size.xy);

    let h = bilinear_sample_height(pos);
    let age = age_display(bilinear_sample_age(pos));
    textureStore(display_out, vec2<i32>(px, py), vec4<f32>(h, age, h, 1.0));

    let h_l = bilinear_sample_height(pos - vec2(dpos.x, 0.0));
    let h_r = bilinear_sample_height(pos + vec2(dpos.x, 0.0));
    let h_d = bilinear_sample_height(pos - vec2(0.0, dpos.y));
    let h_u = bilinear_sample_height(pos + vec2(0.0, dpos.y));
    let dx = h_r - h_l;
    let dy = h_u - h_d;
    let n = normalize(vec3<f32>(-dx, 1.0, -dy));
    let enc = n * 0.5 + vec3<f32>(0.5, 0.5, 0.5);
    textureStore(normal_out, vec2<i32>(px, py), vec4<f32>(enc, 1.0));

    let flow_mag = bilinear_sample_flow(pos);
    let sediment = bilinear_sample_deposition(pos);
    var erosion = bilinear_sample_erosion(pos);
    if (params.do_rivers != 0u) {
        erosion = bilinear_sample_river_momentum(pos) * 0.02;
    }
    let curvature_raw = h_l + h_r + h_d + h_u - 4.0 * h;
    let curvature = curvature_raw * 50.0 + 0.5;
    textureStore(
        analysis_out,
        vec2<i32>(px, py),
        vec4<f32>(flow_mag, sediment, erosion, curvature),
    );
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
    let dims = textureDimensions(ao_out);
    if (any(gid.xy >= dims)) { return; }
    
    let size_x = i32(params.map_size.x);
    let size_y = i32(params.map_size.y);
    let px = i32(gid.x + params.pad_offset.x);
    let py = i32(gid.y + params.pad_offset.y);
    let idx = u32(py * size_x + px);
    
    let current_height = f32(atomicLoad(&height[idx])) / HEIGHT_SCALE;
    
    // AO parameters (matching the default values from Rust)
    let sample_count = 64u; // Increased for smoother result
    let sample_radius_px = 0.08 * f32(dims.x); // Convert UV radius to pixels
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
        let world_distance = (px_distance / f32(dims.x)) * terrain_size;
        
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
    textureStore(ao_out, vec2<i32>(gid.xy), vec4<f32>(ao, 0.0, 0.0, 0.0));
}

// Gaussian blur pass for smoothing AO
// Temporary storage texture for separable blur
@group(1) @binding(4) var ao_temp: texture_storage_2d<r16float, read_write>;

@compute @workgroup_size(8,8,1)
fn blur_ao_horizontal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(ao_out);
    if (any(gid.xy >= dims)) { return; }
    
    let px = i32(gid.x);
    let py = i32(gid.y);
    let size_x = i32(dims.x);
    
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
    let dims = textureDimensions(ao_out);
    if (any(gid.xy >= dims)) { return; }
    
    let px = i32(gid.x);
    let py = i32(gid.y);
    let size_y = i32(dims.y);
    
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

