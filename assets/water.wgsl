// Water bodies: lakes and rivers from a shallow-water solver (virtual pipes, Mei et al. 2007).
//
// Runs on the base map against the finest pyramid level's heights. Water fills depressions and
// spills downhill, so pools and rivers follow the terrain; there is no water level to set.
// Coupling to the erosion:
//   * flowing water dissolves terrain up to its sediment capacity and still water deposits it
//     (Mei et al. capacity model), which carves channels and flattens pool beds;
//   * wet cells are KTT outflow (`masks.y`), so particle trails drain into them;
//   * the optional ocean (terrain below `sea_level`) and the map border drain water away.
// Every dispatch of the solver is one fixed time step (`WATER_STEP` in `lib.rs`); the rates in
// `WaterParams` are already per step.
//
// With `WaterSolver::Swe` the pipe kernels are replaced by bevy_fft's staggered shallow-water
// solver, which `lib.rs` steps between `swe_export` (bed and sources in) and `swe_gather` (depth
// and velocity out). The solver works in cells: heights are scaled by `CELLS_PER_HEIGHT`, so
// depths and the bed share the texel unit of the grid spacing.

const HEIGHT_SCALE: f32 = 1000000.0;
// dt * g * pipe area / pipe length per substep (stable below 0.25 on a 4-neighbour grid).
const PIPE_GRAVITY: f32 = 0.2;
// Depth below which a cell counts as dry.
const WET_DEPTH: f32 = 0.0005;
// Depth at which a cell is full outflow and flow reaches full sediment capacity.
const FULL_DEPTH: f32 = 0.004;
const DISSOLVE_RATE: f32 = 0.3;
const DEPOSIT_RATE: f32 = 0.3;
// Heights are ~1 unit over ~100 texels of rise; scale gradients to a tangent.
const SLOPE_SCALE: f32 = 100.0;
const MIN_TILT: f32 = 0.05;
// Height units to solver cells (the same vertical scale as `SLOPE_SCALE`).
const CELLS_PER_HEIGHT: f32 = 100.0;
// Most a substep may cut from one cell.
const MAX_CUT: f32 = 0.0001;
// Bank slumping in and next to water: height step per texel above which material slides down,
// and the fraction of the excess moved per substep.
const WET_TALUS: f32 = 0.003;
const SLUMP_RATE: f32 = 0.1;

// Must match `MAX_WATER_SOURCES` in `lib.rs`.
const MAX_SOURCES: u32 = 17u;

struct WaterParams {
    size: vec2<u32>,
    pad_offset: vec2<u32>,
    grid_size: vec2<u32>,
    clear: u32,
    // Fraction of the depth lost per step.
    evaporation: f32,
    capacity: f32,
    sea_level: f32,
    // Depth added everywhere per step.
    rain: f32,
    // Seconds per step.
    step_seconds: f32,
    source_count: u32,
    // Springs and the brush: xy = disc center (texels), z = radius, w = depth per step at the
    // center (negative removes water).
    sources: array<vec4<f32>, MAX_SOURCES>,
}

@group(0) @binding(0) var<storage, read_write> height: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> depth: array<f32>;
// Outflow through each face: x = left (-x), y = right (+x), z = top (-y), w = bottom (+y).
@group(0) @binding(2) var<storage, read_write> flux: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> sediment: array<f32>;
@group(0) @binding(4) var<storage, read_write> sediment_tmp: array<f32>;
@group(0) @binding(5) var<storage, read_write> velocity: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> masks: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> source_masks: array<vec4<f32>>;
@group(0) @binding(8) var<uniform> params: WaterParams;
// R = depth, GB = velocity (texels per step), A = suspended sediment (for rendering).
@group(1) @binding(0) var water_out: texture_storage_2d<rgba16float, write>;
// bevy_fft solver inputs: bed height in cells, and sources (x = depth per second in cells,
// y = sink weight).
@group(1) @binding(1) var swe_bed: texture_storage_2d<r32float, write>;
@group(1) @binding(2) var swe_source: texture_storage_2d<rgba32float, write>;
// bevy_fft solver state: x = bed, y = depth (cells), and the cell-centred velocity (cells per
// second).
@group(1) @binding(3) var swe_water: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(4) var swe_velocity: texture_2d<f32>;

fn inside(p: vec2<i32>) -> bool {
    return all(p >= vec2<i32>(0)) && all(p < vec2<i32>(params.size));
}

fn widx(p: vec2<i32>) -> u32 {
    return u32(p.y) * params.size.x + u32(p.x);
}

// Index into the (possibly padded) finest level.
fn gidx(p: vec2<i32>) -> u32 {
    let q = vec2<u32>(p) + params.pad_offset;
    return q.y * params.grid_size.x + q.x;
}

fn terrain(p: vec2<i32>) -> f32 {
    return f32(atomicLoad(&height[gidx(p)])) / HEIGHT_SCALE;
}

fn is_ocean(h: f32) -> bool {
    return params.sea_level > 0.0 && h < params.sea_level;
}

const DIRS = array<vec2<i32>, 4>(vec2<i32>(-1, 0), vec2<i32>(1, 0), vec2<i32>(0, -1), vec2<i32>(0, 1));

// Remove all water.
@compute @workgroup_size(8, 8, 1)
fn water_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p)) { return; }
    let i = widx(p);
    depth[i] = 0.0;
    flux[i] = vec4<f32>(0.0);
    sediment[i] = 0.0;
    sediment_tmp[i] = 0.0;
    velocity[i] = vec2<f32>(0.0);
}

// Water added (or removed) this step by rain, springs and the brush.
fn source_depth(p: vec2<i32>) -> f32 {
    var add = params.rain;
    for (var k = 0u; k < params.source_count; k = k + 1u) {
        let s = params.sources[k];
        let r = distance(vec2<f32>(p) + 0.5, s.xy) / s.z;
        if (r < 1.0) {
            add += s.w * (1.0 - r * r) * (1.0 - r * r);
        }
    }
    return add;
}

// Pipe fluxes from the water surface difference to each neighbour, scaled so a cell never sends
// more water than it holds. Off-map neighbours sit at this cell's ground: the border drains.
@compute @workgroup_size(8, 8, 1)
fn water_flux(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p)) { return; }
    let i = widx(p);
    let ground = terrain(p);
    let d = depth[i];
    let surface = ground + d;
    var f = flux[i];
    for (var k = 0; k < 4; k = k + 1) {
        let q = p + DIRS[k];
        var other = ground;
        if (inside(q)) {
            other = terrain(q) + depth[widx(q)];
        }
        f[k] = max(f[k] + PIPE_GRAVITY * (surface - other), 0.0);
    }
    let total = f.x + f.y + f.z + f.w;
    if (total > 0.0) {
        f *= min(1.0, d / total);
    }
    flux[i] = f;
}

// Move water along the fluxes, derive velocity, couple to KTT outflow, write the render texture.
@compute @workgroup_size(8, 8, 1)
fn water_depth(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p)) { return; }
    let i = widx(p);
    let f = flux[i];
    // Inflow from each neighbour's face pointing at this cell.
    var fin = vec4<f32>(0.0); // from left, right, top, bottom
    if (inside(p + DIRS[0])) { fin.x = flux[widx(p + DIRS[0])].y; }
    if (inside(p + DIRS[1])) { fin.y = flux[widx(p + DIRS[1])].x; }
    if (inside(p + DIRS[2])) { fin.z = flux[widx(p + DIRS[2])].w; }
    if (inside(p + DIRS[3])) { fin.w = flux[widx(p + DIRS[3])].z; }
    let d_old = depth[i];
    var d = d_old + (fin.x + fin.y + fin.z + fin.w) - (f.x + f.y + f.z + f.w);
    d = max(d + source_depth(p), 0.0) * (1.0 - params.evaporation);
    let ground = terrain(p);
    if (is_ocean(ground)) {
        d = 0.0;
    }
    depth[i] = d;

    let mean_depth = max(0.5 * (d + d_old), WET_DEPTH);
    var v = vec2<f32>(fin.x - f.x + f.y - fin.y, fin.z - f.z + f.w - fin.w) * 0.5 / mean_depth;
    let speed = length(v);
    if (speed > 1.0) {
        v /= speed;
    }
    velocity[i] = v;

    let wet = smoothstep(WET_DEPTH, FULL_DEPTH, d);
    let gi = gidx(p);
    var m = masks[gi];
    m.y = max(source_masks[i].y, wet);
    masks[gi] = m;

    textureStore(water_out, p, vec4<f32>(d, v, sediment[i]));
}

// Sediment capacity C = K * tilt * speed (Mei et al.): dissolve below capacity, deposit above.
@compute @workgroup_size(8, 8, 1)
fn water_erode(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p)) { return; }
    let i = widx(p);
    let d = depth[i];
    let s = sediment[i];
    var near_water = d >= WET_DEPTH;
    var nb: array<vec2<i32>, 4>;
    for (var k = 0; k < 4; k = k + 1) {
        nb[k] = clamp(p + DIRS[k], vec2<i32>(0), vec2<i32>(params.size) - 1);
        near_water = near_water || depth[widx(nb[k])] >= WET_DEPTH;
    }
    if (!near_water && s == 0.0) { return; }

    let ground = terrain(p);
    let hl = terrain(nb[0]);
    let hr = terrain(nb[1]);
    let ht = terrain(nb[2]);
    let hb = terrain(nb[3]);
    let slope = length(vec2<f32>(hr - hl, hb - ht)) * 0.5 * SLOPE_SCALE;
    let tilt = max(slope / sqrt(1.0 + slope * slope), MIN_TILT);
    let capacity = params.capacity * tilt * length(velocity[i]) * min(d / FULL_DEPTH, 1.0);

    // Positive dh deposits onto the terrain, negative dissolves it.
    var dh = DEPOSIT_RATE * (s - capacity);
    if (capacity > s) {
        dh = -min(DISSOLVE_RATE * (capacity - s), MAX_CUT);
    }
    if (d < WET_DEPTH) {
        dh = DEPOSIT_RATE * s; // drying out: settle what is left
    }
    let from_sediment = dh;

    // Slumping: steep banks in and next to water slide toward the lower side. Every cell applies
    // its own share of each pairwise exchange, so material is (nearly) conserved.
    if (near_water) {
        for (var k = 0; k < 4; k = k + 1) {
            let diff = terrain(nb[k]) - ground;
            dh += SLUMP_RATE * sign(diff) * max(abs(diff) - WET_TALUS, 0.0) * 0.25;
        }
    }
    let new_ground = max(ground + dh, 0.0);
    atomicStore(&height[gidx(p)], u32(new_ground * HEIGHT_SCALE));
    sediment[i] = max(s - from_sediment, 0.0);
}

fn sample_sediment(pos: vec2<f32>) -> f32 {
    let q = clamp(pos - 0.5, vec2<f32>(0.0), vec2<f32>(params.size) - 1.0);
    let i0 = vec2<i32>(floor(q));
    let i1 = min(i0 + 1, vec2<i32>(params.size) - 1);
    let w = q - floor(q);
    let a = mix(sediment[widx(i0)], sediment[widx(vec2<i32>(i1.x, i0.y))], w.x);
    let b = mix(sediment[widx(vec2<i32>(i0.x, i1.y))], sediment[widx(i1)], w.x);
    return mix(a, b, w.y);
}

// Semi-Lagrangian sediment transport along the velocity.
@compute @workgroup_size(8, 8, 1)
fn water_advect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p)) { return; }
    let i = widx(p);
    sediment_tmp[i] = sample_sediment(vec2<f32>(p) + 0.5 - velocity[i]);
}

@compute @workgroup_size(8, 8, 1)
fn water_copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p)) { return; }
    let i = widx(p);
    sediment[i] = sediment_tmp[i];
}

// Bed and sources for the bevy_fft solver. The map border and the ocean are full sinks.
@compute @workgroup_size(8, 8, 1)
fn swe_export(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p)) { return; }
    let ground = terrain(p);
    textureStore(swe_bed, p, vec4<f32>(ground * CELLS_PER_HEIGHT, 0.0, 0.0, 0.0));
    let edge = any(p == vec2<i32>(0)) || any(p == vec2<i32>(params.size) - 1);
    let sink = select(0.0, 1.0, edge || is_ocean(ground));
    let rate = source_depth(p) / params.step_seconds * CELLS_PER_HEIGHT;
    textureStore(swe_source, p, vec4<f32>(rate, sink, 0.0, 0.0));
}

// Depth and velocity back from the bevy_fft solver, then the same outputs as `water_depth`:
// KTT outflow and the render texture.
@compute @workgroup_size(8, 8, 1)
fn swe_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p)) { return; }
    let i = widx(p);
    let d = textureLoad(swe_water, p).y / CELLS_PER_HEIGHT;
    depth[i] = d;
    var v = textureLoad(swe_velocity, p, 0).xy * params.step_seconds;
    let speed = length(v);
    if (speed > 1.0) {
        v /= speed;
    }
    velocity[i] = v;

    let wet = smoothstep(WET_DEPTH, FULL_DEPTH, d);
    let gi = gidx(p);
    var m = masks[gi];
    m.y = max(source_masks[i].y, wet);
    masks[gi] = m;

    textureStore(water_out, p, vec4<f32>(d, v, sediment[i]));
}

// Hand this crate's depth to the bevy_fft solver (after bed shaping, or when switching solvers).
@compute @workgroup_size(8, 8, 1)
fn swe_import(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p)) { return; }
    var w = textureLoad(swe_water, p);
    w.y = depth[widx(p)] * CELLS_PER_HEIGHT;
    textureStore(swe_water, p, w);
}
