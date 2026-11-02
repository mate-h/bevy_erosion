// Morphodynamics of the water (Phases 4b, 4c and 6a of docs/WATER_PLAN.md): the shared
// discharge field, helical flow in bends, and bed load moved by Exner on the cell faces.
//
// Runs on the base map against the finest pyramid level's heights, after the water steps of
// `water.wgsl`. Units follow the solver: depths and the bed in cells (height × CELLS_PER_HEIGHT),
// velocities in cells per second, so the Shields number and the bed load are in grid units.
//   * morpho_discharge (per frame): time-averaged unit discharge q = h·u and depth, and KTT's
//     river momentum scaled to the same units.
//   * morpho_helical (per frame): helical flow intensity I_s, advected along the flow and relaxed
//     toward h·κ (κ = streamline curvature).
//   * morpho_flux (per step): bed load on the +x and +y faces of every cell (Shields, Meyer-Peter
//     Müller), turned toward the inner bank by the helical flow and down the cross-slope by
//     Talmon's gravity term.
//   * morpho_exner (per step): Δz = −mor_fac · Δt · div(q_b), accumulated in a float and written
//     to the fixed-point heights in whole quanta.

const HEIGHT_SCALE: f32 = 1000000.0;
const CELLS_PER_HEIGHT: f32 = 100.0;
// Must match `water.wgsl`.
const WET_DEPTH: f32 = 0.0005;
const FULL_DEPTH: f32 = 0.004;
// Most of the shallower side's depth one face may move per step (keeps a large morphological
// factor from emptying a cell in one step).
const MAX_FACE_CUT: f32 = 0.02;
// Largest helical flow deflection of the bed load (radians) and streamline curvature (1/texel).
const MAX_DEFLECTION: f32 = 0.5;
const MAX_CURVATURE: f32 = 0.3;
// Speed (cells per second) below which water has no direction for the bed load or curvature.
const MIN_SPEED: f32 = 0.05;

struct MorphoParams {
    size: vec2<u32>,
    pad_offset: vec2<u32>,
    grid_size: vec2<u32>,
    // Seconds per solver step, and solver time simulated this frame.
    step_seconds: f32,
    frame_seconds: f32,
    // Discharge averaging time (seconds).
    discharge_time: f32,
    // KTT river momentum to unit discharge (height units × texels per second).
    ktt_scale: f32,
    // KTT keeps a momentum vector (momentum coherence on) instead of a scalar in `mx`.
    ktt_vector: u32,
    sea_level: f32,
    manning_n: f32,
    // Submerged grain weight (R·D, cells) in θ = n²|u|² / (grain · h^(1/3)).
    grain: f32,
    bed_load_k: f32,
    critical_shields: f32,
    // Talmon's A_s: larger is a weaker cross-slope pull.
    slope_effect: f32,
    // A_h in δ = atan(A_h · I_s).
    helical_strength: f32,
    // Helical flow adaptation length (texels).
    helical_length: f32,
    // Morphological factor (0 = bed load off).
    mor_fac: f32,
    // Helical slot holding the latest intensity (the other one is written this frame).
    parity: u32,
}

@group(0) @binding(0) var<storage, read_write> height: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> depth: array<f32>;
// Texels per solver step.
@group(0) @binding(2) var<storage, read> velocity: array<vec2<f32>>;
// x, y = averaged unit discharge h·u (height units × texels per second), z = averaged depth,
// w = KTT river discharge in the same units.
@group(0) @binding(3) var<storage, read_write> discharge: array<vec4<f32>>;
// KTT river momentum of the finest work set (f32 bits; see `erosion.wgsl`).
@group(0) @binding(4) var<storage, read> mx: array<u32>;
@group(0) @binding(5) var<storage, read> my: array<u32>;
// Bed load through the +x (x) and +y (y) faces of each cell (cells² per second).
@group(0) @binding(6) var<storage, read_write> bed_load: array<vec2<f32>>;
// Four floats per cell: helical intensity in two slots (ping-pong), the bed change not yet
// written to the heights (height units), and a spare.
@group(0) @binding(7) var<storage, read_write> state: array<f32>;
@group(0) @binding(8) var<uniform> params: MorphoParams;

const DIRS = array<vec2<i32>, 4>(vec2<i32>(-1, 0), vec2<i32>(1, 0), vec2<i32>(0, -1), vec2<i32>(0, 1));

fn inside(p: vec2<i32>) -> bool {
    return all(p >= vec2<i32>(0)) && all(p < vec2<i32>(params.size));
}

fn widx(p: vec2<i32>) -> u32 {
    return u32(p.y) * params.size.x + u32(p.x);
}

fn gidx(p: vec2<i32>) -> u32 {
    let q = vec2<u32>(p) + params.pad_offset;
    return q.y * params.grid_size.x + q.x;
}

fn terrain(p: vec2<i32>) -> f32 {
    return f32(atomicLoad(&height[gidx(p)])) / HEIGHT_SCALE;
}

fn clamp_map(p: vec2<i32>) -> vec2<i32> {
    return clamp(p, vec2<i32>(0), vec2<i32>(params.size) - 1);
}

// Cells per second.
fn flow(p: vec2<i32>) -> vec2<f32> {
    return velocity[widx(p)] / params.step_seconds;
}

fn helical(p: vec2<i32>, slot: u32) -> f32 {
    return state[4u * widx(p) + slot];
}

// The map border ring and the ocean drain water; bed load that reaches them leaves the map.
fn is_sink(p: vec2<i32>) -> bool {
    let edge = any(p == vec2<i32>(0)) || any(p == vec2<i32>(params.size) - 1);
    return edge || (params.sea_level > 0.0 && terrain(p) < params.sea_level);
}

@compute @workgroup_size(8, 8, 1)
fn morpho_discharge(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p)) { return; }
    let i = widx(p);
    let h = depth[i];
    let q = h * flow(p);
    let a = 1.0 - exp(-params.frame_seconds / max(params.discharge_time, 1e-3));
    var d = discharge[i];
    d = vec4<f32>(mix(d.xy, q, a), mix(d.z, h, a), d.w);
    let g = gidx(p);
    var m = bitcast<f32>(mx[g]);
    if (params.ktt_vector != 0u) {
        m = length(vec2<f32>(m, bitcast<f32>(my[g])));
    }
    d.w = params.ktt_scale * max(m, 0.0);
    discharge[i] = d;
}

fn sample_helical(pos: vec2<f32>, slot: u32) -> f32 {
    let q = clamp(pos - 0.5, vec2<f32>(0.0), vec2<f32>(params.size) - 1.0);
    let i0 = vec2<i32>(floor(q));
    let i1 = min(i0 + 1, vec2<i32>(params.size) - 1);
    let w = q - floor(q);
    let a = mix(helical(i0, slot), helical(vec2<i32>(i1.x, i0.y), slot), w.x);
    let b = mix(helical(vec2<i32>(i0.x, i1.y), slot), helical(i1, slot), w.x);
    return mix(a, b, w.y);
}

// Streamline curvature κ = (u × (u·∇)u) / |u|³: positive where the flow turns left
// (counter-clockwise), toward the inner bank of the bend.
fn curvature(p: vec2<i32>, u: vec2<f32>) -> f32 {
    let speed = length(u);
    if (speed < MIN_SPEED) { return 0.0; }
    let dudx = 0.5 * (flow(clamp_map(p + DIRS[1])) - flow(clamp_map(p + DIRS[0])));
    let dudy = 0.5 * (flow(clamp_map(p + DIRS[3])) - flow(clamp_map(p + DIRS[2])));
    let a = u.x * dudx + u.y * dudy;
    let k = (u.x * a.y - u.y * a.x) / (speed * speed * speed);
    return clamp(k, -MAX_CURVATURE, MAX_CURVATURE);
}

// Helical flow intensity I_s (terracraft's `main.rs`): semi-Lagrangian advection along the
// flow, relaxation toward I_eq = h·κ over the adaptation length, and a little diffusion.
@compute @workgroup_size(8, 8, 1)
fn morpho_helical(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p)) { return; }
    let src = params.parity;
    let dst = 1u - params.parity;
    let i = widx(p);
    let h = depth[i];
    if (h < WET_DEPTH) {
        state[4u * i + dst] = 0.0;
        return;
    }
    let u = flow(p);
    let speed = length(u);
    let i_eq = h * CELLS_PER_HEIGHT * curvature(p, u);
    let advected = sample_helical(vec2<f32>(p) + 0.5 - u * params.frame_seconds, src);
    let relax = 1.0 - exp(-speed * params.frame_seconds / max(params.helical_length, 1.0));
    var s = mix(advected, i_eq, relax);
    var mean = 0.0;
    for (var k = 0; k < 4; k = k + 1) {
        mean += helical(clamp_map(p + DIRS[k]), src);
    }
    s = mix(s, 0.25 * mean, 0.05);
    state[4u * i + dst] = clamp(s, -1.0, 1.0);
}

// Bed load (cells² per second) through the face from `a` to its neighbour `a + e`, along `e`.
// Its size and direction come from the upwind cell (first-order upwind Exner): averaging the two
// cells instead is a central scheme, which grows two-texel oscillations in the bed.
fn face_load(a: vec2<i32>, e: vec2<i32>) -> f32 {
    let b = a + e;
    if (!inside(b)) { return 0.0; }
    let ha = depth[widx(a)];
    let hb = depth[widx(b)];
    if (min(ha, hb) < WET_DEPTH) { return 0.0; }
    let ua = flow(a);
    let ub = flow(b);
    let downwind = dot(ua + ub, vec2<f32>(e)) >= 0.0;
    let up = select(b, a, downwind);
    let h = select(hb, ha, downwind);
    let u = select(ub, ua, downwind);
    // Thin sheet flow keeps the capacity model of `water_erode`.
    let w = smoothstep(0.5 * FULL_DEPTH, FULL_DEPTH, h);
    if (w <= 0.0) { return 0.0; }
    let speed = length(u);
    if (speed < MIN_SPEED) { return 0.0; }
    let h_cells = h * CELLS_PER_HEIGHT;
    let n = params.manning_n;
    let theta = n * n * speed * speed / (max(params.grain, 1e-6) * pow(h_cells, 1.0 / 3.0));
    let excess = theta - params.critical_shields;
    if (excess <= 0.0) { return 0.0; }
    let q = params.bed_load_k * pow(excess, 1.5) * w;

    // Direction: along the flow, turned toward the inner bank by the helical flow.
    let delta = clamp(
        atan(params.helical_strength * helical(up, params.parity)),
        -MAX_DEFLECTION,
        MAX_DEFLECTION,
    );
    let c = cos(delta);
    let sn = sin(delta);
    let along = u / speed;
    let dir = vec2<f32>(c * along.x - sn * along.y, sn * along.x + c * along.y);

    // Talmon: bed load slides down the cross-slope only, so the long profile keeps its steps.
    let perp = vec2<i32>(abs(e.y), abs(e.x));
    let ga = terrain(clamp_map(a + perp)) - terrain(clamp_map(a - perp));
    let gb = terrain(clamp_map(b + perp)) - terrain(clamp_map(b - perp));
    let normal_grad = terrain(b) - terrain(a);
    let tangent_grad = 0.25 * (ga + gb);
    let grad = (vec2<f32>(e) * normal_grad + vec2<f32>(perp) * tangent_grad) * CELLS_PER_HEIGHT;
    let across = vec2<f32>(-dir.y, dir.x);
    let pull = q / (max(params.slope_effect, 1e-3) * sqrt(theta)) * dot(grad, across);
    let load = q * dir - pull * across;

    let qn = dot(load, vec2<f32>(e));
    let most = MAX_FACE_CUT * min(ha, hb) * CELLS_PER_HEIGHT
        / max(params.mor_fac * params.step_seconds, 1e-9);
    return clamp(qn, -most, most);
}

@compute @workgroup_size(8, 8, 1)
fn morpho_flux(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p)) { return; }
    bed_load[widx(p)] = vec2<f32>(face_load(p, DIRS[1]), face_load(p, DIRS[3]));
}

@compute @workgroup_size(8, 8, 1)
fn morpho_exner(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p)) { return; }
    let i = widx(p);
    let out = bed_load[i];
    var div = out.x + out.y;
    if (p.x > 0) { div -= bed_load[widx(p + DIRS[0])].x; }
    if (p.y > 0) { div -= bed_load[widx(p + DIRS[2])].y; }
    if (is_sink(p)) { return; }
    let dz = -params.mor_fac * params.step_seconds * div / CELLS_PER_HEIGHT;
    var pending = state[4u * i + 2u] + dz;
    let quanta = i32(round(pending * HEIGHT_SCALE));
    if (quanta != 0) {
        let g = gidx(p);
        let old = i32(atomicLoad(&height[g]));
        atomicStore(&height[g], u32(max(old + quanta, 0)));
        pending -= f32(quanta) / HEIGHT_SCALE;
    }
    state[4u * i + 2u] = pending;
}
