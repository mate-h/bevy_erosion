// Bed reconstruction for lakes and rivers (Phase 2 of docs/WATER_PLAN.md), ported from
// terracraft's riverbed solver (`solve_tau_sor`, `solve_lake_bathy_sor`, stage 5 bank grading).
//
// Runs on demand on the wet domain the shallow-water solver produced; there is no painted mask
// and no water level. The water surface η = terrain + depth stays where it is: the bed is
// rebuilt as η − d with a smooth depth d, and the water depth follows.
//   1. bed_init:   wet domain, shore seeds for the bank distance.
//   2. bed_tau:    bank distance τ, ∇²τ = −2 inside, τ = 0 on the banks (red-black SOR).
//   3. bed_prior:  depth prior from τ: half-width b = sqrt(τ + |∇τ|²/4), cross-channel
//                  coordinate n = |∇τ| / 2b; rivers get a (1 − n²)^profile cross-section,
//                  still water a bowl that deepens with the distance to the shore.
//   4. bed_smooth: screened Poisson ∇²d = λ(d − prior), so there are no rings or corners.
//   5. bed_write:  bed = η − d (blended by `strength`), only ever deepening, so shaping never
//                  makes water shallower; the depth keeps the surface in place.
//   6. bed_dilate: carry the nearest shore's η and distance out onto the dry banks.
//   7. bed_grade:  cap the banks with a quintic S5 ramp z_max = η + rise + slope·distance.

const HEIGHT_SCALE: f32 = 1000000.0;
// Depth above which a cell belongs to the wet domain (above `water.wgsl`'s WET_DEPTH, so thin
// sheet flow on hillslopes is left alone).
const DOMAIN_DEPTH: f32 = 0.002;
// Solver speed (texels per step) below which water counts as still (a lake) / above which it is
// a river.
const STILL_SPEED: f32 = 0.05;
const RIVER_SPEED: f32 = 0.25;
// Screened Poisson prior weight (terracraft's soft guide weight).
const PRIOR_WEIGHT: f32 = 0.045;
const FAR: f32 = 1.0e6;

struct BedParams {
    size: vec2<u32>,
    pad_offset: vec2<u32>,
    grid_size: vec2<u32>,
    // Red-black colour (SOR) or ping-pong direction (dilation).
    parity: u32,
    omega: f32,
    max_depth: f32,
    shore_slope: f32,
    profile: f32,
    strength: f32,
    bank_width: f32,
    bank_rise: f32,
    bank_slope: f32,
}

@group(0) @binding(0) var<storage, read_write> height: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> depth: array<f32>;
@group(0) @binding(2) var<storage, read> velocity: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> tau: array<f32>;
@group(0) @binding(4) var<storage, read_write> bed_depth: array<f32>;
@group(0) @binding(5) var<storage, read_write> prior: array<f32>;
// Nearest shore: x = water surface η there, y = distance in texels (0 in the wet domain).
@group(0) @binding(6) var<storage, read_write> shore_a: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read_write> shore_b: array<vec2<f32>>;
@group(0) @binding(8) var<uniform> params: BedParams;

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

fn wet(p: vec2<i32>) -> bool {
    return depth[widx(p)] > DOMAIN_DEPTH;
}

fn red_black(p: vec2<i32>) -> bool {
    return (u32(p.x + p.y) & 1u) == params.parity;
}

@compute @workgroup_size(8, 8, 1)
fn bed_init(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p)) { return; }
    let i = widx(p);
    tau[i] = 0.0;
    bed_depth[i] = 0.0;
    prior[i] = 0.0;
    if (wet(p)) {
        shore_a[i] = vec2<f32>(terrain(p) + depth[i], 0.0);
    } else {
        shore_a[i] = vec2<f32>(0.0, FAR);
    }
}

// Poisson torsion: τ = 0 on dry neighbours (the banks), mirrored at the map border.
@compute @workgroup_size(8, 8, 1)
fn bed_tau(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p) || !red_black(p) || !wet(p)) { return; }
    let i = widx(p);
    let c = tau[i];
    var sum = 0.0;
    for (var k = 0; k < 4; k = k + 1) {
        let q = p + DIRS[k];
        if (!inside(q)) {
            sum += c;
        } else if (wet(q)) {
            sum += tau[widx(q)];
        }
    }
    tau[i] = max(c + params.omega * ((sum + 2.0) * 0.25 - c), 0.0);
}

fn tau_at(q: vec2<i32>, c: f32) -> f32 {
    if (!inside(q)) { return c; }
    if (!wet(q)) { return 0.0; }
    return tau[widx(q)];
}

// Depth that starts out at `d` and levels off toward `max_depth`, so beds stay rounded bowls
// instead of flat pans with a steep rim.
fn soft_cap(d: f32) -> f32 {
    let m = max(params.max_depth, 1e-6);
    return m * (1.0 - exp(-d / m));
}

@compute @workgroup_size(8, 8, 1)
fn bed_prior(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p) || !wet(p)) { return; }
    let i = widx(p);
    let c = tau[i];
    let grad = 0.5 * vec2<f32>(
        tau_at(p + DIRS[1], c) - tau_at(p + DIRS[0], c),
        tau_at(p + DIRS[3], c) - tau_at(p + DIRS[2], c),
    );
    let g2 = dot(grad, grad);
    // Local half-width and normalized cross-channel coordinate (0 on the thalweg, 1 on a bank).
    let b = max(sqrt(c + 0.25 * g2), 0.5);
    let n = clamp(0.5 * sqrt(g2) / b, 0.0, 1.0);
    let river = soft_cap(params.shore_slope * b) * pow(max(1.0 - n * n, 0.0), params.profile);
    // sqrt(2τ) grows like the distance to the shore near the banks and is the radius at the
    // centre of a round lake.
    let lake = soft_cap(params.shore_slope * sqrt(2.0 * c));
    let still = 1.0 - smoothstep(STILL_SPEED, RIVER_SPEED, length(velocity[i]));
    let d = mix(river, lake, still);
    prior[i] = d;
    bed_depth[i] = d;
}

// Screened Poisson smoothing of the depth: d = 0 on the banks, softly pulled to the prior.
@compute @workgroup_size(8, 8, 1)
fn bed_smooth(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p) || !red_black(p) || !wet(p)) { return; }
    let i = widx(p);
    let c = bed_depth[i];
    var sum = 0.0;
    for (var k = 0; k < 4; k = k + 1) {
        let q = p + DIRS[k];
        if (!inside(q)) {
            sum += c;
        } else if (wet(q)) {
            sum += bed_depth[widx(q)];
        }
    }
    let target_d = (sum + PRIOR_WEIGHT * prior[i]) / (4.0 + PRIOR_WEIGHT);
    bed_depth[i] = max(c + params.omega * (target_d - c), 0.0);
}

@compute @workgroup_size(8, 8, 1)
fn bed_write(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p) || !wet(p)) { return; }
    let i = widx(p);
    let ground = terrain(p);
    let eta = ground + depth[i];
    let bed = max(mix(ground, min(ground, eta - bed_depth[i]), params.strength), 0.0);
    atomicStore(&height[gidx(p)], u32(round(bed * HEIGHT_SCALE)));
    depth[i] = max(eta - bed, 0.0);
}

const DIRS8 = array<vec2<i32>, 8>(
    vec2<i32>(-1, 0), vec2<i32>(1, 0), vec2<i32>(0, -1), vec2<i32>(0, 1),
    vec2<i32>(-1, -1), vec2<i32>(1, -1), vec2<i32>(-1, 1), vec2<i32>(1, 1),
);

fn shore_src(i: u32) -> vec2<f32> {
    if (params.parity == 0u) { return shore_a[i]; }
    return shore_b[i];
}

// One step of a chamfer distance transform from the shore (parity 0: a → b, 1: b → a).
@compute @workgroup_size(8, 8, 1)
fn bed_dilate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p)) { return; }
    let i = widx(p);
    var best = shore_src(i);
    for (var k = 0; k < 8; k = k + 1) {
        let q = p + DIRS8[k];
        if (!inside(q)) { continue; }
        let s = shore_src(widx(q));
        let d = s.y + select(1.0, 1.41421, k >= 4);
        if (d < best.y) {
            best = vec2<f32>(s.x, d);
        }
    }
    if (params.parity == 0u) {
        shore_b[i] = best;
    } else {
        shore_a[i] = best;
    }
}

fn s5(t: f32) -> f32 {
    let x = clamp(t, 0.0, 1.0);
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0);
}

// Banks just outside the domain: lower anything above a gentle ramp up from the shore, blending
// back to the untouched terrain by `bank_width` texels.
@compute @workgroup_size(8, 8, 1)
fn bed_grade(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p) || wet(p)) { return; }
    let i = widx(p);
    let shore = shore_a[i];
    if (shore.y >= params.bank_width) { return; }
    let ground = terrain(p);
    let z_max = shore.x + params.bank_rise + params.bank_slope * shore.y;
    let graded = min(ground, mix(z_max, ground, s5(shore.y / params.bank_width)));
    let z = mix(ground, graded, params.strength);
    atomicStore(&height[gidx(p)], u32(round(max(z, 0.0) * HEIGHT_SCALE)));
}
