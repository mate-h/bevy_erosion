// Bed reconstruction for lakes and rivers (Phases 2 and 4f of docs/WATER_PLAN.md), ported from
// terracraft's riverbed solver (`solve_tau_sor`, `solve_lake_bathy_sor`, stage 5 bank grading).
//
// Runs on demand on the domain the water produced: lakes (water deeper than `DOMAIN_DEPTH`) and
// channels (averaged unit discharge above `channel_discharge`, so thin streams count too). There
// is no painted mask and no water level. The bed is cut to `η_ref − d`, where the reference
// surface η_ref comes from the banks (lakes: the water surface) and the depth d from the
// discharge, so the result does not depend on how far the water has drained into an earlier cut:
// a second press changes nothing.
//   1. bed_init:   domain, unit discharge, and bank seeds (dry cells next to the domain).
//   2. bed_tau:    bank distance τ, ∇²τ = −2 inside, τ = 0 on the banks (red-black SOR).
//   3. bed_dilate: nearest bank (its height and distance), inward for η_ref and outward for the
//                  bank grading.
//   4. bed_prior:  depth prior. Rivers: hydraulic geometry, depth ∝ Q^0.4 and half-width ∝ Q^0.5
//                  for the discharge Q through the cross-section, with a (1 − n²)^profile cross-section (n from τ).
//                  Still water: a bowl that deepens with the distance to the shore.
//   5. bed_smooth: screened Poisson ∇·(A∇d) = λ(d − prior), smoothing across the flow only, so
//                  the long profile keeps its steps.
//   6. bed_write:  bed = min(ground, η_ref − d) (blended by `strength`), only ever deepening; the
//                  depth keeps the surface in place.
//   7. bed_grade:  cap the banks beyond the first ring with a quintic S5 ramp up from the bank,
//                  z_max = bank + slope·distance. The first ring is left alone: it sets η_ref.

const HEIGHT_SCALE: f32 = 1000000.0;
// Depth above which still water belongs to the domain (above `water.wgsl`'s WET_DEPTH, so thin
// sheet flow on hillslopes only joins where it carries enough discharge).
const DOMAIN_DEPTH: f32 = 0.002;
const WET_DEPTH: f32 = 0.0005;
// Averaged speed (texels per second) below which water counts as still (a lake) / above which it
// is a river.
const STILL_SPEED: f32 = 2.0;
const RIVER_SPEED: f32 = 8.0;
// Channel discharge (height units × texels² per second) at which rivers get `river_depth` and
// `river_width`: about a default demo spring.
const REFERENCE_DISCHARGE: f32 = 1.0;
// Screened Poisson prior weight (terracraft's soft guide weight) for lakes, whose bowl prior
// has kinks along τ's medial axis. River profiles come out of the hydraulic geometry smooth, so
// rivers only round off their corners.
const PRIOR_WEIGHT: f32 = 0.045;
const RIVER_PRIOR_WEIGHT: f32 = 2.0;
// How much of the smoothing along the flow is removed in rivers.
const ALONG_DAMPING: f32 = 0.95;
const FAR: f32 = 1.0e6;
// Farthest a cross-section is followed to each side (texels).
const SECTION_REACH: i32 = 32;

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
    bank_slope: f32,
    river_depth: f32,
    river_width: f32,
    channel_discharge: f32,
}

@group(0) @binding(0) var<storage, read_write> height: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> depth: array<f32>;
// Shared discharge field (`water_morpho.wgsl`): xy = averaged h·u, z = averaged depth.
@group(0) @binding(2) var<storage, read> discharge: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> tau: array<f32>;
@group(0) @binding(4) var<storage, read_write> bed_depth: array<f32>;
@group(0) @binding(5) var<storage, read_write> prior: array<f32>;
// Nearest bank: x = its terrain height, y = distance in texels (FAR where none is in reach).
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

fn unit_discharge(i: u32) -> f32 {
    return length(discharge[i].xy);
}

// Lakes and channels. Shaping only deepens the water in it, so it stays the same domain.
fn wet(p: vec2<i32>) -> bool {
    let i = widx(p);
    return depth[i] > DOMAIN_DEPTH || unit_discharge(i) > params.channel_discharge;
}

// Dry cell next to the domain.
fn is_bank(p: vec2<i32>) -> bool {
    if (wet(p)) { return false; }
    for (var k = 0; k < 4; k = k + 1) {
        let q = p + DIRS[k];
        if (inside(q) && wet(q)) { return true; }
    }
    return false;
}

// 1 for still water (lakes), 0 for rivers, from the averaged flow.
fn stillness(i: u32) -> f32 {
    let d = discharge[i];
    let speed = length(d.xy) / max(d.z, WET_DEPTH);
    return 1.0 - smoothstep(STILL_SPEED, RIVER_SPEED, speed);
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
    prior[i] = 0.0;
    bed_depth[i] = 0.0;
    if (is_bank(p)) {
        shore_a[i] = vec2<f32>(terrain(p), 0.0);
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

// Discharge through the cross-section at `p`: the averaged flow summed along the line across
// it, out to the banks on both sides.
fn section_discharge(p: vec2<i32>) -> f32 {
    let q = discharge[widx(p)].xy;
    let len = length(q);
    if (len <= 0.0) { return 0.0; }
    let along = q / len;
    let across = vec2<f32>(-along.y, along.x);
    var total = len;
    for (var side = -1.0; side <= 1.0; side += 2.0) {
        for (var k = 1; k <= SECTION_REACH; k = k + 1) {
            let c = vec2<i32>(floor(vec2<f32>(p) + 0.5 + across * side * f32(k)));
            if (!inside(c) || !wet(c)) { break; }
            total += max(dot(discharge[widx(c)].xy, along), 0.0);
        }
    }
    return total;
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
    // Hydraulic geometry from the channel discharge. Where the flowing domain is wider than the
    // channel the discharge asks for (sheet flow), the channel takes its middle.
    let flow = section_discharge(p) / REFERENCE_DISCHARGE;
    let thalweg = params.river_depth * pow(flow, 0.4);
    let half_width = clamp(params.river_width * pow(flow, 0.5), 0.5, b);
    let x = min(n * b / half_width, 1.0);
    let river = soft_cap(thalweg) * pow(max(1.0 - x * x, 0.0), params.profile);
    // sqrt(2τ) grows like the distance to the shore near the banks and is the radius at the
    // centre of a round lake.
    let lake = soft_cap(params.shore_slope * sqrt(2.0 * c));
    let d = mix(river, lake, stillness(i));
    prior[i] = d;
    bed_depth[i] = d;
}

// Screened Poisson smoothing of the depth: d = 0 on the banks, softly pulled to the prior. In
// rivers the coupling along the flow is mostly removed, so only the cross-section is smoothed.
@compute @workgroup_size(8, 8, 1)
fn bed_smooth(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p) || !red_black(p) || !wet(p)) { return; }
    let i = widx(p);
    let still = stillness(i);
    // Flowing water outside the channel's hydraulic width stays uncut: if the smoothing spread
    // the channel into it, the next press after the flow narrows would find lower banks there
    // and cut deeper.
    if (prior[i] <= 0.0 && still < 0.5) {
        bed_depth[i] = 0.0;
        return;
    }
    let c = bed_depth[i];
    let q = discharge[i].xy;
    let along = select(vec2<f32>(0.0), q / length(q), length(q) > 0.0);
    let r = ALONG_DAMPING * (1.0 - still);
    let wx = 1.0 - r * along.x * along.x;
    let wy = 1.0 - r * along.y * along.y;
    var sum = 0.0;
    for (var k = 0; k < 4; k = k + 1) {
        let q = p + DIRS[k];
        let w = select(wy, wx, k < 2);
        if (!inside(q)) {
            sum += w * c;
        } else if (wet(q)) {
            sum += w * bed_depth[widx(q)];
        }
    }
    let lambda = mix(RIVER_PRIOR_WEIGHT, PRIOR_WEIGHT, still);
    let target_d = (sum + lambda * prior[i]) / (2.0 * (wx + wy) + lambda);
    bed_depth[i] = max(c + params.omega * (target_d - c), 0.0);
}

@compute @workgroup_size(8, 8, 1)
fn bed_write(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p) || !wet(p)) { return; }
    let i = widx(p);
    let ground = terrain(p);
    let eta = ground + depth[i];
    // Rivers cut below their nearest (lowest) bank, lakes below their surface; neither moves
    // when the bed is cut, so pressing again cuts to the same bed.
    let bank = shore_a[i];
    let eta_ref = mix(select(eta, bank.x, bank.y < FAR), eta, stillness(i));
    let target_bed = min(ground, eta_ref - bed_depth[i]);
    let bed = max(mix(ground, target_bed, params.strength), 0.0);
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

// One step of a chamfer distance transform from the banks (parity 0: a → b, 1: b → a). Banks
// at the same distance keep the lower height: water would leave over the lower one.
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
        if (s.y >= FAR) { continue; }
        let d = s.y + select(1.0, 1.41421, k >= 4);
        // A loose tie would let the lower bank downstream win step after step and pull the
        // reference down the valley.
        if (d < best.y - 0.01) {
            best = vec2<f32>(s.x, d);
        } else if (d < best.y + 0.01) {
            best.x = min(best.x, s.x);
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

// Banks beyond the first ring: lower anything above a gentle ramp up from the nearest bank,
// blending back to the untouched terrain by `bank_width` texels.
@compute @workgroup_size(8, 8, 1)
fn bed_grade(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p) || wet(p)) { return; }
    let i = widx(p);
    let bank = shore_a[i];
    if (bank.y < 0.5 || bank.y >= params.bank_width) { return; }
    let ground = terrain(p);
    let z_max = bank.x + params.bank_slope * bank.y;
    let graded = min(ground, mix(z_max, ground, s5(bank.y / params.bank_width)));
    let z = mix(ground, graded, params.strength);
    atomicStore(&height[gidx(p)], u32(round(max(z, 0.0) * HEIGHT_SCALE)));
}
