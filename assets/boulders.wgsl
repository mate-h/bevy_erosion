// Boulders as bodies (Phase 8 of docs/UNIFIED_PLAN.md): spheres that sit on the heightfield and
// move when the flow's drag and lift, and gravity down the slope, beat their friction.
//   * boulder_move (per frame, one thread per boulder): forces in SI from the water's depth and
//     velocity under the boulder, integrated in substeps short enough that no step crosses half a
//     texel. A boulder that leaves the map or sinks into the ocean is removed.
//   * obstacle_clear and obstacle_stamp (per frame): each boulder adds a bump of its own volume to
//     the obstacle field, which only the shallow-water solver's bed sees (`swe_export`). The water
//     flows around it and speeds up beside it, so the bed load scours there; the erosion heights
//     themselves stay the terrain's.

const HEIGHT_SCALE: f32 = 1000000.0;
const CELLS_PER_HEIGHT: f32 = 100.0;
const PI: f32 = 3.14159265;
const WATER_DENSITY: f32 = 1000.0;
// Speed below which a boulder is at rest (m/s), and the fastest it may go (rockfall).
const REST_SPEED: f32 = 0.001;
const MAX_SPEED: f32 = 30.0;
const MAX_SUBSTEPS: u32 = 32u;
// Half-width of the stamp workgroup: boulders cover at most this many texels from their centre.
const STAMP_REACH: i32 = 8;

struct Boulder {
    // Centre in map texels.
    pos: vec2<f32>,
    // Velocity (m/s).
    vel: vec2<f32>,
    // Radius (m).
    radius: f32,
    // Ground under the centre (height units).
    ground: f32,
    // Distance rolled (m), for the drawn rotation.
    rolled: f32,
    // 0 = free slot.
    alive: u32,
}

struct BoulderParams {
    size: vec2<u32>,
    pad_offset: vec2<u32>,
    grid_size: vec2<u32>,
    // Slots in use (the rest are free).
    slots: u32,
    // Water time this frame (s).
    seconds: f32,
    // Seconds per solver step: the velocity buffer is in texels per step.
    step_seconds: f32,
    metres_per_texel: f32,
    metres_per_height: f32,
    gravity: f32,
    // Rock density over water's.
    density: f32,
    drag: f32,
    lift: f32,
    // Coulomb friction while at rest (the pivot angle's tangent) and while moving.
    friction_static: f32,
    friction_kinetic: f32,
    sea_level: f32,
}

@group(0) @binding(0) var<storage, read> height: array<u32>;
@group(0) @binding(1) var<storage, read> depth: array<f32>;
@group(0) @binding(2) var<storage, read> velocity: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> boulders: array<Boulder>;
@group(0) @binding(4) var<uniform> params: BoulderParams;
// Height the boulders add to the solver's bed (height units).
@group(0) @binding(5) var obstacle: texture_storage_2d<r32float, read_write>;

fn widx(p: vec2<i32>) -> u32 {
    return u32(p.y) * params.size.x + u32(p.x);
}

fn gidx(p: vec2<i32>) -> u32 {
    let q = vec2<u32>(p) + params.pad_offset;
    return q.y * params.grid_size.x + q.x;
}

fn clamp_map(p: vec2<i32>) -> vec2<i32> {
    return clamp(p, vec2<i32>(0), vec2<i32>(params.size) - 1);
}

fn terrain(p: vec2<i32>) -> f32 {
    return f32(height[gidx(clamp_map(p))]) / HEIGHT_SCALE;
}

// Bilinear weights and corners of `pos` (texels, between texel centres).
struct Taps {
    p: vec2<i32>,
    f: vec2<f32>,
}

fn taps(pos: vec2<f32>) -> Taps {
    let q = pos - 0.5;
    return Taps(vec2<i32>(floor(q)), q - floor(q));
}

fn ground_at(pos: vec2<f32>) -> f32 {
    let t = taps(pos);
    let a = mix(terrain(t.p), terrain(t.p + vec2<i32>(1, 0)), t.f.x);
    let b = mix(terrain(t.p + vec2<i32>(0, 1)), terrain(t.p + vec2<i32>(1, 1)), t.f.x);
    return mix(a, b, t.f.y);
}

fn sample_depth(q: vec2<i32>) -> f32 {
    return depth[widx(clamp_map(q))];
}

fn sample_velocity(q: vec2<i32>) -> vec2<f32> {
    return velocity[widx(clamp_map(q))];
}

@compute @workgroup_size(64, 1, 1)
fn boulder_move(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.slots) { return; }
    var b = boulders[i];
    if (b.alive == 0u) { return; }
    let dx = params.metres_per_texel;
    let g = params.gravity;
    let r = max(b.radius, 0.01);
    let volume = 4.0 / 3.0 * PI * r * r * r;
    let rock = params.density * WATER_DENSITY;

    var pos = b.pos;
    var vel = b.vel;
    var left = params.seconds;
    for (var k = 0u; k < MAX_SUBSTEPS && left > 0.0; k += 1u) {
        let t = taps(pos);
        let c00 = t.p;
        let c10 = t.p + vec2<i32>(1, 0);
        let c01 = t.p + vec2<i32>(0, 1);
        let c11 = t.p + vec2<i32>(1, 1);
        // Water depth (m) and velocity (m/s) under the boulder.
        let h = mix(
            mix(sample_depth(c00), sample_depth(c10), t.f.x),
            mix(sample_depth(c01), sample_depth(c11), t.f.x),
            t.f.y,
        ) * params.metres_per_height;
        let u = mix(
            mix(sample_velocity(c00), sample_velocity(c10), t.f.x),
            mix(sample_velocity(c01), sample_velocity(c11), t.f.x),
            t.f.y,
        ) * dx / max(params.step_seconds, 1e-6);
        // Bed slope (m per m) across the boulder's footprint.
        let p = vec2<i32>(floor(pos));
        let reach = max(i32(round(r / dx)), 1);
        let e = vec2<i32>(reach, 0);
        let n = vec2<i32>(0, reach);
        let grad = vec2<f32>(terrain(p + e) - terrain(p - e), terrain(p + n) - terrain(p - n))
            * params.metres_per_height / (2.0 * f32(reach) * dx);
        let cos_slope = inverseSqrt(1.0 + dot(grad, grad));

        // Submerged share of the boulder, its frontal area in the flow, and the forces (N).
        let wet = clamp(h / (2.0 * r), 0.0, 1.0);
        let area = 2.0 * r * min(h, 2.0 * r);
        let weight = (rock - WATER_DENSITY * wet) * g * volume;
        let rel = u - vel;
        let drag = 0.5 * WATER_DENSITY * params.drag * area * length(rel) * rel;
        let lift = 0.5 * WATER_DENSITY * params.lift * area * dot(u, u);
        let downhill = -weight * grad * cos_slope;
        let normal = max(weight * cos_slope - lift, 0.0);
        let drive = drag + downhill;
        let mass = rock * volume + 0.5 * WATER_DENSITY * volume * wet;

        // Substep: at most half a texel of travel.
        let fastest = max(max(length(vel), length(u)), 1.0);
        let dt = min(left, 0.5 * dx / fastest);
        left -= dt;

        let speed = length(vel);
        if (speed < REST_SPEED) {
            // At rest until the drive beats static friction.
            if (length(drive) <= max(params.friction_static * normal, 1e-6)) {
                vel = vec2<f32>(0.0);
                continue;
            }
        }
        let dir = select(normalize(drive), vel / max(speed, 1e-6), speed >= REST_SPEED);
        var next = vel + (drive - params.friction_kinetic * normal * dir) / mass * dt;
        // Friction stops a boulder; it never turns it round.
        if (dot(next, dir) <= 0.0 && length(drive) <= params.friction_static * normal) {
            next = vec2<f32>(0.0);
        }
        let s = length(next);
        if (s > MAX_SPEED) { next *= MAX_SPEED / s; }
        vel = next;
        pos += vel * dt / dx;
        b.rolled += length(vel) * dt;
    }
    b.pos = pos;
    b.vel = vel;
    let size = vec2<f32>(params.size);
    let ground = ground_at(pos);
    b.ground = ground;
    if (any(pos < vec2<f32>(1.0)) || any(pos > size - 1.0)
        || (params.sea_level > 0.0 && ground < params.sea_level)) {
        b.alive = 0u;
    }
    boulders[i] = b;
}

@compute @workgroup_size(8, 8, 1)
fn obstacle_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (any(p >= vec2<i32>(params.size))) { return; }
    textureStore(obstacle, p, vec4<f32>(0.0));
}

// One workgroup per boulder covers the texels within `STAMP_REACH` of its centre. The bump has
// the boulder's volume (its peak no higher than the boulder) and a smooth rim at the larger of
// its radius and one texel, so boulders smaller than a texel still turn the flow a little.
// Overlapping boulders keep the higher bump.
@compute @workgroup_size(16, 16, 1)
fn obstacle_stamp(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let i = wg.x;
    if (i >= params.slots) { return; }
    let b = boulders[i];
    if (b.alive == 0u) { return; }
    let r = b.radius / params.metres_per_texel;
    let rim = clamp(max(r, 1.0), 1.0, f32(STAMP_REACH));
    let p = vec2<i32>(floor(b.pos)) + vec2<i32>(local.xy) - vec2<i32>(STAMP_REACH);
    if (any(p < vec2<i32>(0)) || any(p >= vec2<i32>(params.size))) { return; }
    let d = distance(vec2<f32>(p) + 0.5, b.pos) / rim;
    if (d >= 1.0) { return; }
    let w = (1.0 - d * d) * (1.0 - d * d);
    // (1 − d²)² over the disc integrates to π·rim²/3.
    let peak = min(2.0 * r, 4.0 * r * r * r / (rim * rim));
    let bump = peak * w / CELLS_PER_HEIGHT;
    let old = textureLoad(obstacle, p).x;
    if (bump > old) {
        textureStore(obstacle, p, vec4<f32>(bump, 0.0, 0.0, 0.0));
    }
}
