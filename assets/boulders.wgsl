// Boulders as bodies (Phase 8 of docs/UNIFIED_PLAN.md): spheres that sit on the heightfield and
// move when the flow's drag and lift, and gravity down the slope, beat their friction.
//   * boulder_forces, boulder_contacts, boulder_integrate (per substep, one thread per boulder):
//     forces in SI from the water's depth and velocity under each boulder, soft contacts between
//     overlapping boulders, then Coulomb friction and the move. Positions change only in the last
//     pass, so the contacts see one consistent set. A boulder that leaves the map or sinks into
//     the ocean is removed.
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
// Contact spring over the pair's mass / dt² (below 1 for a stable explicit step), and the
// damping ratio on the approach.
const CONTACT_STIFFNESS: f32 = 0.2;
const CONTACT_DAMPING: f32 = 0.5;
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
    // Scratch of the current substep: the flow's and the slope's push (N), the contacts' push
    // (N), the normal force on the bed (N), and the submerged share.
    drive: vec2<f32>,
    contact: vec2<f32>,
    normal: f32,
    wet: f32,
    pad: vec2<f32>,
}

struct BoulderParams {
    size: vec2<u32>,
    pad_offset: vec2<u32>,
    grid_size: vec2<u32>,
    // Slots in use (the rest are free).
    slots: u32,
    // Water time per substep (s); the passes run several substeps a frame.
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

// Mass with the added mass of the water it displaces (kg), and the submerged share of the boulder.
fn mass_of(b: Boulder) -> f32 {
    let r = max(b.radius, 0.01);
    let volume = 4.0 / 3.0 * PI * r * r * r;
    return (params.density * WATER_DENSITY + 0.5 * WATER_DENSITY * b.wet) * volume;
}

// Pass 1 of a substep: the flow's drag and lift, buoyancy and gravity down the slope under each
// boulder, from the water's depth and velocity and the bed around it.
@compute @workgroup_size(64, 1, 1)
fn boulder_forces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.slots) { return; }
    var b = boulders[i];
    if (b.alive == 0u) { return; }
    let dx = params.metres_per_texel;
    let r = max(b.radius, 0.01);
    let volume = 4.0 / 3.0 * PI * r * r * r;
    let t = taps(b.pos);
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
    let p = vec2<i32>(floor(b.pos));
    let reach = max(i32(round(r / dx)), 1);
    let e = vec2<i32>(reach, 0);
    let n = vec2<i32>(0, reach);
    let grad = vec2<f32>(terrain(p + e) - terrain(p - e), terrain(p + n) - terrain(p - n))
        * params.metres_per_height / (2.0 * f32(reach) * dx);
    let cos_slope = inverseSqrt(1.0 + dot(grad, grad));

    // Submerged share, frontal area in the flow, and the forces (N).
    b.wet = clamp(h / (2.0 * r), 0.0, 1.0);
    let area = 2.0 * r * min(h, 2.0 * r);
    let weight = (params.density - b.wet) * WATER_DENSITY * params.gravity * volume;
    let rel = u - b.vel;
    let drag = 0.5 * WATER_DENSITY * params.drag * area * length(rel) * rel;
    let lift = 0.5 * WATER_DENSITY * params.lift * area * dot(u, u);
    b.drive = drag - weight * grad * cos_slope;
    b.normal = max(weight * cos_slope - lift, 0.0);
    boulders[i] = b;
}

// Pass 2: contacts. Boulders are discs in plan view; each overlapping pair pushes apart with a
// spring and a damper on their approach (soft contacts, as in DEM). The spring is as stiff as
// the substep allows (a fifth of the pair's mass over dt²), so a pushed boulder passes the push
// on, and a group whose friction together beats the flow holds: a jam.
@compute @workgroup_size(64, 1, 1)
fn boulder_contacts(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.slots) { return; }
    var b = boulders[i];
    if (b.alive == 0u) { return; }
    let dt = max(params.seconds, 1e-6);
    let m = mass_of(b);
    var force = vec2<f32>(0.0);
    for (var j = 0u; j < params.slots; j += 1u) {
        if (j == i) { continue; }
        let o = boulders[j];
        if (o.alive == 0u) { continue; }
        let d = (b.pos - o.pos) * params.metres_per_texel;
        let gap = length(d);
        let overlap = b.radius + o.radius - gap;
        if (overlap <= 0.0) { continue; }
        // Coincident centres: push apart along the slot order.
        let normal = select(vec2<f32>(select(-1.0, 1.0, i > j), 0.0), d / gap, gap > 1e-4);
        let mo = mass_of(o);
        let pair = m * mo / (m + mo);
        let k = CONTACT_STIFFNESS * pair / (dt * dt);
        let damping = CONTACT_DAMPING * 2.0 * sqrt(k * pair);
        let approach = dot(b.vel - o.vel, normal);
        force += max(k * overlap - damping * approach, 0.0) * normal;
    }
    b.contact = force;
    boulders[i] = b;
}

// Pass 3: Coulomb friction and the move. A boulder at rest stays while the flow, the slope and
// its neighbours together push it less than static friction holds it.
@compute @workgroup_size(64, 1, 1)
fn boulder_integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.slots) { return; }
    var b = boulders[i];
    if (b.alive == 0u) { return; }
    let dx = params.metres_per_texel;
    let dt = params.seconds;
    let drive = b.drive + b.contact;
    let normal = b.normal;
    let mass = mass_of(b);
    var vel = b.vel;
    let speed = length(vel);
    let holds = length(drive) <= max(params.friction_static * normal, 1e-6);
    if (speed < REST_SPEED && holds) {
        vel = vec2<f32>(0.0);
    } else {
        let dir = select(normalize(drive), vel / max(speed, 1e-6), speed >= REST_SPEED);
        var next = vel + (drive - params.friction_kinetic * normal * dir) / mass * dt;
        // Friction stops a boulder; it never turns it round.
        if (dot(next, dir) <= 0.0 && holds) {
            next = vec2<f32>(0.0);
        }
        let s = length(next);
        if (s > MAX_SPEED) { next *= MAX_SPEED / s; }
        vel = next;
    }
    b.pos += vel * dt / dx;
    b.vel = vel;
    b.rolled += length(vel) * dt;
    let size = vec2<f32>(params.size);
    b.ground = ground_at(b.pos);
    if (any(b.pos < vec2<f32>(1.0)) || any(b.pos > size - 1.0)
        || (params.sea_level > 0.0 && b.ground < params.sea_level)) {
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
