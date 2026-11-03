// Landslides with MPM (Phase 9 of docs/UNIFIED_PLAN.md), as local events.
//
// A release takes a slab off the terrain (a paraboloid slip surface under a disc) and turns it into
// particles in a local 3D grid, half a texel per cell. MLS-MPM (Hu et al. 2018) moves them as a
// frictional granular continuum: pressure resists compaction only (no tension), and the shear
// stress is a viscosity that holds it at μ·p (Drucker–Prager in its simplest form), capped for a
// stable explicit step. The terrain is the boundary: nodes in the ground lose their velocity into
// it, and Coulomb friction slows their slide along it.
//
// The moving material is written into the heights every frame and taken out again before the next
// (exactly, through `stamp`), so the terrain, the water and the drawing see it. When the event
// ends, the last write stays: the deposit.
//
//   * slide_reset: zero the particle count and the stamp record (before a release).
//   * slide_release (per box texel): lower the terrain by the slab and spawn its particles.
//   * slide_unstamp (per box texel): take last frame's debris back out of the heights.
//   * grid_clear, slide_p2g, grid_update, slide_g2p: one MPM substep.
//   * slide_stamp (per particle): write the debris into the heights, recording it per texel.
//   * slide_settle (per box texel): forget the record, so the last write stays.

const HEIGHT_SCALE: f32 = 1000000.0;
// Fixed point of the grid's atomic sums (mass in m³, momentum in m⁴/s).
const GRID_SCALE: f32 = 10000.0;
// Particles per texel side, and the vertical spacing in cells.
const SUB: u32 = 4u;

struct Particle {
    // Position (m, box frame) and J (volume ratio).
    pos: vec4<f32>,
    // Velocity (m/s) and volume (m³).
    vel: vec4<f32>,
    // APIC affine matrix C (rows).
    c0: vec4<f32>,
    c1: vec4<f32>,
    c2: vec4<f32>,
}

struct SlideParams {
    size: vec2<u32>,
    pad_offset: vec2<u32>,
    grid_size: vec2<u32>,
    // First map texel of the box, and its size in texels.
    box_origin: vec2<i32>,
    box_texels: vec2<u32>,
    // Grid nodes per axis (x, y, z; y is up).
    nodes: vec3<u32>,
    max_particles: u32,
    // Height of the box floor (m), texel size (m), metres per height unit, cell size (m).
    floor: f32,
    metres_per_texel: f32,
    metres_per_height: f32,
    cell: f32,
    // Substep (s), gravity (m/s²), bulk stiffness c² (m²/s²), internal and basal friction.
    dt: f32,
    gravity: f32,
    stiffness: f32,
    friction: f32,
    basal_friction: f32,
    // Largest shear viscosity (m²/s) the explicit step allows.
    max_viscosity: f32,
    // Release: centre (texels), radius (texels), depth at the centre (m).
    centre: vec2<f32>,
    radius: f32,
    depth: f32,
}

@group(0) @binding(0) var<storage, read_write> height: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
// Per node: mass, momentum x, y, z (fixed point).
@group(0) @binding(2) var<storage, read_write> grid: array<atomic<i32>>;
// Per box texel: height quanta the debris added this frame. Last slot: the particle count.
@group(0) @binding(3) var<storage, read_write> stamp: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: SlideParams;

fn gidx(p: vec2<i32>) -> u32 {
    let q = vec2<u32>(p) + params.pad_offset;
    return q.y * params.grid_size.x + q.x;
}

fn clamp_map(p: vec2<i32>) -> vec2<i32> {
    return clamp(p, vec2<i32>(0), vec2<i32>(params.size) - 1);
}

// Terrain height (m) at map texel `p`.
fn ground_texel(p: vec2<i32>) -> f32 {
    return f32(atomicLoad(&height[gidx(clamp_map(p))])) / HEIGHT_SCALE * params.metres_per_height;
}

// Terrain height (m) under a box-frame point (x, z in m), between texel centres, relative to the
// box floor.
fn ground_at(xz: vec2<f32>) -> f32 {
    let t = xz / params.metres_per_texel + vec2<f32>(params.box_origin) - 0.5;
    let p = vec2<i32>(floor(t));
    let f = t - floor(t);
    let a = mix(ground_texel(p), ground_texel(p + vec2<i32>(1, 0)), f.x);
    let b = mix(ground_texel(p + vec2<i32>(0, 1)), ground_texel(p + vec2<i32>(1, 1)), f.x);
    return mix(a, b, f.y) - params.floor;
}

fn count_slot() -> u32 {
    return params.box_texels.x * params.box_texels.y;
}

fn particle_count() -> u32 {
    return min(atomicLoad(&stamp[count_slot()]), params.max_particles);
}

fn box_texel(gid: vec3<u32>) -> bool {
    return all(gid.xy < params.box_texels);
}

@compute @workgroup_size(8, 8, 1)
fn slide_reset(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!box_texel(gid)) { return; }
    let i = gid.y * params.box_texels.x + gid.x;
    atomicStore(&stamp[i], 0u);
    if (i == 0u) { atomicStore(&stamp[count_slot()], 0u); }
}

// The slab under texel `gid`: the paraboloid depth × (1 − (d/R)²), lowered out of the terrain and
// spawned as SUB × SUB columns of particles a quarter texel apart, layered half a cell apart.
@compute @workgroup_size(8, 8, 1)
fn slide_release(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!box_texel(gid)) { return; }
    let p = params.box_origin + vec2<i32>(gid.xy);
    if (any(p < vec2<i32>(1)) || any(p >= vec2<i32>(params.size) - 1)) { return; }
    let d = distance(vec2<f32>(p) + 0.5, params.centre) / max(params.radius, 1e-3);
    if (d >= 1.0) { return; }
    let ground = ground_texel(p);
    let thick = min(params.depth * (1.0 - d * d), ground);
    let spacing = 0.5 * params.cell;
    let layers = u32(ceil(thick / spacing));
    if (layers == 0u) { return; }
    let n = layers * SUB * SUB;
    let first = atomicAdd(&stamp[count_slot()], n);
    if (first + n > params.max_particles) {
        // Out of particles: leave this texel as it is.
        atomicSub(&stamp[count_slot()], n);
        return;
    }
    let quanta = u32(round(thick / params.metres_per_height * HEIGHT_SCALE));
    atomicSub(&height[gidx(p)], min(quanta, atomicLoad(&height[gidx(p)])));
    // Each particle carries its share of the column, so the slab's volume is conserved.
    let column = thick * params.metres_per_texel * params.metres_per_texel;
    let volume = column / f32(n);
    let top = ground - params.floor;
    let dx = params.metres_per_texel;
    for (var k = 0u; k < n; k += 1u) {
        let layer = k / (SUB * SUB);
        let sx = k % SUB;
        let sz = (k / SUB) % SUB;
        let xz = (vec2<f32>(gid.xy) + (vec2<f32>(f32(sx), f32(sz)) + 0.5) / f32(SUB)) * dx;
        let y = top - thick * (f32(layer) + 0.5) / f32(layers);
        var q: Particle;
        q.pos = vec4<f32>(xz.x, y, xz.y, 1.0);
        q.vel = vec4<f32>(0.0, 0.0, 0.0, volume);
        q.c0 = vec4<f32>(0.0);
        q.c1 = vec4<f32>(0.0);
        q.c2 = vec4<f32>(0.0);
        particles[first + k] = q;
    }
}

@compute @workgroup_size(8, 8, 1)
fn slide_unstamp(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!box_texel(gid)) { return; }
    let i = gid.y * params.box_texels.x + gid.x;
    let q = atomicExchange(&stamp[i], 0u);
    if (q == 0u) { return; }
    let p = params.box_origin + vec2<i32>(gid.xy);
    let g = gidx(p);
    atomicSub(&height[g], min(q, atomicLoad(&height[g])));
}

fn node_index(n: vec3<i32>) -> u32 {
    let s = vec3<i32>(params.nodes);
    return u32((n.y * s.z + n.z) * s.x + n.x);
}

fn inside_grid(n: vec3<i32>) -> bool {
    return all(n >= vec3<i32>(0)) && all(n < vec3<i32>(params.nodes));
}

@compute @workgroup_size(64, 1, 1)
fn grid_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.nodes.x * params.nodes.y * params.nodes.z;
    let i = gid.x + gid.y * 65535u * 64u;
    if (i >= total) { return; }
    for (var c = 0u; c < 4u; c += 1u) {
        atomicStore(&grid[4u * i + c], 0);
    }
}

// Quadratic B-spline weights of the 3×3×3 nodes around `x` (in cells).
struct Stencil {
    base: vec3<i32>,
    w: array<vec3<f32>, 3>,
}

fn stencil(x: vec3<f32>) -> Stencil {
    let base = vec3<i32>(floor(x - 0.5));
    let f = x - vec3<f32>(base);
    var s: Stencil;
    s.base = base;
    s.w[0] = 0.5 * (1.5 - f) * (1.5 - f);
    s.w[1] = 0.75 - (f - 1.0) * (f - 1.0);
    s.w[2] = 0.5 * (f - 0.5) * (f - 0.5);
    return s;
}

fn matrix(q: Particle) -> mat3x3<f32> {
    // Columns of C from its rows.
    return transpose(mat3x3<f32>(q.c0.xyz, q.c1.xyz, q.c2.xyz));
}

// Stress over density (m²/s²): pressure from compaction, and a frictional shear viscosity that
// holds the shear stress at μ·p.
fn stress(q: Particle) -> mat3x3<f32> {
    let j = q.pos.w;
    let pressure = params.stiffness * max(1.0 / j - 1.0, 0.0);
    let c = matrix(q);
    let d = 0.5 * (c + transpose(c));
    let trace = d[0][0] + d[1][1] + d[2][2];
    let identity = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
    let dev = d - identity * (trace / 3.0);
    let rate = sqrt(2.0 * (dot(dev[0], dev[0]) + dot(dev[1], dev[1]) + dot(dev[2], dev[2])));
    let viscosity = min(params.friction * pressure / max(rate, 1e-4), params.max_viscosity);
    return dev * (2.0 * viscosity) - identity * pressure;
}

@compute @workgroup_size(64, 1, 1)
fn slide_p2g(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 65535u * 64u;
    if (i >= particle_count()) { return; }
    let q = particles[i];
    let h = params.cell;
    let x = q.pos.xyz / h;
    let s = stencil(x);
    let volume = q.vel.w * q.pos.w;
    let mass = q.vel.w;
    // MLS-MPM: the stress enters as an affine momentum term.
    let affine = stress(q) * (-params.dt * volume * 4.0 / (h * h)) + matrix(q) * mass;
    for (var a = 0; a < 3; a += 1) {
        for (var b = 0; b < 3; b += 1) {
            for (var c = 0; c < 3; c += 1) {
                let n = s.base + vec3<i32>(a, b, c);
                if (!inside_grid(n)) { continue; }
                let w = s.w[a].x * s.w[b].y * s.w[c].z;
                let dpos = (vec3<f32>(n) - x) * h;
                let momentum = w * (mass * q.vel.xyz + affine * dpos);
                let k = 4u * node_index(n);
                atomicAdd(&grid[k], i32(round(w * mass * GRID_SCALE)));
                atomicAdd(&grid[k + 1u], i32(round(momentum.x * GRID_SCALE)));
                atomicAdd(&grid[k + 2u], i32(round(momentum.y * GRID_SCALE)));
                atomicAdd(&grid[k + 3u], i32(round(momentum.z * GRID_SCALE)));
            }
        }
    }
}

// Node velocities, gravity, and the terrain and box walls as boundaries. The velocity is written
// back into the momentum slots (m/s, fixed point) for `slide_g2p`.
@compute @workgroup_size(64, 1, 1)
fn grid_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.nodes.x * params.nodes.y * params.nodes.z;
    let i = gid.x + gid.y * 65535u * 64u;
    if (i >= total) { return; }
    let k = 4u * i;
    let mass = f32(atomicLoad(&grid[k])) / GRID_SCALE;
    if (mass <= 1e-6) { return; }
    var v = vec3<f32>(
        f32(atomicLoad(&grid[k + 1u])),
        f32(atomicLoad(&grid[k + 2u])),
        f32(atomicLoad(&grid[k + 3u])),
    ) / GRID_SCALE / mass;
    v.y -= params.gravity * params.dt;
    let nx = params.nodes.x;
    let nz = params.nodes.z;
    let n = vec3<i32>(i32(i % nx), i32(i / (nx * nz)), i32((i / nx) % nz));
    let pos = vec3<f32>(n) * params.cell;
    // Terrain: inside the ground (or within half a cell of it), velocity into it is removed and
    // Coulomb friction slows the slide along it.
    let ground = ground_at(pos.xz);
    if (pos.y <= ground + 0.5 * params.cell) {
        let e = 0.5 * params.metres_per_texel;
        let gx = ground_at(pos.xz + vec2<f32>(e, 0.0)) - ground_at(pos.xz - vec2<f32>(e, 0.0));
        let gz = ground_at(pos.xz + vec2<f32>(0.0, e)) - ground_at(pos.xz - vec2<f32>(0.0, e));
        let normal = normalize(vec3<f32>(-gx, 2.0 * e, -gz));
        let vn = dot(v, normal);
        if (vn < 0.0) {
            let vt = v - vn * normal;
            let st = length(vt);
            let keep = max(0.0, 1.0 - params.basal_friction * -vn / max(st, 1e-6));
            v = vt * keep;
        }
        if (pos.y < ground - params.cell) {
            v = vec3<f32>(0.0);
        }
    }
    // Box walls (two cells thick): no flow out.
    let last = vec3<i32>(params.nodes) - 3;
    if (n.x < 2 && v.x < 0.0) || (n.x > last.x && v.x > 0.0) { v.x = 0.0; }
    if (n.y < 2 && v.y < 0.0) || (n.y > last.y && v.y > 0.0) { v.y = 0.0; }
    if (n.z < 2 && v.z < 0.0) || (n.z > last.z && v.z > 0.0) { v.z = 0.0; }
    atomicStore(&grid[k + 1u], i32(round(v.x * GRID_SCALE)));
    atomicStore(&grid[k + 2u], i32(round(v.y * GRID_SCALE)));
    atomicStore(&grid[k + 3u], i32(round(v.z * GRID_SCALE)));
}

@compute @workgroup_size(64, 1, 1)
fn slide_g2p(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 65535u * 64u;
    if (i >= particle_count()) { return; }
    var q = particles[i];
    let h = params.cell;
    let x = q.pos.xyz / h;
    let s = stencil(x);
    var v = vec3<f32>(0.0);
    var c = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    for (var a = 0; a < 3; a += 1) {
        for (var b = 0; b < 3; b += 1) {
            for (var cc = 0; cc < 3; cc += 1) {
                let n = s.base + vec3<i32>(a, b, cc);
                if (!inside_grid(n)) { continue; }
                let w = s.w[a].x * s.w[b].y * s.w[cc].z;
                let k = 4u * node_index(n);
                let gv = vec3<f32>(
                    f32(atomicLoad(&grid[k + 1u])),
                    f32(atomicLoad(&grid[k + 2u])),
                    f32(atomicLoad(&grid[k + 3u])),
                ) / GRID_SCALE;
                let dpos = vec3<f32>(n) - x;
                v += w * gv;
                // C = Σ w · 4/h² · v ⊗ (x_i − x_p), with dpos in cells.
                c += mat3x3<f32>(gv * dpos.x, gv * dpos.y, gv * dpos.z) * (w * 4.0 / h);
            }
        }
    }
    var pos = q.pos.xyz + params.dt * v;
    let lo = vec3<f32>(2.0 * h);
    let hi = (vec3<f32>(params.nodes) - 3.0) * h;
    pos = clamp(pos, lo, hi);
    // Particles never end up under the terrain.
    let ground = ground_at(pos.xz);
    if (pos.y < ground) {
        pos.y = ground;
        v.y = max(v.y, 0.0);
    }
    let trace = c[0][0] + c[1][1] + c[2][2];
    let j = clamp(q.pos.w * (1.0 + params.dt * trace), 0.3, 3.0);
    q.pos = vec4<f32>(pos, j);
    q.vel = vec4<f32>(v, q.vel.w);
    let ct = transpose(c);
    q.c0 = vec4<f32>(ct[0], 0.0);
    q.c1 = vec4<f32>(ct[1], 0.0);
    q.c2 = vec4<f32>(ct[2], 0.0);
    particles[i] = q;
}

// The debris on the terrain: each particle's volume over its texel, added to the heights and to
// the record that `slide_unstamp` takes back out.
@compute @workgroup_size(64, 1, 1)
fn slide_stamp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 65535u * 64u;
    if (i >= particle_count()) { return; }
    let q = particles[i];
    let t = vec2<i32>(floor(q.pos.xz / params.metres_per_texel));
    if (any(t < vec2<i32>(0)) || any(t >= vec2<i32>(params.box_texels))) { return; }
    let area = params.metres_per_texel * params.metres_per_texel;
    let quanta = u32(round(q.vel.w / area / params.metres_per_height * HEIGHT_SCALE));
    let p = params.box_origin + t;
    atomicAdd(&height[gidx(p)], quanta);
    atomicAdd(&stamp[u32(t.y) * params.box_texels.x + u32(t.x)], quanta);
}

@compute @workgroup_size(8, 8, 1)
fn slide_settle(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!box_texel(gid)) { return; }
    atomicStore(&stamp[gid.y * params.box_texels.x + gid.x], 0u);
}
