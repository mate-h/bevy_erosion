// Morphodynamics of the water (Phases 4b, 4c and 6a of docs/WATER_PLAN.md): the shared
// discharge field, helical flow in bends, and bed load moved by Exner on the cell faces.
//
// Runs on the base map against the finest pyramid level's heights, after the water steps of
// `water.wgsl`. Units follow the solver: depths and the bed in cells (height × CELLS_PER_HEIGHT),
// velocities in cells per second, so the Shields number and the bed load are in grid units.
//   * morpho_discharge (per frame): time-averaged unit discharge q = h·u and depth, and KTT's
//     river momentum scaled to the same units.
//   * morpho_steer (per frame, KTT rivers on): KTT's river momentum is raised to the water's
//     discharge where the water carries more, so new trails follow the rivers that exist.
//   * morpho_helical (per frame): helical flow intensity I_s, advected along the flow and relaxed
//     toward h·κ (κ = streamline curvature).
//   * morpho_flux (per step): bed load on the +x and +y faces of every cell (Shields, Meyer-Peter
//     Müller), turned toward the inner bank by the helical flow and down the cross-slope by
//     Talmon's gravity term.
//   * morpho_exner (per step): Δz = −mor_fac · Δt · div(q_b), accumulated in a float and written
//     to the fixed-point heights in whole quanta.

const HEIGHT_SCALE: f32 = 1000000.0;
const CELLS_PER_HEIGHT: f32 = 100.0;
// (ρ_s − ρ) / ρ: `grain` is R·D in cells.
const SUBMERGED_DENSITY: f32 = 1.65;
// Most of the shallower side's depth one face may move per step (keeps a large morphological
// factor from emptying a cell in one step; four faces can take at most 40%).
const MAX_FACE_CUT: f32 = 0.1;
// Largest helical flow deflection of the bed load (radians) and streamline curvature (1/texel).
const MAX_DEFLECTION: f32 = 0.5;
const MAX_CURVATURE: f32 = 0.3;

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
    // Manning n in cells (n · dx^(1/3)).
    manning_n: f32,
    // Dry and fully wet depths (height units), as in `water.wgsl`.
    wet_depth: f32,
    full_depth: f32,
    // Speed (cells per second) below which water has no direction for the bed load or curvature.
    min_speed: f32,
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
    // KTT keeps a bedrock layer that the bed may not be cut below.
    use_bedrock: u32,
    // Critical Shields multiplier on the hardest rock (softness mask 0) is 1 + rock_hardness.
    rock_hardness: f32,
    // Boulders: R·D (cells) and bed load coefficient (cells² per second) as for the fines.
    boulder_grain: f32,
    boulder_k: f32,
    // Substrate boulder fraction: `boulder_base`, plus `boulder_rock` on the hardest rock, plus
    // `boulder_scree` on slopes past the talus (tangent 0.6 to 1).
    boulder_base: f32,
    boulder_rock: f32,
    boulder_scree: f32,
    // γ in θ_c,i = θ_c (D_i / D_m)^(−γ).
    hiding: f32,
    // Active layer thickness in mean grain diameters.
    active_layer: f32,
    // Fraction of the boulder load that wears into fines crossing one face (Sternberg).
    abrasion: f32,
    // 1 for one frame after the boulder settings changed: the surface layer starts again from
    // the substrate.
    reset_boulders: u32,
}

@group(0) @binding(0) var<storage, read_write> height: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> depth: array<f32>;
// Texels per solver step.
@group(0) @binding(2) var<storage, read> velocity: array<vec2<f32>>;
// x, y = averaged unit discharge h·u (height units × texels per second), z = averaged depth,
// w = KTT river discharge in the same units.
@group(0) @binding(3) var<storage, read_write> discharge: array<vec4<f32>>;
// KTT river momentum of the finest work set, x and y per cell (f32 bits; see `erosion.wgsl`).
@group(0) @binding(4) var<storage, read_write> m: array<u32>;
// Bed load through the +x and +y faces of each cell (cells² per second): x, y fines, z, w
// boulders.
@group(0) @binding(5) var<storage, read_write> bed_load: array<vec4<f32>>;
// `STATE_STRIDE` floats per cell: helical intensity in two slots (ping-pong), the bed change not
// yet written to the heights (height units), the height when the water frame began
// (`water_splat.wgsl`), and the surface and substrate boulder fractions plus one (0 until the
// first water frame sets them from the terrain).
@group(0) @binding(6) var<storage, read_write> state: array<f32>;
@group(0) @binding(7) var<uniform> params: MorphoParams;
// KTT's per-cell masks (x = softness, 1 where unused) and layers (x = bedrock) of the finest work
// set. Only the bed load kernels bind the layers and the river band, and they leave out
// `discharge` and `m`, so each pipeline stays within eight storage buffers.
@group(0) @binding(8) var<storage, read> masks: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read> layers: array<vec2<f32>>;
// Armour band around the river brush's beds (`river.wgsl`), per texel: x = floor, y = ceiling;
// (0, 0) holds nothing.
@group(0) @binding(10) var<storage, read> river_band: array<vec2<f32>>;

// Floats per cell in `state` (must match `MORPHO_STATE_STRIDE` in `lib.rs`).
const STATE_STRIDE: u32 = 8u;
// Slots of the surface and substrate boulder fractions (plus one).
const BOULDER_SLOT: u32 = 4u;
const SUBSTRATE_SLOT: u32 = 5u;

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

// Boulder fraction of the material under `p` from the terrain as it is: more on hard rock and on
// steep slopes.
fn initial_boulders(p: vec2<i32>) -> f32 {
    let soft = clamp(masks[gidx(p)].x, 0.0, 1.0);
    let gx = terrain(clamp_map(p + DIRS[1])) - terrain(clamp_map(p + DIRS[0]));
    let gy = terrain(clamp_map(p + DIRS[3])) - terrain(clamp_map(p + DIRS[2]));
    let slope = 0.5 * length(vec2<f32>(gx, gy)) * CELLS_PER_HEIGHT;
    let scree = smoothstep(0.6, 1.0, slope);
    return clamp(
        params.boulder_base + params.boulder_rock * (1.0 - soft) + params.boulder_scree * scree,
        0.0,
        1.0,
    );
}

// Boulder fraction of the substrate under `p`, as the first water frame found it: the material of
// a riser stays scree after the flow rounds it.
fn substrate_boulders(p: vec2<i32>) -> f32 {
    let stored = state[STATE_STRIDE * widx(p) + SUBSTRATE_SLOT];
    return clamp(select(initial_boulders(p), stored - 1.0, stored >= 0.5), 0.0, 1.0);
}

// Boulder fraction of the active layer at `p`. It is set from the substrate on the first water
// frame and from then on belongs to the bed, so a riser that the flow rounds below the talus
// keeps the boulders it had.
fn surface_boulders(p: vec2<i32>) -> f32 {
    let stored = state[STATE_STRIDE * widx(p) + BOULDER_SLOT];
    return clamp(select(substrate_boulders(p), stored - 1.0, stored >= 0.5), 0.0, 1.0);
}

fn helical(p: vec2<i32>, slot: u32) -> f32 {
    return state[STATE_STRIDE * widx(p) + slot];
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
    var river = bitcast<f32>(m[2u * g]);
    if (params.ktt_vector != 0u) {
        river = length(vec2<f32>(river, bitcast<f32>(m[2u * g + 1u])));
    }
    d.w = params.ktt_scale * max(river, 0.0);
    discharge[i] = d;
    let slot = STATE_STRIDE * i + BOULDER_SLOT;
    if (params.reset_boulders != 0u || state[slot] < 0.5) {
        let initial = initial_boulders(p) + 1.0;
        state[slot] = initial;
        state[STATE_STRIDE * i + SUBSTRATE_SLOT] = initial;
    }
}

@compute @workgroup_size(8, 8, 1)
fn morpho_steer(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p)) { return; }
    let target_m = discharge[widx(p)].xy / max(params.ktt_scale, 1e-9);
    let g = gidx(p);
    let river = vec2<f32>(bitcast<f32>(m[2u * g]), bitcast<f32>(m[2u * g + 1u]));
    if (params.ktt_vector != 0u) {
        // Momentum coherence: trails turn along the vector.
        if (length(target_m) > length(river)) {
            m[2u * g] = bitcast<u32>(target_m.x);
            m[2u * g + 1u] = bitcast<u32>(target_m.y);
        }
    } else {
        // Scalar water accumulation in x only.
        m[2u * g] = bitcast<u32>(max(river.x, length(target_m)));
    }
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
    if (speed < params.min_speed) { return 0.0; }
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
    if (h < params.wet_depth) {
        state[STATE_STRIDE * i + dst] = 0.0;
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
    state[STATE_STRIDE * i + dst] = clamp(s, -1.0, 1.0);
}

// Bed load (cells² per second) through the face from `a` to its neighbour `a + e`, along `e`:
// x fines, y boulders. Its size and direction come from the upwind cell (first-order upwind
// Exner): averaging the two cells instead is a central scheme, which grows two-texel
// oscillations in the bed.
fn face_load(a: vec2<i32>, e: vec2<i32>) -> vec2<f32> {
    let b = a + e;
    if (!inside(b)) { return vec2<f32>(0.0); }
    let ha = depth[widx(a)];
    let hb = depth[widx(b)];
    if (min(ha, hb) < params.wet_depth) { return vec2<f32>(0.0); }
    let ua = flow(a);
    let ub = flow(b);
    let downwind = dot(ua + ub, vec2<f32>(e)) >= 0.0;
    let up = select(b, a, downwind);
    let h = select(hb, ha, downwind);
    let u = select(ub, ua, downwind);
    // Thin sheet flow keeps the capacity model of `water_erode`.
    let w = smoothstep(0.5 * params.full_depth, params.full_depth, h);
    if (w <= 0.0) { return vec2<f32>(0.0); }
    let speed = length(u);
    if (speed < params.min_speed) { return vec2<f32>(0.0); }
    let h_cells = h * CELLS_PER_HEIGHT;
    let n = params.manning_n;
    // Bed shear over the submerged weight, per grain size: θ_i = τ* / (R D_i).
    let tau = n * n * speed * speed / pow(h_cells, 1.0 / 3.0);
    let theta = tau / vec2<f32>(max(params.grain, 1e-6), max(params.boulder_grain, 1e-6));
    // Hard rock (KTT's softness mask) takes a stronger flow to move, so steps and pools lock
    // onto it instead of cutting evenly everywhere. Hiding: grains finer than the mean are
    // sheltered by the coarser, and coarser ones stick out into the flow.
    let soft = clamp(masks[gidx(up)].x, 0.0, 1.0);
    let fb = surface_boulders(up);
    let fractions = vec2<f32>(1.0 - fb, fb);
    let d = vec2<f32>(params.grain, params.boulder_grain);
    let d_mean = dot(fractions, d);
    let critical = params.critical_shields * (1.0 + params.rock_hardness * (1.0 - soft))
        * pow(max(d, vec2<f32>(1e-6)) / max(d_mean, 1e-6), vec2<f32>(-params.hiding));
    let excess = max(theta - critical, vec2<f32>(0.0));
    if (all(excess <= vec2<f32>(0.0))) { return vec2<f32>(0.0); }
    let q = fractions * vec2<f32>(params.bed_load_k, params.boulder_k) * pow(excess, vec2<f32>(1.5)) * w;

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
    let pull = q / (max(params.slope_effect, 1e-3) * sqrt(max(theta, vec2<f32>(1e-6))))
        * dot(grad, across);
    let qn = q * dot(dir, vec2<f32>(e)) - pull * dot(across, vec2<f32>(e));

    let per_step = max(params.mor_fac * params.step_seconds, 1e-9);
    var most = MAX_FACE_CUT * min(ha, hb) * CELLS_PER_HEIGHT / per_step;
    let total = qn.x + qn.y;
    if (params.use_bedrock != 0u) {
        // The cell the load leaves keeps a quarter of its loose layer above the bedrock per
        // face, so no step cuts below it (and the load is still conserved).
        let source = select(b, a, total > 0.0);
        let g = gidx(source);
        let loose = max(terrain(source) - layers[g].x, 0.0);
        most = min(most, 0.25 * loose * CELLS_PER_HEIGHT / per_step);
    }
    // The same for the armour band of a brushed river bed: the cell the load leaves keeps above
    // its floor, and the cell it enters stays below its ceiling.
    let source = select(b, a, total > 0.0);
    let target_cell = select(a, b, total > 0.0);
    let leaving = river_band[widx(source)];
    if (leaving.y > 0.0) {
        let loose = max(terrain(source) - leaving.x, 0.0);
        most = min(most, 0.25 * loose * CELLS_PER_HEIGHT / per_step);
    }
    let entering = river_band[widx(target_cell)];
    if (entering.y > 0.0) {
        let room = max(entering.y - terrain(target_cell), 0.0);
        most = min(most, 0.25 * room * CELLS_PER_HEIGHT / per_step);
    }
    // Both fractions scale down together, so the limit keeps their mix.
    let size = abs(qn.x) + abs(qn.y);
    return qn * min(1.0, most / max(size, 1e-12));
}

@compute @workgroup_size(8, 8, 1)
fn morpho_flux(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p)) { return; }
    let x = face_load(p, DIRS[1]);
    let y = face_load(p, DIRS[3]);
    bed_load[widx(p)] = vec4<f32>(x.x, y.x, x.y, y.y);
}

@compute @workgroup_size(8, 8, 1)
fn morpho_exner(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p)) { return; }
    let i = widx(p);
    let out = bed_load[i];
    // Divergence per fraction: x fines, y boulders.
    var div = out.xz + out.yw;
    if (p.x > 0) { div -= bed_load[widx(p + DIRS[0])].xz; }
    if (p.y > 0) { div -= bed_load[widx(p + DIRS[2])].yw; }
    if (is_sink(p)) { return; }
    // Abrasion (Sternberg): of the boulders arriving through a face, a part has worn into fines
    // on the way across the texel. The total is kept.
    var arriving = max(-out.z, 0.0) + max(-out.w, 0.0);
    if (p.x > 0) { arriving += max(bed_load[widx(p + DIRS[0])].z, 0.0); }
    if (p.y > 0) { arriving += max(bed_load[widx(p + DIRS[2])].w, 0.0); }
    let worn = params.abrasion * arriving;
    div += vec2<f32>(-worn, worn);
    let change = -params.mor_fac * params.step_seconds * div;
    let dz = (change.x + change.y) / CELLS_PER_HEIGHT;
    // Active layer (Hirano): its boulder fraction gains the boulders that arrive less those
    // leaving, and trades with what lies below as the bed moves through it: its own mix when
    // the bed rises, the substrate's when it cuts.
    let substrate = substrate_boulders(p);
    let fb = surface_boulders(p);
    let d_mean = mix(params.grain, params.boulder_grain, fb) / SUBMERGED_DENSITY;
    let layer = max(params.active_layer * d_mean, 1e-4);
    let traded = select(substrate, fb, change.x + change.y > 0.0);
    let fb_new = clamp(fb + (change.y - traded * (change.x + change.y)) / layer, 0.0, 1.0);
    state[STATE_STRIDE * i + BOULDER_SLOT] = fb_new + 1.0;
    var pending = state[STATE_STRIDE * i + 2u] + dz;
    let quanta = i32(round(pending * HEIGHT_SCALE));
    let g = gidx(p);
    var h = i32(atomicLoad(&height[g]));
    let old = h;
    if (quanta != 0) {
        h = max(h + quanta, 0);
        pending -= f32(quanta) / HEIGHT_SCALE;
    }
    if (params.use_bedrock != 0u) {
        // Bed load itself stops at the bedrock (`face_load`), but wet slumping can still pull
        // the face of a step below it; restore it, as KTT's `write_back` does.
        h = max(h, i32(ceil(layers[g].x * HEIGHT_SCALE)));
    }
    let band = river_band[i];
    if (band.y > 0.0) {
        // Likewise for the armour band, but never past where the bed already was.
        h = max(h, min(i32(ceil(band.x * HEIGHT_SCALE)), old));
        h = min(h, max(i32(floor(band.y * HEIGHT_SCALE)), old));
    }
    if (h != old) {
        atomicStore(&height[g], u32(h));
    }
    state[STATE_STRIDE * i + 2u] = pending;
}

// Debug views (`TerrainView`), on the water grid.
// x, y = averaged unit discharge, z = averaged depth, w = KTT river discharge.
@group(1) @binding(0) var discharge_out: texture_storage_2d<rgba16float, write>;
// x = bed load through the cell (cells² per second), y = helical intensity, z = bed change not yet
// written to the heights (height units).
@group(1) @binding(1) var bed_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn morpho_debug(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p)) { return; }
    let i = widx(p);
    textureStore(discharge_out, p, discharge[i]);
    // Face fluxes averaged onto the cell centre.
    let out = bed_load[i].xy + bed_load[i].zw;
    let left = bed_load[widx(clamp_map(p + DIRS[0]))];
    let below = bed_load[widx(clamp_map(p + DIRS[2]))];
    let qb = 0.5 * (out + vec2<f32>(left.x + left.z, below.y + below.w));
    // The helical pass this frame wrote the slot the parity does not name.
    let helical_now = state[STATE_STRIDE * i + 1u - params.parity];
    textureStore(
        bed_out,
        p,
        vec4<f32>(length(qb), helical_now, state[STATE_STRIDE * i + 2u], surface_boulders(p)),
    );
}
