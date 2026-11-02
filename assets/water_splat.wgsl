// KTT trails and the water (Phases 6b and 6d of docs/WATER_PLAN.md), once per water frame on the
// base map against the finest pyramid level:
//   * water_splat (first): the water rain trails left along their path and the sediment of
//     trails that ended in water (`erosion.wgsl` `splat`) join the water's depth and suspended
//     load. Over the ocean they leave the map, as KTT's outflow does.
//   * water_record (last): the height change the water made this frame (Exner, the capacity
//     model, slumping, Clear Water) goes into KTT's deposition and erosion buffers, so loose
//     sediment the water dropped erodes like KTT's own and the analysis views show both.

const HEIGHT_SCALE: f32 = 1000000.0;
// Must match `erosion.wgsl`.
const SPLAT_SCALE: f32 = 100000000.0;
const DEPOSITION_SCALE: f32 = 100000.0;
const EROSION_SCALE: f32 = 100000.0;
const CELLS_PER_HEIGHT: f32 = 100.0;

// Must match `MAX_WATER_SOURCES` in `lib.rs`.
const MAX_SOURCES: u32 = 17u;

// Same layout as in `water.wgsl`; only the grid and the sea level are read here.
struct WaterParams {
    size: vec2<u32>,
    pad_offset: vec2<u32>,
    grid_size: vec2<u32>,
    clear: u32,
    evaporation: f32,
    capacity: f32,
    sea_level: f32,
    rain: f32,
    step_seconds: f32,
    bed_load: u32,
    source_count: u32,
    sources: array<vec4<f32>, MAX_SOURCES>,
}

@group(0) @binding(0) var<storage, read_write> splat: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> depth: array<f32>;
@group(0) @binding(2) var<storage, read_write> sediment: array<f32>;
@group(0) @binding(3) var<storage, read_write> height: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> deposition: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> erosion_buffer: array<atomic<u32>>;
// `water_morpho.wgsl` state; the fourth float per cell is this pass's: the height when the
// frame began while the water runs, and the change not yet recorded between frames.
@group(0) @binding(6) var<storage, read_write> state: array<f32>;
@group(0) @binding(7) var<uniform> params: WaterParams;
// bevy_fft solver state (y = depth in cells).
@group(0) @binding(8) var swe_water: texture_storage_2d<rgba32float, read_write>;

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

fn terrain(g: u32) -> f32 {
    return f32(atomicLoad(&height[g])) / HEIGHT_SCALE;
}

@compute @workgroup_size(8, 8, 1)
fn water_splat(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p)) { return; }
    let i = widx(p);
    let g = gidx(p);
    let water = f32(atomicExchange(&splat[2u * g], 0u)) / SPLAT_SCALE;
    let load = f32(atomicExchange(&splat[2u * g + 1u], 0u)) / SPLAT_SCALE;
    let ground = terrain(g);
    let ocean = params.sea_level > 0.0 && ground < params.sea_level;
    if (!ocean) {
        if (water > 0.0) {
            depth[i] += water;
            var w = textureLoad(swe_water, p);
            w.y += water * CELLS_PER_HEIGHT;
            textureStore(swe_water, p, w);
        }
        sediment[i] += load;
    }
    // Baseline for `water_record`, less what is still unrecorded.
    state[4u * i + 3u] = ground - state[4u * i + 3u];
}

@compute @workgroup_size(8, 8, 1)
fn water_record(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p)) { return; }
    let i = widx(p);
    let g = gidx(p);
    let dh = terrain(g) - state[4u * i + 3u];
    // Whole deposition quanta only; the rest waits for the next frame.
    let quanta = i32(round(dh * DEPOSITION_SCALE));
    if (quanta > 0) {
        atomicAdd(&deposition[g], quanta);
    } else if (quanta < 0) {
        // Cuts take loose sediment first, as KTT's trails do.
        let loose = max(atomicLoad(&deposition[g]), 0);
        atomicSub(&deposition[g], min(-quanta, loose));
        atomicAdd(&erosion_buffer[g], u32(-quanta) * u32(EROSION_SCALE / DEPOSITION_SCALE));
    }
    state[4u * i + 3u] = dh - f32(quanta) / DEPOSITION_SCALE;
}
