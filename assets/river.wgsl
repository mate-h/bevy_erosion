// River brush beds (`src/river.rs`) on the finest work set.
//
// `river_carve` lowers the terrain to the brushed rivers' beds and grades their banks, then
// writes each cell's armour band: the carved bed, give or take the part of the channel depth
// the armour leaves to the water. `river_armour` runs between `water_erode` and `water_apply` and
// keeps the water's change (dissolving, settling and wet slumping) inside that band, so the steps
// stay up and the pools stay open; what it holds back goes back into the suspended sediment.
// Bed load keeps to the same band in `water_morpho.wgsl`. Off the rivers the band is (0, 0),
// which holds nothing.

const HEIGHT_SCALE: f32 = 1000000.0;
// Must match `river::NO_RIVER`.
const NO_RIVER: f32 = 1.0e6;

// The start of `water.wgsl`'s `WaterParams`, whose buffer this binds.
struct RiverParams {
    size: vec2<u32>,
    pad_offset: vec2<u32>,
    grid_size: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> height: array<atomic<u32>>;
// Per texel: x = target height (NO_RIVER for none), y = blend back toward the ground (0 in the
// channel), z = armour, w = depth the armour protects.
@group(0) @binding(1) var<storage, read> river: array<vec4<f32>>;
// Armour band per texel: x = floor, y = ceiling (0 when off a river).
@group(0) @binding(2) var<storage, read_write> river_band: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> sediment: array<f32>;
// The height change `water_erode` left for `water_apply`.
@group(0) @binding(4) var<storage, read_write> sediment_tmp: array<f32>;
@group(0) @binding(5) var<uniform> params: RiverParams;

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

@compute @workgroup_size(8, 8, 1)
fn river_carve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p)) { return; }
    let i = widx(p);
    let r = river[i];
    if (r.x >= 0.5 * NO_RIVER) {
        river_band[i] = vec2<f32>(0.0);
        return;
    }
    let ground = terrain(p);
    // Only ever lower: the bed in the channel, a blend back to the ground across the banks.
    let z = max(min(ground, mix(r.x, ground, r.y)), 0.0);
    atomicStore(&height[gidx(p)], u32(round(z * HEIGHT_SCALE)));
    let give = (1.0 - r.z) * r.w;
    river_band[i] = select(vec2<f32>(0.0), vec2<f32>(max(z - give, 0.0), z + give), r.z > 0.0);
}

@compute @workgroup_size(8, 8, 1)
fn river_armour(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(gid.xy);
    if (!inside(p)) { return; }
    let i = widx(p);
    let band = river_band[i];
    let dh = sediment_tmp[i];
    if (band.y <= 0.0 || dh == 0.0) { return; }
    // Only stop a change at the band, never pull the ground into it.
    let ground = terrain(p);
    let kept = clamp(dh, min(band.x - ground, 0.0), max(band.y - ground, 0.0));
    sediment_tmp[i] = kept;
    sediment[i] = max(sediment[i] + dh - kept, 0.0);
}
