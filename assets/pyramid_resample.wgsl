// KTT pyramid level transition for height and deposition (see `ref/KTT_PYRAMID.md`).
//
// heightfield_resample2 : gauss-upsample the previous level ×2 (filter scale 1.5).
// Update_Upscaled_Fields: height = source(P) + upsampled(mask), mask = eroded − source(P).
// Smooth_Flows          : deep deposits blend toward the previous level's smooth surface.

const HEIGHT_SCALE: f32 = 1000000.0;
const DEPOSITION_SCALE: f32 = 100000.0;
// Houdini gauss filter with filterscale 1.5 (source voxels) ≈ sigma 0.75.
const GAUSS_SIGMA: f32 = 0.75;

struct PyramidResampleParams {
    src_size: vec2<u32>,
    dst_size: vec2<u32>,
    orig_size: vec2<u32>,
    first_level: u32,
    dst_voxel_size: f32,
    detail_smoothing: f32,
}

@group(0) @binding(0) var<storage, read> src_height: array<u32>;
@group(0) @binding(1) var<storage, read> src_deposition: array<i32>;
@group(0) @binding(2) var<storage, read> orig_height: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst_height: array<u32>;
@group(0) @binding(4) var<storage, read_write> dst_deposition: array<i32>;
@group(0) @binding(5) var<uniform> resample: PyramidResampleParams;

fn clamp_texel(q: vec2<i32>, size: vec2<u32>) -> u32 {
    let c = clamp(q, vec2<i32>(0), vec2<i32>(size) - 1);
    return u32(c.y) * size.x + u32(c.x);
}

fn load_src_height(q: vec2<i32>) -> f32 {
    return f32(src_height[clamp_texel(q, resample.src_size)]) / HEIGHT_SCALE;
}

fn load_orig_height(q: vec2<i32>) -> f32 {
    return f32(orig_height[clamp_texel(q, resample.orig_size)]) / HEIGHT_SCALE;
}

// Bilinear sample at a continuous texel-space position (cell centers at i + 0.5), like volumesample.
fn bilinear_orig(pos: vec2<f32>) -> f32 {
    let p = pos - 0.5;
    let i = vec2<i32>(floor(p));
    let w = p - floor(p);
    let a = mix(load_orig_height(i), load_orig_height(i + vec2(1, 0)), w.x);
    let b = mix(load_orig_height(i + vec2(0, 1)), load_orig_height(i + vec2(1, 1)), w.x);
    return mix(a, b, w.y);
}

fn bilinear_src(pos: vec2<f32>) -> f32 {
    let p = pos - 0.5;
    let i = vec2<i32>(floor(p));
    let w = p - floor(p);
    let a = mix(load_src_height(i), load_src_height(i + vec2(1, 0)), w.x);
    let b = mix(load_src_height(i + vec2(0, 1)), load_src_height(i + vec2(1, 1)), w.x);
    return mix(a, b, w.y);
}

@compute @workgroup_size(8, 8, 1)
fn pyramid_upscale_height(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= resample.dst_size.x || gid.y >= resample.dst_size.y) {
        return;
    }
    let dst_idx = gid.y * resample.dst_size.x + gid.x;
    let center = vec2<f32>(gid.xy) + 0.5;
    let dst_size = vec2<f32>(resample.dst_size);
    let orig_size = vec2<f32>(resample.orig_size);
    var h = bilinear_orig(center * orig_size / dst_size);
    var dep = 0.0;

    if (resample.first_level == 0u) {
        let src_size = vec2<f32>(resample.src_size);
        let sp = center * src_size / dst_size;
        let base = vec2<i32>(floor(sp - 0.5));
        var wsum = 0.0;
        var mask = 0.0;
        var dsum = 0.0;
        for (var j = -1; j <= 2; j = j + 1) {
            for (var i = -1; i <= 2; i = i + 1) {
                let q = base + vec2<i32>(i, j);
                let d = vec2<f32>(q) + 0.5 - sp;
                let w = exp(-0.5 * dot(d, d) / (GAUSS_SIGMA * GAUSS_SIGMA));
                let qc = clamp(q, vec2<i32>(0), vec2<i32>(resample.src_size) - 1);
                let q_orig = (vec2<f32>(qc) + 0.5) * orig_size / src_size;
                mask += w * (load_src_height(qc) - bilinear_orig(q_orig));
                dsum += w * f32(src_deposition[clamp_texel(qc, resample.src_size)]) / DEPOSITION_SCALE;
                wsum += w;
            }
        }
        h += mask / wsum;
        dep = max(dsum / wsum, 0.0);

        // Smooth_Flows: lerp(prev_height, height, 0.1^(deposition * smoothing / voxel)).
        let bias = pow(0.1, dep * resample.detail_smoothing / resample.dst_voxel_size);
        h = mix(bilinear_src(sp), h, bias);
    }

    dst_height[dst_idx] = u32(max(h, 0.0) * HEIGHT_SCALE);
    dst_deposition[dst_idx] = i32(dep * DEPOSITION_SCALE);
}
