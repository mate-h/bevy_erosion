// KTT pyramid level transition for the source masks (Update_Upscaled_Fields):
//   __Softness_Mask       = clamp(source softness(P), 0, 1)
//   __Outflow             = clamp(expand-blurred source outflow(P), 0, 1)
//   __Outflow_Accumulation = clamp(max(__Outflow, upsampled previous accumulation), 0, 1)
// The source outflow goes through heightfield_blur1 ("expand", radius VoxelSize / 4, 10 iterations)
// at every level, so thin outflows survive on coarse grids; approximated here by a distance-weighted
// max over the level cell's source footprint.

struct PyramidResampleParams {
    src_size: vec2<u32>,
    dst_size: vec2<u32>,
    orig_size: vec2<u32>,
    first_level: u32,
    dst_voxel_size: f32,
    detail_smoothing: f32,
    outflow_accumulate: u32,
}

@group(0) @binding(0) var<storage, read> src_outflow_acc: array<f32>;
@group(0) @binding(1) var<storage, read> orig_softness: array<f32>;
@group(0) @binding(2) var<storage, read> orig_outflow: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst_softness: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst_outflow: array<f32>;
@group(0) @binding(5) var<storage, read_write> dst_outflow_acc: array<f32>;
@group(0) @binding(6) var<uniform> resample: PyramidResampleParams;

fn clamp_texel(q: vec2<i32>, size: vec2<u32>) -> u32 {
    let c = clamp(q, vec2<i32>(0), vec2<i32>(size) - 1);
    return u32(c.y) * size.x + u32(c.x);
}

fn bilinear_orig_softness(pos: vec2<f32>) -> f32 {
    let p = pos - 0.5;
    let i = vec2<i32>(floor(p));
    let w = p - floor(p);
    let s = resample.orig_size;
    let a = mix(orig_softness[clamp_texel(i, s)], orig_softness[clamp_texel(i + vec2(1, 0), s)], w.x);
    let b = mix(orig_softness[clamp_texel(i + vec2(0, 1), s)], orig_softness[clamp_texel(i + vec2(1, 1), s)], w.x);
    return mix(a, b, w.y);
}

fn bilinear_src_acc(pos: vec2<f32>) -> f32 {
    let p = pos - 0.5;
    let i = vec2<i32>(floor(p));
    let w = p - floor(p);
    let s = resample.src_size;
    let a = mix(src_outflow_acc[clamp_texel(i, s)], src_outflow_acc[clamp_texel(i + vec2(1, 0), s)], w.x);
    let b = mix(src_outflow_acc[clamp_texel(i + vec2(0, 1), s)], src_outflow_acc[clamp_texel(i + vec2(1, 1), s)], w.x);
    return mix(a, b, w.y);
}

// Expand blur: values only grow. Source texels inside the level cell count fully, then the weight
// falls off over about one level voxel outside it.
fn expanded_orig_outflow(pos: vec2<f32>, ratio: f32) -> f32 {
    let r = i32(ceil(ratio));
    let center = vec2<i32>(floor(pos));
    let core = 0.5 * ratio;
    let falloff = ratio + 1.0;
    var o = 0.0;
    for (var j = -r; j <= r; j = j + 1) {
        for (var i = -r; i <= r; i = i + 1) {
            let q = center + vec2<i32>(i, j);
            let d = length(vec2<f32>(q) + 0.5 - pos);
            let w = clamp((falloff - d) / (falloff - core), 0.0, 1.0);
            o = max(o, orig_outflow[clamp_texel(q, resample.orig_size)] * w);
        }
    }
    return o;
}

@compute @workgroup_size(8, 8, 1)
fn pyramid_upscale_masks(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= resample.dst_size.x || gid.y >= resample.dst_size.y) {
        return;
    }
    let dst_idx = gid.y * resample.dst_size.x + gid.x;
    let center = vec2<f32>(gid.xy) + 0.5;
    let dst_size = vec2<f32>(resample.dst_size);
    let orig_pos = center * vec2<f32>(resample.orig_size) / dst_size;
    let ratio = f32(resample.orig_size.x) / dst_size.x;

    let outflow = clamp(expanded_orig_outflow(orig_pos, ratio), 0.0, 1.0);
    var acc = outflow;
    if (resample.outflow_accumulate != 0u && resample.first_level == 0u) {
        let src_pos = center * vec2<f32>(resample.src_size) / dst_size;
        acc = clamp(max(outflow, bilinear_src_acc(src_pos)), 0.0, 1.0);
    }

    dst_softness[dst_idx] = clamp(bilinear_orig_softness(orig_pos), 0.0, 1.0);
    dst_outflow[dst_idx] = outflow;
    dst_outflow_acc[dst_idx] = acc;
}
