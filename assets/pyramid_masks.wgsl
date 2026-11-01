// KTT pyramid level transition for the source masks and layers (Update_Upscaled_Fields):
//   __Softness_Mask        = clamp(source softness(P), 0, 1)
//   __Outflow              = clamp(expand-blurred source outflow(P), 0, 1)
//   __Erode_Mask           = clamp(source erode mask(P), 0, 1)
//   __Uplift               = source uplift map(P)
//   __Outflow_Accumulation = clamp(max(__Outflow, upsampled previous accumulation), 0, 1)
//   __Bedrock              = source bedrock(P)
//   __Age                  = gauss-upsampled previous level (heightfield_resample2), 0 on the first level
// The source outflow goes through heightfield_blur1 ("expand", radius VoxelSize / 4, 10 iterations)
// at every level, so thin outflows survive on coarse grids; approximated here by a distance-weighted
// max over the level cell's source footprint.

// Houdini gauss filter with filterscale 1.5 (source voxels) ≈ sigma 0.75.
const GAUSS_SIGMA: f32 = 0.75;

struct PyramidResampleParams {
    src_size: vec2<u32>,
    dst_size: vec2<u32>,
    orig_size: vec2<u32>,
    first_level: u32,
    dst_voxel_size: f32,
    detail_smoothing: f32,
    outflow_accumulate: u32,
    use_bedrock: u32,
    detail_scale: f32,
    granularity_amount: f32,
    granularity_scale: f32,
    granularity_strength: f32,
}

@group(0) @binding(0) var<storage, read> src_outflow_acc: array<f32>;
@group(0) @binding(1) var<storage, read> src_layers: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> orig_masks: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> orig_layers: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> dst_masks: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> dst_outflow_acc: array<f32>;
@group(0) @binding(6) var<storage, read_write> dst_layers: array<vec2<f32>>;
@group(0) @binding(7) var<uniform> resample: PyramidResampleParams;

fn clamp_texel(q: vec2<i32>, size: vec2<u32>) -> u32 {
    let c = clamp(q, vec2<i32>(0), vec2<i32>(size) - 1);
    return u32(c.y) * size.x + u32(c.x);
}

// Bilinear sample at a continuous texel-space position (cell centers at i + 0.5), like volumesample.
fn bilinear_orig_masks(pos: vec2<f32>) -> vec4<f32> {
    let p = pos - 0.5;
    let i = vec2<i32>(floor(p));
    let w = p - floor(p);
    let s = resample.orig_size;
    let a = mix(orig_masks[clamp_texel(i, s)], orig_masks[clamp_texel(i + vec2(1, 0), s)], w.x);
    let b = mix(orig_masks[clamp_texel(i + vec2(0, 1), s)], orig_masks[clamp_texel(i + vec2(1, 1), s)], w.x);
    return mix(a, b, w.y);
}

fn bilinear_orig_bedrock(pos: vec2<f32>) -> f32 {
    let p = pos - 0.5;
    let i = vec2<i32>(floor(p));
    let w = p - floor(p);
    let s = resample.orig_size;
    let a = mix(orig_layers[clamp_texel(i, s)].x, orig_layers[clamp_texel(i + vec2(1, 0), s)].x, w.x);
    let b = mix(orig_layers[clamp_texel(i + vec2(0, 1), s)].x, orig_layers[clamp_texel(i + vec2(1, 1), s)].x, w.x);
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

fn gauss_src_age(sp: vec2<f32>) -> f32 {
    let base = vec2<i32>(floor(sp - 0.5));
    var wsum = 0.0;
    var age = 0.0;
    for (var j = -1; j <= 2; j = j + 1) {
        for (var i = -1; i <= 2; i = i + 1) {
            let q = base + vec2<i32>(i, j);
            let d = vec2<f32>(q) + 0.5 - sp;
            let w = exp(-0.5 * dot(d, d) / (GAUSS_SIGMA * GAUSS_SIGMA));
            age += w * src_layers[clamp_texel(q, resample.src_size)].y;
            wsum += w;
        }
    }
    return age / wsum;
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
            o = max(o, orig_masks[clamp_texel(q, resample.orig_size)].y * w);
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
    let src_pos = center * vec2<f32>(resample.src_size) / dst_size;
    let ratio = f32(resample.orig_size.x) / dst_size.x;

    let outflow = clamp(expanded_orig_outflow(orig_pos, ratio), 0.0, 1.0);
    var acc = outflow;
    var age = 0.0;
    if (resample.first_level == 0u) {
        if (resample.outflow_accumulate != 0u) {
            acc = clamp(max(outflow, bilinear_src_acc(src_pos)), 0.0, 1.0);
        }
        age = gauss_src_age(src_pos);
    }

    let m = bilinear_orig_masks(orig_pos);
    dst_masks[dst_idx] = vec4<f32>(clamp(m.x, 0.0, 1.0), outflow, clamp(m.z, 0.0, 1.0), m.w);
    dst_outflow_acc[dst_idx] = acc;
    dst_layers[dst_idx] = vec2<f32>(bilinear_orig_bedrock(orig_pos), age);
}
