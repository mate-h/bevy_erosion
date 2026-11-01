// KTT pyramid level transition for river fields (`__m` is carried through heightfield_resample2)
// and per-level analysis buffers (`__FlowLines` is reset to 0 in Update_Upscaled_Fields).

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

@group(0) @binding(0) var<storage, read> src_mx: array<f32>;
@group(0) @binding(1) var<storage, read> src_my: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst_mx: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst_my: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst_flow: array<u32>;
@group(0) @binding(5) var<storage, read_write> dst_erosion: array<u32>;
@group(0) @binding(6) var<uniform> resample: PyramidResampleParams;

@compute @workgroup_size(8, 8, 1)
fn pyramid_upscale_rivers(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= resample.dst_size.x || gid.y >= resample.dst_size.y) {
        return;
    }
    let dst_idx = gid.y * resample.dst_size.x + gid.x;
    var m = vec2<f32>(0.0);

    if (resample.first_level == 0u) {
        let src_size = vec2<f32>(resample.src_size);
        let sp = (vec2<f32>(gid.xy) + 0.5) * src_size / vec2<f32>(resample.dst_size);
        let base = vec2<i32>(floor(sp - 0.5));
        var wsum = 0.0;
        for (var j = -1; j <= 2; j = j + 1) {
            for (var i = -1; i <= 2; i = i + 1) {
                let q = base + vec2<i32>(i, j);
                let d = vec2<f32>(q) + 0.5 - sp;
                let w = exp(-0.5 * dot(d, d) / (GAUSS_SIGMA * GAUSS_SIGMA));
                let qc = clamp(q, vec2<i32>(0), vec2<i32>(resample.src_size) - 1);
                let si = u32(qc.y) * resample.src_size.x + u32(qc.x);
                m += w * vec2<f32>(src_mx[si], src_my[si]);
                wsum += w;
            }
        }
        m /= wsum;
    }

    dst_mx[dst_idx] = m.x;
    dst_my[dst_idx] = m.y;
    dst_flow[dst_idx] = 0u;
    dst_erosion[dst_idx] = 0u;
}
