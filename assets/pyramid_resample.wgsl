// Multi-resolution pyramid resampling for coarse-to-fine erosion.
// Downscale: bilinear sample at destination cell center (Houdini / KTT heightfield resample).

const HEIGHT_SCALE: f32 = 1000000.0;

struct PyramidResampleParams {
    src_size: vec2<u32>,
    dst_size: vec2<u32>,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> src_height: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> dst_height: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> resample_params: PyramidResampleParams;

// Houdini / KTT heightfield resample: bilinear at the destination cell center mapped into source space.
fn sample_src_height_bilinear(src_pos: vec2<f32>, src_sx: u32, src_sy: u32) -> f32 {
    let res = vec2<f32>(f32(src_sx), f32(src_sy));
    var p = src_pos - 0.5;
    p = vec2<f32>(clamp(p.x, 0.0, res.x - 1.00001), clamp(p.y, 0.0, res.y - 1.00001));
    let w = vec2<f32>(p.x - floor(p.x), p.y - floor(p.y));
    let size = i32(src_sx);
    let ix = i32(floor(p.x));
    let iy = i32(floor(p.y));
    let h00 = f32(atomicLoad(&src_height[u32(iy * size + ix)])) / HEIGHT_SCALE;
    let h10 = f32(atomicLoad(&src_height[u32(iy * size + min(ix + 1, size - 1))])) / HEIGHT_SCALE;
    let h01 = f32(atomicLoad(&src_height[u32(min(iy + 1, size - 1) * size + ix)])) / HEIGHT_SCALE;
    let h11 = f32(atomicLoad(&src_height[u32(min(iy + 1, size - 1) * size + min(ix + 1, size - 1))])) / HEIGHT_SCALE;
    return h00 * (1.0 - w.x) * (1.0 - w.y) + h10 * w.x * (1.0 - w.y) + h01 * (1.0 - w.x) * w.y + h11 * w.x * w.y;
}

@compute @workgroup_size(8, 8, 1)
fn pyramid_downscale(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= resample_params.dst_size.x || gid.y >= resample_params.dst_size.y) {
        return;
    }
    let dst_x = gid.x;
    let dst_y = gid.y;
    let src_sx = resample_params.src_size.x;
    let src_sy = resample_params.src_size.y;
    let dst_sx = resample_params.dst_size.x;
    let dst_sy = resample_params.dst_size.y;
    let scale_x = f32(src_sx) / f32(dst_sx);
    let scale_y = f32(src_sy) / f32(dst_sy);
    let src_pos = vec2<f32>((f32(dst_x) + 0.5) * scale_x, (f32(dst_y) + 0.5) * scale_y);
    let h = sample_src_height_bilinear(src_pos, src_sx, src_sy);
    let dst_idx = dst_y * dst_sx + dst_x;
    atomicStore(&dst_height[dst_idx], u32(h * HEIGHT_SCALE));
}
