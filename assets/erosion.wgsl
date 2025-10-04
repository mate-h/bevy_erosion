// Erosion compute for a heightmap using droplet particles.
// Pass 1: init_fbm - initialize heightmap with FBM noise into storage texture.
// Pass 2: erode - run droplets using structured buffers (map as storage texture).
// Pass 3: blit_to_texture - optional, copy height to rgba for display.

@group(0) @binding(0) var<storage, read_write> height: array<f32>;

// For init_fbm we only need output; height_in is ignored.

struct ErodeParams {
    map_size: vec2<u32>,
    max_lifetime: u32,
    border_size: u32,
    inertia: f32,
    sediment_capacity_factor: f32,
    min_sediment_capacity: f32,
    deposit_speed: f32,
    erode_speed: f32,
    evaporate_speed: f32,
    gravity: f32,
    start_speed: f32,
    start_water: f32,
    brush_length: u32,
    num_particles: u32,
}

@group(0) @binding(1) var<uniform> params: ErodeParams;

@group(0) @binding(2) var<storage, read> random_indices: array<u32>;
@group(0) @binding(3) var<storage, read> brush_indices: array<i32>;
@group(0) @binding(4) var<storage, read> brush_weights: array<f32>;

fn rand_hash(v: u32) -> u32 {
    var x = v;
    x ^= 2747636419u;
    x *= 2654435769u;
    x ^= x >> 16u;
    x *= 2654435769u;
    x ^= x >> 16u;
    x *= 2654435769u;
    return x;
}

fn rand01(v: u32) -> f32 {
    return f32(rand_hash(v)) / 4294967295.0;
}

fn noise_hash(p: vec2<i32>) -> f32 {
    // Simple value noise hash in [0,1)
    let h = u32(p.x) * 374761393u + u32(p.y) * 668265263u;
    return rand01(h);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = fract(p);

    let a = noise_hash(i);
    let b = noise_hash(i + vec2<i32>(1,0));
    let c = noise_hash(i + vec2<i32>(0,1));
    let d = noise_hash(i + vec2<i32>(1,1));

    let u = f * f * (3.0 - 2.0 * f);
    let x1 = mix(a, b, u.x);
    let x2 = mix(c, d, u.x);
    return mix(x1, x2, u.y);
}

fn fbm(p: vec2<f32>) -> f32 {
    var total = 0.0;
    var amp = 0.5;
    var freq = 0.008;
    for (var i = 0; i < 6; i = i + 1) {
        total += value_noise(p * freq) * amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    return total;
}

@compute @workgroup_size(8,8,1)
fn init_fbm(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.map_size.x || gid.y >= params.map_size.y) { return; }
    let uv = vec2<f32>(gid.xy);
    let hval = fbm(uv);
    let idx = gid.y * params.map_size.x + gid.x;
    height[idx] = hval;
}

// No copy pass needed with in-place buffer

fn load_height(p: vec2<i32>) -> f32 {
    let size = i32(params.map_size.x);
    let idx = u32(p.y * size + p.x);
    return height[idx];
}

fn sample_height_and_gradient(pos: vec2<f32>) -> vec3<f32> {
    let size = vec2<i32>(i32(params.map_size.x), i32(params.map_size.y));
    let cx = clamp(i32(floor(pos.x)), 0, size.x - 2);
    let cy = clamp(i32(floor(pos.y)), 0, size.y - 2);
    let x = pos.x - f32(cx);
    let y = pos.y - f32(cy);
    let idx = vec2<i32>(cx, cy);
    let hNW = load_height(idx);
    let hNE = load_height(idx + vec2<i32>(1,0));
    let hSW = load_height(idx + vec2<i32>(0,1));
    let hSE = load_height(idx + vec2<i32>(1,1));
    let gradX = (hNE - hNW) * (1.0 - y) + (hSE - hSW) * y;
    let gradY = (hSW - hNW) * (1.0 - x) + (hSE - hNE) * x;
    let h = hNW * (1.0 - x) * (1.0 - y) + hNE * x * (1.0 - y) + hSW * (1.0 - x) * y + hSE * x * y;
    return vec3<f32>(gradX, gradY, h);
}

@compute @workgroup_size(1024,1,1)
fn erode(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.num_particles) { return; }

    let size = i32(params.map_size.x);
    let index = i32(random_indices[gid.x]);
    var posX = f32(index % size);
    var posY = f32(index / size);
    var dirX = 0.0;
    var dirY = 0.0;
    var speed = params.start_speed;
    var water = params.start_water;
    var sediment = 0.0;

    for (var life: u32 = 0u; life < params.max_lifetime; life = life + 1u) {
        let nodeX = i32(floor(posX));
        let nodeY = i32(floor(posY));
        let dropletIndex = nodeY * size + nodeX;
        let cellOffsetX = posX - f32(nodeX);
        let cellOffsetY = posY - f32(nodeY);

        let hg = sample_height_and_gradient(vec2<f32>(posX, posY));
        dirX = dirX * params.inertia - hg.x * (1.0 - params.inertia);
        dirY = dirY * params.inertia - hg.y * (1.0 - params.inertia);
        let len = max(0.01, length(vec2<f32>(dirX, dirY)));
        dirX /= len; dirY /= len;
        posX += dirX; posY += dirY;

        if ((dirX == 0.0 && dirY == 0.0) ||
            posX < f32(params.border_size) || posX > f32(size - i32(params.border_size)) ||
            posY < f32(params.border_size) || posY > f32(size - i32(params.border_size))) {
            break;
        }

        let newH = sample_height_and_gradient(vec2<f32>(posX, posY)).z;
        let deltaH = newH - hg.z;

        let cap = max(-deltaH * speed * water * params.sediment_capacity_factor, params.min_sediment_capacity);
        if (sediment > cap || deltaH > 0.0) {
            var deposit = select((sediment - cap) * params.deposit_speed, min(deltaH, sediment), deltaH > 0.0);
            sediment -= deposit;
            // bilinear deposit to 4 corners
            let wNW = (1.0 - cellOffsetX) * (1.0 - cellOffsetY);
            let wNE = cellOffsetX * (1.0 - cellOffsetY);
            let wSW = (1.0 - cellOffsetX) * cellOffsetY;
            let wSE = cellOffsetX * cellOffsetY;
            let size_i = i32(params.map_size.x);
            let base = nodeY * size_i + nodeX;
            let iNW = u32(base);
            let iNE = u32(base + 1);
            let iSW = u32(base + size_i);
            let iSE = u32(base + size_i + 1);
            height[iNW] = height[iNW] + deposit * wNW;
            height[iNE] = height[iNE] + deposit * wNE;
            height[iSW] = height[iSW] + deposit * wSW;
            height[iSE] = height[iSE] + deposit * wSE;
        } else {
            let amountToErode = min((cap - sediment) * params.erode_speed, -deltaH);
            for (var i: u32 = 0u; i < params.brush_length; i = i + 1u) {
                let erodeIndex = dropletIndex + brush_indices[i];
                let idx = u32(erodeIndex);
                let current = height[idx];
                let weighted = amountToErode * brush_weights[i];
                let delta = select(weighted, current, current < weighted);
                height[idx] = current - delta;
                sediment += delta;
            }
        }

        speed = sqrt(max(0.0, speed * speed + deltaH * params.gravity));
        water *= (1.0 - params.evaporate_speed);
    }
}

@group(1) @binding(0) var display_out: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8,8,1)
fn blit_to_texture(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.map_size.x || gid.y >= params.map_size.y) { return; }
    let size = i32(params.map_size.x);
    let p = vec2<i32>(gid.xy);
    let idx = u32(p.y * size + p.x);
    let h = height[idx];
    let c = clamp(h, 0.0, 1.0);
    textureStore(display_out, vec2<i32>(gid.xy), vec4<f32>(c, c, c, 1.0));
}


