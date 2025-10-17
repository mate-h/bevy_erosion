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

// Flow, sediment, and erosion accumulation buffers (vec3 per pixel stored as 3 consecutive atomics)
// Using atomic<u32> to store fixed-point float values (multiply by 10000 to preserve precision)
@group(0) @binding(5) var<storage, read_write> flow_buffer: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> sediment_buffer: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> erosion_buffer: array<atomic<u32>>;

// Color image as read-write storage texture
@group(1) @binding(0) var color_image: texture_storage_2d<rgba16float, read_write>;

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
    height[idx] = hval * 1.0;
    
    // Clear flow, sediment, and erosion buffers
    let idx3 = idx * 3u;
    atomicStore(&flow_buffer[idx3], 0u);
    atomicStore(&flow_buffer[idx3 + 1u], 0u);
    atomicStore(&flow_buffer[idx3 + 2u], 0u);
    atomicStore(&sediment_buffer[idx3], 0u);
    atomicStore(&sediment_buffer[idx3 + 1u], 0u);
    atomicStore(&sediment_buffer[idx3 + 2u], 0u);
    atomicStore(&erosion_buffer[idx3], 0u);
    atomicStore(&erosion_buffer[idx3 + 1u], 0u);
    atomicStore(&erosion_buffer[idx3 + 2u], 0u);
}

fn band_color_from_height(h: f32) -> vec3<f32> {
    // Simple banding: water/sand/grass/rock/snow
    let o = 0.0;
    if (h < 0.25 + o) {
        return vec3<f32>(0.08, 0.35, 0.65); // water/deep
    } else if (h < 0.35 + o) {
        return vec3<f32>(0.76, 0.70, 0.50); // sand
    } else if (h < 0.60 + o) {
        return vec3<f32>(0.20, 0.55, 0.25) * 0.5; // grass
    } else if (h < 0.80 + o) {
        return vec3<f32>(0.40, 0.40, 0.40) * 0.5; // rock
    } else {
        return vec3<f32>(0.95, 0.95, 0.95); // snow
    }
}

@compute @workgroup_size(8,8,1)
fn init_color_bands(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.map_size.x || gid.y >= params.map_size.y) { return; }
    let size = i32(params.map_size.x);
    let idx = u32(gid.y * params.map_size.x + gid.x);
    let h = height[idx];
    let c = band_color_from_height(h);
    textureStore(color_image, vec2<i32>(gid.xy), vec4<f32>(c, 1.0));
}

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
    var posY = f32(index) / f32(size);
    var dirX = 0.0;
    var dirY = 0.0;
    var speed = params.start_speed;
    var water = params.start_water;
    var sediment = 0.0;
    var total_erosion = 0.0;
    var total_deposition = 0.0;

    // droplet carries color sampled from the initial position
    var droplet_color = textureLoad(
        color_image,
        vec2<i32>(i32(floor(posX)), i32(floor(posY)))
    ).xyz;

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
        let prevX = posX; let prevY = posY;
        posX += dirX; posY += dirY;

        // Stop if not moving or if leaving the safe interior region.
        // Match reference: rely on a one-cell margin around edges for bilinear sampling.
        if ((dirX == 0.0 && dirY == 0.0) ||
            posX < f32(params.border_size) || posX >= f32(size - 1 - i32(params.border_size)) ||
            posY < f32(params.border_size) || posY >= f32(size - 1 - i32(params.border_size))) {
            break;
        }

        let newH = sample_height_and_gradient(vec2<f32>(posX, posY)).z;
        let deltaH = newH - hg.z;

        let cap = max(-deltaH * speed * water * params.sediment_capacity_factor, params.min_sediment_capacity);
        if (sediment > cap || deltaH > 0.0) {
            var deposit = select((sediment - cap) * params.deposit_speed, min(deltaH, sediment), deltaH > 0.0);
            sediment -= deposit;
            total_deposition += deposit;
            
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
            
            // Accumulate sediment deposition (convert to fixed-point: multiply by 10000)
            atomicAdd(&sediment_buffer[iNW * 3u], u32(deposit * wNW * 10000.0));
            atomicAdd(&sediment_buffer[iNE * 3u], u32(deposit * wNE * 10000.0));
            atomicAdd(&sediment_buffer[iSW * 3u], u32(deposit * wSW * 10000.0));
            atomicAdd(&sediment_buffer[iSE * 3u], u32(deposit * wSE * 10000.0));

            // Blend carried color into deposited texels
            let pNW = vec2<i32>(nodeX, nodeY);
            let pNE = vec2<i32>(nodeX + 1, nodeY);
            let pSW = vec2<i32>(nodeX, nodeY + 1);
            let pSE = vec2<i32>(nodeX + 1, nodeY + 1);
            let cNW = textureLoad(color_image, pNW).xyz;
            let cNE = textureLoad(color_image, pNE).xyz;
            let cSW = textureLoad(color_image, pSW).xyz;
            let cSE = textureLoad(color_image, pSE).xyz;
            let blend = clamp(deposit * 100.0, 0.0, 1.0);
            textureStore(color_image, pNW, vec4<f32>(mix(cNW, droplet_color, blend * wNW), 1.0));
            textureStore(color_image, pNE, vec4<f32>(mix(cNE, droplet_color, blend * wNE), 1.0));
            textureStore(color_image, pSW, vec4<f32>(mix(cSW, droplet_color, blend * wSW), 1.0));
            textureStore(color_image, pSE, vec4<f32>(mix(cSE, droplet_color, blend * wSE), 1.0));
        } else {
            let amountToErode = min((cap - sediment) * params.erode_speed, -deltaH);
            total_erosion += amountToErode;
            let size_i = i32(params.map_size.x);
            let map_len = size_i * size_i;
            for (var i: u32 = 0u; i < params.brush_length; i = i + 1u) {
                let erodeIndex = dropletIndex + brush_indices[i];
                if (erodeIndex >= 0 && erodeIndex < map_len) {
                    let idx = u32(erodeIndex);
                    let current = height[idx];
                    let weighted = amountToErode * brush_weights[i];
                    let delta = select(weighted, current, current < weighted);
                    height[idx] = current - delta;
                    sediment += delta;
                    
                    // Accumulate erosion (convert to fixed-point: multiply by 10000)
                    atomicAdd(&erosion_buffer[idx * 3u], u32(delta * 10000.0));
                }
            }
            // Pull color from eroded neighborhood toward carried color
            let p = vec2<i32>(nodeX, nodeY);
            let c_here = textureLoad(color_image, p).xyz;
            let pull = clamp(amountToErode * 6.0, 0.0, 1.0);
            droplet_color = mix(droplet_color, c_here, 0.10);
            textureStore(color_image, p, vec4<f32>(mix(c_here, droplet_color, pull), 1.0));
        }

        speed = sqrt(max(0.0, speed * speed + deltaH * params.gravity));
        water *= (1.0 - params.evaporate_speed);
        
        // Accumulate flow (direction + magnitude) at current position
        // Ensure dropletIndex is valid for the buffer size
        let map_size_total = i32(params.map_size.x * params.map_size.y);
        if (dropletIndex >= 0 && dropletIndex < map_size_total) {
            let flowIdx = u32(dropletIndex) * 3u;
            let flow_magnitude = speed * water;
        //     // Convert to fixed-point with offset for signed values: (value + 1000) * 10
        //     atomicAdd(&flow_buffer[flowIdx], u32((dirX * flow_magnitude + 1000.0) * 10.0));       // x direction
        //     atomicAdd(&flow_buffer[flowIdx + 1u], u32((dirY * flow_magnitude + 1000.0) * 10.0));  // y direction
        //     atomicAdd(&flow_buffer[flowIdx + 2u], u32(flow_magnitude * 10000.0));                 // magnitude
        }
    }
}

@group(1) @binding(0) var display_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var normal_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var analysis_out: texture_storage_2d<rgba16float, write>;  // R=flow_mag, G=sediment, B=erosion, A=flow_angle

@compute @workgroup_size(8,8,1)
fn blit_to_texture(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.map_size.x || gid.y >= params.map_size.y) { return; }
    let size_x = i32(params.map_size.x);
    let size_y = i32(params.map_size.y);
    let px = i32(gid.x);
    let py = i32(gid.y);
    let idx = u32(py * size_x + px);
    let h = height[idx];

    // grayscale height to display
    let c = h;
    textureStore(display_out, vec2<i32>(px, py), vec4<f32>(c, c, c, 1.0));

    // compute normals from central differences of height field
    let xm1 = max(px - 1, 0);
    let xp1 = min(px + 1, size_x - 1);
    let ym1 = max(py - 1, 0);
    let yp1 = min(py + 1, size_y - 1);

    let hL = height[u32(py * size_x + xm1)];
    let hR = height[u32(py * size_x + xp1)];
    let hD = height[u32(ym1 * size_x + px)];
    let hU = height[u32(yp1 * size_x + px)];

    let dx = hR - hL;
    let dy = hU - hD;
    let n = normalize(vec3<f32>(-dx, 1.0, -dy));
    let enc = n * 0.5 + vec3<f32>(0.5, 0.5, 0.5);
    textureStore(normal_out, vec2<i32>(px, py), vec4<f32>(enc, 1.0));
    
    // Pack flow, sediment, and erosion into single analysis texture
    let idx3 = idx * 3u;
    // Convert from fixed-point back to float
    let flow_x_raw = f32(atomicLoad(&flow_buffer[idx3])) / 10.0 - 1000.0;
    let flow_y_raw = f32(atomicLoad(&flow_buffer[idx3 + 1u])) / 10.0 - 1000.0;
    let flow_mag = f32(atomicLoad(&flow_buffer[idx3 + 2u])) / 10000.0;
    
    // Calculate flow direction angle (0-1 normalized from 0-2π)
    var flow_angle = 0.5;  // default to 0.5 (no flow)
    if (flow_mag > 0.001) {
        let angle_rad = atan2(flow_y_raw, flow_x_raw);
        flow_angle = (angle_rad + 3.14159265) / (2.0 * 3.14159265);  // normalize to 0-1
    }
    
    // Sediment and erosion values (convert from fixed-point: divide by 10000, then scale for visibility)
    // These values accumulate over multiple frames, so we scale them for better visualization
    let sediment = f32(atomicLoad(&sediment_buffer[idx3])) / 10000.0;
    let erosion = f32(atomicLoad(&erosion_buffer[idx3])) / 10000.0;
    
    // Pack into analysis texture: R=flow_mag, G=sediment, B=erosion, A=flow_angle
    let analysis = vec4<f32>(
        clamp(flow_mag * 0.1, 0.0, 1.0),
        sediment,
        erosion,
        flow_angle
    );
    textureStore(analysis_out, vec2<i32>(px, py), analysis);
}


