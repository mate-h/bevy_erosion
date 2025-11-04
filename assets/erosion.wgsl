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

// Accumulation buffers using atomic<u32> to store fixed-point float values (multiply by 10000)
// All buffers store 1 u32 per pixel (single scalar value)
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
    
    // Clear flow, sediment, and erosion buffers (all store 1 value per pixel)
    atomicStore(&flow_buffer[idx], 0u);
    atomicStore(&sediment_buffer[idx], 0u);
    atomicStore(&erosion_buffer[idx], 0u);
}

fn band_color_from_height(h: f32) -> vec3<f32> {
    // Smooth banding with feathering: water/sand/grass/rock/snow
    let o = 0.0;
    let feather = 0.05; // feather width for blending between layers
    
    let water = vec3<f32>(0.08, 0.35, 0.65);
    let sand = vec3<f32>(0.76, 0.70, 0.50);
    let grass = vec3<f32>(0.20, 0.55, 0.25) * 0.5;
    let rock = vec3<f32>(0.40, 0.40, 0.40) * 0.5;
    let snow = vec3<f32>(0.95, 0.95, 0.95);
    
    // Define thresholds
    let t1 = 0.25 + o; // water -> sand
    let t2 = 0.35 + o; // sand -> grass
    let t3 = 0.60 + o; // grass -> rock
    let t4 = 0.80 + o; // rock -> snow
    
    // Blend between adjacent colors
    if (h < t1 - feather) {
        return water;
    } else if (h < t1 + feather) {
        let t = (h - (t1 - feather)) / (2.0 * feather);
        return mix(water, sand, smoothstep(0.0, 1.0, t));
    } else if (h < t2 - feather) {
        return sand;
    } else if (h < t2 + feather) {
        let t = (h - (t2 - feather)) / (2.0 * feather);
        return mix(sand, grass, smoothstep(0.0, 1.0, t));
    } else if (h < t3 - feather) {
        return grass;
    } else if (h < t3 + feather) {
        let t = (h - (t3 - feather)) / (2.0 * feather);
        return mix(grass, rock, smoothstep(0.0, 1.0, t));
    } else if (h < t4 - feather) {
        return rock;
    } else if (h < t4 + feather) {
        let t = (h - (t4 - feather)) / (2.0 * feather);
        return mix(rock, snow, smoothstep(0.0, 1.0, t));
    } else {
        return snow;
    }
}

@compute @workgroup_size(8,8,1)
fn init_color_bands(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.map_size.x || gid.y >= params.map_size.y) { return; }
    let size = i32(params.map_size.x);
    let idx = u32(gid.y * params.map_size.x + gid.x);
    let h = height[idx];
    let c = band_color_from_height(h);
    
    // Generate FBM noise for black overlay (using different scale/offset from height field)
    let uv = vec2<f32>(gid.xy);
    let noise = fbm(uv * 0.015 + vec2<f32>(100.0, 100.0)); // Different scale and offset
    let noise_factor = clamp(noise, 0.0, 1.0);
    let black = vec3<f32>(0.0, 0.0, 0.0);
    
    // Blend black FBM noise on top of color bands
    let blended = mix(c, black, noise_factor); // 30% max darkening
    
    textureStore(color_image, vec2<i32>(gid.xy), vec4<f32>(blended, 1.0));
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
    var posY = f32(index / size);  // Use integer division before float conversion
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
            atomicAdd(&sediment_buffer[iNW], u32(deposit * wNW * 10000.0));
            atomicAdd(&sediment_buffer[iNE], u32(deposit * wNE * 10000.0));
            atomicAdd(&sediment_buffer[iSW], u32(deposit * wSW * 10000.0));
            atomicAdd(&sediment_buffer[iSE], u32(deposit * wSE * 10000.0));

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
                    
                    // Accumulate erosion - use larger multiplier to avoid truncation with fine brush
                    // With brush radius 16, delta can be ~1/800th of amountToErode, so we need a bigger scale
                    atomicAdd(&erosion_buffer[idx], u32(delta * 100000.0));
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
        
        let flow_magnitude = speed * water;
        atomicAdd(&flow_buffer[dropletIndex], u32(flow_magnitude * 10000.0));
    }
}

@group(1) @binding(0) var display_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var normal_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var analysis_out: texture_storage_2d<rgba16float, write>;  // R=flow_mag, G=sediment, B=erosion, A=flow_angle
@group(1) @binding(3) var ao_out: texture_storage_2d<r16float, read_write>;

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
    let flow_mag = f32(atomicLoad(&flow_buffer[idx])) / 10000.0;
    let sediment = f32(atomicLoad(&sediment_buffer[idx])) / 10000.0;
    let erosion = f32(atomicLoad(&erosion_buffer[idx])) / 100000.0;
    
    // Pack into analysis texture: R=flow_mag, G=sediment, B=erosion
    let analysis = vec4<f32>(
        flow_mag,
        sediment,
        erosion,
        0.0
    );
    textureStore(analysis_out, vec2<i32>(px, py), analysis);
}

// Hammersley low-discrepancy sequence for uniform sampling
// Based on Bevy's hammersley_2d implementation
fn radical_inverse_VdC(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10; // / 0x100000000
}

fn hammersley_2d(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(n), radical_inverse_VdC(i));
}

// PCG hash for noise generation
fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Interleaved gradient noise for dithering
fn interleaved_gradient_noise(pixel_coordinates: vec2<f32>, frame: u32) -> f32 {
    let xy = pixel_coordinates + 5.588238 * f32(frame % 64u);
    return fract(52.9829189 * fract(0.06711056 * xy.x + 0.00583715 * xy.y));
}

const PI = 3.14159265359;
const HALF_PI = 1.57079632679;
const PI_2 = 6.28318530718;

// Compute ambient occlusion from heightmap
// Sample count, radius, strength are hardcoded for compute shader
@compute @workgroup_size(8,8,1)
fn compute_ao(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.map_size.x || gid.y >= params.map_size.y) { return; }
    
    let size_x = i32(params.map_size.x);
    let size_y = i32(params.map_size.y);
    let px = i32(gid.x);
    let py = i32(gid.y);
    let idx = u32(py * size_x + px);
    
    let current_height = height[idx];
    
    // AO parameters (matching the default values from Rust)
    let sample_count = 64u; // Increased for smoother result
    let sample_radius_px = 0.08 * f32(params.map_size.x); // Convert UV radius to pixels
    let strength = 1.5;
    let bias = 0.0;
    let height_scale = 50.0; // Match terrain height scale
    let terrain_size = 512.0; // Terrain world size
    
    var occlusion = 0.0;
    
    for (var i = 0u; i < sample_count; i = i + 1u) {
        // Use Hammersley sequence for well-distributed samples (no noise for consistency)
        let h = hammersley_2d(i, sample_count);
        
        // Angle from Hammersley sequence only
        let angle = h.x * PI_2;
        
        // Radius using sqrt for uniform disk distribution
        let radius_t = h.y;
        let radius_px = sqrt(radius_t) * sample_radius_px;
        
        // Sample offset in pixel space
        let offset = vec2<f32>(cos(angle), sin(angle)) * radius_px;
        let sample_x = clamp(px + i32(round(offset.x)), 0, size_x - 1);
        let sample_y = clamp(py + i32(round(offset.y)), 0, size_y - 1);
        let sample_idx = u32(sample_y * size_x + sample_x);
        
        // Sample height
        let sample_height = height[sample_idx];
        
        // Height difference (scaled to world space)
        let height_diff = (sample_height - current_height) * height_scale;
        
        // Distance in pixel space
        let px_distance = length(offset);
        
        // Convert to world distance
        let world_distance = (px_distance / f32(params.map_size.x)) * terrain_size;
        
        // Skip if distance is too small
        if (world_distance < 0.1) {
            continue;
        }
        
        // Calculate occlusion: samples higher than current point block sky light
        if (height_diff > 0.001) {
            // Calculate the horizon angle
            let elevation_angle = atan(height_diff / world_distance);
            
            // Distance falloff with smoothstep
            let t = px_distance / sample_radius_px;
            let falloff = 1.0 - (t * t * (3.0 - 2.0 * t));
            
            // Accumulate occlusion
            let contribution = (elevation_angle / HALF_PI) * falloff;
            occlusion = occlusion + contribution;
        }
    }
    
    // Average and scale occlusion
    occlusion = (occlusion / f32(sample_count)) * strength * 8.0;
    
    // Convert to AO value (1.0 = bright, 0.0 = dark)
    let ao_value = 1.0 - clamp(occlusion, 0.0, 1.0);
    
    // Apply power curve for contrast
    let ao_final = pow(ao_value, 0.8);
    
    // Apply bias
    let ao = max(ao_final, bias);
    
    // Write AO value to texture (single channel R16Float)
    textureStore(ao_out, vec2<i32>(px, py), vec4<f32>(ao, 0.0, 0.0, 0.0));
}

// Gaussian blur pass for smoothing AO
// Temporary storage texture for separable blur
@group(1) @binding(4) var ao_temp: texture_storage_2d<r16float, read_write>;

@compute @workgroup_size(8,8,1)
fn blur_ao_horizontal(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.map_size.x || gid.y >= params.map_size.y) { return; }
    
    let px = i32(gid.x);
    let py = i32(gid.y);
    let size_x = i32(params.map_size.x);
    
    // 5-tap Gaussian kernel weights (approximation: 1, 4, 6, 4, 1) / 16
    let weights = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);
    
    var sum = 0.0;
    for (var i = 0; i < 5; i = i + 1) {
        let offset = i - 2;
        let sample_x = clamp(px + offset, 0, size_x - 1);
        let ao_val = textureLoad(ao_out, vec2<i32>(sample_x, py)).x;
        sum = sum + ao_val * weights[i];
    }
    
    // Write to temp texture
    textureStore(ao_temp, vec2<i32>(px, py), vec4<f32>(sum, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8,8,1)
fn blur_ao_vertical(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.map_size.x || gid.y >= params.map_size.y) { return; }
    
    let px = i32(gid.x);
    let py = i32(gid.y);
    let size_y = i32(params.map_size.y);
    
    // 5-tap Gaussian kernel weights
    let weights = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);
    
    var sum = 0.0;
    for (var i = 0; i < 5; i = i + 1) {
        let offset = i - 2;
        let sample_y = clamp(py + offset, 0, size_y - 1);
        let ao_val = textureLoad(ao_temp, vec2<i32>(px, sample_y)).x;
        sum = sum + ao_val * weights[i];
    }
    
    // Write back to AO output
    textureStore(ao_out, vec2<i32>(px, py), vec4<f32>(sum, 0.0, 0.0, 0.0));
}

