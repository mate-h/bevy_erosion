//! Interactive water placement on the terrain with the mouse, in two modes:
//!
//! * [`BrushMode::Rain`] releases erosion trails under the brush ([`BrushStroke`]); they flow
//!   downhill and carve the current terrain live, on top of whatever the cascade produced.
//! * [`BrushMode::Sea`] paints [`ErosionImages::water`] into the KTT `__Outflow` mask: erosion
//!   drains into it and its cells keep the source terrain. Every stroke restarts the cascade,
//!   whose coarse levels finish in one frame.
//!
//! Put [`TerrainSurface`] on the displaced terrain mesh so clicks can be ray-cast against it.

use crate::{
    BrushStroke, ErosionConfig, ErosionImages, ResetSim, camera::OrbitController,
    camera::OrbitInputBlocked,
};
use bevy::{
    prelude::*,
    render::gpu_readback::{Readback, ReadbackComplete},
    window::PrimaryWindow,
};

pub struct WaterBrushPlugin;

impl Plugin for WaterBrushPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WaterBrush>()
            .init_resource::<HeightCache>()
            .init_resource::<BrushHit>()
            .add_systems(
                Update,
                (
                    sync_height_readback,
                    swap_orbit_button,
                    pick_terrain,
                    paint_water,
                    draw_brush,
                )
                    .chain(),
            );
    }
}

/// Rain trails per brush texel per dispatch at strength 1 (KTT's map-wide density is 0.01 at the
/// default Erosion_Amount).
const RAIN_DENSITY: f32 = 0.1;
/// Rain dispatches per frame while the button is held.
const RAIN_PASSES: u32 = 8;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BrushMode {
    /// Release erosion trails under the brush.
    #[default]
    Rain,
    /// Paint outflow (sea / lake) that erosion drains into; restarts the cascade.
    Sea,
}

#[derive(Resource, Clone, Debug)]
pub struct WaterBrush {
    /// Paint on left click (orbit moves to the right button while enabled).
    pub enabled: bool,
    pub mode: BrushMode,
    /// Brush radius in map texels.
    pub radius: f32,
    /// Rain: trail density (× [`RAIN_DENSITY`]). Sea: water added per frame at the center (0..1).
    pub strength: f32,
    /// Set to clear all painted water on the next frame.
    pub clear_requested: bool,
}

impl Default for WaterBrush {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: BrushMode::Rain,
            radius: 12.0,
            strength: 0.35,
            clear_requested: false,
        }
    }
}

/// The displaced terrain mesh: a `size` plane centered on its transform, with
/// `height × height_scale` added along local +Y (see `terrain_extended.wgsl`).
#[derive(Component, Clone, Copy, Debug)]
pub struct TerrainSurface {
    pub size: Vec2,
    pub height_scale: f32,
}

/// Where the cursor ray meets the terrain this frame.
#[derive(Resource, Default, Clone, Copy, Debug)]
pub struct BrushHit {
    pub world: Option<Vec3>,
    /// Map texel coordinates of the hit.
    pub texel: Vec2,
}

/// CPU copy of the displayed heights, read back while the brush is on.
#[derive(Resource, Default)]
struct HeightCache {
    size: UVec2,
    heights: Vec<f32>,
}

impl HeightCache {
    fn sample(&self, texel: Vec2) -> f32 {
        if self.heights.is_empty() {
            return 0.0;
        }
        let max = (self.size - 1).as_vec2();
        let p = (texel - 0.5).clamp(Vec2::ZERO, max);
        let i = p.floor().as_uvec2();
        let j = (i + 1).min(self.size - 1);
        let f = p - p.floor();
        let at = |x: u32, y: u32| self.heights[(y * self.size.x + x) as usize];
        let a = at(i.x, i.y).lerp(at(j.x, i.y), f.x);
        let b = at(i.x, j.y).lerp(at(j.x, j.y), f.x);
        a.lerp(b, f.y)
    }
}

#[derive(Component)]
struct HeightReadback;

fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((bits >> 10) & 0x1f);
    let mant = f32::from(bits & 0x3ff);
    match exp {
        0 => sign * mant * 2f32.powi(-24),
        31 => f32::NAN,
        _ => sign * (1.0 + mant / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// Keep a GPU readback of the height texture alive only while painting is enabled.
fn sync_height_readback(
    mut commands: Commands,
    brush: Res<WaterBrush>,
    images: Option<Res<ErosionImages>>,
    config: Res<ErosionConfig>,
    existing: Query<Entity, With<HeightReadback>>,
) {
    let Some(images) = images else { return };
    match (brush.enabled, existing.single()) {
        (true, Err(_)) => {
            let size = config.map_size;
            commands
                .spawn((Readback::texture(images.height.clone()), HeightReadback))
                .observe(
                    move |ev: On<ReadbackComplete>, mut cache: ResMut<HeightCache>| {
                        // Rgba16Float rows, possibly padded: R holds the height.
                        let stride = ev.data.len() / size.y as usize;
                        cache.size = size;
                        cache.heights.resize((size.x * size.y) as usize, 0.0);
                        for y in 0..size.y as usize {
                            let row = &ev.data[y * stride..];
                            for x in 0..size.x as usize {
                                let bits = u16::from_le_bytes([row[x * 8], row[x * 8 + 1]]);
                                cache.heights[y * size.x as usize + x] = f16_to_f32(bits);
                            }
                        }
                    },
                );
        }
        (false, Ok(entity)) => commands.entity(entity).despawn(),
        _ => {}
    }
}

/// Free the left button for painting: orbit with the right button while the brush is on.
fn swap_orbit_button(brush: Res<WaterBrush>, mut orbits: Query<&mut OrbitController>) {
    if !brush.is_changed() {
        return;
    }
    let button = if brush.enabled {
        MouseButton::Right
    } else {
        MouseButton::Left
    };
    for mut orbit in &mut orbits {
        orbit.rotate_button = button;
    }
}

/// Ray-march the cursor ray against the heightfield (bilinear heights, then a bisection).
fn pick_terrain(
    brush: Res<WaterBrush>,
    cache: Res<HeightCache>,
    config: Res<ErosionConfig>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    terrain: Query<(&TerrainSurface, &GlobalTransform)>,
    mut hit: ResMut<BrushHit>,
) {
    hit.world = None;
    if !brush.enabled {
        return;
    }
    let (Ok(window), Ok((camera, cam_tf)), Ok((surface, terrain_tf))) =
        (windows.single(), cameras.single(), terrain.single())
    else {
        return;
    };
    let Some(ray) = window
        .cursor_position()
        .and_then(|c| camera.viewport_to_world(cam_tf, c).ok())
    else {
        return;
    };
    let to_local = terrain_tf.affine().inverse();
    let origin = to_local.transform_point3(ray.origin);
    let dir = to_local.transform_vector3(*ray.direction).normalize();
    let map = config.map_size.as_vec2();
    if let Some(local) = ray_hit(origin, dir, surface, map, |t| cache.sample(t)) {
        hit.world = Some(terrain_tf.transform_point(local));
        hit.texel = surface.texel_of(local, map);
    }
}

impl TerrainSurface {
    /// Map texel coordinates under a terrain-local point.
    pub fn texel_of(&self, local: Vec3, map: Vec2) -> Vec2 {
        (Vec2::new(local.x, local.z) / self.size + 0.5) * map
    }
}

/// First point (terrain-local) where the ray dips below the heightfield. Marches in half-texel
/// steps across the terrain footprint, then bisects.
fn ray_hit(
    origin: Vec3,
    dir: Vec3,
    surface: &TerrainSurface,
    map: Vec2,
    height: impl Fn(Vec2) -> f32,
) -> Option<Vec3> {
    let below = |p: Vec3| p.y <= height(surface.texel_of(p, map)) * surface.height_scale;

    // Clip the ray to the terrain footprint.
    let half = surface.size * 0.5;
    let (mut t0, mut t1) = (0.0f32, f32::MAX);
    for (o, d, h) in [(origin.x, dir.x, half.x), (origin.z, dir.z, half.y)] {
        if d.abs() < 1e-8 {
            if o.abs() > h {
                return None;
            }
            continue;
        }
        let (a, b) = ((-h - o) / d, (h - o) / d);
        t0 = t0.max(a.min(b));
        t1 = t1.min(a.max(b));
    }
    if t0 > t1 {
        return None;
    }
    let step = (surface.size.x / map.x) * 0.5;
    let mut prev = t0;
    let mut t = t0;
    while t <= t1 {
        if below(origin + dir * t) {
            let (mut lo, mut hi) = (prev, t);
            for _ in 0..16 {
                let mid = 0.5 * (lo + hi);
                if below(origin + dir * mid) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            return Some(origin + dir * hi);
        }
        prev = t;
        t += step;
    }
    None
}

/// Left drag rains (or paints sea, Shift + left drag erases sea and restarts the erosion).
#[allow(clippy::too_many_arguments)]
fn paint_water(
    mut brush: ResMut<WaterBrush>,
    mut stroke: ResMut<BrushStroke>,
    hit: Res<BrushHit>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    blocked: Res<OrbitInputBlocked>,
    erosion_images: Option<Res<ErosionImages>>,
    mut images: ResMut<Assets<Image>>,
    mut reset: ResMut<ResetSim>,
) {
    let Some(erosion_images) = erosion_images else {
        return;
    };
    let clear = std::mem::take(&mut brush.clear_requested);
    let painting = brush.enabled && !blocked.0 && buttons.pressed(MouseButton::Left);
    let rain = painting && hit.world.is_some() && brush.mode == BrushMode::Rain;
    let area = std::f32::consts::PI * brush.radius * brush.radius;
    *stroke = BrushStroke {
        active: rain,
        center: hit.texel,
        radius: brush.radius,
        particles: (brush.strength * RAIN_DENSITY * area).ceil() as u32,
        passes: RAIN_PASSES,
    };
    let sea = painting && hit.world.is_some() && brush.mode == BrushMode::Sea;
    if !clear && !sea {
        return;
    }
    let Some(mut image) = images.get_mut(&erosion_images.water) else {
        return;
    };
    let size = image.size();
    let Some(data) = image.data.as_mut() else {
        return;
    };
    if clear {
        data.fill(0);
    } else {
        let erase = keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]);
        let r = brush.radius.max(0.5);
        let lo = (hit.texel - r).floor().max(Vec2::ZERO).as_uvec2();
        let hi = (hit.texel + r).ceil().as_uvec2().min(size - 1);
        for y in lo.y..=hi.y {
            for x in lo.x..=hi.x {
                let d = (Vec2::new(x as f32, y as f32) + 0.5).distance(hit.texel) / r;
                if d >= 1.0 {
                    continue;
                }
                // Smooth falloff toward the rim.
                let w = brush.strength * (1.0 - d * d).powi(2);
                let o = ((y * size.x + x) * 4) as usize;
                let mut v = f32::from_le_bytes(data[o..o + 4].try_into().unwrap());
                v = if erase {
                    (v - w).max(0.0)
                } else {
                    (v + w).min(1.0)
                };
                data[o..o + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
    }
    reset.generation = reset.generation.wrapping_add(1);
}

/// Ring that follows the terrain under the cursor.
fn draw_brush(
    brush: Res<WaterBrush>,
    hit: Res<BrushHit>,
    cache: Res<HeightCache>,
    config: Res<ErosionConfig>,
    keys: Res<ButtonInput<KeyCode>>,
    terrain: Query<(&TerrainSurface, &GlobalTransform)>,
    mut gizmos: Gizmos,
) {
    let (Some(_), Ok((surface, tf))) = (hit.world, terrain.single()) else {
        return;
    };
    if !brush.enabled {
        return;
    }
    let map = config.map_size.as_vec2();
    let color = if keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]) {
        Color::srgb(1.0, 0.45, 0.2)
    } else {
        Color::srgb(0.3, 0.7, 1.0)
    };
    let points = (0..=64).map(|i| {
        let a = i as f32 / 64.0 * std::f32::consts::TAU;
        let texel = hit.texel + Vec2::new(a.cos(), a.sin()) * brush.radius;
        let xz = (texel / map - 0.5) * surface.size;
        let y = cache.sample(texel) * surface.height_scale + 0.3;
        tf.transform_point(Vec3::new(xz.x, y, xz.y))
    });
    gizmos.linestrip(points, color);
}

#[cfg(test)]
mod tests {
    use super::*;

    const SURFACE: TerrainSurface = TerrainSurface {
        size: Vec2::splat(256.0),
        height_scale: 50.0,
    };
    const MAP: Vec2 = Vec2::splat(512.0);

    #[test]
    fn straight_down_hits_flat_ground() {
        let p = ray_hit(
            Vec3::new(64.0, 100.0, -32.0),
            Vec3::NEG_Y,
            &SURFACE,
            MAP,
            |_| 0.5,
        )
        .expect("hit");
        assert!((p.y - 25.0).abs() < 1e-3, "{p}");
        let texel = SURFACE.texel_of(p, MAP);
        assert!(texel.distance(Vec2::new(384.0, 192.0)) < 1e-3, "{texel}");
    }

    #[test]
    fn slanted_ray_stops_at_first_ridge() {
        // A wall of height 1 (50 world units) for texel x >= 300, flat 0 before it.
        let wall = |t: Vec2| if t.x >= 300.0 { 1.0 } else { 0.0 };
        let dir = Vec3::new(1.0, -0.1, 0.0).normalize();
        let p = ray_hit(Vec3::new(-128.0, 40.0, 0.0), dir, &SURFACE, MAP, wall).expect("hit");
        // The wall starts at local x = 300 / 512 * 256 - 128 = 22.
        assert!((p.x - 22.0).abs() < 0.3, "{p}");
    }

    #[test]
    fn ray_outside_the_footprint_misses() {
        let miss = ray_hit(Vec3::new(500.0, 10.0, 0.0), Vec3::X, &SURFACE, MAP, |_| 0.0);
        assert!(miss.is_none());
    }
}
