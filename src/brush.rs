//! Mouse brushes on the terrain:
//!
//! * [`BrushMode::Rain`] releases erosion trails under the brush ([`BrushStroke`]); they flow
//!   downhill and carve the current terrain live, on top of whatever the cascade produced.
//! * [`BrushMode::Terrain`] edits the source terrain ([`ErosionImages::terrain_edit`]): raise,
//!   lower, flatten or smooth. Every stroke restarts the cascade, whose coarse levels finish in
//!   one frame.
//! * [`BrushMode::WaterBody`] pours water into the shallow-water solver ([`WaterControl`]); it
//!   pools in depressions and runs off downhill as rivers, with no water level to set.
//!
//! Shift + drag erases (removes water, or undoes terrain edits). Put [`TerrainSurface`] on the
//! displaced terrain mesh so clicks can be ray-cast against it.

use crate::{
    BrushStroke, ErosionConfig, ErosionImages, ResetSim, WaterControl, camera::OrbitController,
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

/// Rain trail starts per brush texel per second at strength 1. The cascade's finest level starts
/// about 2 per texel in total (200 dispatches at KTT's 1% density), so one second of rain at full
/// strength erodes like one more finest-level batch under the brush.
const RAIN_STARTS_PER_SECOND: f32 = 2.0;
/// Rain dispatches per frame (one ridge-biased, one flow-biased, like the cascade alternates).
const RAIN_PASSES: u32 = 2;

/// Water depth poured per second at the brush center at strength 1 (height units).
const WATER_DEPTH_PER_SECOND: f32 = 0.02;
/// Terrain raise / lower per second at strength 1 (height units).
const TERRAIN_HEIGHT_PER_SECOND: f32 = 0.1;
/// Flatten / smooth weight gained per second at strength 1.
const TERRAIN_BLEND_PER_SECOND: f32 = 2.0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BrushMode {
    /// Release erosion trails under the brush.
    #[default]
    Rain,
    /// Edit the source terrain.
    Terrain(TerrainTool),
    /// Pour water that pools into lakes and runs off as rivers.
    WaterBody,
}

impl BrushMode {
    /// Rain → Terrain → Water Body → Rain.
    pub fn next(self) -> Self {
        match self {
            Self::Rain => Self::Terrain(TerrainTool::Raise),
            Self::Terrain(_) => Self::WaterBody,
            Self::WaterBody => Self::Rain,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Rain => "Rain",
            Self::Terrain(TerrainTool::Raise) => "Terrain: Raise",
            Self::Terrain(TerrainTool::Lower) => "Terrain: Lower",
            Self::Terrain(TerrainTool::Flatten) => "Terrain: Flatten",
            Self::Terrain(TerrainTool::Smooth) => "Terrain: Smooth",
            Self::WaterBody => "Water Body",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TerrainTool {
    #[default]
    Raise,
    Lower,
    /// Pull toward the height under the cursor where the stroke started.
    Flatten,
    /// Blend toward a blurred version of the source noise.
    Smooth,
}

impl TerrainTool {
    pub fn next(self) -> Self {
        match self {
            Self::Raise => Self::Lower,
            Self::Lower => Self::Flatten,
            Self::Flatten => Self::Smooth,
            Self::Smooth => Self::Raise,
        }
    }
}

#[derive(Resource, Clone, Debug)]
pub struct WaterBrush {
    /// Paint on left click (orbit moves to the right button while enabled).
    pub enabled: bool,
    pub mode: BrushMode,
    /// Brush radius in map texels.
    pub radius: f32,
    /// Rate: rain trails (× [`RAIN_STARTS_PER_SECOND`]), terrain change, or water poured.
    pub strength: f32,
    /// Set to remove all water on the next frame.
    pub clear_water: bool,
    /// Set to undo all terrain edits on the next frame.
    pub clear_terrain: bool,
    /// Flatten target captured when the stroke started.
    flatten_target: Option<f32>,
}

impl WaterBrush {
    /// Rain for one frame of `dt` seconds centered on `center` (map texels).
    pub fn rain_stroke(&self, center: Vec2, dt: f32) -> BrushStroke {
        let area = std::f32::consts::PI * self.radius * self.radius;
        let starts = self.strength * RAIN_STARTS_PER_SECOND * area * dt;
        BrushStroke {
            active: true,
            center,
            radius: self.radius,
            particles: (starts / RAIN_PASSES as f32).ceil() as u32,
            passes: RAIN_PASSES,
        }
    }
}

impl Default for WaterBrush {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: BrushMode::Rain,
            radius: 12.0,
            strength: 0.35,
            clear_water: false,
            clear_terrain: false,
            flatten_target: None,
        }
    }
}

/// The displaced terrain mesh: a `size` plane centered on its transform, with
/// `height × height_scale` added along local +Y (see `terrain_extended.wesl`).
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

/// Applies the active brush for this frame (see the module docs).
#[allow(clippy::too_many_arguments)]
fn paint_water(
    mut brush: ResMut<WaterBrush>,
    mut stroke: ResMut<BrushStroke>,
    mut water: ResMut<WaterControl>,
    time: Res<Time>,
    hit: Res<BrushHit>,
    cache: Res<HeightCache>,
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
    let dt = time.delta_secs();
    let erase = keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]);
    let painting =
        brush.enabled && !blocked.0 && buttons.pressed(MouseButton::Left) && hit.world.is_some();
    if !buttons.pressed(MouseButton::Left) {
        brush.flatten_target = None;
    }

    *stroke = if painting && brush.mode == BrushMode::Rain {
        brush.rain_stroke(hit.texel, dt)
    } else {
        BrushStroke::default()
    };

    if std::mem::take(&mut brush.clear_water) {
        water.clear_generation = water.clear_generation.wrapping_add(1);
    }
    water.stroke_active = painting && brush.mode == BrushMode::WaterBody;
    water.center = hit.texel;
    water.radius = brush.radius;
    let pour = brush.strength * WATER_DEPTH_PER_SECOND * dt;
    water.rate = if erase { -4.0 * pour } else { pour };

    let clear_terrain = std::mem::take(&mut brush.clear_terrain);
    let tool = match brush.mode {
        BrushMode::Terrain(tool) if painting => Some(tool),
        _ => None,
    };
    if !clear_terrain && tool.is_none() {
        return;
    }
    let Some(mut image) = images.get_mut(&erosion_images.terrain_edit) else {
        return;
    };
    let size = image.size();
    let Some(data) = image.data.as_mut() else {
        return;
    };
    if clear_terrain {
        data.fill(0);
    }
    if let Some(tool) = tool {
        let target = *brush
            .flatten_target
            .get_or_insert_with(|| cache.sample(hit.texel));
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
                let w = brush.strength * dt * (1.0 - d * d).powi(2);
                let o = ((y * size.x + x) * 16) as usize;
                let mut e = Vec4::from_array(std::array::from_fn(|c| {
                    f32::from_le_bytes(data[o + c * 4..o + c * 4 + 4].try_into().unwrap())
                }));
                let blend = w * TERRAIN_BLEND_PER_SECOND;
                match (tool, erase) {
                    (TerrainTool::Raise | TerrainTool::Lower, true) => {
                        e.x -= e.x.signum() * (w * TERRAIN_HEIGHT_PER_SECOND).min(e.x.abs());
                    }
                    (TerrainTool::Raise, false) => e.x += w * TERRAIN_HEIGHT_PER_SECOND,
                    (TerrainTool::Lower, false) => e.x -= w * TERRAIN_HEIGHT_PER_SECOND,
                    (TerrainTool::Flatten, true) => e.y = (e.y - blend).max(0.0),
                    (TerrainTool::Flatten, false) => {
                        e.y = (e.y + blend).min(1.0);
                        e.z = target;
                    }
                    (TerrainTool::Smooth, true) => e.w = (e.w - blend).max(0.0),
                    (TerrainTool::Smooth, false) => e.w = (e.w + blend).min(1.0),
                }
                for (c, v) in e.to_array().into_iter().enumerate() {
                    data[o + c * 4..o + c * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
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
        match brush.mode {
            BrushMode::Rain => Color::srgb(0.75, 0.9, 1.0),
            BrushMode::Terrain(_) => Color::srgb(0.95, 0.8, 0.4),
            BrushMode::WaterBody => Color::srgb(0.3, 0.7, 1.0),
        }
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
