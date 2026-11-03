//! Landslides with MPM (Phase 9 of `docs/UNIFIED_PLAN.md`), as local events.
//!
//! A click with the Landslide brush releases a slab of the terrain under the brush: a paraboloid
//! slip surface [`LandslideSettings::depth`] deep at the centre. The GPU (`assets/landslide.wgsl`)
//! turns it into particles in a local grid of [`NODES`] cells, half a texel each, and moves them
//! as a frictional granular continuum (MLS-MPM) over the terrain until the event's time is up.
//! Meanwhile the moving debris is written into the heights every frame, so the water and the
//! drawing see it; at the end it stays as the deposit.

use crate::{
    ErosionConfig, WaterControl, WaterFields, WorldScale,
    brush::{BrushHit, BrushMode, WaterBrush},
    drainage::DrainageArea,
};
use bevy::{
    input::mouse::MouseButton,
    prelude::*,
    render::{
        RenderApp,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        gpu_readback::{Readback, ReadbackComplete},
    },
};

/// Grid nodes of the local box (x, y up, z). Its cells are half a texel, so the box is
/// `NODES.x / 2` texels across.
pub const NODES: UVec3 = UVec3::new(128, 64, 128);
/// Most particles in one event.
pub const MAX_PARTICLES: u32 = 262_144;
/// MPM substep (s).
pub const SLIDE_STEP: f32 = 0.02;
/// Most substeps per frame.
pub const MAX_SLIDE_SUBSTEPS: u32 = 16;
/// Unit weight of the soil (N/m³).
pub const SOIL_WEIGHT: f32 = 20_000.0;
/// Real seconds after a slide settles before the next one may release on its own.
const AUTO_PAUSE: f32 = 2.0;

/// How landslides release and run. In SI.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LandslideSettings {
    /// Depth of the slip surface at the release's centre (m).
    pub depth: f32,
    /// Internal friction angle (degrees): the debris holds shear stress up to tan(this) × its
    /// pressure, so it comes to rest at about this slope.
    pub friction_angle: f32,
    /// Friction angle of the debris sliding on the terrain (degrees).
    pub basal_angle: f32,
    /// Seconds of slide each event runs before it settles.
    pub duration: f32,
    /// Slide seconds per real second.
    pub speed: f32,
    /// Speed of sound in the debris (m/s): how stiffly it resists compaction. The substep is
    /// sized for it.
    pub sound_speed: f32,
    /// Release slides where slopes fail on their own: where the infinite-slope factor of safety
    /// of a slab [`Self::depth`] deep falls below 1 (see `safety` in `landslide.wgsl`).
    pub auto: bool,
    /// Friction angle of the intact slope (degrees).
    pub soil_friction_angle: f32,
    /// Cohesion of the soil (kPa), and what the hardest rock of KTT's softness mask adds.
    pub cohesion: f32,
    pub rock_cohesion: f32,
    /// Radius (texels) of the slides that release on their own.
    pub auto_radius: f32,
    /// How fast water drains through the soil along the slope (m²/day): rain saturates slopes
    /// where it arrives faster than this carries it away (SHALSTAB). Soils measure 1–100.
    pub transmissivity: f32,
}

impl Default for LandslideSettings {
    fn default() -> Self {
        Self {
            depth: 25.0,
            friction_angle: 30.0,
            basal_angle: 20.0,
            duration: 60.0,
            speed: 2.0,
            sound_speed: 40.0,
            // Dry slopes up to about 45° hold; saturated ones fail from about 30°.
            auto: true,
            soil_friction_angle: 40.0,
            cohesion: 50.0,
            rock_cohesion: 500.0,
            auto_radius: 8.0,
            transmissivity: 65.0,
        }
    }
}

/// One release.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlideEvent {
    /// Centre and radius in map texels.
    pub centre: Vec2,
    pub radius: f32,
    /// Slip surface depth at the centre (m).
    pub depth: f32,
    /// First map texel of the local box.
    pub box_origin: IVec2,
    /// Height of the box floor (m).
    pub floor: f32,
}

/// The latest release (the settings are [`WaterControl::landslide`]); the render world starts an event each time
/// [`Self::generation`] changes.
#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct Landslides {
    pub event: Option<SlideEvent>,
    pub generation: u32,
    /// A release (centre, radius, depth) waiting for the terrain's heights to be read back.
    pub pending: Option<(Vec2, f32, f32)>,
}

/// The stability scan's latest result, read back from the GPU.
#[derive(Resource, Default, Clone, Copy, Debug)]
pub struct SlopeFailure {
    /// The least stable texel and its factor of safety, where one is below 1.
    pub weakest: Option<(UVec2, f32)>,
    /// Texels below 1.
    pub failing: u32,
}

impl Landslides {
    /// Release a slide of `radius` texels at `centre`, `depth` m deep at the centre, its box
    /// placed over the terrain `heights` (height units, `size` texels; the drainage readback).
    pub fn release(
        &mut self,
        centre: Vec2,
        radius: f32,
        depth: f32,
        heights: &[f32],
        size: UVec2,
        scale: &WorldScale,
    ) {
        if heights.is_empty() || heights.len() != (size.x * size.y) as usize {
            self.pending = Some((centre, radius, depth));
            return;
        }
        let texels = (NODES.xz() / 2).as_ivec2();
        let max = (size.as_ivec2() - texels).max(IVec2::ZERO);
        let box_origin = (centre.as_ivec2() - texels / 2).clamp(IVec2::ZERO, max);
        let metres = |h: f32| h * scale.metres_per_height();
        let mut lowest = f32::MAX;
        for y in box_origin.y..(box_origin.y + texels.y).min(size.y as i32) {
            for x in box_origin.x..(box_origin.x + texels.x).min(size.x as i32) {
                lowest = lowest.min(heights[(y as u32 * size.x + x as u32) as usize]);
            }
        }
        let c = centre.max(Vec2::ZERO).as_uvec2().min(size - 1);
        let at_centre = heights[(c.y * size.x + c.x) as usize];
        // The floor sits below the lowest ground in the box, but no lower than leaves the
        // release inside the box's height.
        let cell = 0.5 * scale.metres_per_texel;
        let reach = (NODES.y as f32 - 8.0) * cell;
        let floor = metres(lowest).max(metres(at_centre) - 0.7 * reach) - 3.0 * cell;
        self.event = Some(SlideEvent {
            centre,
            radius,
            depth,
            box_origin,
            floor,
        });
        self.generation = self.generation.wrapping_add(1);
        self.pending = None;
        info!(
            "Landslide at {centre} ({:.0} m across, {:.0} m deep)",
            2.0 * radius * scale.metres_per_texel,
            depth
        );
    }
}

pub struct LandslidePlugin;

impl Plugin for LandslidePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Landslides>()
            .init_resource::<SlopeFailure>()
            .add_plugins(ExtractResourcePlugin::<Landslides>::default())
            .add_systems(
                Update,
                (sync_failure_readback, release_pending, click_landslide, release_failures),
            );
    }
}

/// Release a slide that came before the heights did.
fn release_pending(
    drainage: Res<DrainageArea>,
    scale: Res<WorldScale>,
    mut slides: ResMut<Landslides>,
) {
    if let Some((centre, radius, depth)) = slides.pending
        && !drainage.heights.is_empty()
    {
        slides.release(centre, radius, depth, &drainage.heights, drainage.size, &scale);
    }
}

/// A click with the Landslide brush releases a slide under it.
fn click_landslide(
    brush: Option<Res<WaterBrush>>,
    hit: Option<Res<BrushHit>>,
    buttons: Option<Res<ButtonInput<MouseButton>>>,
    drainage: Res<DrainageArea>,
    scale: Res<WorldScale>,
    water: Res<WaterControl>,
    mut slides: ResMut<Landslides>,
) {
    let (Some(brush), Some(hit), Some(buttons)) = (brush, hit, buttons) else {
        return;
    };
    if brush.enabled
        && brush.mode == BrushMode::Landslide
        && hit.world.is_some()
        && buttons.just_pressed(MouseButton::Left)
    {
        let depth = water.landslide.depth;
        slides.release(hit.texel, brush.radius, depth, &drainage.heights, drainage.size, &scale);
    }
}

#[derive(Component)]
struct FailureReadback;

/// Read the stability scan back every frame while slopes may fail on their own.
fn sync_failure_readback(
    mut commands: Commands,
    water: Res<WaterControl>,
    fields: Option<Res<WaterFields>>,
    config: Res<ErosionConfig>,
    mut failure: ResMut<SlopeFailure>,
    existing: Query<Entity, With<FailureReadback>>,
) {
    let Some(fields) = fields else { return };
    if water.landslide.auto && existing.is_empty() {
        let width = config.map_size.x;
        commands
            .spawn((Readback::buffer(fields.slope_failure.clone()), FailureReadback))
            .observe(move |ev: On<ReadbackComplete>, mut failure: ResMut<SlopeFailure>| {
                let words: Vec<u32> = bytemuck::pod_collect_to_vec(&ev.data);
                let [fs, index, failing, ..] = words[..] else {
                    return;
                };
                failure.failing = failing;
                failure.weakest = (index > 0).then(|| {
                    let i = index - 1;
                    (UVec2::new(i % width, i / width), f32::from_bits(fs))
                });
            });
    } else if !water.landslide.auto && !existing.is_empty() {
        for entity in &existing {
            commands.entity(entity).despawn();
        }
        *failure = SlopeFailure::default();
    }
}

/// Release a slide at the least stable texel once the last one has settled.
fn release_failures(
    time: Res<Time>,
    water: Res<WaterControl>,
    failure: Res<SlopeFailure>,
    drainage: Res<DrainageArea>,
    scale: Res<WorldScale>,
    mut slides: ResMut<Landslides>,
    mut busy_until: Local<f32>,
    mut seen: Local<u32>,
) {
    let s = &water.landslide;
    let now = time.elapsed_secs();
    // Any release (the brush's too) keeps the slope still until its slide has settled.
    if slides.generation != *seen {
        *seen = slides.generation;
        *busy_until = now + s.duration / s.speed.max(1e-3) + AUTO_PAUSE;
    }
    if !s.auto || now < *busy_until || slides.pending.is_some() || drainage.heights.is_empty() {
        return;
    }
    if let Some((texel, fs)) = failure.weakest {
        info!(
            "Slope failure at {texel}: factor of safety {fs:.2} ({} texels below 1)",
            failure.failing
        );
        let centre = texel.as_vec2() + 0.5;
        slides.release(centre, s.auto_radius, s.depth, &drainage.heights, drainage.size, &scale);
    }
}
