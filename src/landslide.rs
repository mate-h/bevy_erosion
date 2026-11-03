//! Landslides with MPM (Phase 9 of `docs/UNIFIED_PLAN.md`), as local events.
//!
//! A click with the Landslide brush releases a slab of the terrain under the brush: a paraboloid
//! slip surface [`LandslideSettings::depth`] deep at the centre. The GPU (`assets/landslide.wgsl`)
//! turns it into particles in a local grid of [`NODES`] cells, half a texel each, and moves them
//! as a frictional granular continuum (MLS-MPM) over the terrain until the event's time is up.
//! Meanwhile the moving debris is written into the heights every frame, so the water and the
//! drawing see it; at the end it stays as the deposit.

use crate::{
    WorldScale,
    brush::{BrushHit, BrushMode, WaterBrush},
    drainage::DrainageArea,
};
use bevy::{
    input::mouse::MouseButton,
    prelude::*,
    render::{
        RenderApp,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
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

/// The landslide settings and the latest release; the render world starts an event each time
/// [`Self::generation`] changes.
#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct Landslides {
    pub settings: LandslideSettings,
    pub event: Option<SlideEvent>,
    pub generation: u32,
    /// A release (centre, radius) waiting for the terrain's heights to be read back.
    pub pending: Option<(Vec2, f32)>,
}

impl Landslides {
    /// Release a slide of `radius` texels at `centre`, its box placed over the terrain `heights`
    /// (height units, `size` texels; the drainage readback).
    pub fn release(
        &mut self,
        centre: Vec2,
        radius: f32,
        heights: &[f32],
        size: UVec2,
        scale: &WorldScale,
    ) {
        if heights.is_empty() || heights.len() != (size.x * size.y) as usize {
            self.pending = Some((centre, radius));
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
            depth: self.settings.depth,
            box_origin,
            floor,
        });
        self.generation = self.generation.wrapping_add(1);
        self.pending = None;
        info!(
            "Landslide at {centre} ({:.0} m across, {:.0} m deep)",
            2.0 * radius * scale.metres_per_texel,
            self.settings.depth
        );
    }
}

pub struct LandslidePlugin;

impl Plugin for LandslidePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Landslides>()
            .add_plugins(ExtractResourcePlugin::<Landslides>::default())
            .add_systems(Update, (release_pending, click_landslide));
    }
}

/// Release a slide that came before the heights did.
fn release_pending(
    drainage: Res<DrainageArea>,
    scale: Res<WorldScale>,
    mut slides: ResMut<Landslides>,
) {
    if let Some((centre, radius)) = slides.pending
        && !drainage.heights.is_empty()
    {
        slides.release(centre, radius, &drainage.heights, drainage.size, &scale);
    }
}

/// A click with the Landslide brush releases a slide under it.
fn click_landslide(
    brush: Option<Res<WaterBrush>>,
    hit: Option<Res<BrushHit>>,
    buttons: Option<Res<ButtonInput<MouseButton>>>,
    drainage: Res<DrainageArea>,
    scale: Res<WorldScale>,
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
        slides.release(hit.texel, brush.radius, &drainage.heights, drainage.size, &scale);
    }
}
