//! River brush geometry, on the CPU: a click on the eroded terrain becomes a river bed.
//!
//! 1. [`route`]: a priority flood from the clicked source. It always grows into the lowest cell
//!    next to what it has reached, so it runs down the valley the erosion made, fills any
//!    hollow on the way up to its rim and leaves over the lowest point of it. It stops at the
//!    sea, the map edge, standing water or another river.
//! 2. [`smooth_path`]: the 8-connected cell path, smoothed and resampled every texel.
//! 3. [`long_profile`]: the bank level along the path, made to run downhill. A rise up to
//!    `breach_depth` is cut through; a higher one is left for the water to fill as a lake.
//! 4. [`rasterize`]: the channel and its banks, per texel, sized by [`crate::ChannelModel`] for
//!    the bankfull discharge along the path: that of its drainage area
//!    ([`crate::drainage::DrainageArea`]) plus the catchment upstream of the click. Shape Beds'
//!    cross-section and bank grading ([`crate::BedShaping`]) shape the banks. The bed
//!    detail follows the reach: step-pools above about 3 % slope, riffle-pools above about 1 %,
//!    and point bars on the inside of bends everywhere.
//!
//! The routing, profile, channel model and raster live in the `river_geometry` crate, shared
//! with bevy_fft (its `docs/river_ecosystem.md`); this module adapts them to the brush.
//!
//! The raster ([`RiverCell`] per texel) goes to `river_carve` in `river.wgsl`, which lowers the
//! terrain to it and writes the armour band (a floor and a ceiling around the carved bed) that
//! water erosion, settling, slumping and bed load keep to.

use crate::{
    BedShaping, CELLS_PER_HEIGHT, ChannelModel, ErosionConfig, WaterControl, WorldScale,
    brush::WaterBrush, drainage::DrainageArea,
};
use bevy::{
    prelude::*,
    render::{RenderApp, extract_resource::ExtractResource},
};
use std::sync::Arc;

/// Water deeper than this (texels of depth, as [`crate::WET_DEPTH`]) ends a route: a lake or
/// another river.
const STOP_DEPTH: f32 = 0.2 / CELLS_PER_HEIGHT;
/// Water this close to the source (texels) does not end the route: it is the source's own.
const SOURCE_REACH: f32 = 6.0;
pub use river_geometry::NO_RIVER;

/// One texel of the river raster: the target height (or [`NO_RIVER`]), how far to blend back
/// toward the ground (0 in the channel, rising to 1 across the banks), the armour (0 to 1) and
/// the depth the armour protects below the carved bed (height units).
pub type RiverCell = Vec4;

/// The rivers' raster for the render world, which carves it (`river_carve`) on the finest level
/// whenever `generation` changes.
#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct RiverBed {
    pub generation: u32,
    /// One [`RiverCell`] per map texel; empty before the first river.
    pub cells: Arc<Vec<RiverCell>>,
}

/// River brush settings.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct RiverSettings {
    /// Catchment upstream of the click beyond what the map drains to it (km²): a river that
    /// comes from off the map, or from a larger range than the map shows.
    pub catchment: f32,
    /// How much of the carved bed the water may not rework: 1 keeps every step up and every pool
    /// open, 0 leaves the bed to the water, which wears the steps down and fills the pools.
    pub armour: f32,
    /// Strength of the steps, pools and point bars, 0 for a plain bed.
    pub bed_detail: f32,
    /// Highest rise (height units) the bed cuts through; behind a higher one the water forms a
    /// lake and spills over.
    pub breach_depth: f32,
}

impl Default for RiverSettings {
    fn default() -> Self {
        Self {
            catchment: 100.0,
            armour: 1.0,
            bed_detail: 1.0,
            breach_depth: 0.03,
        }
    }
}

/// A river placed with the brush: where it starts, its path and the bank level along it.
#[derive(Clone, Debug, PartialEq)]
pub struct River {
    pub source: Vec2,
    /// Catchment upstream of the source (m²), beyond the map's own drainage.
    pub catchment: f32,
    /// Bankfull discharge (m³/s) at each path sample.
    pub flow: Vec<f32>,
    /// Seeds the step and pool spacing.
    pub seed: u32,
    /// Centreline, one sample per texel, in map texels.
    pub path: Vec<Vec2>,
    /// Bank level (height units) at each path sample, running downhill.
    pub profile: Vec<f32>,
}

/// Hydraulic geometry ([`ChannelModel`]) and bank grading ([`BedShaping`]).
#[derive(Clone, Copy, Debug)]
pub struct ChannelShape {
    pub model: ChannelModel,
    pub scale: WorldScale,
    pub profile: f32,
    pub max_depth: f32,
    pub bank_width: f32,
    pub bank_slope: f32,
}

impl ChannelShape {
    pub fn new(water: &WaterControl, scale: &WorldScale) -> Self {
        let bed: &BedShaping = &water.bed;
        Self {
            model: water.channel,
            scale: *scale,
            profile: bed.profile,
            max_depth: bed.max_depth,
            bank_width: bed.bank_width.clamp(1.0, crate::MAX_BANK_WIDTH),
            bank_slope: bed.bank_slope,
        }
    }

    /// Half-width (texels) and depth (height units) of the channel for `q` m³/s.
    pub fn section(&self, q: f32) -> (f32, f32) {
        self.model
            .half_width_and_depth(q, self.scale.metres_per_texel, CELLS_PER_HEIGHT)
    }
}

/// The rivers placed with the brush. They survive reruns of the cascade: once it finishes,
/// [`refresh_after_cascade`] gives them the new terrain's bank levels and flows and carves them
/// again.
#[derive(Resource, Default)]
pub struct Rivers {
    pub list: Vec<River>,
    next_seed: u32,
}

impl Rivers {
    pub fn next_seed(&mut self) -> u32 {
        self.next_seed = self.next_seed.wrapping_add(1);
        self.next_seed
    }
}

/// Rasterize `rivers` for the render world, which carves them on its next finished frame.
pub fn rebuild(
    rivers: &Rivers,
    shape: &ChannelShape,
    settings: &RiverSettings,
    size: UVec2,
    out: &mut RiverBed,
) {
    out.cells = Arc::new(rasterize(&rivers.list, shape, settings, size));
    out.generation = out.generation.wrapping_add(1);
}

/// The channel is sized for bankfull flow, which comes about every year or two; the spring
/// feeds this share of it, so the river runs inside its banks.
const SPRING_SHARE: f32 = 0.5;

/// The spring that feeds a river, over a disc about the channel's width.
pub fn spring_for(river: &River, shape: &ChannelShape) -> crate::Spring {
    let q = river.flow.first().copied().unwrap_or(0.0);
    let radius = shape.section(q).0.max(2.0);
    // In height units × texels² per second.
    let scale = &shape.scale;
    let q = q / (scale.metres_per_height() * scale.metres_per_texel * scale.metres_per_texel);
    // A spring adds `rate` at its centre, falling off as (1 − r²)², which integrates to πR²/3.
    crate::Spring {
        center: river.source,
        radius,
        rate: SPRING_SHARE * q * 3.0 / (std::f32::consts::PI * radius * radius),
    }
}

pub struct RiverPlugin;

impl Plugin for RiverPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Rivers>()
            .add_systems(Update, (refresh_after_cascade, follow_armour));
    }
}

/// When a rerun of the cascade finishes and its drainage area is in, give every river the new
/// terrain's bank levels and flows (its path stays). The raster then goes to the render world.
fn refresh_after_cascade(
    drainage: Res<DrainageArea>,
    mut rivers: ResMut<Rivers>,
    brush: Res<WaterBrush>,
    water: Res<WaterControl>,
    scale: Res<WorldScale>,
    mut bed: ResMut<RiverBed>,
) {
    if !drainage.is_changed() || drainage.area.is_empty() || rivers.list.is_empty() {
        return;
    }
    let shape = ChannelShape::new(&water, &scale);
    for river in &mut rivers.list {
        reprofile(
            river,
            &drainage.heights,
            drainage.size,
            brush.river.breach_depth,
        );
        set_flow(river, &drainage, &shape.model);
        info!(
            "River from {}: {:.1} m³/s at the source, {:.1} m³/s at the mouth",
            river.source,
            river.flow[0],
            river.flow.last().unwrap()
        );
    }
    rebuild(&rivers, &shape, &brush.river, drainage.size, &mut bed);
}

/// Armour only sets the band `river_carve` writes, so a change re-sends the raster. (Bed Detail
/// and Breach Depth change the beds themselves and rerun the cascade instead.)
fn follow_armour(
    brush: Res<WaterBrush>,
    rivers: Res<Rivers>,
    water: Res<WaterControl>,
    scale: Res<WorldScale>,
    config: Res<ErosionConfig>,
    mut bed: ResMut<RiverBed>,
    mut last: Local<Option<f32>>,
) {
    let armour = brush.river.armour;
    if last.replace(armour).is_some_and(|a| a != armour) && !rivers.list.is_empty() {
        let shape = ChannelShape::new(&water, &scale);
        rebuild(&rivers, &shape, &brush.river, config.map_size, &mut bed);
    }
}

/// Cells from `source` down to where the river ends (see the module docs). `heights` and
/// `water` are per texel; `taken` marks texels of earlier rivers, which a new one joins.
pub fn route(
    heights: &[f32],
    water: &[f32],
    taken: &[bool],
    size: UVec2,
    source: Vec2,
    sea_level: f32,
) -> Vec<UVec2> {
    let ends = |i: usize| {
        let c = UVec2::new(i as u32 % size.x, i as u32 / size.x);
        let away = (c.as_vec2() + 0.5).distance(source) > SOURCE_REACH;
        (sea_level > 0.0 && heights[i] < sea_level)
            || (away && (water.get(i).is_some_and(|&d| d > STOP_DEPTH) || taken[i]))
    };
    river_geometry::route(heights, size.to_array(), source.to_array(), ends)
        .into_iter()
        .map(UVec2::from)
        .collect()
}

/// Texel centres of `cells`, smoothed (twice, over `radius` samples each side, ends kept) and
/// resampled one texel apart.
pub fn smooth_path(cells: &[UVec2], radius: usize) -> Vec<Vec2> {
    let cells: Vec<[u32; 2]> = cells.iter().map(|c| c.to_array()).collect();
    river_geometry::smooth_path(&cells, radius)
        .into_iter()
        .map(Vec2::from)
        .collect()
}

/// Bank level along a path whose ground is `ground`: it never rises, except over a rise higher
/// than `breach_depth`, which it follows (the water fills a lake behind it and spills over).
/// A rise it cuts through gets a steady slope from where it starts to where the ground is back
/// below it.
pub fn long_profile(ground: &[f32], breach_depth: f32) -> Vec<f32> {
    river_geometry::long_profile(ground, breach_depth)
}

/// The rivers' beds, banks and armour, one [`RiverCell`] per texel of a `size` map.
pub fn rasterize(
    rivers: &[River],
    shape: &ChannelShape,
    settings: &RiverSettings,
    size: UVec2,
) -> Vec<RiverCell> {
    let rivers: Vec<river_geometry::River> = rivers
        .iter()
        .map(|r| river_geometry::River {
            path: r.path.iter().map(|p| p.to_array()).collect(),
            profile: r.profile.clone(),
            flow: r.flow.clone(),
            seed: r.seed,
        })
        .collect();
    let shape = river_geometry::ChannelShape {
        model: shape.model,
        metres_per_texel: shape.scale.metres_per_texel,
        cells_per_height: CELLS_PER_HEIGHT,
        profile: shape.profile,
        max_depth: shape.max_depth,
        bank_width: shape.bank_width,
        bank_slope: shape.bank_slope,
    };
    river_geometry::rasterize(
        &rivers,
        &shape,
        settings.bed_detail,
        settings.armour,
        size.to_array(),
    )
    .into_iter()
    .map(Vec4::from)
    .collect()
}

/// Route a river from `source` and give it its bank level and flow, or `None` if it goes
/// nowhere. `drainage` sizes it; before the first one is in, only the upstream catchment does.
#[allow(clippy::too_many_arguments)]
pub fn place(
    heights: &[f32],
    water: &[f32],
    rivers: &[River],
    size: UVec2,
    source: Vec2,
    sea_level: f32,
    settings: &RiverSettings,
    shape: &ChannelShape,
    drainage: &DrainageArea,
    seed: u32,
) -> Option<River> {
    let mut taken = vec![false; (size.x * size.y) as usize];
    for p in rivers.iter().flat_map(|r| &r.path) {
        let c = p.floor().as_uvec2().min(size - 1);
        taken[(c.y * size.x + c.x) as usize] = true;
    }
    let cells = route(heights, water, &taken, size, source, sea_level);
    let catchment = settings.catchment.max(0.0) * 1.0e6;
    let q = shape.model.discharge(catchment + drainage.at(source));
    let radius = shape.section(q).0.round().max(2.0) as usize;
    let path = smooth_path(&cells, radius);
    if path.len() < 4 {
        return None;
    }
    let mut river = River {
        source,
        catchment,
        flow: Vec::new(),
        seed,
        path,
        profile: Vec::new(),
    };
    reprofile(&mut river, heights, size, settings.breach_depth);
    set_flow(&mut river, drainage, &shape.model);
    Some(river)
}

/// Bankfull discharge along `river` from its drainage area, which only grows downstream (the
/// multiple-flow-direction area dips where flow spreads across the path), plus its upstream
/// catchment.
pub fn set_flow(river: &mut River, drainage: &DrainageArea, model: &ChannelModel) {
    let mut area = 0.0f32;
    river.flow = river
        .path
        .iter()
        .map(|p| {
            area = area.max(drainage.at(*p));
            model.discharge(river.catchment + area)
        })
        .collect();
}

/// Bank level of `river` on the terrain `heights` (after the terrain changed under it).
pub fn reprofile(river: &mut River, heights: &[f32], size: UVec2, breach_depth: f32) {
    let ground: Vec<f32> = river
        .path
        .iter()
        .map(|p| bilinear(heights, size, *p))
        .collect();
    river.profile = long_profile(&ground, breach_depth);
}

/// `field` at a point in texels, between texel centres.
pub fn bilinear(field: &[f32], size: UVec2, p: Vec2) -> f32 {
    river_geometry::bilinear(field, size.to_array(), p.to_array())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: UVec2 = UVec2::new(64, 64);

    fn shape() -> ChannelShape {
        ChannelShape {
            model: ChannelModel::default(),
            scale: WorldScale::default(),
            profile: 0.8,
            max_depth: 0.025,
            bank_width: 4.0,
            bank_slope: 0.0015,
        }
    }

    /// A valley along y = 32 falling toward +x.
    fn valley(x: u32, y: u32) -> f32 {
        0.5 - 0.002 * x as f32 + 0.0005 * (y as f32 - 32.0).abs()
    }

    fn field(f: impl Fn(u32, u32) -> f32) -> Vec<f32> {
        (0..SIZE.x * SIZE.y)
            .map(|i| f(i % SIZE.x, i / SIZE.x))
            .collect()
    }

    #[test]
    fn route_runs_down_the_valley_to_the_edge() {
        let h = field(valley);
        let cells = route(
            &h,
            &[],
            &vec![false; h.len()],
            SIZE,
            Vec2::new(5.5, 20.5),
            0.0,
        );
        let end = *cells.last().unwrap();
        assert_eq!(end.x, SIZE.x - 1, "ended at {end}");
        // It reaches the valley floor and follows it.
        assert!(cells.iter().any(|c| c.y == 32));
        assert!((end.y as i32 - 32).abs() <= 1);
    }

    #[test]
    fn route_spills_out_of_a_hollow() {
        // A pit at x = 20 in the valley: the route crosses it and carries on.
        let h = field(|x, y| valley(x, y) - 0.01 * (-((x as f32 - 20.0).powi(2)) / 8.0).exp());
        let cells = route(
            &h,
            &[],
            &vec![false; h.len()],
            SIZE,
            Vec2::new(5.5, 32.5),
            0.0,
        );
        assert_eq!(cells.last().unwrap().x, SIZE.x - 1);
    }

    #[test]
    fn route_stops_at_water_and_other_rivers() {
        let h = field(valley);
        let mut water = vec![0.0; h.len()];
        water[(32 * SIZE.x + 40) as usize] = 0.01;
        let cells = route(
            &h,
            &water,
            &vec![false; h.len()],
            SIZE,
            Vec2::new(5.5, 32.5),
            0.0,
        );
        assert_eq!(*cells.last().unwrap(), UVec2::new(40, 32));
    }

    #[test]
    fn smoothed_path_is_evenly_spaced() {
        let cells: Vec<UVec2> = (0..20).map(|i| UVec2::new(i, i / 2)).collect();
        let path = smooth_path(&cells, 3);
        for w in path.windows(2).take(path.len() - 2) {
            assert!((w[0].distance(w[1]) - 1.0).abs() < 1e-3);
        }
        assert_eq!(path[0], Vec2::new(0.5, 0.5));
    }

    #[test]
    fn profile_cuts_low_rises_and_keeps_high_ones() {
        let ground = [1.0, 0.9, 0.95, 0.85, 0.8, 1.2, 0.7];
        let z = long_profile(&ground, 0.1);
        assert!(z.windows(2).take(4).all(|w| w[1] <= w[0]), "{z:?}");
        assert!(z.iter().zip(&ground).all(|(z, g)| z <= g));
        // The 0.4 rise at index 5 is higher than the breach depth: the bed follows it.
        assert_eq!(z[5], 1.2);
    }

    #[test]
    fn raster_cuts_a_channel_with_graded_banks() {
        let h = field(valley);
        let settings = RiverSettings::default();
        let river = place(
            &h,
            &[],
            &[],
            SIZE,
            Vec2::new(5.5, 32.5),
            0.0,
            &settings,
            &shape(),
            &DrainageArea::default(),
            1,
        )
        .unwrap();
        let cells = rasterize(&[river], &shape(), &settings, SIZE);
        let at = |x: u32, y: u32| cells[(y * SIZE.x + x) as usize];
        // On the thalweg: below the ground, fully armoured, no blend.
        let mid = at(30, 32);
        assert!(mid.x < valley(30, 32), "{mid}");
        assert_eq!(mid.y, 0.0);
        assert_eq!(mid.z, 1.0);
        // On the bank: blending back toward the ground; far away: untouched.
        let side = at(30, 32 + 5);
        assert!(side.y > 0.0 && side.y < 1.0, "{side}");
        assert_eq!(at(30, 5).x, NO_RIVER);
    }
}
