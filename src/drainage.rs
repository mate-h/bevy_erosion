//! Drainage area, on the CPU: the area of the map (and of any catchment beyond its edge) whose
//! rain runs through each texel. It sizes the rivers (see `docs/UNIFIED_PLAN.md`).
//!
//! 1. [`fill`]: priority-flood depression filling (Barnes et al. 2014) from the outlets (the map
//!    border and the sea), raising every pit to its spill level plus a tiny gradient, so all
//!    water reaches an outlet.
//! 2. [`accumulate`]: multiple-flow-direction accumulation (Quinn et al. 1991 / Freeman 1991)
//!    on the filled surface, highest cell first. Each cell passes its area to all lower
//!    neighbours in proportion to slope^[`MFD_EXPONENT`]: single-direction (D8) routing draws
//!    rivers along the grid axes and diagonals.
//!
//! The KTT erosion does not give drainage area: its flow buffer counts deposition per trail step,
//! and its river momentum is trail water capped by Flow Length.

use crate::{ErodeParams, ErosionConfig, ErosionImages, PyramidProgress, WorldScale};
use bevy::{
    prelude::*,
    render::gpu_readback::{Readback, ReadbackComplete},
};
use std::{cmp::Reverse, collections::BinaryHeap, sync::Arc};

/// Flow partition exponent: 1.1 is Freeman's; larger concentrates flow into fewer neighbours.
const MFD_EXPONENT: f64 = 1.1;
/// Height added per step away from an outlet across filled flats (height units), so they drain.
const FLAT_GRADIENT: f64 = 1e-9;

const NEIGHBOURS: [(i32, i32); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// Drainage area of the eroded terrain, recomputed each time the cascade finishes.
#[derive(Resource, Clone, Default)]
pub struct DrainageArea {
    pub size: UVec2,
    /// The heights it was computed on (height units per map texel).
    pub heights: Arc<Vec<f32>>,
    /// m² per map texel; empty before the first cascade has finished.
    pub area: Arc<Vec<f32>>,
    /// Bumped on every update.
    pub generation: u32,
}

impl DrainageArea {
    /// Drainage area (m²) at a point in texels, or 0 before it is known.
    pub fn at(&self, p: Vec2) -> f32 {
        crate::river::bilinear(&self.area, self.size, p)
    }
}

pub struct DrainagePlugin;

impl Plugin for DrainagePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DrainageArea>()
            .add_systems(Update, refresh_drainage);
    }
}

#[derive(Component)]
struct DrainageReadback;

/// When the cascade finishes, read its heights back and compute the drainage area, for the
/// rivers and the Drainage view (`ErosionImages::drainage`).
fn refresh_drainage(
    mut commands: Commands,
    progress: Option<Res<PyramidProgress>>,
    images: Option<Res<ErosionImages>>,
    config: Res<ErosionConfig>,
    pending: Query<(), With<DrainageReadback>>,
    mut was_finished: Local<bool>,
) {
    let (Some(progress), Some(images)) = (progress, images) else {
        return;
    };
    let finished = progress.get().finished;
    let rising = finished && !*was_finished;
    *was_finished = finished;
    if !rising || !pending.is_empty() {
        return;
    }
    let size = config.map_size;
    commands
        .spawn((Readback::texture(images.height.clone()), DrainageReadback))
        .observe(
            move |ev: On<ReadbackComplete>,
                  mut commands: Commands,
                  params: Res<ErodeParams>,
                  scale: Res<WorldScale>,
                  images: Res<ErosionImages>,
                  mut image_assets: ResMut<Assets<Image>>,
                  mut drainage: ResMut<DrainageArea>| {
                commands.entity(ev.entity).despawn();
                let mut heights = Vec::new();
                crate::brush::read_red_channel(&ev.data, size, &mut heights);
                let sea_level = if params.use_outflow != 0 {
                    params.outflow_sea_level
                } else {
                    0.0
                };
                let area = drainage_area(&heights, size, scale.metres_per_texel, sea_level, &[]);
                if let Some(mut image) = image_assets.get_mut(&images.drainage) {
                    image.data = Some(area.iter().flat_map(|a| a.to_le_bytes()).collect());
                }
                drainage.size = size;
                drainage.area = Arc::new(area);
                drainage.heights = Arc::new(heights);
                drainage.generation = drainage.generation.wrapping_add(1);
            },
        );
}

/// Water entering the map from a catchment beyond its edge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Inlet {
    /// Map texel it enters at.
    pub texel: UVec2,
    /// Upstream catchment (m²).
    pub area: f32,
}

/// Whether texel `i` drains away: the map border, or below the sea (`sea_level` > 0).
fn is_outlet(heights: &[f32], size: UVec2, sea_level: f32, i: usize) -> bool {
    let (x, y) = (i as u32 % size.x, i as u32 / size.x);
    x == 0
        || y == 0
        || x == size.x - 1
        || y == size.y - 1
        || (sea_level > 0.0 && heights[i] < sea_level)
}

/// `heights` (height units) with every depression filled to its spill level, plus a tiny
/// gradient toward the outlets so filled flats drain.
pub fn fill(heights: &[f32], size: UVec2, sea_level: f32) -> Vec<f64> {
    let n = (size.x * size.y) as usize;
    let mut filled: Vec<f64> = heights.iter().map(|&h| h as f64).collect();
    let mut done = vec![false; n];
    let mut open = BinaryHeap::new();
    let key = |h: f64| Reverse(h.to_bits());
    for i in 0..n {
        if is_outlet(heights, size, sea_level, i) {
            done[i] = true;
            open.push((key(filled[i].max(0.0)), i));
        }
    }
    while let Some((_, i)) = open.pop() {
        let (x, y) = ((i as u32 % size.x) as i32, (i as u32 / size.x) as i32);
        for (dx, dy) in NEIGHBOURS {
            let (qx, qy) = (x + dx, y + dy);
            if qx < 0 || qy < 0 || qx >= size.x as i32 || qy >= size.y as i32 {
                continue;
            }
            let j = (qy as u32 * size.x + qx as u32) as usize;
            if done[j] {
                continue;
            }
            done[j] = true;
            filled[j] = filled[j].max(filled[i] + FLAT_GRADIENT);
            open.push((key(filled[j].max(0.0)), j));
        }
    }
    filled
}

/// Drainage area (m²) of every texel on the filled surface `filled`, for texels `metres_per_texel`
/// wide, with the `inlets`' catchments added where they enter.
pub fn accumulate(
    filled: &[f64],
    size: UVec2,
    metres_per_texel: f32,
    inlets: &[Inlet],
) -> Vec<f32> {
    let n = (size.x * size.y) as usize;
    let cell = (metres_per_texel as f64).powi(2);
    let mut area = vec![cell; n];
    for inlet in inlets {
        let t = inlet.texel.min(size - 1);
        area[(t.y * size.x + t.x) as usize] += inlet.area as f64;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| filled[b].total_cmp(&filled[a]));
    let mut weights = [0.0f64; 8];
    for i in order {
        let (x, y) = ((i as u32 % size.x) as i32, (i as u32 / size.x) as i32);
        let mut total = 0.0;
        for (k, (dx, dy)) in NEIGHBOURS.iter().enumerate() {
            weights[k] = 0.0;
            let (qx, qy) = (x + dx, y + dy);
            if qx < 0 || qy < 0 || qx >= size.x as i32 || qy >= size.y as i32 {
                continue;
            }
            let j = (qy as u32 * size.x + qx as u32) as usize;
            let drop = filled[i] - filled[j];
            if drop > 0.0 {
                let run = if dx * dy != 0 { 2f64.sqrt() } else { 1.0 };
                // Freeman weights the slope by the contour length the flow crosses.
                let contour = if dx * dy != 0 { 0.354 } else { 0.5 };
                weights[k] = (drop / run).powf(MFD_EXPONENT) * contour;
                total += weights[k];
            }
        }
        if total <= 0.0 {
            continue;
        }
        let a = area[i];
        for (k, (dx, dy)) in NEIGHBOURS.iter().enumerate() {
            if weights[k] > 0.0 {
                let j = ((y + dy) as u32 * size.x + (x + dx) as u32) as usize;
                area[j] += a * weights[k] / total;
            }
        }
    }
    area.into_iter().map(|a| a as f32).collect()
}

/// Drainage area (m²) of every texel of `heights` (height units): [`fill`] then [`accumulate`].
pub fn drainage_area(
    heights: &[f32],
    size: UVec2,
    metres_per_texel: f32,
    sea_level: f32,
    inlets: &[Inlet],
) -> Vec<f32> {
    accumulate(
        &fill(heights, size, sea_level),
        size,
        metres_per_texel,
        inlets,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: UVec2 = UVec2::new(64, 64);

    fn field(f: impl Fn(u32, u32) -> f32) -> Vec<f32> {
        (0..SIZE.x * SIZE.y)
            .map(|i| f(i % SIZE.x, i / SIZE.x))
            .collect()
    }

    fn at(v: &[f32], x: u32, y: u32) -> f32 {
        v[(y * SIZE.x + x) as usize]
    }

    #[test]
    fn plane_area_grows_downslope() {
        // Tilted toward x = 0. The y borders are outlets too, so measure the middle row, far from
        // them.
        let heights = field(|x, _| 1.0 + 0.01 * x as f32);
        let area = drainage_area(&heights, SIZE, 1.0, 0.0, &[]);
        let mid = SIZE.y / 2;
        // Each texel gathers the strip upslope of it: about its distance from the high edge.
        for x in [8, 24, 40] {
            let upslope = (SIZE.x - 1 - x) as f32;
            let a = at(&area, x, mid);
            assert!((a - (upslope + 1.0)).abs() < 0.2 * upslope, "x {x}: {a}");
        }
    }

    #[test]
    fn pit_fills_and_drains() {
        // A bowl: everything ends at the border after the pit fills.
        let c = SIZE.as_vec2() / 2.0;
        let heights = field(|x, y| 1.0 + 0.001 * Vec2::new(x as f32, y as f32).distance(c));
        let filled = fill(&heights, SIZE, 0.0);
        // The bowl fills to its lowest rim and no higher.
        let rim = (0..SIZE.x * SIZE.y)
            .filter(|&i| is_outlet(&heights, SIZE, 0.0, i as usize))
            .map(|i| heights[i as usize])
            .fold(f32::INFINITY, f32::min) as f64;
        let centre = filled[(SIZE.y / 2 * SIZE.x + SIZE.x / 2) as usize];
        assert!(
            centre >= rim && centre < rim + 1e-4,
            "centre {centre}, rim {rim}"
        );
        let area = accumulate(&filled, SIZE, 1.0, &[]);
        let total_out: f32 = (0..SIZE.x * SIZE.y)
            .filter(|&i| is_outlet(&heights, SIZE, 0.0, i as usize))
            .map(|i| area[i as usize])
            .sum();
        // Outlets receive the interior's area once (plus their own).
        let cells = (SIZE.x * SIZE.y) as f32;
        assert!(
            total_out > 0.9 * cells,
            "outlets got {total_out} of {cells}"
        );
    }

    #[test]
    fn valley_gathers_its_slopes() {
        // A V valley running toward y = 0: the thalweg near the outlet holds most of the map.
        let heights =
            field(|x, y| 1.0 + 0.002 * y as f32 + 0.01 * (x as f32 - (SIZE.x / 2) as f32).abs());
        let area = drainage_area(&heights, SIZE, 10.0, 0.0, &[]);
        let thalweg = at(&area, SIZE.x / 2, 1);
        let side = at(&area, SIZE.x / 2 + 10, 1);
        assert!(thalweg > 20.0 * side, "thalweg {thalweg}, side {side}");
        // In m²: 10 m texels.
        assert!(thalweg > 100.0 * 100.0, "{thalweg}");
    }

    #[test]
    fn inlet_adds_its_catchment() {
        let heights = field(|x, _| 1.0 + 0.01 * x as f32);
        let inlet = Inlet {
            texel: UVec2::new(SIZE.x - 2, SIZE.y / 2),
            area: 1.0e6,
        };
        let area = drainage_area(&heights, SIZE, 1.0, 0.0, &[inlet]);
        // Downstream of the inlet the flow carries its catchment, which MFD spreads across the
        // plane by about 4.5 texels over these 42.
        let row: f32 = (SIZE.y / 2 - 14..=SIZE.y / 2 + 14)
            .map(|y| at(&area, 20, y))
            .sum();
        assert!(row > 0.8e6, "{row}");
    }
}
