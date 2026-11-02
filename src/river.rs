//! River brush geometry, on the CPU: a click on the eroded terrain becomes a river bed.
//!
//! 1. [`route`]: a priority flood from the clicked source. It always grows into the lowest cell
//!    next to what it has reached, so it runs down the valley the erosion made, fills any
//!    hollow on the way up to its rim and leaves over the lowest point of it. It stops at the
//!    sea, the map edge, standing water or another river.
//! 2. [`smooth_path`]: the 8-connected cell path, smoothed and resampled every texel.
//! 3. [`long_profile`]: the bank level along the path, made to run downhill. A rise up to
//!    `breach_depth` is cut through; a higher one is left for the water to fill as a lake.
//! 4. [`rasterize`]: the channel and its banks, per texel. Depth ∝ Q^0.4 and half-width
//!    ∝ Q^0.5 with Shape Beds' cross-section and bank grading ([`crate::BedShaping`]). The bed
//!    detail follows the reach: step-pools above about 3 % slope, riffle-pools above about 1 %,
//!    and point bars on the inside of bends everywhere.
//!
//! The raster ([`RiverCell`] per texel) goes to `river_carve` in `river.wgsl`, which lowers the
//! terrain to it and writes the armour band (a floor and a ceiling around the carved bed) that
//! water erosion, settling, slumping and bed load keep to.

use crate::{BedShaping, ErosionConfig, ErosionImages, PyramidProgress, WaterControl, brush::WaterBrush};
use bevy::{
    prelude::*,
    render::{
        RenderApp,
        extract_resource::ExtractResource,
        gpu_readback::{Readback, ReadbackComplete},
    },
};
use std::{cmp::Reverse, collections::BinaryHeap, sync::Arc};

/// Map texels per height unit, as drawn (and as `water.wgsl` counts cells).
const CELLS_PER_HEIGHT: f32 = 100.0;
/// Water deeper than this (height units) ends a route: a lake or another river.
const STOP_DEPTH: f32 = 0.002;
/// Water this close to the source (texels) does not end the route: it is the source's own.
const SOURCE_REACH: f32 = 6.0;
/// Discharge doubles over this many texels downstream.
const DOUBLING_LENGTH: f32 = 256.0;
/// Channel discharge at which rivers get Shape Beds' `river_depth` and `river_width`
/// (`REFERENCE_DISCHARGE` in `water_bed.wgsl`).
const REFERENCE_DISCHARGE: f32 = 1.0;
/// Local slope (cells per texel) where step-pools take over from riffle-pools, and where
/// riffle-pools take over from a plain bed.
const STEP_POOL_SLOPE: (f32, f32) = (0.02, 0.04);
const RIFFLE_POOL_SLOPE: (f32, f32) = (0.001, 0.004);

/// No river at this texel.
pub const NO_RIVER: f32 = 1.0e6;

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
    /// Discharge at the source (height units × texels² per second); 1 gives Shape Beds' River
    /// Depth and River Width.
    pub discharge: f32,
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
            discharge: 1.0,
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
    pub discharge: f32,
    /// Seeds the step and pool spacing.
    pub seed: u32,
    /// Centreline, one sample per texel, in map texels.
    pub path: Vec<Vec2>,
    /// Bank level (height units) at each path sample, running downhill.
    pub profile: Vec<f32>,
}

/// Hydraulic geometry and bank grading, from [`crate::BedShaping`].
#[derive(Clone, Copy, Debug)]
pub struct ChannelShape {
    pub river_depth: f32,
    pub river_width: f32,
    pub profile: f32,
    pub max_depth: f32,
    pub bank_width: f32,
    pub bank_slope: f32,
}

impl From<&BedShaping> for ChannelShape {
    fn from(bed: &BedShaping) -> Self {
        Self {
            river_depth: bed.river_depth,
            river_width: bed.river_width,
            profile: bed.profile,
            max_depth: bed.max_depth,
            bank_width: bed.bank_width.clamp(1.0, crate::MAX_BANK_WIDTH),
            bank_slope: bed.bank_slope,
        }
    }
}

/// The rivers placed with the brush. They survive reruns of the cascade: once it finishes,
/// [`refresh_after_cascade`] gives them the new terrain's bank levels and carves them again.
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
pub fn rebuild(rivers: &Rivers, bed: &BedShaping, settings: &RiverSettings, size: UVec2, out: &mut RiverBed) {
    out.cells = Arc::new(rasterize(&rivers.list, &bed.into(), settings, size));
    out.generation = out.generation.wrapping_add(1);
}

/// The channel is sized for bankfull flow, which comes about every year or two; the spring
/// feeds this share of it, so the river runs inside its banks.
const SPRING_SHARE: f32 = 0.5;

/// The spring that feeds a river, over a disc about the channel's width.
pub fn spring_for(river: &River, bed: &BedShaping) -> crate::Spring {
    let radius = (bed.river_width * river.discharge.max(1e-6).sqrt()).max(2.0);
    // A spring adds `rate` at its centre, falling off as (1 − r²)², which integrates to πR²/3.
    crate::Spring {
        center: river.source,
        radius,
        rate: SPRING_SHARE * river.discharge * 3.0 / (std::f32::consts::PI * radius * radius),
    }
}

pub struct RiverPlugin;

impl Plugin for RiverPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Rivers>()
            .add_systems(Update, (refresh_after_cascade, follow_armour));
    }
}

#[derive(Component)]
struct RiverReadback;

/// When a rerun of the cascade finishes, read its heights back and give every river the new
/// terrain's bank levels (its path stays). The raster then goes to the render world.
fn refresh_after_cascade(
    mut commands: Commands,
    progress: Option<Res<PyramidProgress>>,
    rivers: Res<Rivers>,
    images: Option<Res<ErosionImages>>,
    config: Res<ErosionConfig>,
    pending: Query<(), With<RiverReadback>>,
    mut was_finished: Local<bool>,
) {
    let (Some(progress), Some(images)) = (progress, images) else {
        return;
    };
    let finished = progress.get().finished;
    let rising = finished && !*was_finished;
    *was_finished = finished;
    if !rising || rivers.list.is_empty() || !pending.is_empty() {
        return;
    }
    let size = config.map_size;
    commands
        .spawn((Readback::texture(images.height.clone()), RiverReadback))
        .observe(
            move |ev: On<ReadbackComplete>,
                  mut commands: Commands,
                  mut rivers: ResMut<Rivers>,
                  brush: Res<WaterBrush>,
                  water: Res<WaterControl>,
                  mut bed: ResMut<RiverBed>| {
                commands.entity(ev.entity).despawn();
                let mut heights = Vec::new();
                crate::brush::read_red_channel(&ev.data, size, &mut heights);
                for river in &mut rivers.list {
                    reprofile(river, &heights, size, brush.river.breach_depth);
                }
                rebuild(&rivers, &water.bed, &brush.river, size, &mut bed);
            },
        );
}

/// Armour only sets the band `river_carve` writes, so a change re-sends the raster. (Bed Detail
/// and Breach Depth change the beds themselves and rerun the cascade instead.)
fn follow_armour(
    brush: Res<WaterBrush>,
    rivers: Res<Rivers>,
    water: Res<WaterControl>,
    config: Res<ErosionConfig>,
    mut bed: ResMut<RiverBed>,
    mut last: Local<Option<f32>>,
) {
    let armour = brush.river.armour;
    if last.replace(armour).is_some_and(|a| a != armour) && !rivers.list.is_empty() {
        rebuild(&rivers, &water.bed, &brush.river, config.map_size, &mut bed);
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
    let n = (size.x * size.y) as usize;
    if heights.len() != n || size.min_element() < 3 {
        return Vec::new();
    }
    let at = |c: UVec2| (c.y * size.x + c.x) as usize;
    let start = source.floor().as_uvec2().min(size - 1);
    let ends = |c: UVec2| {
        let i = at(c);
        let edge = c.x == 0 || c.y == 0 || c.x == size.x - 1 || c.y == size.y - 1;
        let away = (c.as_vec2() + 0.5).distance(source) > SOURCE_REACH;
        edge || (sea_level > 0.0 && heights[i] < sea_level)
            || (away && (water.get(i).is_some_and(|&d| d > STOP_DEPTH) || taken[i]))
    };
    let mut parent = vec![u32::MAX; n];
    let mut seen = vec![false; n];
    // Lowest first; ties in the order reached, so flats fill outward evenly. Heights are not
    // negative, so their bits sort like the values.
    let mut open = BinaryHeap::new();
    let mut order = 0u32;
    seen[at(start)] = true;
    open.push(Reverse((heights[at(start)].to_bits(), order, start.x, start.y)));
    let mut end = start;
    let mut lowest = f32::INFINITY;
    while let Some(Reverse((_, _, x, y))) = open.pop() {
        let c = UVec2::new(x, y);
        if heights[at(c)] < lowest {
            lowest = heights[at(c)];
            end = c;
        }
        if c != start && ends(c) {
            end = c;
            break;
        }
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let q = c.as_ivec2() + IVec2::new(dx, dy);
                if (dx, dy) == (0, 0) || q.cmplt(IVec2::ZERO).any() || q.cmpge(size.as_ivec2()).any()
                {
                    continue;
                }
                let q = q.as_uvec2();
                if !seen[at(q)] {
                    seen[at(q)] = true;
                    parent[at(q)] = at(c) as u32;
                    order += 1;
                    open.push(Reverse((heights[at(q)].max(0.0).to_bits(), order, q.x, q.y)));
                }
            }
        }
    }
    let mut cells = vec![end];
    let mut i = at(end);
    while parent[i] != u32::MAX {
        i = parent[i] as usize;
        cells.push(UVec2::new(i as u32 % size.x, i as u32 / size.x));
    }
    cells.reverse();
    cells
}

/// Texel centres of `cells`, smoothed (twice, over `radius` samples each side, ends kept) and
/// resampled one texel apart.
pub fn smooth_path(cells: &[UVec2], radius: usize) -> Vec<Vec2> {
    let mut points: Vec<Vec2> = cells.iter().map(|c| c.as_vec2() + 0.5).collect();
    if points.len() < 3 {
        return points;
    }
    for _ in 0..2 {
        let prev = points.clone();
        let last = prev.len() - 1;
        for (i, p) in points.iter_mut().enumerate().take(last).skip(1) {
            let r = radius.min(i).min(last - i);
            let sum: Vec2 = prev[i - r..=i + r].iter().sum();
            *p = sum / (2 * r + 1) as f32;
        }
    }
    let mut out = vec![points[0]];
    let mut carry = 0.0;
    for w in points.windows(2) {
        let (a, b) = (w[0], w[1]);
        let len = a.distance(b);
        let mut t = 1.0 - carry;
        while t <= len {
            out.push(a.lerp(b, t / len));
            t += 1.0;
        }
        carry = len - (t - 1.0);
    }
    if out.last().is_some_and(|p| p.distance(*points.last().unwrap()) > 0.25) {
        out.push(*points.last().unwrap());
    }
    out
}

/// Bank level along a path whose ground is `ground`: it never rises, except over a rise higher
/// than `breach_depth`, which it follows (the water fills a lake behind it and spills over).
/// A rise it cuts through gets a steady slope from where it starts to where the ground is back
/// below it.
pub fn long_profile(ground: &[f32], breach_depth: f32) -> Vec<f32> {
    let mut z = ground.to_vec();
    let mut i = 1;
    while i < z.len() {
        let level = z[i - 1];
        if ground[i] <= level {
            z[i] = ground[i];
            i += 1;
            continue;
        }
        let back = (i..ground.len()).find(|&j| ground[j] <= level);
        let peak = ground[i..back.unwrap_or(ground.len())]
            .iter()
            .fold(0.0f32, |m, &g| m.max(g - level));
        match back {
            Some(j) if peak <= breach_depth => {
                let end = ground[j];
                for (k, v) in z.iter_mut().enumerate().take(j + 1).skip(i) {
                    *v = level + (end - level) * (k + 1 - i) as f32 / (j + 1 - i) as f32;
                }
                i = j + 1;
            }
            _ => {
                z[i] = ground[i];
                i += 1;
            }
        }
    }
    z
}

/// Deterministic 0..1 from a seed and an index.
fn hash01(seed: u32, i: u32) -> f32 {
    let mut x = seed.wrapping_mul(0x9e37_79b9) ^ i.wrapping_mul(0x85eb_ca6b);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x as f32 / u32::MAX as f32
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Quintic ramp, as Shape Beds grades its banks.
fn s5(x: f32) -> f32 {
    let t = x.clamp(0.0, 1.0);
    t * t * t * (t * (6.0 * t - 15.0) + 10.0)
}

/// Per-sample channel along a river.
struct Section {
    /// Half-width (texels).
    width: f32,
    /// Depth below the bank level on the thalweg, for the discharge (height units).
    depth: f32,
    /// What steps and pools add to it on the thalweg (height units). It fades out toward the
    /// banks, so they do not become walls.
    extra: f32,
    /// How far the step lips bow downstream in the middle of the channel (channel widths).
    bow: f32,
    /// How sharply the channel bends, -1 to 1 (+ turning left). The thalweg moves toward the
    /// outer bank under a steep cut bank, and a gently sloping point bar builds on the inner one.
    bend: f32,
    tangent: Vec2,
}

fn sections(river: &River, shape: &ChannelShape, detail: f32) -> Vec<Section> {
    let n = river.path.len();
    let soft_cap = |d: f32| {
        let m = shape.max_depth.max(1e-6);
        m * (1.0 - (-d / m).exp())
    };
    let flow = |i: usize| river.discharge * (1.0 + i as f32 / DOUBLING_LENGTH) / REFERENCE_DISCHARGE;
    let width = |i: usize| (shape.river_width * flow(i).max(1e-6).sqrt()).max(1.0);
    let base = |i: usize| soft_cap(shape.river_depth * flow(i).max(1e-6).powf(0.4));
    let tangent = |i: usize| {
        let a = river.path[i.saturating_sub(1)];
        let b = river.path[(i + 1).min(n - 1)];
        (b - a).normalize_or(Vec2::X)
    };
    // Local slope of the bank level (cells per texel) over a few channel widths.
    let slope = |i: usize| {
        let k = (2.0 * width(i)).round().max(2.0) as usize;
        let (a, b) = (i.saturating_sub(k), (i + k).min(n - 1));
        ((river.profile[a] - river.profile[b]) * CELLS_PER_HEIGHT / (b - a).max(1) as f32).max(0.0)
    };

    // Steps, as depth added to the channel's (positive, a pool) or taken from it (a lip).
    // Between steps at `s` and `e` the bed is a nearly flat tread ending at the lip `e`; at each
    // lip the bed drops about 0.9 of the fall since the last one into a pool scoured about 0.6
    // of the fall deeper, so a step's height comes to about 1.5 × its spacing × the slope
    // (Abrahams et al. 1995). Steps about 1.5 × the channel depth high then sit
    // H / (1.5 S) apart, 1 to 4 channel widths (Chin 1999): closer on steeper reaches. Each lip
    // sits 0.7 of the channel depth below the banks, so the channel stays continuous and holds
    // its flow, and its pool hangs below that.
    let mut extra = vec![0.0f32; n];
    // How far each step's lip bows downstream in the middle, in channel widths.
    let mut bow = vec![0.4f32; n];
    let mut s = 0usize;
    let mut k = 0u32;
    while s + 1 < n {
        let w = 2.0 * width(s);
        let fall_rate = (slope(s) / CELLS_PER_HEIGHT).max(1e-6);
        let jitter = 0.6 + 0.8 * hash01(river.seed, k);
        let spacing = (base(s) / fall_rate * jitter).clamp(w, 4.0 * w).max(3.0) as usize;
        let e = (s + spacing).min(n - 1);
        let steep = smoothstep(STEP_POOL_SLOPE.0, STEP_POOL_SLOPE.1, slope(s)) * detail;
        if steep > 0.0 && e > s {
            let fall = (river.profile[s] - river.profile[e]).max(0.0);
            let drop = (0.9 * (0.75 + 0.5 * hash01(river.seed ^ 0x51, k))).min(1.0);
            let reach = (0.25 * spacing as f32).max(1.0);
            let step: Vec<f32> = (s..e)
                .map(|j| {
                    let left = (e - j) as f32 / (e - s) as f32;
                    let tread = river.profile[e] + (1.0 - drop) * fall * left;
                    let pool = 0.6 * fall * (-((j - s) as f32 / reach).powi(2)).exp();
                    (river.profile[j] - tread).max(0.0) + pool
                })
                .collect();
            let lip = step.iter().fold(f32::MAX, |m, &d| m.min(d)) + 0.3 * base(s);
            for (v, d) in extra[s..e].iter_mut().zip(&step) {
                *v += steep * (d - lip);
            }
            bow[s..e].fill(0.2 + 0.5 * hash01(river.seed ^ 0x3c, k));
        }
        s = e;
        k += 1;
    }
    // Riffle-pools: a pool every 5 to 7 channel widths (Leopold et al. 1964), from about 0.1 %
    // slope to where step-pools take over. Each pool reaches about 0.8 of the channel depth
    // deeper and the riffles between are a little shallower, so pools come to about twice the
    // riffles' depth.
    let mut s = 0usize;
    while s + 1 < n {
        let w = width(s);
        let spacing = (2.0 * w * (5.0 + 2.0 * hash01(river.seed ^ 0xa7, k))).max(4.0) as usize;
        let e = (s + spacing).min(n - 1);
        let c = (s + e) / 2;
        let gentle = smoothstep(RIFFLE_POOL_SLOPE.0, RIFFLE_POOL_SLOPE.1, slope(c))
            * (1.0 - smoothstep(STEP_POOL_SLOPE.0, STEP_POOL_SLOPE.1, slope(c)))
            * detail;
        if gentle > 0.0 {
            let pool: Vec<f32> = (s..e)
                .map(|j| {
                    let t = (j as f32 - c as f32) / w;
                    0.8 * base(c) * (-t * t).exp()
                })
                .collect();
            let mean = pool.iter().sum::<f32>() / pool.len() as f32;
            for (v, d) in extra[s..e].iter_mut().zip(&pool) {
                *v += gentle * (d - mean);
            }
        }
        s = e;
        k += 1;
    }

    // A riser falls over a few texels rather than one (a cascade over a boulder line). No pool
    // goes deeper than 3 × the channel, and no lip comes closer than 0.7 of it to the banks.
    let extra: Vec<f32> = (0..n)
        .map(|i| {
            let (a, b) = (i.saturating_sub(1), (i + 1).min(n - 1));
            let mean = extra[a..=b].iter().sum::<f32>() / (b - a + 1) as f32;
            mean.clamp(-0.3 * base(i), 3.0 * base(i))
        })
        .collect();

    (0..n)
        .map(|i| {
            let w = width(i);
            // Curvature from the turn of the tangent over about a channel width each side.
            let r = (w.round() as usize).max(2);
            let (a, b) = (i.saturating_sub(r), (i + r).min(n - 1));
            let turn = tangent(a).perp_dot(tangent(b));
            let curvature = turn / (b - a).max(1) as f32;
            let bend = (curvature * w * 4.0 * detail).clamp(-1.0, 1.0);
            Section {
                // Pools are wider as well as deeper.
                width: w * (1.0 + 0.5 * (extra[i] / base(i).max(1e-6)).clamp(0.0, 1.0)),
                bow: bow[i],
                depth: base(i),
                extra: extra[i],
                bend,
                tangent: tangent(i),
            }
        })
        .collect()
}

/// [`Section::extra`] between samples, at a fractional sample index.
fn sample_extra(secs: &[Section], at: f32) -> f32 {
    let at = at.clamp(0.0, (secs.len() - 1) as f32);
    let i = at.floor() as usize;
    let j = (i + 1).min(secs.len() - 1);
    let f = at - i as f32;
    secs[i].extra + (secs[j].extra - secs[i].extra) * f
}

/// `values` between samples, at a fractional sample index.
fn sample(values: &[f32], at: f32) -> f32 {
    let at = at.clamp(0.0, (values.len() - 1) as f32);
    let i = at.floor() as usize;
    let j = (i + 1).min(values.len() - 1);
    values[i] + (values[j] - values[i]) * (at - i as f32)
}

/// The rivers' beds, banks and armour, one [`RiverCell`] per texel of a `size` map.
pub fn rasterize(rivers: &[River], shape: &ChannelShape, settings: &RiverSettings, size: UVec2) -> Vec<RiverCell> {
    let n = (size.x * size.y) as usize;
    let mut cells = vec![Vec4::new(NO_RIVER, 1.0, 0.0, 0.0); n];
    let bank = shape.bank_width.max(1.0);
    for river in rivers.iter().filter(|r| r.path.len() >= 2) {
        let secs = sections(river, shape, settings.bed_detail);
        // Nearest sample per texel, for this river.
        let mut nearest = vec![(f32::INFINITY, u32::MAX); n];
        for (i, (p, sec)) in river.path.iter().zip(&secs).enumerate() {
            let reach = sec.width + bank + 1.0;
            let lo = (*p - reach).floor().max(Vec2::ZERO).as_uvec2();
            let hi = (*p + reach).ceil().as_uvec2().min(size - 1);
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let d = (Vec2::new(x as f32, y as f32) + 0.5).distance_squared(*p);
                    let j = (y * size.x + x) as usize;
                    if d < nearest[j].0 {
                        nearest[j] = (d, i as u32);
                    }
                }
            }
        }
        for (j, &(_, i)) in nearest.iter().enumerate() {
            if i == u32::MAX {
                continue;
            }
            let (i, sec) = (i as usize, &secs[i as usize]);
            let d = Vec2::new((j as u32 % size.x) as f32, (j as u32 / size.x) as f32) + 0.5
                - river.path[i];
            // Past either end of the river.
            if d.dot(sec.tangent).abs() > 1.0
                && (i == 0 || i == river.path.len() - 1)
            {
                continue;
            }
            // Across the channel from its centreline (+ to the left), and from the thalweg, which
            // a bend moves toward its outer bank. The banks stay at ±width from the centreline:
            // the outer side gets narrower and steep (the cut bank), the inner side wider and
            // shallow (the point bar).
            let centre = sec.tangent.perp_dot(d);
            let thalweg = -0.35 * sec.bend * sec.width;
            let across = centre - thalweg;
            let bar_side = across * sec.bend > 0.0;
            let half = if bar_side {
                sec.width + thalweg.abs()
            } else {
                sec.width - thalweg.abs()
            };
            let x = across.abs() / half;
            // At least about parabolic: a boxier section would stand its banks up as walls
            // wherever a pool deepens the channel.
            let rounded = shape.profile.max(1.2);
            let profile = if bar_side {
                rounded * (1.0 + 2.5 * sec.bend.abs())
            } else {
                rounded
            };
            // Bank level at the texel's own place along the path, not its nearest sample's, so
            // a steep bed does not come out in stripes.
            let level = sample(&river.profile, i as f32 + d.dot(sec.tangent));
            let cell = if x < 1.0 {
                let inner = (1.0 - x * x).max(0.0);
                // Step lips bow downstream in the middle of the channel.
                let along = i as f32 + d.dot(sec.tangent) - sec.bow * sec.width * inner;
                let extra = sample_extra(&secs, along);
                let cut = sec.depth * inner.powf(profile) + extra * inner.powf(2.0);
                Vec4::new(level - cut, 0.0, settings.armour, sec.depth + extra)
            } else {
                let out = centre.abs() - sec.width;
                if out >= bank {
                    continue;
                }
                let blend = s5(out / bank);
                Vec4::new(
                    level + shape.bank_slope * out,
                    blend,
                    settings.armour * (1.0 - blend),
                    sec.depth * (1.0 - blend),
                )
            };
            let old = cells[j];
            // The channel wins over another river's bank; between two channels, the deeper.
            if cell.y < old.y || (cell.y == old.y && cell.x < old.x) {
                cells[j] = Vec4::new(cell.x, cell.y, cell.z.max(old.z), cell.w.max(old.w));
            } else {
                cells[j].z = old.z.max(cell.z);
            }
        }
    }
    cells
}

/// Route a river from `source` and give it its bank level, or `None` if it goes nowhere.
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
    seed: u32,
) -> Option<River> {
    let mut taken = vec![false; (size.x * size.y) as usize];
    for p in rivers.iter().flat_map(|r| &r.path) {
        let c = p.floor().as_uvec2().min(size - 1);
        taken[(c.y * size.x + c.x) as usize] = true;
    }
    let cells = route(heights, water, &taken, size, source, sea_level);
    let radius = (shape.river_width * settings.discharge.max(1e-6).sqrt()).round().max(2.0) as usize;
    let path = smooth_path(&cells, radius);
    if path.len() < 4 {
        return None;
    }
    let mut river = River {
        source,
        discharge: settings.discharge,
        seed,
        path,
        profile: Vec::new(),
    };
    reprofile(&mut river, heights, size, settings.breach_depth);
    Some(river)
}

/// Bank level of `river` on the terrain `heights` (after the terrain changed under it).
pub fn reprofile(river: &mut River, heights: &[f32], size: UVec2, breach_depth: f32) {
    let ground: Vec<f32> = river.path.iter().map(|p| bilinear(heights, size, *p)).collect();
    river.profile = long_profile(&ground, breach_depth);
}

/// `field` at a point in texels, between texel centres.
pub fn bilinear(field: &[f32], size: UVec2, p: Vec2) -> f32 {
    if field.len() != (size.x * size.y) as usize {
        return 0.0;
    }
    let q = (p - 0.5).clamp(Vec2::ZERO, (size - 1).as_vec2());
    let i = q.floor().as_uvec2();
    let j = (i + 1).min(size - 1);
    let f = q - q.floor();
    let at = |x: u32, y: u32| field[(y * size.x + x) as usize];
    let a = at(i.x, i.y) + (at(j.x, i.y) - at(i.x, i.y)) * f.x;
    let b = at(i.x, j.y) + (at(j.x, j.y) - at(i.x, j.y)) * f.x;
    a + (b - a) * f.y
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: UVec2 = UVec2::new(64, 64);

    fn shape() -> ChannelShape {
        ChannelShape {
            river_depth: 0.008,
            river_width: 3.0,
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
        (0..SIZE.x * SIZE.y).map(|i| f(i % SIZE.x, i / SIZE.x)).collect()
    }

    #[test]
    fn route_runs_down_the_valley_to_the_edge() {
        let h = field(valley);
        let cells = route(&h, &[], &vec![false; h.len()], SIZE, Vec2::new(5.5, 20.5), 0.0);
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
        let cells = route(&h, &[], &vec![false; h.len()], SIZE, Vec2::new(5.5, 32.5), 0.0);
        assert_eq!(cells.last().unwrap().x, SIZE.x - 1);
    }

    #[test]
    fn route_stops_at_water_and_other_rivers() {
        let h = field(valley);
        let mut water = vec![0.0; h.len()];
        water[(32 * SIZE.x + 40) as usize] = 0.01;
        let cells = route(&h, &water, &vec![false; h.len()], SIZE, Vec2::new(5.5, 32.5), 0.0);
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
    fn steep_rivers_get_steps_and_pools() {
        let path: Vec<Vec2> = (0..200).map(|i| Vec2::new(i as f32 + 0.5, 32.5)).collect();
        let steep = |s: f32| River {
            source: path[0],
            discharge: 1.0,
            seed: 7,
            path: path.clone(),
            // `s` cells per texel.
            profile: (0..200).map(|i| 1.0 - s * i as f32 / CELLS_PER_HEIGHT).collect(),
        };
        // Depth the steps and pools add to the plain bed.
        let depths = |r: &River| -> Vec<f32> {
            sections(r, &shape(), 1.0).iter().map(|s| s.extra).collect()
        };
        let base = shape().river_depth;
        // From the shallowest lip to the deepest pool.
        let spread = |d: &[f32]| {
            d.iter().fold(f32::MIN, |m, &v| m.max(v)) - d.iter().fold(f32::MAX, |m, &v| m.min(v))
        };
        let step_pool = depths(&steep(0.1));
        let plain = depths(&steep(0.001));
        assert!(spread(&step_pool) > 0.5 * base, "steps vary the depth: {}", spread(&step_pool));
        assert!(spread(&plain) < 0.1 * base, "a gentle bed stays plain: {}", spread(&plain));
    }

    #[test]
    fn raster_cuts_a_channel_with_graded_banks() {
        let h = field(valley);
        let settings = RiverSettings::default();
        let river = place(&h, &[], &[], SIZE, Vec2::new(5.5, 32.5), 0.0, &settings, &shape(), 1).unwrap();
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
