//! Scenes: scripted runs that show the features, from the Scenes tab of the panel (or the demo's
//! `--showcase`, which plays [`SceneKind::Tour`]).
//!
//! A scene first finds its sites on the eroded map: a river source with a good catchment, a
//! stretch of that river, and the steepest slope above it that faces the river. Then it plays its
//! stages, each with a camera shot and a caption:
//!
//! 1. The eroded terrain (KTT), from above.
//! 2. Its drainage area, which sizes every channel.
//! 3. A river placed at the source (or the first river already there).
//! 4. Rain: the shallow-water solver and bed load reshape the beds.
//! 5. Boulders dropped into the river: the flow carries the small ones and the large ones jam.
//! 6. The bed's boulder fraction: armour, scree and abrasion.
//! 7. A landslide released into the valley: MPM debris runs down toward the river.
//! 8. Slopes failing on their own, then control goes back to you.
//!
//! The full tour plays them all; the other scenes play the stages they need. Escape (or Stop in
//! the tab) ends a scene.

use bevy::prelude::*;
use crate::{
    PreviewMode, PyramidProgress, TerrainView, WaterControl, WorldScale,
    boulders::{Boulder, BoulderEdits, BoulderList},
    brush::{BrushMode, WaterBrush},
    camera::OrbitController,
    drainage::DrainageArea,
    landslide::Landslides,
    river::Rivers,
};

/// The scenes the Scenes tab offers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneKind {
    /// Every stage, about a minute and a half.
    Tour,
    /// The terrain and its drainage area.
    Terrain,
    /// A river placed with the brush, then rain on it.
    River,
    /// Boulders dropped into a river, and the bed's boulder fraction.
    Boulders,
    /// A landslide into a river valley, and slopes failing on their own.
    Landslide,
}

impl SceneKind {
    pub const ALL: [Self; 5] = [Self::Tour, Self::Terrain, Self::River, Self::Boulders, Self::Landslide];

    pub fn label(self) -> &'static str {
        match self {
            Self::Tour => "Full Tour",
            Self::Terrain => "Terrain & Drainage",
            Self::River => "River & Rain",
            Self::Boulders => "Boulders",
            Self::Landslide => "Landslide",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Tour => "Every feature in turn, with captions (about 90 s).",
            Self::Terrain => "The eroded terrain and the drainage area that sizes its channels.",
            Self::River => "A river placed at a good source, then rain and bed load on it.",
            Self::Boulders => "Rocks dropped into a river: the flow moves the small ones, the large ones jam.",
            Self::Landslide => "The steepest slope above a river lets go and runs into the valley.",
        }
    }

    fn stages(self) -> &'static [Stage] {
        use Stage::*;
        match self {
            Self::Tour => &[Terrain, Drainage, River, Rain, Boulders, Grain, Landslide, Aftermath],
            Self::Terrain => &[Terrain, Drainage],
            Self::River => &[River, Rain],
            Self::Boulders => &[River, Boulders, Grain],
            Self::Landslide => &[River, Landslide, Aftermath],
        }
    }
}

/// Set to start a scene (the Scenes tab does); `None` again once it has started.
#[derive(Resource, Default)]
pub struct SceneRequest {
    pub start: Option<SceneKind>,
    pub stop: bool,
}

pub struct ScenesPlugin;

impl Plugin for ScenesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Tour>()
            .init_resource::<SceneRequest>()
            .init_resource::<SceneStatus>()
            .add_systems(Startup, spawn_caption)
            .add_systems(
                Update,
                (start_scene, stop_on_escape, run_tour, report_status).chain(),
            );
    }
}

/// What is playing, for other systems (a caption elsewhere, screenshots).
#[derive(Resource, Default, Clone, Debug)]
pub struct SceneStatus {
    pub scene: Option<SceneKind>,
    /// The stage's caption title, its index in the scene, and seconds into it out of its length.
    pub stage: &'static str,
    pub index: usize,
    pub time: f32,
    pub seconds: f32,
}

/// Where the tour looks, in map texels, found once the terrain is done.
#[derive(Clone, Copy, Debug)]
struct Sites {
    /// River source: high ground with a catchment worth a river.
    source: Vec2,
    /// A stretch of the river (until it is placed, the source).
    reach: Vec2,
    /// The steepest slope near the lower river that faces it.
    slope: Vec2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Waiting,
    Terrain,
    Drainage,
    River,
    Rain,
    Boulders,
    Grain,
    Landslide,
    Aftermath,
    Done,
}

impl Stage {
    /// Real seconds the stage lasts.
    fn seconds(self) -> f32 {
        match self {
            Self::Waiting | Self::Done => f32::INFINITY,
            Self::Terrain => 9.0,
            Self::Drainage => 7.0,
            Self::River => 10.0,
            Self::Rain => 12.0,
            Self::Boulders => 14.0,
            Self::Grain => 7.0,
            Self::Landslide => 22.0,
            Self::Aftermath => 10.0,
        }
    }

    fn caption(self) -> (&'static str, &'static str) {
        match self {
            Self::Waiting => (
                "Eroding the terrain",
                "The KTT cascade erodes the landscape coarse to fine. The scene starts when it is done.",
            ),
            Self::Terrain => (
                "Eroded terrain",
                "5 km across at 10 m per texel. Ridges, valleys and fans come from KTT's stream-power erosion, run coarse to fine.",
            ),
            Self::Drainage => (
                "Drainage area",
                "Every texel's upstream area (priority-flood fill, multiple-flow-direction routing). It sizes every channel: Q = k·A^0.8, then width and depth from Q.",
            ),
            Self::River => (
                "River brush",
                "A river placed at a source routes itself down the valleys, with its bed carved for its bankfull discharge: steps and pools where steep, riffles and bars where gentle.",
            ),
            Self::Rain => (
                "Shallow water and bed load",
                "Rain on the whole map, in metres and seconds. The shallow-water solver carries it into the channels, and bed load (Wong & Parker, with helical flow in bends) reshapes their beds.",
            ),
            Self::Boulders => (
                "Boulders as bodies",
                "Rocks from 1 to 8 m dropped into the river. The flow's drag and lift move them once they beat their friction; the small ones travel, the large ones stay, and those that touch jam together.",
            ),
            Self::Grain => (
                "Boulders in the bed",
                "The bed's boulder fraction: scree on steep slopes, armour where the flow carried the fines away, and finer deposits downstream as boulders wear down.",
            ),
            Self::Landslide => (
                "Landslide",
                "A slab of the steepest slope above the river lets go. MPM moves the debris as a granular flow over the terrain, down toward the river, where it settles as a deposit.",
            ),
            Self::Aftermath => (
                "Slopes fail on their own",
                "Every slope's factor of safety is checked as the water runs: wet slopes past about 30° fail by themselves.",
            ),
            Self::Done => ("", ""),
        }
    }

    /// The view the stage shows (`None`: the demo's own shaded view).
    fn view(self) -> Option<PreviewMode> {
        match self {
            Self::Drainage => Some(PreviewMode::Drainage),
            Self::Grain => Some(PreviewMode::Boulders),
            _ => None,
        }
    }
}

#[derive(Resource)]
struct Tour {
    scene: Option<SceneKind>,
    /// Index of the stage in the scene's list.
    index: usize,
    stage: Stage,
    /// Seconds into the stage.
    time: f32,
    sites: Option<Sites>,
    /// The view to come back to after a data view.
    shaded: Option<PreviewMode>,
    /// Rainfall before the tour turned it up.
    rainfall: Option<f32>,
    /// Whether slopes failed on their own before the scene held them still (until its last stage
    /// about them), so its own slide is the one shown.
    auto_slides: Option<bool>,
}

impl Default for Tour {
    fn default() -> Self {
        Self {
            scene: None,
            index: 0,
            stage: Stage::Done,
            time: 0.0,
            sites: None,
            shaded: None,
            rainfall: None,
            auto_slides: None,
        }
    }
}

#[derive(Component)]
struct Caption;

#[derive(Component)]
struct CaptionTitle;

#[derive(Component)]
struct CaptionBody;

fn spawn_caption(mut commands: Commands) {
    commands
        .spawn((
            Caption,
            Visibility::Hidden,
            Node {
                position_type: PositionType::Absolute,
                left: percent(25.0),
                right: percent(10.0),
                bottom: px(32.0),
                padding: UiRect::axes(px(20.0), px(14.0)),
                flex_direction: FlexDirection::Column,
                row_gap: px(6.0),
                border_radius: BorderRadius::all(px(8.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.02, 0.03, 0.05, 0.72)),
        ))
        .with_children(|c| {
            c.spawn((
                CaptionTitle,
                Text::new(Stage::Waiting.caption().0),
                TextFont::from_font_size(24.0),
                TextColor(Color::WHITE),
            ));
            c.spawn((
                CaptionBody,
                Text::new(Stage::Waiting.caption().1),
                TextFont::from_font_size(16.0),
                TextColor(Color::srgb(0.85, 0.88, 0.92)),
            ));
        });
}

/// Start a requested scene from its first stage (it waits for the terrain first).
fn start_scene(
    mut request: ResMut<SceneRequest>,
    mut tour: ResMut<Tour>,
    mut water: ResMut<WaterControl>,
    mut caption: Query<&mut Visibility, With<Caption>>,
    mut title: Query<&mut Text, (With<CaptionTitle>, Without<CaptionBody>)>,
    mut body: Query<&mut Text, (With<CaptionBody>, Without<CaptionTitle>)>,
) {
    let Some(scene) = request.start.take() else {
        return;
    };
    info!("Scene: {}", scene.label());
    let shaded = tour.shaded;
    let rainfall = tour.rainfall;
    let auto_slides = tour.auto_slides.or(Some(water.landslide.auto));
    water.landslide.auto = false;
    *tour = Tour {
        scene: Some(scene),
        stage: Stage::Waiting,
        // A scene started over another keeps the settings to come back to.
        shaded,
        rainfall,
        auto_slides,
        ..default()
    };
    for mut v in &mut caption {
        *v = Visibility::Inherited;
    }
    let (t, b) = Stage::Waiting.caption();
    for mut text in &mut title {
        text.0 = t.to_string();
    }
    for mut text in &mut body {
        text.0 = b.to_string();
    }
}

/// Escape (or the tab's Stop) ends the scene.
fn stop_on_escape(
    keys: Res<ButtonInput<KeyCode>>,
    mut request: ResMut<SceneRequest>,
    mut tour: ResMut<Tour>,
    mut view: ResMut<TerrainView>,
    mut water: ResMut<WaterControl>,
    mut caption: Query<&mut Visibility, With<Caption>>,
) {
    let stop = std::mem::take(&mut request.stop) || keys.just_pressed(KeyCode::Escape);
    if !stop || tour.scene.is_none() {
        return;
    }
    finish(&mut tour, &mut view, &mut water);
    for mut v in &mut caption {
        *v = Visibility::Hidden;
    }
}

fn report_status(tour: Res<Tour>, mut status: ResMut<SceneStatus>) {
    *status = SceneStatus {
        scene: tour.scene,
        stage: tour.stage.caption().0,
        index: tour.index,
        time: tour.time,
        seconds: tour.stage.seconds(),
    };
}

/// Back to the view and rain from before the scene.
fn finish(tour: &mut Tour, view: &mut TerrainView, water: &mut WaterControl) {
    if let Some(shaded) = tour.shaded.take() {
        view.mode = shaded;
    }
    if let Some(rain) = tour.rainfall.take() {
        water.rainfall = rain;
    }
    if let Some(auto) = tour.auto_slides.take() {
        water.landslide.auto = auto;
    }
    tour.scene = None;
    tour.stage = Stage::Done;
}

#[allow(clippy::too_many_arguments)]
fn run_tour(
    time: Res<Time>,
    progress: Res<PyramidProgress>,
    drainage: Res<DrainageArea>,
    scale: Res<WorldScale>,
    rivers: Option<Res<Rivers>>,
    list: Res<BoulderList>,
    mut tour: ResMut<Tour>,
    mut view: ResMut<TerrainView>,
    mut water: ResMut<WaterControl>,
    brush: Option<ResMut<WaterBrush>>,
    mut boulders: ResMut<BoulderEdits>,
    mut slides: ResMut<Landslides>,
    mut cameras: Query<&mut OrbitController>,
    mut title: Query<&mut Text, (With<CaptionTitle>, Without<CaptionBody>)>,
    mut body: Query<&mut Text, (With<CaptionBody>, Without<CaptionTitle>)>,
    mut caption: Query<&mut Visibility, With<Caption>>,
) {
    let tour = &mut *tour;
    let Some(scene) = tour.scene else { return };
    // Scenes place their river with the brush.
    let Some(mut brush) = brush else { return };
    let dt = time.delta_secs();
    tour.time += dt;
    let size = drainage.size;

    // Start once the terrain is eroded and its drainage is known.
    if tour.stage == Stage::Waiting {
        if !progress.get().finished || drainage.area.is_empty() {
            return;
        }
        let mut sites = find_sites(&drainage, &scale);
        // A river already placed is the scene's river.
        if let Some(river) = rivers.as_ref().and_then(|r| r.list.first()) {
            sites.source = river.source;
        }
        tour.sites = Some(sites);
        if tour.shaded.is_none() {
            tour.shaded = Some(view.mode);
        }
        info!("Scene sites: {:?}", tour.sites);
    }
    let Some(mut sites) = tour.sites else { return };

    // The river's path, once it is placed, gives the reach and the slope above it.
    if let Some(river) = rivers.as_ref().and_then(|r| r.list.first())
        && sites.reach == sites.source
        && river.path.len() > 8
    {
        sites.reach = river.path[river.path.len() * 2 / 5];
        let lower = river.path[river.path.len() * 3 / 5];
        sites.slope = steepest_facing(&drainage, &scale, lower).unwrap_or(lower);
        tour.sites = Some(sites);
        info!("Scene river reach {} and slope {}", sites.reach, sites.slope);
        // Placing is done: the brush goes back off.
        if brush.mode == BrushMode::River {
            brush.enabled = false;
            brush.mode = BrushMode::Rain;
        }
    }

    let advance = tour.stage == Stage::Waiting || tour.time >= tour.stage.seconds();
    if advance {
        let stages = scene.stages();
        if tour.stage != Stage::Waiting {
            tour.index += 1;
        }
        tour.stage = stages.get(tour.index).copied().unwrap_or(Stage::Done);
        tour.time = 0.0;
        enter(
            tour,
            &sites,
            &scale,
            &drainage,
            &mut view,
            &mut water,
            &mut brush,
            &mut boulders,
            &mut slides,
            &list,
        );
        let (t, b) = tour.stage.caption();
        for mut text in &mut title {
            text.0 = t.to_string();
        }
        for mut text in &mut body {
            text.0 = b.to_string();
        }
        if tour.stage == Stage::Done {
            finish(tour, &mut view, &mut water);
            for mut v in &mut caption {
                *v = Visibility::Hidden;
            }
            return;
        }
    }

    // The camera eases toward the stage's shot and turns slowly around it.
    let (focus, distance, pitch) = shot(tour.stage, &sites, size);
    let width = scale.map_metres(size).x;
    let ground = crate::river::bilinear(&drainage.heights, size, focus) * scale.metres_per_height();
    let xz = (focus / size.as_vec2() - 0.5) * width;
    let target = Vec3::new(xz.x, ground, xz.y);
    let ease = 1.0 - (-1.2 * dt).exp();
    for mut orbit in &mut cameras {
        orbit.target = orbit.target.lerp(target, ease);
        orbit.distance += (distance * width - orbit.distance) * ease;
        orbit.target_pitch = pitch.to_radians();
        orbit.target_yaw += 0.04 * dt;
    }
}

/// Focus (texels), distance (map widths) and pitch (degrees) of each stage's camera.
fn shot(stage: Stage, sites: &Sites, size: UVec2) -> (Vec2, f32, f32) {
    let centre = size.as_vec2() * 0.5;
    match stage {
        Stage::Waiting | Stage::Terrain | Stage::Done => (centre, 1.15, 40.0),
        Stage::Drainage => (centre, 1.2, 70.0),
        Stage::River => (sites.source.lerp(sites.reach, 0.5), 0.35, 35.0),
        Stage::Rain => (sites.reach, 0.3, 30.0),
        Stage::Boulders => (sites.reach, 0.07, 28.0),
        Stage::Grain => (sites.reach, 0.35, 55.0),
        Stage::Landslide => (sites.slope, 0.1, 38.0),
        Stage::Aftermath => (sites.slope.lerp(sites.reach, 0.5), 0.45, 40.0),
    }
}

/// What starts with each stage.
#[allow(clippy::too_many_arguments)]
fn enter(
    tour: &mut Tour,
    sites: &Sites,
    scale: &WorldScale,
    drainage: &DrainageArea,
    view: &mut TerrainView,
    water: &mut WaterControl,
    brush: &mut WaterBrush,
    boulders: &mut BoulderEdits,
    slides: &mut Landslides,
    list: &BoulderList,
) {
    let stage = tour.stage;
    if let Some(shaded) = tour.shaded {
        view.mode = stage.view().unwrap_or(shaded);
    }
    match stage {
        Stage::River if sites.reach == sites.source => {
            brush.enabled = true;
            brush.mode = BrushMode::River;
            brush.place_river_at = Some(sites.source);
        }
        Stage::Rain => {
            if tour.rainfall.is_none() {
                tour.rainfall = Some(water.rainfall);
            }
            water.rainfall = water.rainfall.max(30.0);
        }
        Stage::Boulders => {
            // Sixty rocks from 1 to 8 m in a disc across the channel.
            for k in 0..60u32 {
                let a = k as f32 * 2.399;
                let r = 5.0 * ((k as f32 + 0.5) / 60.0).sqrt();
                let diameter = 1.0 + 7.0 * ((k * 23) % 60) as f32 / 59.0;
                let at = sites.reach + Vec2::from_angle(a) * r;
                boulders.place(Boulder::at(at, diameter), list);
            }
        }
        Stage::Aftermath => {
            // Slopes may fail on their own again.
            if let Some(auto) = tour.auto_slides.take() {
                water.landslide.auto = auto;
            }
        }
        Stage::Landslide => {
            let depth = water.landslide.depth.max(30.0);
            slides.release(sites.slope, 12.0, depth, &drainage.heights, drainage.size, scale);
        }
        _ => {}
    }
}

/// The river source: among texels well inside the map whose catchment is 1–4 % of the largest,
/// the highest.
fn find_sites(drainage: &DrainageArea, scale: &WorldScale) -> Sites {
    let size = drainage.size;
    let largest = drainage.area.iter().copied().fold(0.0, f32::max);
    // Well inside the map, so the river has room to run.
    let margin = size.x / 4;
    let mut best: Option<(f32, UVec2)> = None;
    for y in margin..size.y - margin {
        for x in margin..size.x - margin {
            let i = (y * size.x + x) as usize;
            let a = drainage.area[i];
            if a < 0.01 * largest || a > 0.04 * largest {
                continue;
            }
            let h = drainage.heights[i];
            if best.is_none_or(|(b, _)| h > b) {
                best = Some((h, UVec2::new(x, y)));
            }
        }
    }
    let source = best.map_or(size.as_vec2() * 0.5, |(_, p)| p.as_vec2() + 0.5);
    Sites {
        source,
        reach: source,
        slope: steepest_facing(drainage, scale, source).unwrap_or(source),
    }
}

/// The steepest texel 10–30 texels from `near` whose downhill direction points at it, so its
/// slide runs toward it.
fn steepest_facing(drainage: &DrainageArea, scale: &WorldScale, near: Vec2) -> Option<Vec2> {
    let size = drainage.size.as_ivec2();
    let h = |x: i32, y: i32| {
        let p = IVec2::new(x, y).clamp(IVec2::ZERO, size - 1);
        drainage.heights[(p.y * size.x + p.x) as usize]
    };
    let c = near.as_ivec2();
    let mut best: Option<(f32, Vec2)> = None;
    // Clear of the map border, whose outflow band stands in cliffs.
    let border = size.x / 12;
    for y in (c.y - 30).max(border)..(c.y + 30).min(size.y - border) {
        for x in (c.x - 30).max(border)..(c.x + 30).min(size.x - border) {
            let p = Vec2::new(x as f32, y as f32) + 0.5;
            let d = p.distance(near);
            if !(10.0..=30.0).contains(&d) {
                continue;
            }
            // Slope (m/m) over two texels, and whether it falls toward the river.
            let g = Vec2::new(h(x + 2, y) - h(x - 2, y), h(x, y + 2) - h(x, y - 2))
                * scale.metres_per_height()
                / (4.0 * scale.metres_per_texel);
            let downhill = -g.normalize_or_zero();
            if downhill.dot((near - p) / d) < 0.6 {
                continue;
            }
            let steep = g.length();
            if best.is_none_or(|(b, _)| steep > b) {
                best = Some((steep, p));
            }
        }
    }
    best.map(|(_, p)| p)
}
