//! Bevy Feathers UI for erosion parameters.
//!
//! A sidebar with a fixed header (simulation controls and progress) over a scrolling list of
//! collapsible sections. The brush and water sections come first and start open; the KTT
//! parameter groups start collapsed.

use bevy::{
    feathers::{
        FeathersPlugins,
        constants::{fonts, icons},
        controls::{
            ButtonVariant, FeathersButton, FeathersCheckbox, FeathersNumberInput,
            FeathersScrollbar, HardLimit, NumberInputPrecision, NumberInputStep, NumberInputValue,
            SoftLimit,
        },
        dark_theme::create_dark_theme,
        display::icon,
        font_styles::InheritableFont,
        theme::{ThemeBackgroundColor, ThemedText, UiTheme},
        tokens,
    },
    input_focus::tab_navigation::TabGroup,
    prelude::*,
    scene::SceneList,
    text::{FontWeight, LineHeight},
    ui::Checked,
    ui_widgets::{
        Activate, Button, Checkbox, ControlOrientation, ScrollArea, ValueChange,
        checkbox_self_update,
    },
};

use crate::camera::OrbitInputBlocked;
use crate::{
    ErodeParams, MAX_BANK_WIDTH, MAX_SPRINGS, PyramidProgress, ResetSim, SimControl, WaterControl,
    WaterSolver,
    brush::{BrushMode, TerrainTool, WaterBrush},
};

/// Marker for the erosion params panel root.
#[derive(Component, Clone, Default)]
pub struct ErosionParamsPanel;

#[derive(Component, Clone, Default)]
struct PyramidLevelCountText;

#[derive(Component, Clone, Default)]
struct PyramidStatusText;

#[derive(Component, Clone, Default)]
struct BrushHintText;

#[derive(Component, Clone, Default)]
struct SpringCountText;

/// Row of terrain tool buttons, shown only while the terrain brush is selected.
#[derive(Component, Clone, Default)]
struct TerrainToolRow;

/// A button that selects a brush mode (or, when `tool`, one exact terrain tool); highlighted
/// while it is active.
#[derive(Component, Clone, Default)]
struct BrushModeButton {
    mode: BrushMode,
    tool: bool,
}

/// Clickable header of a collapsible section.
#[derive(Component, Clone, Default)]
struct SectionHeader {
    open: bool,
}

#[derive(Component, Clone, Default)]
struct SectionBody;

#[derive(Component, Clone, Default)]
struct SectionChevron;

/// Marks erosion param sliders for styling (height, font).
#[derive(Component, Clone, Default)]
struct ErosionSliderField;

#[derive(Clone, Copy, PartialEq)]
enum ErosionSliderFieldKind {
    ErosionStrength,
    RockSoftness,
    ErosionAmount,
    ReferenceDetailScale,
    WearAngle,
    TalusAngle,
    MaxDepositAngle,
    ErosionCutoffAngle,
    HydroSearchDistance,
    FlowLength,
    FlowVolume,
    RidgeErosionSteps,
    RidgeSofteningAmount,
    RidgeErosionAmount,
    SedimentFriction,
    RockFriction,
    SedimentCompaction,
    CompactionThreshold,
    Channeling,
    ChannelingCharacter,
    SedimentRemoval,
    RemovalCharacter,
    SuspendedSediment,
    VelocityRandomness,
    VelocityRandomnessRefinement,
    UpliftAmount,
    RiverScale,
    RiverCharacter,
    RiverVolume,
    RiverFrictionReduction,
    RiverChanneling,
    MeanderLongevity,
    MomentumCoherence,
    UpliftRiverCarving,
    RiverThreshold,
    TrailDamping,
    NoiseFrequency,
    NoiseScale,
    NoiseSeed,
    FeatureSize,
    SourceDetailSmoothing,
    GeologicalAge,
    IterationsPerFrame,
    MaskFrequency,
    OutflowSeaLevel,
    OutflowBorder,
    BedrockDepth,
    GranularityAmount,
    GranularityScale,
    GranularityStrength,
    AgeStrength,
    BrushRadius,
    BrushStrength,
    WaterErosion,
    WaterEvaporation,
    WaterRainfall,
    WaterManning,
    WaterGravity,
    BedMaxDepth,
    BedShoreSlope,
    BedProfile,
    BedStrength,
    BedBankWidth,
    BedRiverDepth,
    BedRiverWidth,
    BedChannelDischarge,
    WaterRainWater,
    BedLoadGrain,
    BedLoadK,
    BedLoadCriticalShields,
    BedLoadSlopeEffect,
    BedLoadHelical,
    BedLoadMorphFactor,
    BedLoadRefineSteps,
    BedLoadRockHardness,
}

impl ErosionSliderFieldKind {
    fn apply(
        &self,
        params: &mut ErodeParams,
        sim: &mut SimControl,
        brush: &mut WaterBrush,
        water: &mut WaterControl,
        value: f32,
    ) {
        match self {
            Self::ErosionStrength => params.erosion_strength = value,
            Self::RockSoftness => params.rock_softness = value,
            Self::ErosionAmount => params.erosion_amount = value,
            Self::ReferenceDetailScale => params.detail_scale = value,
            Self::WearAngle => params.wear_angle = value,
            Self::TalusAngle => params.talus_angle = value,
            Self::MaxDepositAngle => params.max_deposit_angle = value,
            Self::ErosionCutoffAngle => params.erosion_cutoff_angle = value,
            Self::HydroSearchDistance => params.hydro_conditioning_search_distance = value,
            Self::FlowLength => params.flow_length = value,
            Self::FlowVolume => params.flow_volume = value,
            Self::RidgeErosionSteps => params.ridge_erosion_steps = value as u32,
            Self::RidgeSofteningAmount => params.ridge_softening_amount = value,
            Self::RidgeErosionAmount => params.ridge_erosion_amount = value,
            Self::SedimentFriction => params.friction = value,
            Self::RockFriction => params.rock_friction = value,
            Self::SedimentCompaction => params.sediment_compaction = value,
            Self::CompactionThreshold => params.compaction_threshold = value,
            Self::Channeling => params.channeling = value,
            Self::ChannelingCharacter => params.channeling_character = value,
            Self::SedimentRemoval => params.sediment_removal = value,
            Self::RemovalCharacter => params.removal_character = value,
            Self::SuspendedSediment => params.suspended_load = value,
            Self::VelocityRandomness => params.velocity_randomness = value,
            Self::VelocityRandomnessRefinement => params.velocity_randomness_refinement = value,
            Self::UpliftAmount => params.uplift = value,
            Self::RiverScale => params.river_scale = value,
            Self::RiverThreshold => params.river_threshold = value,
            Self::TrailDamping => params.trail_damping = value,
            Self::RiverCharacter => params.river_character = value,
            Self::RiverVolume => params.river_volume = value,
            Self::RiverFrictionReduction => params.river_friction_reduction = value,
            Self::RiverChanneling => params.river_channeling = value,
            Self::MeanderLongevity => params.meander_longevity = value,
            Self::MomentumCoherence => params.momentum_coherence = value,
            Self::UpliftRiverCarving => params.uplift_river_carving = value,
            Self::NoiseFrequency => params.noise_frequency = value,
            Self::NoiseScale => params.noise_scale = value,
            Self::NoiseSeed => params.noise_seed = value as u32,
            Self::FeatureSize => params.feature_size = value,
            Self::SourceDetailSmoothing => params.source_detail_smoothing = value,
            Self::GeologicalAge => params.geological_age = value,
            Self::IterationsPerFrame => sim.iterations_per_frame = value as u32,
            Self::MaskFrequency => params.mask_frequency = value,
            Self::OutflowSeaLevel => params.outflow_sea_level = value,
            Self::OutflowBorder => params.outflow_border = value,
            Self::BedrockDepth => params.bedrock_depth = value,
            Self::GranularityAmount => params.granularity_amount = value,
            Self::GranularityScale => params.granularity_scale = value,
            Self::GranularityStrength => params.granularity_strength = value,
            Self::AgeStrength => params.age_strength = value,
            Self::BrushRadius => brush.radius = value,
            Self::BrushStrength => brush.strength = value,
            Self::WaterErosion => water.sediment_capacity = value,
            // Shown in % per second.
            Self::WaterEvaporation => water.evaporation = value / 100.0,
            // Shown in ‰ of the map height per second.
            Self::WaterRainfall => water.rainfall = value / 1000.0,
            Self::WaterManning => water.manning_n = value,
            Self::WaterGravity => water.gravity = value,
            Self::BedMaxDepth => water.bed.max_depth = value,
            Self::BedShoreSlope => water.bed.shore_slope = value,
            Self::BedProfile => water.bed.profile = value,
            Self::BedStrength => water.bed.strength = value,
            Self::BedBankWidth => water.bed.bank_width = value,
            Self::BedRiverDepth => water.bed.river_depth = value,
            Self::BedRiverWidth => water.bed.river_width = value,
            Self::BedChannelDischarge => water.bed.channel_discharge = value,
            // Shown per thousand trail steps.
            Self::WaterRainWater => water.rain_water = value / 1000.0,
            Self::BedLoadGrain => water.bed_load.grain = value,
            Self::BedLoadK => water.bed_load.k = value,
            Self::BedLoadCriticalShields => water.bed_load.critical_shields = value,
            Self::BedLoadSlopeEffect => water.bed_load.slope_effect = value,
            Self::BedLoadHelical => water.bed_load.helical_strength = value,
            Self::BedLoadMorphFactor => water.bed_load.morph_factor = value,
            Self::BedLoadRefineSteps => water.bed_load.refine_steps = value as u32,
            Self::BedLoadRockHardness => water.bed_load.rock_hardness = value,
        }
    }

    /// Every slider that shapes the erosion reruns the cascade; the rest (brush, water, beds,
    /// views and pacing) apply live.
    fn resets_sim(&self) -> bool {
        !matches!(
            self,
            Self::IterationsPerFrame
                | Self::AgeStrength
                | Self::BrushRadius
                | Self::BrushStrength
                | Self::WaterErosion
                | Self::WaterEvaporation
                | Self::WaterRainfall
                | Self::WaterManning
                | Self::WaterGravity
                | Self::BedMaxDepth
                | Self::BedShoreSlope
                | Self::BedProfile
                | Self::BedStrength
                | Self::BedBankWidth
                | Self::BedRiverDepth
                | Self::BedRiverWidth
                | Self::BedChannelDischarge
                | Self::WaterRainWater
                | Self::BedLoadGrain
                | Self::BedLoadK
                | Self::BedLoadCriticalShields
                | Self::BedLoadSlopeEffect
                | Self::BedLoadHelical
                | Self::BedLoadMorphFactor
                | Self::BedLoadRefineSteps
                | Self::BedLoadRockHardness
        )
    }

    fn is_int(&self) -> bool {
        matches!(
            self,
            Self::RidgeErosionSteps
                | Self::NoiseSeed
                | Self::IterationsPerFrame
                | Self::BrushRadius
                | Self::BedBankWidth
                | Self::WaterGravity
                | Self::BedLoadRefineSteps
        )
    }
}

/// A checkbox kept in step with its setting (so keyboard shortcuts and defaults show).
#[derive(Component, Clone, Default)]
struct Toggle {
    kind: ToggleKind,
}

#[derive(Clone, Copy, PartialEq, Default)]
enum ToggleKind {
    #[default]
    Paused,
    ContinueAfterCascade,
    BrushEnabled,
    RidgeErosion,
    Rivers,
    ClampRiverChanneling,
    HydroConditioning,
    SoftnessMask,
    ErosionMask,
    UpliftMap,
    Outflow,
    AccumulateOutflow,
    Bedrock,
    BedrockMap,
    BedrockMapIsHeight,
    AdditiveSediment,
    NormalizeAge,
    MomentumFlow,
    SecondOrderDepth,
    LiveBedLoad,
}

impl ToggleKind {
    fn get(
        self,
        params: &ErodeParams,
        sim: &SimControl,
        brush: &WaterBrush,
        water: &WaterControl,
    ) -> bool {
        match self {
            Self::Paused => sim.paused,
            Self::ContinueAfterCascade => sim.continue_after_cascade,
            Self::BrushEnabled => brush.enabled,
            Self::RidgeErosion => params.compute_ridge_erosion != 0,
            Self::Rivers => params.do_rivers != 0,
            Self::ClampRiverChanneling => params.clamp_river_channeling != 0,
            Self::HydroConditioning => params.use_hydro_conditioning != 0,
            Self::SoftnessMask => params.use_softness_mask != 0,
            Self::ErosionMask => params.use_erosion_mask != 0,
            Self::UpliftMap => params.use_uplift_map != 0,
            Self::Outflow => params.use_outflow != 0,
            Self::AccumulateOutflow => params.outflow_integration != 0,
            Self::Bedrock => params.use_bedrock != 0,
            Self::BedrockMap => params.use_bedrock_map != 0,
            Self::BedrockMapIsHeight => params.bedrock_mode != 0,
            Self::AdditiveSediment => params.additive_sediment != 0,
            Self::NormalizeAge => params.normalize_age != 0,
            Self::MomentumFlow => water.solver == WaterSolver::Swe,
            Self::SecondOrderDepth => water.second_order_depth,
            Self::LiveBedLoad => water.bed_load.live,
        }
    }

    fn set(
        self,
        params: &mut ErodeParams,
        sim: &mut SimControl,
        brush: &mut WaterBrush,
        water: &mut WaterControl,
        on: bool,
    ) {
        let v = u32::from(on);
        match self {
            Self::Paused => sim.paused = on,
            Self::ContinueAfterCascade => sim.continue_after_cascade = on,
            Self::BrushEnabled => brush.enabled = on,
            Self::RidgeErosion => params.compute_ridge_erosion = v,
            Self::Rivers => params.do_rivers = v,
            Self::ClampRiverChanneling => params.clamp_river_channeling = v,
            Self::HydroConditioning => params.use_hydro_conditioning = v,
            Self::SoftnessMask => params.use_softness_mask = v,
            Self::ErosionMask => params.use_erosion_mask = v,
            Self::UpliftMap => params.use_uplift_map = v,
            Self::Outflow => params.use_outflow = v,
            Self::AccumulateOutflow => params.outflow_integration = v,
            Self::Bedrock => params.use_bedrock = v,
            Self::BedrockMap => params.use_bedrock_map = v,
            Self::BedrockMapIsHeight => params.bedrock_mode = v,
            Self::AdditiveSediment => params.additive_sediment = v,
            Self::NormalizeAge => params.normalize_age = v,
            Self::MomentumFlow => {
                water.solver = if on {
                    WaterSolver::Swe
                } else {
                    WaterSolver::Pipes
                }
            }
            Self::SecondOrderDepth => water.second_order_depth = on,
            Self::LiveBedLoad => water.bed_load.live = on,
        }
    }

    fn resets_sim(self) -> bool {
        matches!(
            self,
            Self::Rivers
                | Self::RidgeErosion
                | Self::ClampRiverChanneling
                | Self::HydroConditioning
                | Self::AccumulateOutflow
                | Self::SoftnessMask
                | Self::ErosionMask
                | Self::UpliftMap
                | Self::Outflow
                | Self::Bedrock
                | Self::BedrockMap
                | Self::BedrockMapIsHeight
                | Self::AdditiveSediment
        )
    }
}

/// Plugin that adds the erosion parameters UI panel using Bevy Feathers.
pub struct ErosionParamsPlugin;

const FONT_SIZE: f32 = 11.0;
const SLIDER_VALUE_FONT_SIZE: f32 = 9.0;
const SLIDER_HEIGHT: f32 = 14.0;
const BUTTON_HEIGHT: f32 = 20.0;
const ROW_GAP: f32 = 4.0;
const PADDING: f32 = 8.0;
const PANEL_WIDTH: f32 = 300.0;
const LABEL_WIDTH: f32 = 120.0;

impl Plugin for ErosionParamsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FeathersPlugins)
            .insert_resource(UiTheme(create_dark_theme()))
            .add_systems(
                PostStartup,
                (
                    spawn_erosion_params_panel,
                    (
                        shrink_slider_heights,
                        shrink_slider_value_font,
                        shrink_button_checkbox_font,
                    )
                        .after(spawn_erosion_params_panel),
                ),
            )
            .init_resource::<WaterBrush>()
            .init_resource::<WaterControl>()
            .add_systems(
                PostUpdate,
                (
                    update_pyramid_ui_labels,
                    sync_toggles,
                    update_brush_ui,
                    update_spring_count,
                    update_section_chevrons,
                ),
            );
    }
}

fn spawn_erosion_params_panel(
    params: Res<ErodeParams>,
    sim: Res<SimControl>,
    mut commands: Commands,
) {
    commands.spawn_scene(erosion_params_panel(params.clone(), sim.clone()));
}

/// Keep every checkbox in step with its setting.
fn sync_toggles(
    mut commands: Commands,
    params: Res<ErodeParams>,
    sim: Res<SimControl>,
    brush: Res<WaterBrush>,
    water: Res<WaterControl>,
    toggles: Query<(Entity, &Toggle, Has<Checked>)>,
    added: Query<(), Added<Toggle>>,
) {
    if !(params.is_changed()
        || sim.is_changed()
        || brush.is_changed()
        || water.is_changed()
        || !added.is_empty())
    {
        return;
    }
    for (entity, toggle, checked) in &toggles {
        let on = toggle.kind.get(&params, &sim, &brush, &water);
        if on && !checked {
            commands.entity(entity).insert(Checked);
        } else if !on && checked {
            commands.entity(entity).remove::<Checked>();
        }
    }
}

/// Highlight the active brush mode, show the terrain tools only for the terrain brush, and say
/// what the mouse does.
fn update_brush_ui(
    brush: Res<WaterBrush>,
    mut buttons: Query<(&BrushModeButton, &mut ButtonVariant)>,
    mut tool_rows: Query<&mut Node, With<TerrainToolRow>>,
    mut hints: Query<&mut Text, With<BrushHintText>>,
) {
    if !brush.is_changed() {
        return;
    }
    for (button, mut variant) in &mut buttons {
        let active = if button.tool {
            button.mode == brush.mode
        } else {
            std::mem::discriminant(&button.mode) == std::mem::discriminant(&brush.mode)
        };
        let v = ButtonVariant::selected(brush.enabled && active);
        if *variant != v {
            *variant = v;
        }
    }
    let terrain = matches!(brush.mode, BrushMode::Terrain(_));
    for mut node in &mut tool_rows {
        node.display = if terrain {
            Display::Flex
        } else {
            Display::None
        };
    }
    let hint = if !brush.enabled {
        "Off: left drag orbits (B or a brush turns it on)"
    } else {
        match brush.mode {
            BrushMode::Rain => "Drag: rain erodes under the brush",
            BrushMode::Terrain(_) => "Drag: edit · Shift+drag: undo edits",
            BrushMode::WaterBody => "Drag: pour · Shift: remove · Alt+click: spring",
        }
    };
    for mut text in &mut hints {
        **text = hint.to_string();
    }
}

fn update_spring_count(water: Res<WaterControl>, mut q: Query<&mut Text, With<SpringCountText>>) {
    if !water.is_changed() {
        return;
    }
    let text = format!("Springs: {} / {MAX_SPRINGS}", water.springs.len());
    for mut t in &mut q {
        if **t != text {
            **t = text.clone();
        }
    }
}

fn toggle_section(
    click: On<PointerClick>,
    mut headers: Query<(&mut SectionHeader, &ChildOf)>,
    children: Query<&Children>,
    mut bodies: Query<&mut Node, With<SectionBody>>,
) {
    let Ok((mut header, parent)) = headers.get_mut(click.entity) else {
        return;
    };
    header.open = !header.open;
    for &child in children.get(parent.parent()).into_iter().flatten() {
        if let Ok(mut node) = bodies.get_mut(child) {
            node.display = if header.open {
                Display::Flex
            } else {
                Display::None
            };
        }
    }
}

fn update_section_chevrons(
    headers: Query<(&SectionHeader, &Children), Changed<SectionHeader>>,
    mut chevrons: Query<&mut UiTransform, With<SectionChevron>>,
) {
    for (header, children) in &headers {
        for &child in children {
            if let Ok(mut transform) = chevrons.get_mut(child) {
                transform.rotation = Rot2::turn_fraction(if header.open { 0.25 } else { 0.0 });
            }
        }
    }
}

fn erosion_params_panel(params: ErodeParams, sim: SimControl) -> impl Scene {
    let water = WaterControl::default();
    let brush = WaterBrush::default();
    bsn! {
        Node {
            position_type: PositionType::Absolute,
            top: px(10.0),
            left: px(10.0),
            bottom: px(10.0),
            width: px(PANEL_WIDTH),
            padding: UiRect::all(px(PADDING)),
            row_gap: px(ROW_GAP),
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Stretch,
            justify_content: JustifyContent::Start,
            border_radius: BorderRadius::all(px(6.0)),
        }
        TabGroup
        ThemeBackgroundColor(tokens::WINDOW_BG)
        ErosionParamsPanel
        ThemedText
        InheritableFont {
            font: fonts::REGULAR,
            font_size: FontSize::Px(FONT_SIZE),
            weight: FontWeight::NORMAL,
        }
        on(|_: On<PointerOver>, mut blocked: ResMut<OrbitInputBlocked>| {
            blocked.0 = true;
        })
        on(|_: On<PointerOut>, mut blocked: ResMut<OrbitInputBlocked>| {
            blocked.0 = false;
        })
        Children [
            @header_controls()
            --
            // Scrolling list of sections, with room on the right for the scrollbar.
            Node {
                flex_grow: 1.0,
                min_height: px(0.0),
                display: Display::Flex,
                flex_direction: FlexDirection::Column,
                padding: UiRect::right(px(10.0)),
            }
            ThemedText
            Children [
                #sections
                Node {
                    flex_grow: 1.0,
                    min_height: px(0.0),
                    display: Display::Flex,
                    flex_direction: FlexDirection::Column,
                    row_gap: px(ROW_GAP),
                    overflow: Overflow::scroll_y(),
                }
                ScrollArea
                ThemedText
                Children [
                    @section("Brush", true, bsn_list! {
                        @toggle_row("Brush On (B)", ToggleKind::BrushEnabled)
                        --
                        @button_row(bsn_list! {
                            @brush_mode_button("Rain", BrushMode::Rain)
                            --
                            @brush_mode_button("Terrain", BrushMode::Terrain(TerrainTool::Raise))
                            --
                            @brush_mode_button("Water", BrushMode::WaterBody)
                        })
                        --
                        Node {
                            display: Display::None,
                            flex_direction: FlexDirection::Row,
                            column_gap: px(ROW_GAP),
                        }
                        TerrainToolRow
                        ThemedText
                        Children [
                            @tool_button("Raise", TerrainTool::Raise)
                            --
                            @tool_button("Lower", TerrainTool::Lower)
                            --
                            @tool_button("Flatten", TerrainTool::Flatten)
                            --
                            @tool_button("Smooth", TerrainTool::Smooth)
                        ]
                        --
                        @note_text("")
                        BrushHintText
                        --
                        @slider_row("Radius (texels)", 2.0, 128.0, brush.radius, ErosionSliderFieldKind::BrushRadius)
                        --
                        @slider_row("Strength", 0.01, 5.0, brush.strength, ErosionSliderFieldKind::BrushStrength)
                    })
                    --
                    @section("Water", true, bsn_list! {
                        @slider_row("Erosion", 0.0, 0.01, water.sediment_capacity, ErosionSliderFieldKind::WaterErosion)
                        --
                        @slider_row("Evaporation %/s", 0.0, 10.0, water.evaporation * 100.0, ErosionSliderFieldKind::WaterEvaporation)
                        --
                        @slider_row("Rainfall ‰/s", 0.0, 1.0, water.rainfall * 1000.0, ErosionSliderFieldKind::WaterRainfall)
                        --
                        @slider_row("Rain Water (per 1000 trail steps)", 0.0, 2.0, water.rain_water * 1000.0, ErosionSliderFieldKind::WaterRainWater)
                        --
                        @toggle_row("Momentum Flow (SWE)", ToggleKind::MomentumFlow)
                        --
                        @toggle_row("Second-Order Depth Flux", ToggleKind::SecondOrderDepth)
                        --
                        @slider_row("Bed Friction (Manning n)", 0.002, 0.1, water.manning_n, ErosionSliderFieldKind::WaterManning)
                        --
                        @slider_row("Gravity (cells/s²)", 250.0, 8000.0, water.gravity, ErosionSliderFieldKind::WaterGravity)
                        --
                        @note_text("Springs: 0")
                        SpringCountText
                        --
                        @button_row(bsn_list! {
                            @FeathersButton {
                                @caption: bsn! { Text("Clear Water (C)") ThemedText },
                            }
                            on(|_: On<Activate>, mut brush: ResMut<WaterBrush>| {
                                brush.clear_water = true;
                            })
                            --
                            @FeathersButton {
                                @caption: bsn! { Text("Clear Springs") ThemedText },
                            }
                            on(|_: On<Activate>, mut water: ResMut<WaterControl>| {
                                water.springs.clear();
                            })
                        })
                    })
                    --
                    @section("Lake & River Beds", true, bsn_list! {
                        @note_text("Cuts channels sized by their discharge, and lake bowls, under the current water.")
                        --
                        @FeathersButton {
                            @caption: bsn! { Text("Shape Beds (G)") ThemedText },
                            @variant: ButtonVariant::Primary,
                        }
                        on(|_: On<Activate>, mut water: ResMut<WaterControl>| {
                            water.shape_generation = water.shape_generation.wrapping_add(1);
                        })
                        --
                        @slider_row("River Depth", 0.0, 0.03, water.bed.river_depth, ErosionSliderFieldKind::BedRiverDepth)
                        --
                        @slider_row("River Width", 1.0, 16.0, water.bed.river_width, ErosionSliderFieldKind::BedRiverWidth)
                        --
                        @slider_row("Channel Discharge", 0.0, 0.2, water.bed.channel_discharge, ErosionSliderFieldKind::BedChannelDischarge)
                        --
                        @slider_row("Max Depth", 0.0, 0.1, water.bed.max_depth, ErosionSliderFieldKind::BedMaxDepth)
                        --
                        @slider_row("Shore Slope", 0.0, 0.01, water.bed.shore_slope, ErosionSliderFieldKind::BedShoreSlope)
                        --
                        @slider_row("River Profile", 0.2, 3.0, water.bed.profile, ErosionSliderFieldKind::BedProfile)
                        --
                        @slider_row("Strength", 0.0, 1.0, water.bed.strength, ErosionSliderFieldKind::BedStrength)
                        --
                        @slider_row("Bank Width", 1.0, MAX_BANK_WIDTH, water.bed.bank_width, ErosionSliderFieldKind::BedBankWidth)
                    })
                    --
                    @section("Bed Load", true, bsn_list! {
                        @note_text("Flowing water moves its bed: bars, pools, steps and deltas.")
                        --
                        @toggle_row("Live Bed Load", ToggleKind::LiveBedLoad)
                        --
                        @FeathersButton {
                            @caption: bsn! { Text("Refine Beds (M)") ThemedText },
                        }
                        on(|_: On<Activate>, mut water: ResMut<WaterControl>| {
                            water.refine_generation = water.refine_generation.wrapping_add(1);
                        })
                        --
                        @slider_row("Grain", 0.02, 2.0, water.bed_load.grain, ErosionSliderFieldKind::BedLoadGrain)
                        --
                        @slider_row("Bed Load K", 0.0, 20.0, water.bed_load.k, ErosionSliderFieldKind::BedLoadK)
                        --
                        @slider_row("Critical Shields", 0.0, 0.2, water.bed_load.critical_shields, ErosionSliderFieldKind::BedLoadCriticalShields)
                        --
                        @slider_row("Rock Hardness", 0.0, 20.0, water.bed_load.rock_hardness, ErosionSliderFieldKind::BedLoadRockHardness)
                        --
                        @slider_row("Slope Effect", 0.1, 5.0, water.bed_load.slope_effect, ErosionSliderFieldKind::BedLoadSlopeEffect)
                        --
                        @slider_row("Helical Strength", 0.0, 60.0, water.bed_load.helical_strength, ErosionSliderFieldKind::BedLoadHelical)
                        --
                        @slider_row("Morph. Factor", 1.0, 100.0, water.bed_load.morph_factor, ErosionSliderFieldKind::BedLoadMorphFactor)
                        --
                        @slider_row("Refine Steps", 50.0, 4000.0, water.bed_load.refine_steps as f32, ErosionSliderFieldKind::BedLoadRefineSteps)
                    })
                    --
                    @section("Terrain Source", false, bsn_list! {
                        @slider_row("Noise Freq", 0.001, 0.05, params.noise_frequency, ErosionSliderFieldKind::NoiseFrequency)
                        --
                        @slider_row("Noise Scale", 0.5, 5.0, params.noise_scale, ErosionSliderFieldKind::NoiseScale)
                        --
                        @slider_row("Seed", 0.0, 100.0, params.noise_seed as f32, ErosionSliderFieldKind::NoiseSeed)
                        --
                        @FeathersButton {
                            @caption: bsn! { Text("Reset Terrain Edits (X)") ThemedText },
                        }
                        on(|_: On<Activate>, mut brush: ResMut<WaterBrush>| {
                            brush.clear_terrain = true;
                        })
                    })
                    --
                    @section("Erosion", true, bsn_list! {
                        @slider_row("Erosion Strength", 0.0, 1.0, params.erosion_strength, ErosionSliderFieldKind::ErosionStrength)
                        --
                        @slider_row("Rock Softness", 0.0, 1.0, params.rock_softness, ErosionSliderFieldKind::RockSoftness)
                        --
                        @slider_row("Erosion Amount", 0.0, 1.0, params.erosion_amount, ErosionSliderFieldKind::ErosionAmount)
                        --
                        @slider_row("Detail Scale", 0.25, 4.0, params.detail_scale, ErosionSliderFieldKind::ReferenceDetailScale)
                    })
                    --
                    @section("Pyramid", false, bsn_list! {
                        @note_text(format!("Pyramid Levels: {}", params.pyramid_levels().len()))
                        PyramidLevelCountText
                        --
                        @slider_row("Feature Size", 0.5, 128.0, params.feature_size, ErosionSliderFieldKind::FeatureSize)
                        --
                        @slider_row("Detail Smoothing", 0.0, 4.0, params.source_detail_smoothing, ErosionSliderFieldKind::SourceDetailSmoothing)
                        --
                        @slider_row("Geological Age", 0.0, 50.0, params.geological_age, ErosionSliderFieldKind::GeologicalAge)
                        --
                        @slider_row("Granularity Amount", 0.0, 10.0, params.granularity_amount, ErosionSliderFieldKind::GranularityAmount)
                        --
                        @slider_row("Granularity Scale", 0.0, 2.0, params.granularity_scale, ErosionSliderFieldKind::GranularityScale)
                        --
                        @slider_row("Granularity Strength", 0.0, 1.0, params.granularity_strength, ErosionSliderFieldKind::GranularityStrength)
                        --
                        @slider_row("Iterations / Frame", 1.0, 64.0, sim.iterations_per_frame as f32, ErosionSliderFieldKind::IterationsPerFrame)
                        --
                        @toggle_row("Continue At Finest", ToggleKind::ContinueAfterCascade)
                    })
                    --
                    @section("Angles (deg)", false, bsn_list! {
                        @slider_row("Wear Angle", 0.0, 90.0, params.wear_angle, ErosionSliderFieldKind::WearAngle)
                        --
                        @slider_row("Talus Angle", 0.0, 90.0, params.talus_angle, ErosionSliderFieldKind::TalusAngle)
                        --
                        @slider_row("Max Deposit Angle", 0.0, 90.0, params.max_deposit_angle, ErosionSliderFieldKind::MaxDepositAngle)
                        --
                        @slider_row("Erosion Cutoff", 0.0, 90.0, params.erosion_cutoff_angle, ErosionSliderFieldKind::ErosionCutoffAngle)
                    })
                    --
                    @section("Ridge Erosion", false, bsn_list! {
                        @toggle_row("Do Ridge Erosion", ToggleKind::RidgeErosion)
                        --
                        @slider_row("Steps", 1.0, 100.0, params.ridge_erosion_steps as f32, ErosionSliderFieldKind::RidgeErosionSteps)
                        --
                        @slider_row("Amount", 0.0, 2.0, params.ridge_erosion_amount, ErosionSliderFieldKind::RidgeErosionAmount)
                        --
                        @slider_row("Softening", 0.0, 5.0, params.ridge_softening_amount, ErosionSliderFieldKind::RidgeSofteningAmount)
                    })
                    --
                    @section("Flow", false, bsn_list! {
                        @slider_row("Flow Length", 16.0, 512.0, params.flow_length, ErosionSliderFieldKind::FlowLength)
                        --
                        @slider_row("Flow Volume", 0.0, 2.0, params.flow_volume, ErosionSliderFieldKind::FlowVolume)
                        --
                        @slider_row("Sediment Friction", 0.0, 1.0, params.friction, ErosionSliderFieldKind::SedimentFriction)
                        --
                        @slider_row("Rock Friction", 0.0, 1.0, params.rock_friction, ErosionSliderFieldKind::RockFriction)
                        --
                        @slider_row("Velocity Random", 0.0, 1.0, params.velocity_randomness, ErosionSliderFieldKind::VelocityRandomness)
                        --
                        @slider_row("Random Refinement", 0.0, 1.0, params.velocity_randomness_refinement, ErosionSliderFieldKind::VelocityRandomnessRefinement)
                    })
                    --
                    @section("Sediment", false, bsn_list! {
                        @slider_row("Compaction", 0.0, 1.0, params.sediment_compaction, ErosionSliderFieldKind::SedimentCompaction)
                        --
                        @slider_row("Compaction Threshold", 0.0, 1.0, params.compaction_threshold, ErosionSliderFieldKind::CompactionThreshold)
                        --
                        @slider_row("Suspended Load", 0.0, 1.0, params.suspended_load, ErosionSliderFieldKind::SuspendedSediment)
                    })
                    --
                    @section("Effects", false, bsn_list! {
                        @slider_row("Channeling", 0.0, 1.0, params.channeling, ErosionSliderFieldKind::Channeling)
                        --
                        @slider_row("Channeling Character", 0.0, 2.0, params.channeling_character, ErosionSliderFieldKind::ChannelingCharacter)
                        --
                        @slider_row("Sediment Removal", 0.0, 1.0, params.sediment_removal, ErosionSliderFieldKind::SedimentRemoval)
                        --
                        @slider_row("Removal Character", 0.0, 2.0, params.removal_character, ErosionSliderFieldKind::RemovalCharacter)
                        --
                        @slider_row("Uplift", 0.0, 0.01, params.uplift, ErosionSliderFieldKind::UpliftAmount)
                    })
                    --
                    @section("Rivers", false, bsn_list! {
                        @toggle_row("Do Rivers", ToggleKind::Rivers)
                        --
                        @slider_row("River Scale", 0.0, 1.0, params.river_scale, ErosionSliderFieldKind::RiverScale)
                        --
                        @slider_row("River Threshold", 0.0, 200.0, params.river_threshold, ErosionSliderFieldKind::RiverThreshold)
                        --
                        @slider_row("Trail Damping", 0.0, 1.0, params.trail_damping, ErosionSliderFieldKind::TrailDamping)
                        --
                        @slider_row("River Character", 0.0, 2.0, params.river_character, ErosionSliderFieldKind::RiverCharacter)
                        --
                        @slider_row("River Volume", 0.0, 2.0, params.river_volume, ErosionSliderFieldKind::RiverVolume)
                        --
                        @slider_row("Friction Reduction", 0.0, 2.0, params.river_friction_reduction, ErosionSliderFieldKind::RiverFrictionReduction)
                        --
                        @slider_row("River Channeling", 0.0, 1.0, params.river_channeling, ErosionSliderFieldKind::RiverChanneling)
                        --
                        @toggle_row("Clamp River Channeling", ToggleKind::ClampRiverChanneling)
                        --
                        @slider_row("Meander Longevity", 0.0, 1.0, params.meander_longevity, ErosionSliderFieldKind::MeanderLongevity)
                        --
                        @slider_row("Momentum Coherence", 0.0, 10.0, params.momentum_coherence, ErosionSliderFieldKind::MomentumCoherence)
                        --
                        @slider_row("Uplift River Carving", 0.0, 2.0, params.uplift_river_carving, ErosionSliderFieldKind::UpliftRiverCarving)
                        --
                        @toggle_row("Hydro Conditioning", ToggleKind::HydroConditioning)
                        --
                        @slider_row("Search Distance", 1.0, 64.0, params.hydro_conditioning_search_distance, ErosionSliderFieldKind::HydroSearchDistance)
                    })
                    --
                    @section("Masks & Outflow", false, bsn_list! {
                        @slider_row("Mask Noise Freq", 0.001, 0.05, params.mask_frequency, ErosionSliderFieldKind::MaskFrequency)
                        --
                        @toggle_row("Softness Mask", ToggleKind::SoftnessMask)
                        --
                        @toggle_row("Erosion Mask", ToggleKind::ErosionMask)
                        --
                        @toggle_row("Uplift Map", ToggleKind::UpliftMap)
                        --
                        @toggle_row("Use Outflow (ocean)", ToggleKind::Outflow)
                        --
                        @toggle_row("Accumulate Outflow", ToggleKind::AccumulateOutflow)
                        --
                        @slider_row("Sea Level", 0.0, 1.5, params.outflow_sea_level, ErosionSliderFieldKind::OutflowSeaLevel)
                        --
                        @slider_row("Border Width", 0.0, 64.0, params.outflow_border, ErosionSliderFieldKind::OutflowBorder)
                    })
                    --
                    @section("Bedrock", false, bsn_list! {
                        @toggle_row("Use Bedrock", ToggleKind::Bedrock)
                        --
                        @toggle_row("Use Bedrock Map", ToggleKind::BedrockMap)
                        --
                        @toggle_row("Bedrock Map Is Height", ToggleKind::BedrockMapIsHeight)
                        --
                        @toggle_row("Additive Sediment", ToggleKind::AdditiveSediment)
                        --
                        @slider_row("Bedrock Depth", 0.0, 1.5, params.bedrock_depth, ErosionSliderFieldKind::BedrockDepth)
                    })
                    --
                    @section("Age View", false, bsn_list! {
                        @slider_row("Age Strength", -4.0, 2.0, params.age_strength, ErosionSliderFieldKind::AgeStrength)
                        --
                        @toggle_row("Normalize Age", ToggleKind::NormalizeAge)
                    })
                ]
                --
                @FeathersScrollbar {
                    @target: #sections,
                    @orientation: {ControlOrientation::Vertical}
                }
                Node {
                    position_type: PositionType::Absolute,
                    right: px(0),
                    top: px(0),
                    bottom: px(0),
                    width: px(6),
                }
            ]
        ]
    }
}

/// Title, simulation controls and cascade progress; stays put above the scrolling sections.
fn header_controls() -> impl Scene {
    bsn! {
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            row_gap: px(ROW_GAP),
            padding: UiRect::bottom(px(4.0)),
        }
        ThemedText
        Children [
            Node {}
            InheritableFont {
                font: fonts::BOLD,
                font_size: FontSize::Px(14.0),
                weight: FontWeight::BOLD,
            }
            Children [ Text("Erosion") ThemedText ]
            --
            @button_row(bsn_list! {
                @FeathersButton {
                    @caption: bsn! { Text("Reset (R)") ThemedText },
                    @variant: ButtonVariant::Primary,
                }
                on(|_: On<Activate>, mut reset: ResMut<ResetSim>| {
                    reset.generation = reset.generation.wrapping_add(1);
                })
                --
                @toggle_row("Pause (Space)", ToggleKind::Paused)
                --
                @FeathersButton {
                    @caption: bsn! { Text("Step (E)") ThemedText },
                }
                on(|_: On<Activate>, mut sim: ResMut<SimControl>| {
                    if sim.paused {
                        sim.step_counter = sim.step_counter.wrapping_add(1);
                    }
                })
            })
            --
            @note_text("Level –")
            PyramidStatusText
        ]
    }
}

/// A collapsible section: a clickable header with a chevron over `body`.
fn section(title: &str, open: bool, body: impl SceneList) -> impl Scene {
    let title = title.to_string();
    bsn! {
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            row_gap: px(ROW_GAP),
        }
        ThemedText
        Children [
            Node {
                display: Display::Flex,
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: px(6.0),
                min_height: px(22.0),
                padding: UiRect::horizontal(px(4.0)),
                border_radius: BorderRadius::all(px(4.0)),
            }
            SectionHeader { open: {open} }
            ThemeBackgroundColor(tokens::SUBPANE_HEADER_BG)
            ThemedText
            on(toggle_section)
            Children [
                @icon(icons::CHEVRON_RIGHT)
                SectionChevron
                UiTransform { rotation: {Rot2::turn_fraction(if open { 0.25 } else { 0.0 })} }
                --
                Node {}
                InheritableFont {
                    font: fonts::BOLD,
                    font_size: FontSize::Px(FONT_SIZE),
                    weight: FontWeight::BOLD,
                }
                Children [ Text(title) ThemedText ]
            ]
            --
            Node {
                display: {if open { Display::Flex } else { Display::None }},
                flex_direction: FlexDirection::Column,
                row_gap: px(ROW_GAP),
                padding: UiRect::new(px(6.0), px(2.0), px(0.0), px(6.0)),
            }
            SectionBody
            ThemedText
            Children [
                {body}
            ]
        ]
    }
}

fn button_row(items: impl SceneList) -> impl Scene {
    bsn! {
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            flex_wrap: FlexWrap::Wrap,
            align_items: AlignItems::Center,
            column_gap: px(ROW_GAP),
            row_gap: px(ROW_GAP),
        }
        ThemedText
        Children [
            {items}
        ]
    }
}

/// A line of (wrapping) informational text.
fn note_text(text: impl Into<String>) -> impl Scene {
    let text: String = text.into();
    bsn! {
        Node {
            min_height: px(SLIDER_HEIGHT),
        }
        Text(text)
        ThemedText
    }
}

fn toggle_row(label: &str, kind: ToggleKind) -> impl Scene {
    let label = label.to_string();
    bsn! {
        @FeathersCheckbox {
            @caption: bsn! { Text({label.clone()}) ThemedText },
        }
        Toggle { kind: {kind} }
        on(checkbox_self_update)
        on(move |change: On<ValueChange<bool>>, mut params: ResMut<ErodeParams>, mut sim: ResMut<SimControl>, mut brush: ResMut<WaterBrush>, mut water: ResMut<WaterControl>, mut reset: ResMut<ResetSim>| {
            kind.set(&mut params, &mut sim, &mut brush, &mut water, change.value);
            if kind.resets_sim() {
                reset.generation = reset.generation.wrapping_add(1);
            }
        })
    }
}

/// Selects a brush mode and turns the brush on.
fn brush_mode_button(label: &str, mode: BrushMode) -> impl Scene {
    let label = label.to_string();
    bsn! {
        @FeathersButton {
            @caption: bsn! { Text({label.clone()}) ThemedText },
        }
        BrushModeButton { mode: {mode}, tool: false }
        on(move |_: On<Activate>, mut brush: ResMut<WaterBrush>| {
            // Keep the current terrain tool when re-selecting the terrain brush.
            if !(matches!(mode, BrushMode::Terrain(_)) && matches!(brush.mode, BrushMode::Terrain(_))) {
                brush.mode = mode;
            }
            brush.enabled = true;
        })
    }
}

fn tool_button(label: &str, tool: TerrainTool) -> impl Scene {
    let label = label.to_string();
    let mode = BrushMode::Terrain(tool);
    bsn! {
        @FeathersButton {
            @caption: bsn! { Text({label.clone()}) ThemedText },
        }
        BrushModeButton { mode: {mode}, tool: true }
        on(move |_: On<Activate>, mut brush: ResMut<WaterBrush>| {
            brush.mode = mode;
            brush.enabled = true;
        })
    }
}

fn update_pyramid_ui_labels(
    params: Res<ErodeParams>,
    progress: Res<PyramidProgress>,
    mut count_q: Query<&mut Text, (With<PyramidLevelCountText>, Without<PyramidStatusText>)>,
    mut status_q: Query<&mut Text, (With<PyramidStatusText>, Without<PyramidLevelCountText>)>,
) {
    if params.is_changed() {
        if let Ok(mut text) = count_q.single_mut() {
            **text = format!("Pyramid Levels: {}", params.pyramid_levels().len());
        }
    }
    let p = progress.get();
    if p.level_count == 0 {
        return;
    }
    let status = if p.finished {
        format!("Cascade done · {}×{}", p.size.x, p.size.y)
    } else {
        format!(
            "Level {}/{} · {}×{} · {}/{}",
            p.level + 1,
            p.level_count,
            p.size.x,
            p.size.y,
            p.batch_iter,
            p.batch_len
        )
    };
    if let Ok(mut text) = status_q.single_mut() {
        if **text != status {
            **text = status;
        }
    }
}

/// Slider step and shown decimals: about 1% of the range, whole numbers for counts.
fn slider_step(min: f32, max: f32, int: bool) -> (f32, i32) {
    if int {
        return (1.0, 0);
    }
    let exp = (max - min).max(1e-6).log10().floor() as i32 - 2;
    (10f32.powi(exp), (-exp).max(0))
}

fn slider_row(
    label: &str,
    min: f32,
    max: f32,
    value: f32,
    field: ErosionSliderFieldKind,
) -> impl Scene {
    let label = label.to_string();
    let (step, precision) = slider_step(min, max, field.is_int());
    // Shown at the field's precision, so 0.2 does not read 0.19999999.
    let scale = 10f32.powi(precision);
    let value = (value * scale).round() / scale;
    bsn! {
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: px(4.0),
            min_height: px(SLIDER_HEIGHT),
        }
        ThemedText
        Children [
            Node {
                display: Display::Flex,
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                width: px(LABEL_WIDTH),
                flex_shrink: 0.0,
            }
            ThemedText
            Text(label)
            --
            Node {
                flex_grow: 1.0,
                min_width: px(60.0),
            }
            Children [
                // Drag to scrub through the range, or click to type a value (Enter or clicking
                // away applies it).
                @FeathersNumberInput
                NumberInputValue::F32(value)
                SoftLimit::f32(min..=max)
                HardLimit::f32(min..=max)
                NumberInputStep({step as f64})
                NumberInputPrecision(precision)
                ErosionSliderField
                on(move |change: On<ValueChange<f32>>, mut params: ResMut<ErodeParams>, mut sim: ResMut<SimControl>, mut brush: ResMut<WaterBrush>, mut water: ResMut<WaterControl>, mut reset: ResMut<ResetSim>, mut commands: Commands| {
                    let value = if field.is_int() { change.value.round() } else { change.value };
                    commands.entity(change.source).insert(NumberInputValue::F32(value));
                    field.apply(&mut params, &mut sim, &mut brush, &mut water, value);
                    if field.resets_sim() {
                        reset.generation = reset.generation.wrapping_add(1);
                    }
                })
            ]
        ]
    }
}

fn shrink_slider_heights(mut q: Query<&mut Node, With<ErosionSliderField>>) {
    for mut node in &mut q {
        node.height = Val::Px(SLIDER_HEIGHT);
    }
}

fn shrink_slider_value_font(
    q: Query<(Entity, &Children), With<ErosionSliderField>>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
) {
    for (_input, children) in &q {
        // The number input's text field, sized to the row.
        if let Some(&text) = children.first() {
            commands.entity(text).insert((
                InheritableFont {
                    font: asset_server.load(fonts::MONO),
                    font_size: FontSize::Px(SLIDER_VALUE_FONT_SIZE),
                    weight: FontWeight::NORMAL,
                },
                LineHeight::Px(SLIDER_HEIGHT),
            ));
        }
    }
}

fn shrink_button_checkbox_font(
    mut q: Query<(Entity, &mut Node), Or<(With<Button>, With<Checkbox>)>>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
) {
    let small_font = InheritableFont {
        font: asset_server.load(fonts::REGULAR),
        font_size: FontSize::Px(FONT_SIZE),
        weight: FontWeight::NORMAL,
    };
    for (entity, mut node) in &mut q {
        node.height = Val::Px(BUTTON_HEIGHT);
        commands.entity(entity).insert(small_font.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::slider_step;

    #[test]
    fn slider_steps_follow_the_range() {
        assert_eq!(slider_step(0.0, 1.0, false), (0.01, 2));
        assert_eq!(slider_step(0.0, 0.01, false), (0.0001, 4));
        assert_eq!(slider_step(2.0, 128.0, false), (1.0, 0));
        assert_eq!(slider_step(0.0, 90.0, true), (1.0, 0));
    }
}
