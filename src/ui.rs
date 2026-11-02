//! Bevy Feathers UI for erosion parameters.

use bevy::{
    feathers::{
        FeathersPlugins,
        constants::fonts,
        controls::{ButtonVariant, FeathersButton, FeathersCheckbox, FeathersSlider},
        dark_theme::create_dark_theme,
        font_styles::InheritableFont,
        theme::{ThemeBackgroundColor, ThemedText, UiTheme},
        tokens,
    },
    input_focus::tab_navigation::TabGroup,
    prelude::*,
    text::FontWeight,
    ui::Checked,
    ui_widgets::{
        Activate, Button, Checkbox, SliderPrecision, SliderStep, SliderValue, ValueChange,
        checkbox_self_update, slider_self_update,
    },
};

use crate::camera::OrbitInputBlocked;
use crate::{
    ErodeParams, PyramidProgress, ResetSim, SimControl, WaterControl,
    brush::{BrushMode, TerrainTool, WaterBrush},
};

/// Marker for the erosion params panel root.
#[derive(Component, Clone, Default)]
pub struct ErosionParamsPanel;

#[derive(Component, Clone, Default)]
struct EnableRiversCheckbox;

#[derive(Component, Clone, Default)]
struct EnableRidgeCheckbox;

#[derive(Component, Clone, Default)]
struct ClampRiverChannelingCheckbox;

#[derive(Component, Clone, Default)]
struct HydroConditioningCheckbox;

#[derive(Component, Clone, Default)]
struct PyramidLevelCountText;

#[derive(Component, Clone, Default)]
struct PyramidStatusText;

#[derive(Component, Clone, Default)]
struct BrushModeText;

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
    NoiseFrequency,
    NoiseScale,
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
            Self::RiverCharacter => params.river_character = value,
            Self::RiverVolume => params.river_volume = value,
            Self::RiverFrictionReduction => params.river_friction_reduction = value,
            Self::RiverChanneling => params.river_channeling = value,
            Self::MeanderLongevity => params.meander_longevity = value,
            Self::MomentumCoherence => params.momentum_coherence = value,
            Self::UpliftRiverCarving => params.uplift_river_carving = value,
            Self::NoiseFrequency => params.noise_frequency = value,
            Self::NoiseScale => params.noise_scale = value,
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
            Self::WaterEvaporation => water.evaporation = value,
        }
    }

    fn resets_sim(&self) -> bool {
        matches!(
            self,
            Self::NoiseFrequency
                | Self::NoiseScale
                | Self::ReferenceDetailScale
                | Self::FeatureSize
                | Self::SourceDetailSmoothing
                | Self::GeologicalAge
                | Self::MaskFrequency
                | Self::OutflowSeaLevel
                | Self::OutflowBorder
                | Self::BedrockDepth
        )
    }
}

/// Plugin that adds the erosion parameters UI panel using Bevy Feathers.
pub struct ErosionParamsPlugin;

const FONT_SIZE: f32 = 10.0;
const SLIDER_VALUE_FONT_SIZE: f32 = 8.0;
const SLIDER_HEIGHT: f32 = 12.0;
const BUTTON_HEIGHT: f32 = 18.0;
const ROW_GAP: f32 = 4.0;
const PADDING: f32 = 8.0;

impl Plugin for ErosionParamsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FeathersPlugins)
            .insert_resource(UiTheme(create_dark_theme()))
            .add_systems(
                PostStartup,
                (
                    spawn_erosion_params_panel,
                    (
                        init_param_checkboxes,
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
                    sync_water_brush_checkbox,
                    update_brush_mode_label,
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

fn init_param_checkboxes(
    mut commands: Commands,
    params: Res<ErodeParams>,
    rivers: Query<Entity, With<EnableRiversCheckbox>>,
    ridge: Query<Entity, With<EnableRidgeCheckbox>>,
    clamp_channeling: Query<Entity, With<ClampRiverChannelingCheckbox>>,
    hydro: Query<Entity, With<HydroConditioningCheckbox>>,
) {
    if params.clamp_river_channeling != 0 {
        if let Ok(entity) = clamp_channeling.single() {
            commands.entity(entity).insert(Checked);
        }
    }
    if params.use_hydro_conditioning != 0 {
        if let Ok(entity) = hydro.single() {
            commands.entity(entity).insert(Checked);
        }
    }
    if params.do_rivers != 0 {
        if let Ok(entity) = rivers.single() {
            commands.entity(entity).insert(Checked);
        }
    }
    if params.compute_ridge_erosion != 0 {
        if let Ok(entity) = ridge.single() {
            commands.entity(entity).insert(Checked);
        }
    }
}

#[derive(Component, Default, Clone)]
struct WaterBrushCheckbox;

/// Keep the checkbox in step with keyboard toggles.
fn sync_water_brush_checkbox(
    mut commands: Commands,
    brush: Res<WaterBrush>,
    checkbox: Query<(Entity, Has<Checked>), With<WaterBrushCheckbox>>,
) {
    if !brush.is_changed() {
        return;
    }
    for (entity, checked) in &checkbox {
        if brush.enabled && !checked {
            commands.entity(entity).insert(Checked);
        } else if !brush.enabled && checked {
            commands.entity(entity).remove::<Checked>();
        }
    }
}

fn erosion_params_panel(params: ErodeParams, sim: SimControl) -> impl Scene {
    bsn! {
        Node {
            position_type: PositionType::Absolute,
            top: px(10.0),
            left: px(10.0),
            width: px(260.0),
            padding: UiRect::all(px(PADDING)),
            row_gap: px(ROW_GAP),
            column_gap: px(ROW_GAP),
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Stretch,
            justify_content: JustifyContent::Start,
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
            @section_label("Water")
            --
            @FeathersCheckbox {
                @caption: bsn! { Text("Water Brush (B)") ThemedText },
            }
            WaterBrushCheckbox
            on(checkbox_self_update)
            on(|change: On<ValueChange<bool>>, mut brush: ResMut<WaterBrush>| {
                brush.enabled = change.value;
            })
            --
            Node {
                display: Display::Flex,
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                min_height: px(SLIDER_HEIGHT),
            }
            ThemedText
            Children [
                Text("Brush: Rain (off)")
                ThemedText
                BrushModeText
            ]
            --
            Node {
                display: Display::Flex,
                flex_direction: FlexDirection::Row,
                column_gap: px(ROW_GAP),
            }
            ThemedText
            Children [
                @brush_mode_button("Rain", BrushMode::Rain)
                --
                @brush_mode_button("Terrain", BrushMode::Terrain(TerrainTool::Raise))
                --
                @brush_mode_button("Water Body", BrushMode::WaterBody)
            ]
            --
            Node {
                display: Display::Flex,
                flex_direction: FlexDirection::Row,
                column_gap: px(ROW_GAP),
            }
            ThemedText
            Children [
                @brush_mode_button("Raise", BrushMode::Terrain(TerrainTool::Raise))
                --
                @brush_mode_button("Lower", BrushMode::Terrain(TerrainTool::Lower))
                --
                @brush_mode_button("Flatten", BrushMode::Terrain(TerrainTool::Flatten))
                --
                @brush_mode_button("Smooth", BrushMode::Terrain(TerrainTool::Smooth))
            ]
            --
            @slider_row("Brush Radius", 2.0, 128.0, WaterBrush::default().radius, ErosionSliderFieldKind::BrushRadius)
            --
            @slider_row("Brush Strength", 0.01, 5.0, WaterBrush::default().strength, ErosionSliderFieldKind::BrushStrength)
            --
            @slider_row("Water Erosion", 0.0, 0.01, WaterControl::default().sediment_capacity, ErosionSliderFieldKind::WaterErosion)
            --
            @slider_row("Evaporation", 0.0, 0.0002, WaterControl::default().evaporation, ErosionSliderFieldKind::WaterEvaporation)
            --
            Node {
                display: Display::Flex,
                flex_direction: FlexDirection::Row,
                column_gap: px(ROW_GAP),
            }
            ThemedText
            Children [
                @FeathersButton {
                    @caption: bsn! { Text("Clear Water (C)") ThemedText },
                }
                on(|_: On<Activate>, mut brush: ResMut<WaterBrush>| {
                    brush.clear_water = true;
                })
                --
                @FeathersButton {
                    @caption: bsn! { Text("Reset Terrain Edits") ThemedText },
                }
                on(|_: On<Activate>, mut brush: ResMut<WaterBrush>| {
                    brush.clear_terrain = true;
                })
            ]
            --
            @section_label("Noise")
            --
            @slider_row("Noise Freq", 0.001, 0.05, params.noise_frequency, ErosionSliderFieldKind::NoiseFrequency)
            --
            @slider_row("Noise Scale", 0.5, 5.0, params.noise_scale, ErosionSliderFieldKind::NoiseScale)
            --
            @section_label("Main")
            --
            @slider_row("Erosion Strength", 0.0, 1.0, params.erosion_strength, ErosionSliderFieldKind::ErosionStrength)
            --
            @slider_row("Rock Softness", 0.0, 1.0, params.rock_softness, ErosionSliderFieldKind::RockSoftness)
            --
            @slider_row("Erosion Amount", 0.0, 1.0, params.erosion_amount, ErosionSliderFieldKind::ErosionAmount)
            --
            @slider_row("Reference Detail Scale", 0.25, 4.0, params.detail_scale, ErosionSliderFieldKind::ReferenceDetailScale)
            --
            @section_label("Pyramid")
            --
            @pyramid_info_row(params.pyramid_levels().len())
            --
            @slider_row("Feature Size", 0.5, 64.0, params.feature_size, ErosionSliderFieldKind::FeatureSize)
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
            @slider_row_int("Iterations / Frame", 1.0, 64.0, sim.iterations_per_frame as f32, ErosionSliderFieldKind::IterationsPerFrame)
            --
            @FeathersCheckbox {
                @caption: bsn! { Text("Continue At Finest") ThemedText },
            }
            on(checkbox_self_update)
            on(|change: On<ValueChange<bool>>, mut sim: ResMut<SimControl>| {
                sim.continue_after_cascade = change.value;
            })
            --
            @section_label("Angles (°)")
            --
            @slider_row("Wear Angle", 0.0, 90.0, params.wear_angle, ErosionSliderFieldKind::WearAngle)
            --
            @slider_row("Talus Angle", 0.0, 90.0, params.talus_angle, ErosionSliderFieldKind::TalusAngle)
            --
            @slider_row("Max Deposit Angle", 0.0, 90.0, params.max_deposit_angle, ErosionSliderFieldKind::MaxDepositAngle)
            --
            @slider_row("Erosion Cutoff Angle", 0.0, 90.0, params.erosion_cutoff_angle, ErosionSliderFieldKind::ErosionCutoffAngle)
            --
            @section_label("Ridge Erosion")
            --
            @FeathersCheckbox {
                @caption: bsn! { Text("Do Ridge Erosion") ThemedText },
            }
            EnableRidgeCheckbox
            on(checkbox_self_update)
            on(|change: On<ValueChange<bool>>, mut params: ResMut<ErodeParams>| {
                params.compute_ridge_erosion = u32::from(change.value);
            })
            --
            @slider_row("Ridge Erosion Steps", 1.0, 100.0, params.ridge_erosion_steps as f32, ErosionSliderFieldKind::RidgeErosionSteps)
            --
            @slider_row("Ridge Erosion Amount", 0.0, 2.0, params.ridge_erosion_amount, ErosionSliderFieldKind::RidgeErosionAmount)
            --
            @slider_row("Ridge Softening Amount", 0.0, 5.0, params.ridge_softening_amount, ErosionSliderFieldKind::RidgeSofteningAmount)
            --
            @section_label("Flow")
            --
            @slider_row("Flow Length", 16.0, 512.0, params.flow_length, ErosionSliderFieldKind::FlowLength)
            --
            @slider_row("Flow Volume", 0.0, 2.0, params.flow_volume, ErosionSliderFieldKind::FlowVolume)
            --
            @slider_row("Sediment Friction", 0.0, 1.0, params.friction, ErosionSliderFieldKind::SedimentFriction)
            --
            @slider_row("Rock Friction", 0.0, 1.0, params.rock_friction, ErosionSliderFieldKind::RockFriction)
            --
            @slider_row("Velocity Randomness", 0.0, 1.0, params.velocity_randomness, ErosionSliderFieldKind::VelocityRandomness)
            --
            @slider_row("Velocity Rand. Refine", 0.0, 1.0, params.velocity_randomness_refinement, ErosionSliderFieldKind::VelocityRandomnessRefinement)
            --
            @section_label("Sediment")
            --
            @slider_row("Sediment Compaction", 0.0, 1.0, params.sediment_compaction, ErosionSliderFieldKind::SedimentCompaction)
            --
            @slider_row("Compaction Threshold", 0.0, 1.0, params.compaction_threshold, ErosionSliderFieldKind::CompactionThreshold)
            --
            @slider_row("Suspended Sediment", 0.0, 1.0, params.suspended_load, ErosionSliderFieldKind::SuspendedSediment)
            --
            @section_label("Effects")
            --
            @slider_row("Channeling", 0.0, 1.0, params.channeling, ErosionSliderFieldKind::Channeling)
            --
            @slider_row("Channeling Character", 0.0, 2.0, params.channeling_character, ErosionSliderFieldKind::ChannelingCharacter)
            --
            @slider_row("Sediment Removal", 0.0, 1.0, params.sediment_removal, ErosionSliderFieldKind::SedimentRemoval)
            --
            @slider_row("Removal Character", 0.0, 2.0, params.removal_character, ErosionSliderFieldKind::RemovalCharacter)
            --
            @slider_row("Uplift Amount", 0.0, 0.01, params.uplift, ErosionSliderFieldKind::UpliftAmount)
            --
            @section_label("Rivers")
            --
            @FeathersCheckbox {
                @caption: bsn! { Text("Do Rivers") ThemedText },
            }
            EnableRiversCheckbox
            on(checkbox_self_update)
            on(|change: On<ValueChange<bool>>, mut params: ResMut<ErodeParams>, mut reset: ResMut<ResetSim>| {
                params.do_rivers = u32::from(change.value);
                reset.generation = reset.generation.wrapping_add(1);
            })
            --
            @slider_row("River Scale", 0.0, 1.0, params.river_scale, ErosionSliderFieldKind::RiverScale)
            --
            @slider_row("River Character", 0.0, 2.0, params.river_character, ErosionSliderFieldKind::RiverCharacter)
            --
            @slider_row("River Volume", 0.0, 2.0, params.river_volume, ErosionSliderFieldKind::RiverVolume)
            --
            @slider_row("River Friction Reduction", 0.0, 2.0, params.river_friction_reduction, ErosionSliderFieldKind::RiverFrictionReduction)
            --
            @slider_row("River Channeling", 0.0, 1.0, params.river_channeling, ErosionSliderFieldKind::RiverChanneling)
            --
            @FeathersCheckbox {
                @caption: bsn! { Text("Clamp River Channeling") ThemedText },
            }
            ClampRiverChannelingCheckbox
            on(checkbox_self_update)
            on(|change: On<ValueChange<bool>>, mut params: ResMut<ErodeParams>| {
                params.clamp_river_channeling = u32::from(change.value);
            })
            --
            @slider_row("Meander Longevity", 0.0, 1.0, params.meander_longevity, ErosionSliderFieldKind::MeanderLongevity)
            --
            @slider_row("Momentum Coherence", 0.0, 10.0, params.momentum_coherence, ErosionSliderFieldKind::MomentumCoherence)
            --
            @slider_row("Uplift River Carving", 0.0, 2.0, params.uplift_river_carving, ErosionSliderFieldKind::UpliftRiverCarving)
            --
            @section_label("Hydro Conditioning")
            --
            @FeathersCheckbox {
                @caption: bsn! { Text("Use Hydro Conditioning") ThemedText },
            }
            HydroConditioningCheckbox
            on(checkbox_self_update)
            on(|change: On<ValueChange<bool>>, mut params: ResMut<ErodeParams>| {
                params.use_hydro_conditioning = u32::from(change.value);
            })
            --
            @slider_row("Search Distance", 1.0, 64.0, params.hydro_conditioning_search_distance, ErosionSliderFieldKind::HydroSearchDistance)
            --
            @section_label("Masks")
            --
            @FeathersCheckbox {
                @caption: bsn! { Text("Use Softness Mask") ThemedText },
            }
            on(checkbox_self_update)
            on(|change: On<ValueChange<bool>>, mut params: ResMut<ErodeParams>, mut reset: ResMut<ResetSim>| {
                params.use_softness_mask = u32::from(change.value);
                reset.generation = reset.generation.wrapping_add(1);
            })
            --
            @slider_row("Mask Noise Freq", 0.001, 0.05, params.mask_frequency, ErosionSliderFieldKind::MaskFrequency)
            --
            @FeathersCheckbox {
                @caption: bsn! { Text("Use Erosion Mask") ThemedText },
            }
            on(checkbox_self_update)
            on(|change: On<ValueChange<bool>>, mut params: ResMut<ErodeParams>, mut reset: ResMut<ResetSim>| {
                params.use_erosion_mask = u32::from(change.value);
                reset.generation = reset.generation.wrapping_add(1);
            })
            --
            @FeathersCheckbox {
                @caption: bsn! { Text("Use Uplift Map") ThemedText },
            }
            on(checkbox_self_update)
            on(|change: On<ValueChange<bool>>, mut params: ResMut<ErodeParams>, mut reset: ResMut<ResetSim>| {
                params.use_uplift_map = u32::from(change.value);
                reset.generation = reset.generation.wrapping_add(1);
            })
            --
            @FeathersCheckbox {
                @caption: bsn! { Text("Use Outflow") ThemedText },
            }
            on(checkbox_self_update)
            on(|change: On<ValueChange<bool>>, mut params: ResMut<ErodeParams>, mut reset: ResMut<ResetSim>| {
                params.use_outflow = u32::from(change.value);
                reset.generation = reset.generation.wrapping_add(1);
            })
            --
            @FeathersCheckbox {
                @caption: bsn! { Text("Accumulate Outflow") ThemedText },
            }
            on(checkbox_self_update)
            on(|change: On<ValueChange<bool>>, mut params: ResMut<ErodeParams>| {
                params.outflow_integration = u32::from(change.value);
            })
            --
            @slider_row("Sea Level", 0.0, 1.5, params.outflow_sea_level, ErosionSliderFieldKind::OutflowSeaLevel)
            --
            @slider_row("Border Width", 0.0, 64.0, params.outflow_border, ErosionSliderFieldKind::OutflowBorder)
            --
            @section_label("Bedrock")
            --
            @FeathersCheckbox {
                @caption: bsn! { Text("Use Bedrock") ThemedText },
            }
            on(checkbox_self_update)
            on(|change: On<ValueChange<bool>>, mut params: ResMut<ErodeParams>, mut reset: ResMut<ResetSim>| {
                params.use_bedrock = u32::from(change.value);
                reset.generation = reset.generation.wrapping_add(1);
            })
            --
            @FeathersCheckbox {
                @caption: bsn! { Text("Use Bedrock Map") ThemedText },
            }
            on(checkbox_self_update)
            on(|change: On<ValueChange<bool>>, mut params: ResMut<ErodeParams>, mut reset: ResMut<ResetSim>| {
                params.use_bedrock_map = u32::from(change.value);
                reset.generation = reset.generation.wrapping_add(1);
            })
            --
            @FeathersCheckbox {
                @caption: bsn! { Text("Bedrock Map Is Height") ThemedText },
            }
            on(checkbox_self_update)
            on(|change: On<ValueChange<bool>>, mut params: ResMut<ErodeParams>, mut reset: ResMut<ResetSim>| {
                params.bedrock_mode = u32::from(change.value);
                reset.generation = reset.generation.wrapping_add(1);
            })
            --
            @FeathersCheckbox {
                @caption: bsn! { Text("Additive Sediment") ThemedText },
            }
            on(checkbox_self_update)
            on(|change: On<ValueChange<bool>>, mut params: ResMut<ErodeParams>, mut reset: ResMut<ResetSim>| {
                params.additive_sediment = u32::from(change.value);
                reset.generation = reset.generation.wrapping_add(1);
            })
            --
            @slider_row("Bedrock Depth", 0.0, 1.5, params.bedrock_depth, ErosionSliderFieldKind::BedrockDepth)
            --
            @section_label("Age")
            --
            @slider_row("Age Strength", -4.0, 2.0, params.age_strength, ErosionSliderFieldKind::AgeStrength)
            --
            @FeathersCheckbox {
                @caption: bsn! { Text("Normalize Age") ThemedText },
            }
            on(checkbox_self_update)
            on(|change: On<ValueChange<bool>>, mut params: ResMut<ErodeParams>| {
                params.normalize_age = u32::from(change.value);
            })
        ]
    }
}

fn header_controls() -> impl Scene {
    bsn! {
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            row_gap: px(ROW_GAP),
        }
        ThemedText
        Children [
            Text("Erosion Parameters") ThemedText
            --
            Node {
                display: Display::Flex,
                flex_direction: FlexDirection::Row,
                column_gap: px(ROW_GAP),
                align_items: AlignItems::Center,
            }
            ThemedText
            Children [
                @FeathersButton {
                    @caption: bsn! { Text("Reset") ThemedText },
                    @variant: ButtonVariant::Primary,
                }
                on(|_: On<Activate>, mut reset: ResMut<ResetSim>| {
                    reset.generation = reset.generation.wrapping_add(1);
                })
                --
                @FeathersCheckbox {
                    @caption: bsn! { Text("Pause") ThemedText },
                }
                on(checkbox_self_update)
                on(|change: On<ValueChange<bool>>, mut sim: ResMut<SimControl>| {
                    sim.paused = change.value;
                })
                --
                @FeathersButton {
                    @caption: bsn! { Text("Step") ThemedText },
                }
                on(|_: On<Activate>, mut sim: ResMut<SimControl>| {
                    if sim.paused {
                        sim.step_counter = sim.step_counter.wrapping_add(1);
                    }
                })
            ]
        ]
    }
}

fn pyramid_info_row(level_count: usize) -> impl Scene {
    bsn! {
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            row_gap: px(2.0),
        }
        ThemedText
        Children [
            Node {
                display: Display::Flex,
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                min_height: px(SLIDER_HEIGHT),
            }
            ThemedText
            Children [
                Text(format!("Pyramid Levels: {level_count}"))
                ThemedText
                PyramidLevelCountText
            ]
            --
            Node {
                display: Display::Flex,
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                min_height: px(SLIDER_HEIGHT),
            }
            ThemedText
            Children [
                Text("Level –")
                ThemedText
                PyramidStatusText
            ]
        ]
    }
}

/// Selects a brush mode and turns the brush on.
fn brush_mode_button(label: &str, mode: BrushMode) -> impl Scene {
    let label = label.to_string();
    bsn! {
        @FeathersButton {
            @caption: bsn! { Text({label.clone()}) ThemedText },
        }
        on(move |_: On<Activate>, mut brush: ResMut<WaterBrush>| {
            brush.mode = mode;
            brush.enabled = true;
        })
    }
}

fn update_brush_mode_label(brush: Res<WaterBrush>, mut q: Query<&mut Text, With<BrushModeText>>) {
    if !brush.is_changed() {
        return;
    }
    for mut text in &mut q {
        **text = format!(
            "Brush: {}{}",
            brush.mode.label(),
            if brush.enabled { "" } else { " (off)" }
        );
    }
}

fn slider_row_int(
    label: &str,
    min: f32,
    max: f32,
    value: f32,
    field: ErosionSliderFieldKind,
) -> impl Scene {
    let label = label.to_string();
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
                min_width: px(110.0),
            }
            ThemedText
            Text(label)
            --
            Node {
                flex_grow: 1.0,
                min_width: px(60.0),
            }
            Children [
                @FeathersSlider {
                    @min: min,
                    @max: max,
                }
                SliderValue(value)
                SliderStep(1.0)
                SliderPrecision(0)
                ErosionSliderField
                on(slider_self_update)
                on(move |change: On<ValueChange<f32>>, mut params: ResMut<ErodeParams>, mut sim: ResMut<SimControl>, mut brush: ResMut<WaterBrush>, mut water: ResMut<WaterControl>, mut reset: ResMut<ResetSim>| {
                    field.apply(&mut params, &mut sim, &mut brush, &mut water, change.value);
                    if field.resets_sim() {
                        reset.generation = reset.generation.wrapping_add(1);
                    }
                })
            ]
        ]
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
        format!("Done · {}×{}", p.size.x, p.size.y)
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

fn section_label(text: &str) -> impl Scene {
    let text = text.to_string();
    bsn! {
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            margin: UiRect::top(px(4.0)),
            width: percent(100),
        }
        ThemedText
        Text(text)
    }
}

fn slider_row(
    label: &str,
    min: f32,
    max: f32,
    value: f32,
    field: ErosionSliderFieldKind,
) -> impl Scene {
    let label = label.to_string();
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
                min_width: px(110.0),
            }
            ThemedText
            Text(label)
            --
            Node {
                flex_grow: 1.0,
                min_width: px(60.0),
            }
            Children [
                @FeathersSlider {
                    @min: min,
                    @max: max,
                }
                SliderValue(value)
                SliderStep(0.01)
                SliderPrecision(3)
                ErosionSliderField
                on(slider_self_update)
                on(move |change: On<ValueChange<f32>>, mut params: ResMut<ErodeParams>, mut sim: ResMut<SimControl>, mut brush: ResMut<WaterBrush>, mut water: ResMut<WaterControl>, mut reset: ResMut<ResetSim>| {
                    field.apply(&mut params, &mut sim, &mut brush, &mut water, change.value);
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
    for (_slider, children) in &q {
        if let Some(&text_container) = children.first() {
            commands.entity(text_container).insert(InheritableFont {
                font: asset_server.load(fonts::MONO),
                font_size: FontSize::Px(SLIDER_VALUE_FONT_SIZE),
                weight: FontWeight::NORMAL,
            });
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
