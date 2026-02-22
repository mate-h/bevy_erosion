//! Bevy Feathers UI for erosion parameters.

use bevy::{
    feathers::{
        constants::fonts,
        controls::{button, checkbox, slider, ButtonProps, ButtonVariant, SliderProps},
        dark_theme::create_dark_theme,
        font_styles::InheritableFont,
        handle_or_path::HandleOrPath,
        theme::{ThemeBackgroundColor, ThemedText, UiTheme},
        tokens, FeathersPlugins,
    },
    input_focus::tab_navigation::TabGroup,
    prelude::*,
    ui::Checked,
    ui_widgets::{
        checkbox_self_update, observe, slider_self_update, Activate, Button, Checkbox,
        SliderPrecision, SliderStep, ValueChange,
    },
};

use crate::camera::OrbitInputBlocked;
use crate::{ErodeParams, ResetSim, SimControl};

/// Marker for the erosion params panel root.
#[derive(Component)]
pub struct ErosionParamsPanel;

/// Marks erosion param sliders for styling (height, font).
#[derive(Component)]
struct ErosionSliderField;

#[derive(Clone, Copy, PartialEq)]
enum ErosionSliderFieldKind {
    ErosionStrength,
    RockSoftness,
    TrailDensity,
    DetailScale,
    WearAngle,
    TalusAngle,
    MaxDepositAngle,
    FlowLength,
    RidgeErosionSteps,
    RidgeSofteningAmount,
    RidgeErosionAmount,
    Friction,
    RockFriction,
    SedimentCompaction,
    CompactionThreshold,
    Channeling,
    SedimentRemoval,
    Uplift,
}

impl ErosionSliderFieldKind {
    fn apply(&self, params: &mut ErodeParams, value: f32) {
        match self {
            Self::ErosionStrength => params.erosion_strength = value,
            Self::RockSoftness => params.rock_softness = value,
            Self::TrailDensity => params.trail_density = value,
            Self::DetailScale => params.detail_scale = value,
            Self::WearAngle => params.wear_angle = value,
            Self::TalusAngle => params.talus_angle = value,
            Self::MaxDepositAngle => params.max_deposit_angle = value,
            Self::FlowLength => params.flow_length = value,
            Self::RidgeErosionSteps => params.ridge_erosion_steps = value as u32,
            Self::RidgeSofteningAmount => params.ridge_softening_amount = value,
            Self::RidgeErosionAmount => params.ridge_erosion_amount = value,
            Self::Friction => params.friction = value,
            Self::RockFriction => params.rock_friction = value,
            Self::SedimentCompaction => params.sediment_compaction = value,
            Self::CompactionThreshold => params.compaction_threshold = value,
            Self::Channeling => params.channeling = value,
            Self::SedimentRemoval => params.sediment_removal = value,
            Self::Uplift => params.uplift = value,
        }
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
            .add_systems(Startup, spawn_erosion_params_panel)
            .add_systems(
                PostStartup,
                (shrink_slider_heights, shrink_slider_value_font, shrink_button_checkbox_font),
            );
    }
}

fn spawn_erosion_params_panel(mut commands: Commands) {
    let params = ErodeParams::default();

    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            width: Val::Px(260.0),
            padding: UiRect::all(Val::Px(PADDING)),
            row_gap: Val::Px(ROW_GAP),
            column_gap: Val::Px(ROW_GAP),
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Stretch,
            justify_content: JustifyContent::Start,
            ..default()
        },
        TabGroup::default(),
        ThemeBackgroundColor(tokens::WINDOW_BG),
        ErosionParamsPanel,
        InheritableFont {
            font: HandleOrPath::Path(fonts::REGULAR.to_string()),
            font_size: FONT_SIZE.into(),
        },
        observe(|_: On<Pointer<Over>>, mut blocked: ResMut<OrbitInputBlocked>| {
            blocked.0 = true;
        }),
        observe(|_: On<Pointer<Out>>, mut blocked: ResMut<OrbitInputBlocked>| {
            blocked.0 = false;
        }),
        children![
            // Header + sim controls
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(ROW_GAP),
                    ..default()
                },
                ThemedText,
                InheritableFont {
                    font: HandleOrPath::Path(fonts::REGULAR.to_string()),
                    font_size: FONT_SIZE.into(),
                },
                children![
                    (Text::new("Erosion Parameters"), ThemedText),
                    (
                        Node {
                            display: Display::Flex,
                            flex_direction: FlexDirection::Row,
                            column_gap: Val::Px(ROW_GAP),
                            align_items: AlignItems::Center,
                            ..default()
                        },
                        ThemedText,
                        InheritableFont {
                            font: HandleOrPath::Path(fonts::REGULAR.to_string()),
                            font_size: FONT_SIZE.into(),
                        },
                        children![
                            (
                                button(
                                    ButtonProps {
                                        variant: ButtonVariant::Primary,
                                        ..default()
                                    },
                                    (),
                                    Spawn((Text::new("Reset"), ThemedText)),
                                ),
                                observe(|_: On<Activate>, mut reset: ResMut<ResetSim>| {
                                    reset.generation = reset.generation.wrapping_add(1);
                                }),
                            ),
                            (
                                checkbox((), Spawn((Text::new("Pause"), ThemedText))),
                                observe(checkbox_self_update),
                                observe(
                                    |change: On<ValueChange<bool>>, mut sim: ResMut<SimControl>| {
                                        sim.paused = change.value;
                                    },
                                ),
                            ),
                            (
                                button(
                                    ButtonProps::default(),
                                    (),
                                    Spawn((Text::new("Step"), ThemedText)),
                                ),
                                observe(
                                    |_: On<Activate>, mut sim: ResMut<SimControl>| {
                                        if sim.paused {
                                            sim.step_counter = sim.step_counter.wrapping_add(1);
                                        }
                                    },
                                ),
                            ),
                        ],
                    ),
                ],
            ),
            section_label("Main"),
            slider_row("Erosion Strength", 0.0, 1.0, params.erosion_strength, ErosionSliderFieldKind::ErosionStrength),
            slider_row("Rock Softness", 0.0, 1.0, params.rock_softness, ErosionSliderFieldKind::RockSoftness),
            slider_row("Trail Density", 0.01, 0.5, params.trail_density, ErosionSliderFieldKind::TrailDensity),
            slider_row("Detail Scale", 0.5, 16.0, params.detail_scale, ErosionSliderFieldKind::DetailScale),
            section_label("Angles (°)"),
            slider_row("Wear Angle", 0.0, 90.0, params.wear_angle, ErosionSliderFieldKind::WearAngle),
            slider_row("Talus Angle", 0.0, 90.0, params.talus_angle, ErosionSliderFieldKind::TalusAngle),
            slider_row(
                "Max Deposit Angle",
                0.0,
                90.0,
                params.max_deposit_angle,
                ErosionSliderFieldKind::MaxDepositAngle,
            ),
            section_label("Ridge Erosion"),
            (
                checkbox(
                    Checked, // default: ridge erosion on (compute_ridge_erosion=1)
                    Spawn((Text::new("Enable Ridge"), ThemedText)),
                ),
                observe(checkbox_self_update),
                observe(
                    |change: On<ValueChange<bool>>, mut params: ResMut<ErodeParams>| {
                        params.compute_ridge_erosion = if change.value { 1 } else { 0 };
                    },
                ),
            ),
            slider_row(
                "Ridge Steps",
                1.0,
                100.0,
                params.ridge_erosion_steps as f32,
                ErosionSliderFieldKind::RidgeErosionSteps,
            ),
            slider_row(
                "Ridge Amount",
                0.0,
                2.0,
                params.ridge_erosion_amount,
                ErosionSliderFieldKind::RidgeErosionAmount,
            ),
            slider_row(
                "Ridge Softening",
                0.0,
                5.0,
                params.ridge_softening_amount,
                ErosionSliderFieldKind::RidgeSofteningAmount,
            ),
            section_label("Flow"),
            slider_row("Flow Length", 16.0, 512.0, params.flow_length, ErosionSliderFieldKind::FlowLength),
            slider_row("Friction", 0.0, 1.0, params.friction, ErosionSliderFieldKind::Friction),
            slider_row("Rock Friction", 0.0, 1.0, params.rock_friction, ErosionSliderFieldKind::RockFriction),
            section_label("Sediment"),
            slider_row(
                "Compaction",
                0.0,
                1.0,
                params.sediment_compaction,
                ErosionSliderFieldKind::SedimentCompaction,
            ),
            slider_row(
                "Compaction Threshold",
                0.0,
                1.0,
                params.compaction_threshold,
                ErosionSliderFieldKind::CompactionThreshold,
            ),
            section_label("Effects"),
            slider_row("Channeling", 0.0, 1.0, params.channeling, ErosionSliderFieldKind::Channeling),
            slider_row(
                "Sediment Removal",
                0.0,
                1.0,
                params.sediment_removal,
                ErosionSliderFieldKind::SedimentRemoval,
            ),
            slider_row("Uplift", 0.0, 0.01, params.uplift, ErosionSliderFieldKind::Uplift),
        ],
    ));
}

fn shrink_slider_heights(
    mut q: Query<&mut Node, With<ErosionSliderField>>,
) {
    for mut node in &mut q {
        node.height = Val::Px(SLIDER_HEIGHT);
    }
}

fn shrink_slider_value_font(
    q: Query<(Entity, &Children), With<ErosionSliderField>>,
    mut commands: Commands,
) {
    for (_slider, children) in &q {
        if let Some(&text_container) = children.first() {
            commands.entity(text_container).insert(InheritableFont {
                font: HandleOrPath::Path(fonts::MONO.to_string()),
                font_size: SLIDER_VALUE_FONT_SIZE.into(),
            });
        }
    }
}

fn shrink_button_checkbox_font(
    mut q: Query<
        (Entity, &mut Node),
        Or<(With<Button>, With<Checkbox>)>,
    >,
    mut commands: Commands,
) {
    let small_font = InheritableFont {
        font: HandleOrPath::Path(fonts::REGULAR.to_string()),
        font_size: FONT_SIZE.into(),
    };
    for (entity, mut node) in &mut q {
        node.height = Val::Px(BUTTON_HEIGHT);
        commands.entity(entity).insert(small_font.clone());
    }
}

fn section_label(text: &str) -> impl Bundle {
    (
        Text::new(text),
        ThemedText,
        Node {
            margin: UiRect::top(Val::Px(4.0)),
            ..default()
        },
    )
}

fn slider_row(
    label: &str,
    min: f32,
    max: f32,
    value: f32,
    field: ErosionSliderFieldKind,
) -> impl Bundle {
    (
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: Val::Px(4.0),
            min_height: Val::Px(SLIDER_HEIGHT),
            ..default()
        },
        ThemedText,
        InheritableFont {
            font: HandleOrPath::Path(fonts::REGULAR.to_string()),
            font_size: FONT_SIZE.into(),
        },
        children![
            (
                Node {
                    min_width: Val::Px(110.0),
                    ..default()
                },
                (Text::new(label), ThemedText),
            ),
            (
                Node {
                    flex_grow: 1.0,
                    min_width: Val::Px(60.0),
                    ..default()
                },
                children![(
                    slider(
                        SliderProps { value, min, max, ..default() },
                        (
                            SliderStep(0.01),
                            SliderPrecision(3),
                            ErosionSliderField,
                        ),
                    ),
                    observe(slider_self_update),
                    observe(move |change: On<ValueChange<f32>>, mut params: ResMut<ErodeParams>| {
                        field.apply(&mut params, change.value);
                    }),
                )],
            ),
        ],
    )
}
