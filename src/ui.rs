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
        Activate, Button, Checkbox, SliderPrecision, SliderStep, ValueChange, checkbox_self_update,
        slider_self_update,
    },
};

use crate::camera::OrbitInputBlocked;
use crate::{ErodeParams, ResetSim, SimControl};

/// Marker for the erosion params panel root.
#[derive(Component, Clone, Default)]
pub struct ErosionParamsPanel;

/// Marks erosion param sliders for styling (height, font).
#[derive(Component, Clone, Default)]
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
    NoiseFrequency,
    NoiseScale,
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
            Self::NoiseFrequency => params.noise_frequency = value,
            Self::NoiseScale => params.noise_scale = value,
        }
    }

    fn resets_sim(&self) -> bool {
        matches!(self, Self::NoiseFrequency | Self::NoiseScale)
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
                (
                    shrink_slider_heights,
                    shrink_slider_value_font,
                    shrink_button_checkbox_font,
                ),
            );
    }
}

fn spawn_erosion_params_panel(mut commands: Commands) {
    let params = ErodeParams::default();
    commands.spawn_scene(erosion_params_panel(params));
}

fn erosion_params_panel(params: ErodeParams) -> impl Scene {
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
        on(|_: On<Pointer<Over>>, mut blocked: ResMut<OrbitInputBlocked>| {
            blocked.0 = true;
        })
        on(|_: On<Pointer<Out>>, mut blocked: ResMut<OrbitInputBlocked>| {
            blocked.0 = false;
        })
        Children [
            header_controls(),
            section_label("Noise"),
            slider_row("Noise Freq", 0.001, 0.05, params.noise_frequency, ErosionSliderFieldKind::NoiseFrequency),
            slider_row("Noise Scale", 0.5, 5.0, params.noise_scale, ErosionSliderFieldKind::NoiseScale),
            section_label("Main"),
            slider_row("Erosion Strength", 0.0, 1.0, params.erosion_strength, ErosionSliderFieldKind::ErosionStrength),
            slider_row("Rock Softness", 0.0, 1.0, params.rock_softness, ErosionSliderFieldKind::RockSoftness),
            slider_row("Trail Density", 0.01, 0.5, params.trail_density, ErosionSliderFieldKind::TrailDensity),
            slider_row("Detail Scale", 0.5, 16.0, params.detail_scale, ErosionSliderFieldKind::DetailScale),
            section_label("Angles (°)"),
            slider_row("Wear Angle", 0.0, 90.0, params.wear_angle, ErosionSliderFieldKind::WearAngle),
            slider_row("Talus Angle", 0.0, 90.0, params.talus_angle, ErosionSliderFieldKind::TalusAngle),
            slider_row("Max Deposit Angle", 0.0, 90.0, params.max_deposit_angle, ErosionSliderFieldKind::MaxDepositAngle),
            section_label("Ridge Erosion"),
            (
                @FeathersCheckbox {
                    @caption: bsn! { (Text("Enable Ridge") ThemedText) },
                }
                Checked
                on(checkbox_self_update)
                on(|change: On<ValueChange<bool>>, mut params: ResMut<ErodeParams>| {
                    params.compute_ridge_erosion = if change.value { 1 } else { 0 };
                })
            ),
            slider_row("Ridge Steps", 1.0, 100.0, params.ridge_erosion_steps as f32, ErosionSliderFieldKind::RidgeErosionSteps),
            slider_row("Ridge Amount", 0.0, 2.0, params.ridge_erosion_amount, ErosionSliderFieldKind::RidgeErosionAmount),
            slider_row("Ridge Softening", 0.0, 5.0, params.ridge_softening_amount, ErosionSliderFieldKind::RidgeSofteningAmount),
            section_label("Flow"),
            slider_row("Flow Length", 16.0, 512.0, params.flow_length, ErosionSliderFieldKind::FlowLength),
            slider_row("Friction", 0.0, 1.0, params.friction, ErosionSliderFieldKind::Friction),
            slider_row("Rock Friction", 0.0, 1.0, params.rock_friction, ErosionSliderFieldKind::RockFriction),
            section_label("Sediment"),
            slider_row("Compaction", 0.0, 1.0, params.sediment_compaction, ErosionSliderFieldKind::SedimentCompaction),
            slider_row("Compaction Threshold", 0.0, 1.0, params.compaction_threshold, ErosionSliderFieldKind::CompactionThreshold),
            section_label("Effects"),
            slider_row("Channeling", 0.0, 1.0, params.channeling, ErosionSliderFieldKind::Channeling),
            slider_row("Sediment Removal", 0.0, 1.0, params.sediment_removal, ErosionSliderFieldKind::SedimentRemoval),
            slider_row("Uplift", 0.0, 0.01, params.uplift, ErosionSliderFieldKind::Uplift),
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
            (Text("Erosion Parameters") ThemedText),
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    column_gap: px(ROW_GAP),
                    align_items: AlignItems::Center,
                }
                ThemedText
                Children [
                    (
                        @FeathersButton {
                            @caption: bsn! { (Text("Reset") ThemedText) },
                            @variant: ButtonVariant::Primary,
                        }
                        on(|_: On<Activate>, mut reset: ResMut<ResetSim>| {
                            reset.generation = reset.generation.wrapping_add(1);
                        })
                    ),
                    (
                        @FeathersCheckbox {
                            @caption: bsn! { (Text("Pause") ThemedText) },
                        }
                        on(checkbox_self_update)
                        on(|change: On<ValueChange<bool>>, mut sim: ResMut<SimControl>| {
                            sim.paused = change.value;
                        })
                    ),
                    (
                        @FeathersButton {
                            @caption: bsn! { (Text("Step") ThemedText) },
                        }
                        on(|_: On<Activate>, mut sim: ResMut<SimControl>| {
                            if sim.paused {
                                sim.step_counter = sim.step_counter.wrapping_add(1);
                            }
                        })
                    ),
                ]
            ),
        ]
    }
}

fn section_label(text: &str) -> impl Scene {
    let text = Text::new(text.to_string());
    bsn! {
        (
            Node {
                display: Display::Flex,
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                margin: UiRect::top(px(4.0)),
                width: percent(100),
            }
            ThemedText
            template_value(text)
        )
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
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    min_width: px(110.0),
                }
                ThemedText
                template_value(Text::new(label))
            ),
            (
                Node {
                    flex_grow: 1.0,
                    min_width: px(60.0),
                }
                Children [(
                    @FeathersSlider {
                        @value: value,
                        @min: min,
                        @max: max,
                    }
                    SliderStep(0.01)
                    SliderPrecision(3)
                    ErosionSliderField
                    on(slider_self_update)
                    on(move |change: On<ValueChange<f32>>, mut params: ResMut<ErodeParams>, mut reset: ResMut<ResetSim>| {
                        field.apply(&mut params, change.value);
                        if field.resets_sim() {
                            reset.generation = reset.generation.wrapping_add(1);
                        }
                    })
                )]
            ),
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
