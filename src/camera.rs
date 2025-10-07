use bevy::{
    prelude::*,
    input::mouse::{MouseMotion, MouseWheel, MouseScrollUnit},
};

pub struct OrbitCameraPlugin;

impl Plugin for OrbitCameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (orbit_mouse_rotate, orbit_mouse_zoom, apply_orbit_transform));
    }
}

#[derive(Component)]
pub struct OrbitController {
    pub target: Vec3,
    pub distance: f32,
    pub min_distance: f32,
    pub max_distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub rotate_sensitivity: Vec2,
    pub zoom_sensitivity: f32,
    pub rotate_button: MouseButton,
}

impl Default for OrbitController {
    fn default() -> Self {
        Self {
            target: Vec3::new(0.0, 1.0, 0.0),
            distance: 200.0,
            min_distance: 30.0,
            max_distance: 2000.0,
            yaw: std::f32::consts::FRAC_PI_4,      // 45°
            pitch: std::f32::consts::FRAC_PI_6,    // 30°
            rotate_sensitivity: Vec2::new(0.01, 0.01),
            zoom_sensitivity: 2.0, // units per line
            rotate_button: MouseButton::Left,
        }
    }
}

fn orbit_mouse_rotate(
    mut query: Query<&mut OrbitController, With<Camera3d>>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion_events: MessageReader<MouseMotion>,
) {
    let mut delta = Vec2::ZERO;
    for ev in motion_events.read() {
        delta += ev.delta;
    }
    if delta == Vec2::ZERO {
        return;
    }
    for mut ctrl in &mut query {
        if !buttons.pressed(ctrl.rotate_button) {
            continue;
        }
        ctrl.yaw -= delta.x * ctrl.rotate_sensitivity.x;
        ctrl.pitch += delta.y * ctrl.rotate_sensitivity.y;
        let max_pitch = std::f32::consts::FRAC_PI_2 - 0.01;
        let min_pitch = -max_pitch;
        ctrl.pitch = ctrl.pitch.clamp(min_pitch, max_pitch);
    }
}

fn orbit_mouse_zoom(
    mut query: Query<&mut OrbitController, With<Camera3d>>,
    mut wheel_events: MessageReader<MouseWheel>,
) {
    if wheel_events.is_empty() {
        return;
    }
    let mut total_y: f32 = 0.0;
    for ev in wheel_events.read() {
        match ev.unit {
            MouseScrollUnit::Line => {
                total_y += ev.y;
            }
            MouseScrollUnit::Pixel => {
                total_y += ev.y * 0.1; // scale pixel wheels down
            }
        }
    }
    if total_y == 0.0 { return; }
    for mut ctrl in &mut query {
        ctrl.distance = (ctrl.distance - total_y * ctrl.zoom_sensitivity)
            .clamp(ctrl.min_distance, ctrl.max_distance);
    }
}

fn apply_orbit_transform(
    mut query: Query<(&mut Transform, &OrbitController), With<Camera3d>>,
) {
    for (mut transform, ctrl) in &mut query {
        // Spherical coordinates around target
        let x = ctrl.distance * ctrl.pitch.cos() * ctrl.yaw.sin();
        let y = ctrl.distance * ctrl.pitch.sin();
        let z = ctrl.distance * ctrl.pitch.cos() * ctrl.yaw.cos();
        let eye = ctrl.target + Vec3::new(x, y, z);
        *transform = Transform::from_translation(eye).looking_at(ctrl.target, Vec3::Y);
    }
}