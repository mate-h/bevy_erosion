use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy::camera::CameraProjection;

pub struct SunPlugin;

impl Plugin for SunPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, update_sun_position);
    }
}

#[derive(Component)]
pub struct Sun;

#[derive(Component)]
pub struct SunController {
    current_direction: Vec3,
    target_direction: Vec3,
    damping_factor: f32,
}

impl SunController {
    pub fn new(direction: Vec3) -> Self {
        let dir = direction.normalize();
        Self {
            current_direction: dir,
            target_direction: dir,
            damping_factor: 0.15,
        }
    }
}

impl Default for SunController {
    fn default() -> Self {
        let default_dir = Vec3::new(1.0, 0.5, 1.0).normalize();
        Self {
            current_direction: default_dir,
            target_direction: default_dir,
            damping_factor: 0.15,
        }
    }
}

fn screen_to_world_ray(
    camera: &Camera,
    projection: &Projection,
    camera_transform: &GlobalTransform,
    screen_pos: Vec2,
) -> Option<Ray> {
    let viewport_rect = camera.logical_viewport_rect()?;
    
    let ndc_x = ((screen_pos.x - viewport_rect.min.x) / viewport_rect.width()) * 2.0 - 1.0;
    let ndc_y = 1.0 - ((screen_pos.y - viewport_rect.min.y) / viewport_rect.height()) * 2.0;
    
    let clip_from_view = match projection {
        Projection::Perspective(p) => p.get_clip_from_view(),
        Projection::Orthographic(o) => o.get_clip_from_view(),
        Projection::Custom(c) => c.get_clip_from_view(),
    };
    
    let view_from_world = camera_transform.affine().inverse();
    let clip_from_world = clip_from_view * view_from_world;
    let world_from_clip = clip_from_world.inverse();
    
    let near_point = world_from_clip * Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
    let far_point = world_from_clip * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
    
    let near_point = near_point.truncate() / near_point.w;
    let far_point = far_point.truncate() / far_point.w;
    
    let ray_origin = near_point;
    let mut ray_dir = (far_point - near_point).normalize();
    // invert the direction
    ray_dir = -ray_dir;

    
    Some(Ray {
        origin: ray_origin,
        direction: ray_dir,
    })
}

struct Ray {
    origin: Vec3,
    direction: Vec3,
}

fn ray_sphere_intersection(ray: &Ray, sphere_radius: f32) -> Option<Vec3> {
    let a = ray.direction.length_squared();
    let b = 2.0 * ray.origin.dot(ray.direction);
    let c = ray.origin.length_squared() - sphere_radius * sphere_radius;
    let discriminant = b * b - 4.0 * a * c;
    
    if discriminant < 0.0 {
        return None;
    }
    
    let sqrt_discriminant = discriminant.sqrt();
    let t1 = (-b - sqrt_discriminant) / (2.0 * a);
    let t2 = (-b + sqrt_discriminant) / (2.0 * a);
    let t = t1.max(t2);
    
    if t < 0.0 {
        return None;
    }
    
    Some(ray.origin + ray.direction * t)
}

fn update_sun_position(
    mut sun_query: Query<(&mut Transform, &mut SunController), (With<DirectionalLight>, With<Sun>)>,
    camera_query: Query<(&Camera, &Projection, &GlobalTransform), (With<Camera3d>, Without<DirectionalLight>)>,
    windows: Query<&Window, With<PrimaryWindow>>,
    buttons: Res<ButtonInput<MouseButton>>,
) {
    let target_direction = if buttons.pressed(MouseButton::Right) {
        let window = match windows.single() {
            Ok(w) => w,
            Err(_) => return,
        };
        
        let mouse_pos = match window.cursor_position() {
            Some(pos) => pos,
            None => return,
        };
        
        let (camera, projection, camera_transform) = match camera_query.single() {
            Ok(c) => c,
            Err(_) => return,
        };
        
        let ray = match screen_to_world_ray(camera, projection, camera_transform, mouse_pos) {
            Some(r) => r,
            None => return,
        };
        
        let sphere_radius = 10000.0;
        let intersection = match ray_sphere_intersection(&ray, sphere_radius) {
            Some(p) => p,
            None => ray.direction.normalize() * sphere_radius,
        };
        
        Some(intersection.normalize())
    } else {
        None
    };
    
    for (mut transform, mut controller) in sun_query.iter_mut() {
        if let Some(target_dir) = target_direction {
            controller.target_direction = target_dir;
        }
        
        controller.current_direction = controller.current_direction
            + (controller.target_direction - controller.current_direction) * controller.damping_factor;
        controller.current_direction = controller.current_direction.normalize();
        
        let sphere_radius = 10000.0;
        let sun_position = (-controller.current_direction) * sphere_radius;
        *transform = Transform::from_translation(sun_position).looking_at(Vec3::ZERO, Vec3::Y);
    }
}

