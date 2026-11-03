//! Boulders as bodies (Phase 8 of `docs/UNIFIED_PLAN.md`): spheres a metre to tens of metres
//! across that sit on the heightfield and move when the flow beats their friction.
//!
//! The GPU owns them (`assets/boulders.wgsl`): every frame each one feels the water's drag and
//! lift, buoyancy and gravity down the slope, and moves or stays put by Coulomb friction. Each adds
//! a bump of its own volume to the shallow-water solver's bed (not to the terrain), so the water
//! flows around it and scours beside it.
//!
//! The CPU only places and removes them ([`BoulderEdits`], written into free slots of the GPU
//! buffer) and reads them back every frame ([`BoulderList`]) to draw a rock mesh per boulder.

use crate::{
    ErosionConfig, WaterControl, WaterFields,
    brush::{BrushHit, BrushMode, TerrainSurface, WaterBrush},
};
use bevy::{
    asset::RenderAssetUsages,
    input::mouse::MouseButton,
    mesh::{Indices, VertexAttributeValues},
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        gpu_readback::{Readback, ReadbackComplete},
        RenderApp,
        render_resource::ShaderType,
    },
};
use bytemuck::{Pod, Zeroable};

/// Slots in the GPU boulder buffer.
pub const MAX_BOULDERS: usize = 4096;
/// Boulders dropped per second per texel² of brush at strength 1.
const DROPS_PER_SECOND: f32 = 0.05;
/// Rock meshes to pick from.
const ROCK_SHAPES: usize = 4;

/// One boulder, as `Boulder` in `boulders.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable, ShaderType)]
pub struct Boulder {
    /// Centre in map texels.
    pub pos: Vec2,
    /// Velocity (m/s).
    pub vel: Vec2,
    /// Radius (m).
    pub radius: f32,
    /// Ground under the centre (height units), from the GPU.
    pub ground: f32,
    /// Distance rolled (m), from the GPU.
    pub rolled: f32,
    /// 0 = free slot.
    pub alive: u32,
}

impl Boulder {
    pub fn at(pos: Vec2, diameter: f32) -> Self {
        Self {
            pos,
            radius: 0.5 * diameter,
            alive: 1,
            ..default()
        }
    }
}

/// How boulders move. In SI; part of [`WaterControl`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoulderPhysics {
    /// Diameter of the boulders the brush drops (m); each varies by ±30 %.
    pub diameter: f32,
    /// Rock density over water's (granite 2.65).
    pub density: f32,
    /// Drag coefficient on the frontal area in the flow (0.4–1 for boulders on a bed).
    pub drag: f32,
    /// Lift coefficient (about 0.2; Schmeeckle et al. 2007 measure 0.1–0.4).
    pub lift: f32,
    /// Coulomb friction at rest: the tangent of the pivot angle a boulder must be tipped over to
    /// move (about 0.6–1.2 for boulders nested in a bed).
    pub friction_static: f32,
    /// Coulomb friction while moving: rolling and sliding mixed.
    pub friction_kinetic: f32,
}

impl Default for BoulderPhysics {
    fn default() -> Self {
        Self {
            diameter: 4.0,
            density: 2.65,
            drag: 0.5,
            lift: 0.2,
            friction_static: 0.8,
            friction_kinetic: 0.5,
        }
    }
}

/// Changes to the GPU boulder buffer, applied once each time [`Self::generation`] changes.
#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct BoulderEdits {
    /// Slots to overwrite.
    pub writes: Vec<(u32, Boulder)>,
    /// Zero every slot first.
    pub clear: bool,
    pub generation: u64,
    /// Slots in use: the GPU passes cover `0..slots`.
    pub slots: u32,
    /// Next slot to recycle once all are in use (the oldest placed).
    recycle: u32,
    /// Writes (and the clear) that a frame's extraction has seen, dropped on the next frame.
    sent: Option<(usize, bool)>,
}

impl BoulderEdits {
    /// Place `boulder` in a free slot (or the oldest one when all are taken). `list` is the last
    /// readback, so slots the GPU freed (boulders gone off the map) are reused.
    pub fn place(&mut self, boulder: Boulder, list: &BoulderList) {
        let taken = |i: u32| {
            self.writes.iter().any(|(s, b)| *s == i && b.alive != 0)
                || list.boulders.get(i as usize).is_some_and(|b| b.alive != 0)
        };
        let slot = (0..self.slots)
            .find(|&i| !taken(i))
            .or_else(|| ((self.slots as usize) < MAX_BOULDERS).then(|| {
                self.slots += 1;
                self.slots - 1
            }))
            .unwrap_or_else(|| {
                self.recycle = (self.recycle + 1) % MAX_BOULDERS as u32;
                self.recycle
            });
        self.writes.push((slot, boulder));
        self.generation += 1;
    }

    /// Remove the boulders within `radius` texels of `center`.
    pub fn remove_near(&mut self, center: Vec2, radius: f32, list: &BoulderList) {
        for (i, b) in list.boulders.iter().enumerate() {
            if b.alive != 0 && b.pos.distance(center) <= radius {
                self.writes.push((i as u32, Boulder::default()));
                self.generation += 1;
            }
        }
    }

    pub fn clear_all(&mut self) {
        self.writes.clear();
        self.clear = true;
        self.slots = 0;
        self.recycle = 0;
        self.generation += 1;
    }
}

/// The boulders as last read back from the GPU (a frame or two old).
#[derive(Resource, Default)]
pub struct BoulderList {
    pub boulders: Vec<Boulder>,
}

impl BoulderList {
    pub fn alive(&self) -> impl Iterator<Item = (usize, &Boulder)> {
        self.boulders.iter().enumerate().filter(|(_, b)| b.alive != 0)
    }
}

pub struct BoulderPlugin;

impl Plugin for BoulderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BoulderEdits>()
            .init_resource::<BoulderList>()
            .add_plugins(ExtractResourcePlugin::<BoulderEdits>::default())
            .add_systems(PreUpdate, forget_sent_edits)
            .add_systems(Last, mark_sent_edits)
            .add_systems(Startup, setup_rocks)
            .add_systems(
                Update,
                (paint_boulders, sync_readback, draw_boulders).chain(),
            );
    }
}

/// The render world extracts the edits after `Last`.
fn mark_sent_edits(mut edits: ResMut<BoulderEdits>) {
    let edits = edits.bypass_change_detection();
    edits.sent = Some((edits.writes.len(), edits.clear));
}

/// Drop the edits the last frame's extraction took; any made since stay for this frame.
fn forget_sent_edits(mut edits: ResMut<BoulderEdits>) {
    let edits = edits.bypass_change_detection();
    if let Some((writes, clear)) = edits.sent.take() {
        edits.writes.drain(..writes.min(edits.writes.len()));
        edits.clear &= !clear;
    }
}

#[derive(Component)]
struct BoulderReadback;

/// Read the boulder slots back every frame while any are in use.
fn sync_readback(
    mut commands: Commands,
    edits: Res<BoulderEdits>,
    fields: Option<Res<WaterFields>>,
    mut list: ResMut<BoulderList>,
    existing: Query<Entity, With<BoulderReadback>>,
) {
    let Some(fields) = fields else { return };
    let want = edits.slots > 0;
    if want && existing.is_empty() {
        commands
            .spawn((Readback::buffer(fields.boulders.clone()), BoulderReadback))
            .observe(
                |ev: On<ReadbackComplete>, edits: Res<BoulderEdits>, mut list: ResMut<BoulderList>| {
                    let all: Vec<Boulder> = bytemuck::pod_collect_to_vec(&ev.data);
                    list.boulders = all;
                    list.boulders.truncate(edits.slots as usize);
                },
            );
    } else if !want {
        for entity in &existing {
            commands.entity(entity).despawn();
        }
        list.boulders.clear();
    }
}

/// Drag to drop boulders under the brush; Shift+drag removes them.
#[allow(clippy::too_many_arguments)]
fn paint_boulders(
    brush: Option<Res<WaterBrush>>,
    hit: Option<Res<BrushHit>>,
    buttons: Option<Res<ButtonInput<MouseButton>>>,
    keys: Option<Res<ButtonInput<KeyCode>>>,
    time: Res<Time>,
    water: Res<WaterControl>,
    list: Res<BoulderList>,
    mut edits: ResMut<BoulderEdits>,
    mut owed: Local<f32>,
    mut seed: Local<u32>,
) {
    // (Without the brush, as in the tests, boulders are placed through [`BoulderEdits`].)
    let (Some(brush), Some(hit), Some(buttons), Some(keys)) = (brush, hit, buttons, keys) else {
        return;
    };
    let painting = brush.enabled
        && brush.mode == BrushMode::Boulders
        && hit.world.is_some()
        && buttons.pressed(MouseButton::Left);
    if !painting {
        *owed = 0.0;
        return;
    }
    let erase = keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]);
    if erase {
        edits.remove_near(hit.texel, brush.radius, &list);
        return;
    }
    let area = std::f32::consts::PI * brush.radius * brush.radius;
    *owed += brush.strength * DROPS_PER_SECOND * area * time.delta_secs();
    if buttons.just_pressed(MouseButton::Left) {
        *owed = owed.max(1.0);
    }
    while *owed >= 1.0 {
        *owed -= 1.0;
        let r = |s: &mut u32| {
            *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (*s >> 8) as f32 / (1u32 << 24) as f32
        };
        let (a, d, size) = (r(&mut seed), r(&mut seed), r(&mut seed));
        let offset = Vec2::from_angle(a * std::f32::consts::TAU) * d.sqrt() * brush.radius;
        let diameter = water.boulders.diameter * (0.7 + 0.6 * size);
        edits.place(Boulder::at(hit.texel + offset, diameter), &list);
    }
}

#[derive(Resource)]
struct RockAssets {
    meshes: [Handle<Mesh>; ROCK_SHAPES],
    material: Handle<StandardMaterial>,
}

/// A few lumpy icospheres to draw boulders with.
fn setup_rocks(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let shapes = std::array::from_fn(|k| meshes.add(rock_mesh(k as u32 + 1)));
    commands.insert_resource(RockAssets {
        meshes: shapes,
        material: materials.add(StandardMaterial {
            base_color: Color::srgb(0.52, 0.5, 0.47),
            perceptual_roughness: 0.95,
            ..default()
        }),
    });
}

/// A unit icosphere pushed in and out by a few random lobes, with flat-ish facets.
fn rock_mesh(seed: u32) -> Mesh {
    let mut mesh = Sphere::new(1.0).mesh().ico(2).unwrap();
    let mut s = seed.wrapping_mul(2654435761);
    let mut rand = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        (s >> 8) as f32 / (1u32 << 24) as f32
    };
    let lobes: Vec<(Vec3, f32)> = (0..6)
        .map(|_| {
            let dir = Vec3::new(rand() - 0.5, rand() - 0.5, rand() - 0.5).normalize_or(Vec3::Y);
            (dir, 0.25 * (rand() - 0.4))
        })
        .collect();
    if let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION)
    {
        for p in positions.iter_mut() {
            let v = Vec3::from_array(*p);
            let bump: f32 = lobes
                .iter()
                .map(|(d, a)| a * v.dot(*d).max(0.0).powi(3))
                .sum();
            *p = (v * (1.0 + bump)).to_array();
        }
    }
    // Facets: unshare the vertices so each triangle gets its own normal.
    if matches!(mesh.indices(), Some(Indices::U32(_) | Indices::U16(_))) {
        mesh.duplicate_vertices();
    }
    mesh.compute_flat_normals();
    mesh.asset_usage = RenderAssetUsages::RENDER_WORLD;
    mesh
}

#[derive(Component)]
struct BoulderMesh {
    slot: usize,
    rolled: f32,
}

/// One rock per live boulder, sitting on the terrain and rolling as it goes.
fn draw_boulders(
    mut commands: Commands,
    list: Res<BoulderList>,
    rocks: Option<Res<RockAssets>>,
    config: Res<ErosionConfig>,
    terrain: Query<(&TerrainSurface, &GlobalTransform)>,
    mut drawn: Query<(Entity, &mut BoulderMesh, &mut Transform)>,
) {
    let Some(rocks) = rocks else { return };
    let Ok((surface, terrain_tf)) = terrain.single() else {
        return;
    };
    let map = config.map_size.as_vec2();
    let to_world = terrain_tf.affine();
    let mut seen = vec![false; list.boulders.len()];
    for (entity, mut mesh, mut tf) in &mut drawn {
        let Some(b) = list.boulders.get(mesh.slot).filter(|b| b.alive != 0) else {
            commands.entity(entity).despawn();
            continue;
        };
        seen[mesh.slot] = true;
        let rolled = b.rolled - mesh.rolled;
        mesh.rolled = b.rolled;
        let dir = Vec3::new(b.vel.x, 0.0, b.vel.y);
        if rolled > 0.0 && dir.length_squared() > 0.0 {
            let axis = Vec3::Y.cross(dir).normalize();
            tf.rotation = Quat::from_axis_angle(axis, rolled / b.radius.max(0.01)) * tf.rotation;
        }
        tf.translation = to_world.transform_point3(local_centre(b, surface, map));
        tf.scale = Vec3::new(1.0, 0.75, 1.0) * b.radius;
    }
    for (slot, b) in list.alive() {
        if seen.get(slot).copied().unwrap_or(false) {
            continue;
        }
        let spin = (slot as f32 * 2.399).sin_cos();
        commands.spawn((
            BoulderMesh {
                slot,
                rolled: b.rolled,
            },
            Mesh3d(rocks.meshes[slot % ROCK_SHAPES].clone()),
            MeshMaterial3d(rocks.material.clone()),
            Transform {
                translation: to_world.transform_point3(local_centre(b, surface, map)),
                rotation: Quat::from_rotation_y(spin.0 * std::f32::consts::PI),
                scale: Vec3::new(1.0, 0.75, 1.0) * b.radius,
            },
        ));
    }
}

/// The boulder's centre in the terrain's local frame, sunk a quarter of its radius into the bed.
fn local_centre(b: &Boulder, surface: &TerrainSurface, map: Vec2) -> Vec3 {
    let xz = (b.pos / map - 0.5) * surface.size;
    Vec3::new(
        xz.x,
        b.ground * surface.height_scale + 0.75 * 0.75 * b.radius,
        xz.y,
    )
}
