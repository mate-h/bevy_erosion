# One landscape in metres

Plan for turning the KTT cascade, the shallow-water solver, bed load, Shape Beds, Refine Beds and
the river brush into one system with physical units. It follows `WATER_PLAN.md` (Phases 1–6,
done) and keeps its validation (`VALIDATION.md`) as the acceptance bar.

## Goals

- **Metres and seconds everywhere the user looks.** Map size, relief, feature size, river width,
  brush radius, rainfall and discharge are physical quantities. Changing the resolution keeps the
  landscape; changing the scale changes the physics.
- **Mountains and rivers at the same scale.** A river's size follows from the catchment that
  feeds it, so a 5 km map does not grow a river that needs 1500 km² of drainage.
- **One channel model.** Shape Beds, the river brush and the springs size channels from the same
  discharge, through the same hydraulic geometry and the same carving code.
- **Honest time scales.** Hydraulic seconds, morphological acceleration and geological iterations
  are named and kept apart, instead of hidden in gravity and fitted constants.

Non-goals:
- Calibrating KTT to geological years. It stays a shape generator; its knobs stay dimensionless.
- Changing what the water looks like. Roll waves and step-pool stair-stepping stay (see
  [Acceptance](#acceptance)).

## Where things stand

### The solvers

| Piece | Where | What it does | Time |
|---|---|---|---|
| KTT cascade | `erosion.wgsl`, `pyramid_*.wgsl` | Particle trails, coarse to fine over `log2(Feature_Size · Detail_Scale) + 1` levels; `writeBack` (uplift, compaction, channeling) every 20 iterations | Iterations |
| Shallow water | bevy_fft `ShallowWaterSolver`, `water.wgsl` `swe_export` / `swe_gather` | Staggered SWE with Manning friction; pipes as fallback | `WATER_STEP` = 1/480 s, g = 2000 cells/s² |
| Capacity erosion | `water.wgsl` `water_erode` | Mei et al. dissolve and deposit, slumping; only thin sheet flow while bed load runs | Per water step |
| Bed load | `water_morpho.wgsl` | Shields, `q = K (θ − θc)^1.5`, helical deflection, Talmon slope effect, upwind Exner | Per water step × MorFac |
| Shape Beds (G) | `water_bed.wgsl` | One shot: bank distance τ (SOR), prior, smoothing, channels from the averaged discharge (depth ∝ Q^0.4, half-width ∝ Q^0.5), lake bowls, bank grading | Instant |
| Refine Beds (M) | `plan_erosion_frame` | 400 water steps at MorFac 20, 64 per frame | Accelerated |
| River brush | `river.rs` (CPU), `river.wgsl` | Priority-flood route on read-back heights, breached long profile, raster with Shape Beds' geometry plus step-pools, riffle-pools and point bars; `river_carve` lowers the terrain, `river_armour` clamps later change to a band; a spring feeds half the bankfull flow | Instant, then armoured |

### The couplings

1. **Shared heights.** All of them write the finest level's `u32` fixed-point height. Water only
   runs once the cascade reaches the finest level.
2. **Trails into water.** Wet cells are KTT outflow (`masks.y`), so trails end in water and hand
   it their sediment (`water_splat`). Rain-brush trails add water (`rain_water`).
3. **Water into KTT.** `water_record` writes the water's height change into KTT's deposition and
   erosion buffers.
4. **Discharge and momentum.** `morpho_discharge` averages unit discharge over 3 s;
   `morpho_steer` raises KTT's river momentum to it through `KTT_DISCHARGE_SCALE = 0.0044`, fitted
   on one spring-fed valley (`discharge_matches_ktt`).
5. **Shared channel sizing.** Shape Beds and the river brush both read `BedShaping`
   (`river_depth`, `river_width` at `REFERENCE_DISCHARGE = 1`).

Four mechanisms make channels: KTT trails, Shape Beds, the river brush and bed load (with Refine
as its fast-forward). Only bed load is process-based, and the hand-offs between them are fitted.

### Units today

There is one consistent convention already: **1 height unit = 100 texels** (`CELLS_PER_HEIGHT`).
The mesh uses 0.5 world units per texel and `height_scale = 50`, so it draws without vertical
exaggeration, and KTT's angles go through the same constant. Everything else is per texel or per
height unit:

| Quantity | Today | Unit |
|---|---|---|
| Feature Size | 64 | texels |
| Noise Freq | 0.005 | per texel |
| Flow Length, hydro search distance, outflow border | 256, 16, 0 | texels |
| Bedrock Depth | 0.05 | height units |
| Gravity | 2000 | cells/s² |
| Manning n | 0.01 | cell units |
| Grain (R·D) | 0.25 | cells |
| Bed load K | 4 | cells²/s |
| River depth, half-width | 0.008, 5 | height units, texels |
| Bank width, bank slope, shore slope | 8, 0.0015, 0.0015 | texels, height units per texel |
| Rainfall, spring rate | — | height units per second |
| Evaporation | 1 % | of the depth per second |
| Water thresholds (`WET_DEPTH`, `FULL_DEPTH`, `DRY_SINK`, `MAX_CUT`, `WET_TALUS`) | 0.0005–0.02 | height units |
| `MIN_SPEED` | 0.05 | cells per second |
| Brush radius | 2–128 | texels |

## The scale problem, in numbers

Choose a texel size dx. Then 1 height unit is 100·dx metres, and the default river (half-width
5 texels, depth 0.8 cells) is:

| dx | Map (512²) | Default river W × H | Q for that width (W ≈ 4.4 Q^0.5) | Catchment that gives that Q | Whole map |
|---|---|---|---|---|---|
| 1 m | 0.5 km | 10 m × 0.8 m | ≈ 5 m³/s | ≈ 4 km² | 0.26 km² |
| 2 m | 1 km | 20 m × 1.6 m | ≈ 21 m³/s | ≈ 25 km² | 1.05 km² |
| 10 m | 5 km | 100 m × 8 m | ≈ 520 m³/s | ≈ 1500 km² | 26 km² |

(Catchment from a bankfull relation Q ≈ 1.5 A^0.8, A in km²; see [Reference
values](#reference-values) for how rough that is.)

At every scale the default river is 15–60× the area of the map. That is the mismatch between
mountains and rivers. Turned around, a river fed only by this map is about 2–3 texels wide at any
dx: the map's catchment grows as dx², and bankfull width as A^0.4.

So:
- Rivers born on the map are **narrow on the grid**, and need a sub-grid rule (see [Narrow
  rivers](#narrow-rivers)).
- Big rivers **come from off the map**, through an inlet with an upstream catchment.

### What the water is today, in SI

With g = 2000 cells/s², the solver is Froude-similar to real water at a time stretch of
τ = √(2000·dx / 9.81) real seconds per simulated second. Holding u = h^(2/3) S^(1/2) / n under
that stretch gives the equivalent SI roughness n_SI = n_cells · τ · dx^(−1/3). For the grain,
R·D = 0.25 cells, so D = 0.25·dx / 1.65.

| dx | τ (time stretch) | Manning n equivalent | Grain D equivalent |
|---|---|---|---|
| 1 m | 14 × | 0.14 | 0.15 m |
| 2 m | 20 × | 0.16 | 0.30 m |
| 10 m | 45 × | 0.21 | 1.5 m |

The thresholds scale the same way: `WET_DEPTH` = 0.0005 height units is 0.5 m of water at
dx = 10 m, so anything shallower counts as dry, and `FULL_DEPTH` (0.004) is 4 m.

The water is fast-forwarded and very rough (natural channels have n = 0.03–0.07), with cobbles to
boulders on the bed. Every Froude-based result in `VALIDATION.md` holds as is; what changes in SI
is which flows reach those Froude numbers.

## Units

### `WorldScale`

A new resource, the single source of truth:

```rust
pub struct WorldScale {
    /// Texel size in metres (dx).
    pub metres_per_texel: f32,
}
impl WorldScale {
    pub fn metres_per_height(&self) -> f32 { CELLS_PER_HEIGHT * self.metres_per_texel }
    pub fn map_metres(&self, size: UVec2) -> Vec2 { size.as_vec2() * self.metres_per_texel }
}
```

- **Internal state stays as it is.** Heights stay in height units (1 = 100 texels, isotropic);
  the shaders stay in texels. `WorldScale` converts at the boundary: UI, uniforms, rendering.
  This keeps the change small and the shaders' slopes dimensionless.
- **Precision.** `HEIGHT_SCALE = 1e6` per height unit is 100·dx µm: 1 mm at dx = 10 m. The `u32`
  range is 4294 height units, 429 km at dx = 1. Both are fine.
- **Settings hold metres.** The UI value is the physical one; the texel value is derived on
  upload. Changing dx with metres held fixed changes the texel values and reruns the cascade.
- **Default dx: 10 m.** A 5 km map with about 1 km of relief is mountain terrain, where the
  default eroded slopes (5–25 %) and step-pool rivers (`VALIDATION.md`, river brush) belong.

### Settings in metres

| Setting | New unit | Conversion |
|---|---|---|
| Map | km (read-only, N·dx) | — |
| Relief (was Noise Scale) | m | / metres_per_height |
| Noise wavelength (was Noise Freq) | m | frequency = dx / wavelength |
| Feature Size | m | / dx |
| Flow Length, hydro search, outflow border | m | / dx |
| Bedrock Depth | m | / metres_per_height |
| Sea Level | m | / metres_per_height |
| Brush radius | m | / dx |
| Bank width | m | / dx |
| Bank, shore slope | m/m (or %) | / 100 (height units per texel) |
| Breach depth, max depth | m | / metres_per_height |
| Rainfall | mm/h | to metres per simulated second |
| Evaporation | mm/day (depth, not a fraction) | to metres per simulated second |
| Spring, river discharge | m³/s | / (metres_per_height · dx²) |
| Angles (talus, wear, deposit, cutoff) | degrees (already) | unchanged |
| KTT knobs (strength, softness, channeling, river terms, Detail Scale) | dimensionless (already) | unchanged |

The shader constants in height units (`WET_DEPTH`, `FULL_DEPTH`, `DRY_SINK`, `MAX_CUT`,
`WET_TALUS`, `STOP_DEPTH`) become metres in `lib.rs` and reach the shaders through a uniform, so
a thin film is 1 mm whatever dx is.

### Rendering in metres

- Mesh: N·dx metres across (now `.size(half_size, half_size)`, 0.5 per texel), and
  `height_scale = metres_per_height`.
- Camera distances (`min_distance`, `max_distance`), the brush gizmo offset (+0.3), the sun's
  shadow cascades and bevy_fft's `WaterHeightfield` scale with them.
- Bevy's atmosphere and the water optics (refraction depth, foam, absorption) are in metres. The
  terrain is now drawn 256 units wide, a model 256 m across; drawing it at 5 km should give the
  aerial perspective and water colour of a real valley without retuning. Check whether the demo
  sets the atmosphere's scene-units-to-metres factor, and set it to 1.

## Time scales

| Scale | Solver | Unit | User control |
|---|---|---|---|
| Hydraulic | SWE, capacity erosion | seconds | **Time acceleration**: simulated seconds per real second |
| Morphological | bed load, Exner | bed-seconds = water seconds × MorFac | Live: MorFac 1. Refine: "run N hours of bankfull flow" |
| Geological | KTT cascade | iterations | Geological Age, Erosion Amount (as now) |

- **Gravity becomes physical.** g = 9.81 m/s². The Gravity slider is replaced by Time
  Acceleration.
- **The step follows CFL.** dt = C · dx / (|u| + √(g h)) with C ≈ 0.4, from the deepest water
  expected (bounded by Max Depth) or a reduction pass. At dx = 10 m and 5 m of water, dt ≈ 0.5 s,
  against 1/480 s now; 8 steps a frame is then about 4 s of water per frame, 240× real time.
- **The depth limiter** (flux carried up to about 2 / (g·dt)) is restated in metres and checked
  against the new dt.
- **Refine (M) counts bankfull time.** "Run 6 h of floods" sets steps × dt × MorFac. MorFac stays
  the accelerator; the user picks the time, the app picks MorFac within its stable range.

## The channel model

### Drainage area

KTT does not provide drainage area: `flow_buffer` accumulates deposition per trail step, and the
river momentum is trail water capped by Flow Length. A dedicated pass computes it:

- **Where:** on the CPU, after the cascade, on the height readback `river.rs` already makes in
  `refresh_after_cascade`. A 512² priority flood plus accumulation runs in milliseconds; a GPU
  version can come later.
- **How:** priority-flood depression filling (as in `river::route`), then multiple-flow-direction
  accumulation (D∞ or MFD, which avoids the grid-aligned rivers D8 makes) in metres².
- **Inlets:** each map edge can carry inlets with an upstream catchment in km². Their area is
  added where they enter.
- **Output:** `DrainageArea` (one f32 per texel, m²), uploaded for the debug views ("Drainage
  area", log scale).

### From area to channel

| Step | Relation | Default | Setting |
|---|---|---|---|
| Channel heads | A > A_c | 0.05 km² | Channel Head Area |
| Bankfull discharge | Q_bf = k · A^0.8 (A in km²) | k = 1.5 m³/s | Climate (Runoff factor) |
| Width | W = a_W · Q^0.5 | a_W = 4.4 | River Width factor |
| Depth | H = a_H · Q^0.4 | a_H = 0.3 | River Depth factor |
| Bed detail | step-pools above 3 % slope, riffle-pools above 1 %, point bars in bends | as now (`river.rs`) | Bed Detail |

The width and depth factors replace `river_width` and `river_depth`, which today give W/H ≈ 12,
narrower and deeper than the a_W / a_H default (W/H ≈ 15–27 over 1–500 m³/s).

### Narrow rivers

A channel narrower than about 3 texels does not carry its flow in the solver. Below W_min = 3 dx:

- widen it to W_min and reduce the depth so the Manning conveyance W·H^(5/3) is unchanged, so
  the velocity and Froude number stay about right;
- below a second threshold (W < 1 dx, the smallest streams) carve nothing and leave the valley
  KTT made.

This exaggerates the width of small streams on coarse maps; the debug view shows where.

### One carving path

| Today | Unified |
|---|---|
| G: channels from the averaged discharge of the current water, lakes as bowls | G: carve the network (A > A_c) with the model above; lakes stay as now |
| River brush: route from the click, Discharge slider | River brush: click picks the network branch (or an edge inlet); Discharge defaults to Q_bf(A) at the click and can be overridden |
| Springs: radius and rate by hand | River springs: Q_bf × Spring Share (0.5, as now); hand-placed springs in m³/s |
| `BedShaping` river terms, `REFERENCE_DISCHARGE` | `ChannelModel` (k, a_W, a_H, A_c, W_min) |

Both G and the brush build `river::River`s (path + long profile) and rasterize them through
`river::rasterize`. Shape Beds' SOR stays for lakes and bank grading. The armour band stays.

### KTT steering

`morpho_steer` keeps raising KTT's river momentum to the water's discharge. The fitted
`KTT_DISCHARGE_SCALE` is restated in SI (m²/s of unit discharge per unit of momentum) and refit
with `discharge_matches_ktt` at two dx values, to check that it does not depend on dx.

## Bed load in SI

| Term | Today (cells) | SI |
|---|---|---|
| Shields number | θ = n² u² / (grain · h^(1/3)), grain = R·D in cells | θ = n² u² / (R D h^(1/3)), D in m, R = 1.65 |
| Transport | q = K (θ − θc)^1.5, K = 4 cells²/s | q = 3.97 (θ − 0.0495)^1.5 · √(R g D³) (Wong & Parker 2006) |
| Exner | ∂η/∂t = −MorFac · ∇·q | ∂η/∂t = −MorFac · ∇·q / (1 − p), p = 0.4 |
| Rock | θc × (1 + hardness · (1 − softness)) | unchanged |
| Helical, Talmon | A_h, A_s | unchanged (dimensionless) |
| Adaptation length | `HELICAL_LENGTH` = 20 texels | metres (terracraft: 42 m) |

- **Grain** becomes **Bed Material** in mm: sand 0.5, gravel 20, cobbles 100. KTT's softness mask
  can scale it per cell.
- **Armour stays a band**, as `river_armour` does now. A coarser armour grain is a possible
  addition, not a replacement: the band is what keeps the step-pools.
- K = 4 cells²/s today stands in for the whole 3.97·√(RgD³)/(1 − p). In SI it follows from D, so
  bed load rates change with dx at a fixed D. That is the intended physics, and the reason for
  re-validating.

## The frame, unified

```
cascade (KTT, iterations)          ── Geological
   │ finished
   ▼
drainage area (CPU, readback)      ── once per cascade
   │
   ▼
channel network → rivers → carve   ── G, brush; reruns after every cascade (as rivers do now)
   │
   ▼
water (SWE, dt by CFL) ×N          ── Hydraulic, Time Acceleration
   ├─ capacity erosion (sheets)
   ├─ bed load + Exner (× MorFac)  ── Morphological, Refine = run N hours
   ├─ armour band
   └─ discharge → KTT momentum (steer)
   ▼
display (metres)
```

Keys keep their meaning: G carves channels, M runs floods, R reruns the cascade (and the
drainage and carving after it).

## Phases

### Phase 1: `WorldScale`, metric UI and rendering (no change to the simulation)

- `WorldScale` resource, default dx = 10 m; UI field for it.
- Every setting in [Settings in metres](#settings-in-metres) shown and entered in metres,
  converted on upload. Defaults chosen so the uploaded values equal today's.
- Mesh, height scale, camera, gizmos and water heightfield in metres; atmosphere scale checked.
- **Done when:** at dx = 10 m the demo produces the same heights as before (bit-identical
  readback), the UI reads in metres, and the scene renders at real scale.

### Phase 2: water in SI

- bevy_fft runs with `cell_size = dx` and heights in metres (`swe_export` / `swe_gather` scale by
  `metres_per_height`; the pipes solver gets the same factor or is retired, see open questions).
- g = 9.81, Manning n in SI, dt by CFL, Time Acceleration slider.
- Rainfall in mm/h, evaporation in mm/day, springs in m³/s; height-unit thresholds in metres.
- First default: n chosen to match today (0.21 at 10 m) so the look is unchanged. Then sweep
  n = 0.035–0.07 with the acceptance runs and pick the default.
- Re-baseline `tests/water_swe_gpu.rs` and rerun the `validate_*` sweeps; Froude-based results
  must not move.

### Phase 3: bed load in SI

- Bed Material in mm, Wong & Parker with √(RgD³), porosity, adaptation length in metres.
- Refine as "hours of bankfull flow".
- Rerun `bed_load_on_flumes`, `validate_cyclic_steps`, `validate_alternate_bars`,
  `bends_scour_their_outer_banks`, `river_brush_keeps_its_steps` at dx = 2 and 10 m.

### Phase 4: drainage area

- CPU accumulation after the cascade, edge inlets, `DrainageArea` resource and debug view.
- Unit tests on synthetic surfaces: a tilted plane (A grows linearly downslope), a cone (A ∝ r²
  toward the outlet), a V-valley (all area reaches the thalweg).

### Phase 5: one channel model

- `ChannelModel` replaces the river terms of `BedShaping`; G and the brush both build `River`s
  from the network and carve through `river::rasterize`.
- Narrow-river rule; river springs from Q_bf.
- `morpho_steer` in SI, refit at two dx.
- `validate_hydraulic_geometry` and `inspect_river_brush` rerun: width ∝ Q^0.5 and depth ∝
  Q^0.4 now hold by construction for carved rivers; check the water stays in its banks
  (1–2 % of wet cells outside, as now).

### Phase 6: cleanup

- Remove the Gravity slider, `REFERENCE_DISCHARGE`, `KTT_DISCHARGE_SCALE` as a magic number,
  duplicate `CELLS_PER_HEIGHT` constants (`river.rs`, five shaders) in favour of one uniform.
- Update `README.md` (keys, River brush) and `VALIDATION.md` (units line).

## How it was built

Decisions taken while implementing, where they differ from or sharpen the plan above.

- **The solver stays in cells.** Rather than handing bevy_fft metres, the water keeps texels and
  height units and receives SI-derived constants: g = 9.81 / dx cells/s², n · dx^(1/3) for
  Manning, and the step in seconds. This is the same physics with no change to the solver
  kernels (`WaterControl::gravity_cells`, `manning_cells`).
- **The step follows CFL for deep lakes.** It is sized for a gravity wave over 20 texels of
  water (200 m at 10 m texels) at Courant 0.4: 0.09 s at 10 m. A 20 m design depth was not
  enough: lakes on the default terrain are deeper and sloshed.
- **bevy_fft's depth limiter is not dimensionless.** It holds back water above β·dx/(g·dt),
  which is not a length, so moving to SI seconds loosened it about 45×. The solver now runs in a
  time unit (`WaterControl::solver_seconds`) that keeps the cap at the old 0.48 texels; g, dt, n,
  the sources and the velocities are converted at its boundary.
- **Wet depths stay relative to the grid.** Water is dry below 0.05 texels of depth and fully
  wet from 0.4 (0.5 m and 4 m at 10 m texels). These set where KTT trails end, where banks slump
  and where bed load takes over from sheet-flow erosion. Moving them to absolute centimetres was
  tried: rain trails then ended in their own sheets, wet slumping flattened steps on the flumes,
  and bed load in shallow margins stopped channels cutting. Absolute thresholds need their own
  validation sweep.
- **Shape Beds keeps its tuned thresholds** in metres at their old values (2 m lake depth,
  0.44 and 1.8 m/s still and river speeds) until Phase 5 replaces its river sizing.
- **The GPU tests run in old seconds.** The harness sets `time_scale` to `LEGACY_SECONDS` and
  divides rates by it, and runs at the old-equivalent roughness and grain, so they hold the SI
  solver to the old one. The app's defaults are separate.
- **Defaults keep the look.** n = 0.2 and 1.5 m grains at 10 m texels reproduce the old water:
  a boulder-bed mountain torrent, which is what the default terrain's 5–25 % slopes are.

## Acceptance

Checked at dx = 2 m and dx = 10 m, against screenshots and the `validate_*` CSVs:

- **Roll waves stay.** Second-order depth flux on steep chutes still grows roll waves
  (`validate_roll_wave_onset`, onset Fr ≈ 2.2 or lower).
- **Step-pool stair-stepping stays** on brushed and carved rivers (`inspect_river_brush`: H/L/S
  ≈ 1–2, spacing 1–4 widths).
- **No new grid artifacts** in the drainage network (MFD, not D8) or the carved beds.
- **Scale invariance where expected:** at a fixed metres setting, 256² at dx = 20 m and 512² at
  dx = 10 m give the same landscape at the coarser resolution.
- **Scale dependence where expected:** at a fixed texel setting, dx = 2 m and dx = 10 m give
  different rivers (narrower, relatively, at 10 m) and different bed load rates.

## Risks

- **Realistic roughness speeds the water up.** Going from n ≈ 0.2 to 0.035 raises velocities
  about 6× at the same depth and lowers depths. Froude numbers rise, so roll waves and cyclic
  steps get more frequent, and steep reaches may need the depth limiter or a smaller C. That is
  why Phase 2 starts at today's equivalent n.
- **Bed load rates jump with D.** Sand at dx = 10 m moves orders of magnitude faster than today's
  1.5 m equivalent grain. MorFac ranges and `MAX_FACE_CUT` need checking.
- **Narrow rivers.** Most rivers on a 512² map are 2–3 texels wide; the widening rule decides
  how they look. A higher-resolution water grid under rivers is out of scope here.
- **Fixed-point thresholds.** Thresholds in metres become tiny height-unit values at large dx
  (1 mm is 1e-6 height units at 10 m, one fixed-point quantum). Keep wet thresholds ≥ 10 quanta.
- **Readback latency.** Drainage area arrives a few frames after the cascade finishes, as river
  refreshes do now; carving waits for it.

## Open questions

- **Pipes solver:** keep it in metres too, or retire it now that SWE is the default?
- **Heights in metres internally:** worth doing later (one less conversion, KTT constants such as
  Bedrock Depth become meaningful), or is boundary conversion enough? This plan assumes boundary
  conversion.
- **Geological time:** a stream-power calibration (E = K A^m S^n with an uplift rate) could give
  KTT iterations a duration. Not needed for the look; left out.
- **Bankfull relation:** one Climate factor, or a region preset (humid, semi-arid, alpine)?

## Reference values

Order-of-magnitude values to start from; check each against its source before coding the
default.

| Quantity | Value | Source to check |
|---|---|---|
| Bankfull discharge | Q_bf ≈ 1–3 · A^0.75–0.85 m³/s (A km², humid regions); much less in arid ones | Dunne & Leopold 1978; regional curves (USGS) |
| Width | W ≈ 4–5 · Q^0.5 (m, m³/s); gravel-bed bankfull | Leopold & Maddock 1953; Parker et al. 2007 |
| Depth | H ≈ 0.2–0.4 · Q^0.4 | Leopold & Maddock 1953; Parker et al. 2007 |
| Channel heads | A ≈ 0.01–1 km², smaller on steep slopes | Montgomery & Dietrich 1988 |
| Manning n | 0.03 clean channel, 0.04–0.07 mountain streams with boulders | Chow 1959 |
| Critical Shields | 0.03–0.06 | Shields 1936; Wong & Parker 2006 (0.0495) |
| Bed load | q* = 3.97 (θ − 0.0495)^1.5 | Wong & Parker 2006 |
| Porosity | 0.3–0.4 | — |
| Gravity | 9.81 m/s² | — |
