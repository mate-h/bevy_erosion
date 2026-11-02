# Water bodies: lakes, rivers and ocean

Plan for water in `bevy_erosion`. Water should follow the terrain and the erosion physics, and
there is no user-set water level except the optional ocean. Reference implementation for the bed
reconstruction: `~/git/terracraft/crates/riverbed_solver_rs` (`shaders.wgsl`, `main.rs`) and
`~/git/terracraft/docs/chapters/riverbed.typ`.

## Principles

- **The terrain shapes the water.** Lakes form where water collects in depressions, and rivers
  form where it spills downhill. The brush adds water; it never sets a level.
- **The ocean is the only global level.** It is optional (Use Outflow plus Sea Level). Terrain
  below it renders as a flat sea and drains water away.
- **Water is coupled to the erosion physics:**
  - flowing water erodes and still water deposits;
  - wet cells are KTT outflow, so particle trails drain into lakes instead of carving under them.
- **Water covers the terrain.** Under water you see a flat water surface, never the source noise.

## Phase 1: brushes and the water solver (done, PR #2, `88e44c7`)

| Piece | Where |
|---|---|
| Brush types: Rain, Terrain (Raise, Lower, Flatten, Smooth), Water Body | `src/brush.rs` |
| Terrain edit layer (Rgba32Float: offset, flatten weight, flatten target, smooth weight) applied to the source noise in `init_fbm`; each stroke restarts the cascade | `assets/erosion.wgsl` `edited_height` |
| Shallow-water solver (virtual pipes, Mei et al. 2007) on the finest level, 8 substeps per frame once water exists | `assets/water.wgsl` |
| Sediment model: capacity = K × tilt × speed; dissolve below capacity, deposit above; slumping of steep banks in and next to water | `water_erode` |
| Outflow coupling: `masks.y = max(source outflow, wet)` on the finest work set | `water_depth` |
| Ocean and map border are sinks | `water_depth`, `water_flux` |
| Rendering: surface = terrain + depth (or the sea level), colour by depth, low roughness | `assets/terrain_extended.wgsl` |
| UI and keys: B toggles the brush, V cycles the type, T cycles the terrain tool, C clears water, X resets terrain edits | `src/ui.rs`, `examples/demo.rs` |

Verified headless and in the demo window:
- poured water formed a flat lake and channels that follow the valleys;
- a long pour holds its pour point within 0.004;
- the ocean drains water;
- Raise lifts the terrain as expected.

### Known gaps

- **KTT trails do not add water** (see Phase 1.5).
- **Grid artifacts.** Pipe-model erosion still leaves faint grid-aligned grooves.

## Phase 1.5: make water persistent and frame-rate independent (done)

| Piece | Where |
|---|---|
| Springs: Alt + click in Water Body mode places one (brush radius and pour rate), Alt + Shift + click removes those under the brush, C clears them with the water; up to 16, drawn as cyan rings | `src/brush.rs` `paint_water`, `WaterControl::springs` |
| Springs, the brush and rainfall are per-step sources in `water_depth` (uniform array of 17 discs plus a global rain depth) | `assets/water.wgsl` `source_depth` |
| Fixed time step: `WATER_STEP` = 1/480 s with a time accumulator (`floor`, not `ceil`, so no time is over-simulated), capped at 32 steps per frame; all rates are per second | `src/lib.rs` `plan_erosion_frame` |
| Restart: `pyramid_upscale_masks` folds the wet water depth into the expanded source outflow on every level, so restarted coarse levels drain into lakes. This is done per level rather than in `init_fbm`, so cleared water leaves no stale outflow in the source masks | `assets/pyramid_masks.wgsl` |
| UI: Evaporation in %/s (default 1%/s, the old 0.00002 per substep at 60 fps), Rainfall in ‰ of the map height per second | `src/ui.rs` |

Verified headlessly with a spring of rate 0.007 and radius 12:
- the water volume after 20 s is the same at 30 fps (19.23) and 120 fps (19.28);
- it matches the inflow minus evaporation (about 21.1 − 2);
- there are no NaNs.

Rain at 0.0002/s mostly runs off the draining map border (the volume grows by about 7/s against
about 52/s of input).

Not done: **KTT rain trails adding water along their path.** The erode pass already binds all 8
storage buffers per stage, so it cannot write the water depth without repacking a binding. A
rain trail splat buffer read by the water pass is the likely route.

## Phase 2: bed reconstruction (done)

"Shape Beds" (button, or G in the demo) deepens the bed under the current water into smooth
channels and bowls, keeping the water surface where it is. It runs once per request on the wet
domain the solver produced (depth > 0.002), as one frame of about 1,200 dispatches on the base
map. The code is `assets/water_bed.wgsl`, with `BedShaping` in `src/lib.rs`.

| Step | Kernel |
|---|---|
| Wet domain and shore seeds | `bed_init` |
| Bank distance τ, `∇²τ = −2`, τ = 0 on the banks, red-black SOR (400 sweeps, ω = 1.9), from `solve_tau_sor` | `bed_tau` |
| Depth prior: half-width `b = sqrt(τ + ¼|∇τ|²)`, cross-channel `n = |∇τ| / 2b`; rivers `(1 − n²)^profile`, lakes a bowl from `sqrt(2τ)`; blended by solver speed (still → lake) | `bed_prior` |
| Screened Poisson smoothing `∇²d = λ(d − prior)`, prior weight 0.045 (200 sweeps), from `solve_lake_bathy_sor` | `bed_smooth` |
| `bed = min(ground, η − d)` blended by Strength; depth = η − bed | `bed_write` |
| Shore distance and nearest η, chamfer dilation | `bed_dilate` |
| Bank grading, `z ≤ η + rise + slope·distance`, quintic S5 blend over Bank Width (terracraft stage 5) | `bed_grade` |

The UI exposes Max Depth, Shore Slope, River Profile, Strength and Bank Width.

Changes from the plan:
- **No φ or η(φ).** terracraft rebuilds a monotonic water surface because it only has a
  LiDAR DEM. Here the solver already provides a physical surface, so the bed is cut below it and
  φ is not needed.
- **Deepen-only, with a soft depth cap** (`max·(1 − e^(−d/max))`). A hard cap made lakes into
  flat pans with a steep rim, and raising the bed under narrow streams made them shallower until
  they drained.
- **Single grid, no coarse-to-fine.** At 512² a fixed sweep count converges for lakes up to
  about 100 texels across.
- **On demand only,** with no automatic run after the water settles.
- **The bed is not locked against later erosion.** Wet cells are already KTT outflow, but
  `water_erode` still works on the shaped bed.

Verified headlessly with one spring (20 s, then Shape Beds):
- the water surface moves by 0.00017 on average, while the bed drops by 0.014;
- a cross-section is a smooth symmetric bowl (depth 0 → 0.011 → 0.018 → 0.0198 → … → 0);
- no NaNs, and dry cells are unchanged on flat ground.

## Phase 3: rendering (done)

In `assets/terrain_extended.wesl`, `shade_water`:
- **Light through the water:** Beer–Lambert absorption along the view path (down and back up),
  in-scattered deep-water colour, and silt tint from the suspended sediment.
- **Ripples:** flow-map normals advected along the solver velocity, two phases so they never
  stretch. The water texture now stores velocity: R depth, GB velocity, A sediment. The ocean
  gets wind-drifting waves instead.
- **Foam:** on fast flow, on thin water at the shore, and as breaking water on the ocean shore.
- **Reflections:** low roughness and a water F0 (reflectance 0.35), so the atmosphere
  environment map supplies the Fresnel sky reflection.

Picking: the brush reads back the water texture too and ray-casts against the displayed surface
(terrain plus lake, river or ocean). The ring follows the water, and Flatten still targets the
terrain height.

Checked in the demo (autopilot screenshots): a shallow lake shows its bed through teal water; a
shaped lake reads deep blue with a soft shore.

Not done: blending the ocean shoreline into the lakes beyond what the shared shading gives.

## Phase 4 (optional, not started): morphodynamics refinement

Port terracraft's local-inertial shallow water and Exner sediment transport for the final bed
shape.
- **Flow** (Bates 2010 / de Almeida 2012, `step_swe_flux`): staggered-grid flux, Manning
  friction, θ-upwinding.
- **Bed load** (`step_swe_continuity_and_exner`): Meyer-Peter Müller transport, deflected by
  Talmon (1995).
- **Bends:** helical secondary flow from bend curvature (advected `I_s`, on the CPU in
  terracraft, needs porting to the GPU), so pools move to the outer banks and bars to the inner
  ones.
- **Deltas:** where a river meets a lake, as a foreset ramp (quintic Hermite).

This would replace or extend the Phase 1 capacity model on wet cells. Run it for a few hundred
steps after Phase 2.

## Units

terracraft works in metres at 1 m/px. Here the terrain is about 1 height unit over 512 texels
(1 unit ≈ 50 world units ≈ 100 texels of rise). Keep depths and widths relative to the map:
- depth in height units (lakes about 0.01–0.08);
- widths in texels;
- slopes scaled by `SLOPE_SCALE = 100` (as in `water.wgsl`).

## Testing

- **Headless probes** (temporary `examples/zz_*.rs`, deleted after use): pour at fixed points,
  read back the height and water textures, check:
  - wet area, volume and NaNs;
  - surface flatness in lakes;
  - terrain change under water.
- **Demo window:** an autopilot copy of the demo for before/after screenshots
  (`screencapture -l <window id>`).
- **Unit tests:** pure-Rust parts (brush math, schedules, a CPU reference for the SOR solvers).
