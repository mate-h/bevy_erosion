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

- **Grid artifacts.** Pipe-model erosion still leaves faint grid-aligned grooves.
- **No reconstructed bed.** Lake beds are whatever erosion and deposition produced; nothing
  shapes a proper bed profile yet.

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

## Phase 2: bed reconstruction (terracraft port)

Goal: once a lake or river is established, shape its bed like a real channel instead of leaving
eroded noise. This runs on the wet domain that the solver produces. There is no painted mask and
no user level.

1. **Wet domain:** cells with depth above a threshold, split into connected bodies.
   - Still water (low speed) is a lake; flowing water is a river.
2. **Bank distance τ** (Poisson torsion), `∇²τ = −2` inside the domain with τ = 0 on the banks.
   - Red-black SOR, ported from `solve_tau_sor`.
   - Coarse to fine on the existing pyramid grids, so a few hundred dispatches converge.
   - Gives the local half-width `b = sqrt(τ + |∇τ|²/4)` and the normalized cross-channel
     coordinate.
3. **Rivers: stream potential φ**, `∇²φ = 0`, from `solve_phi_sor`.
   - Pinned to 1 at sources (springs or the highest wet cells) and 0 at outlets (lake, ocean or
     border).
   - The flow direction comes from ∇φ (`compute_flow_vectors`).
4. **Water surface η:**
   - lakes: the solver's own (flat) surface;
   - rivers: η(φ) made monotonic, as in terracraft's `wse_lut` (binned low quantile, steady fall,
     smoothed), so there are no pits or dams.
5. **Depth prior:**
   - parabolic cross-section from the normalized cross-channel coordinate, scaled by the
     half-width (Max Depth and Profile Shape sliders, in map-relative units);
   - lakes get a smooth bowl from τ.
6. **Smoothing:** screened Poisson, ported from `solve_lake_bathy_sor`
   (`∇²d = λ(d − d_prior)`, soft prior weight about 0.045), so there are no rings or corners.
7. **Write the bed:** `bed = η − d` into the finest work set.
   - Grade the banks just outside the domain with a quintic S5 ramp
     (terracraft stage 5: `z_max = η_near + a + b·d_shore`).
   - Remember the bed so that erosion does not re-roughen it (e.g. outflow = 1 on the bed).

Driver:
- **When it runs:** on demand ("Shape Beds" button), or automatically after water settles.
- **Pipeline:** a new pipeline group; it reuses the water buffers plus two scratch fields (τ, φ).

## Phase 3: rendering

- **Water surface:**
  - Fresnel and a sky reflection from the atmosphere environment map;
  - depth-based absorption, shore foam from water speed and depth gradient;
  - flow-aligned normal ripples (using the solver velocity, or ∇φ on rivers).
- **Ocean:** waves or a normal map, and shoreline blending with the lakes.
- **Picking:** optionally against the water surface, so a click lands on the water rather than
  the bed.

## Phase 4 (optional): morphodynamics refinement

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
