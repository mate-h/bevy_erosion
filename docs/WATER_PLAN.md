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

## Phase 4 (planned): morphodynamics

Goal: on wet cells, replace the Phase 1 capacity model with bed load that moves along the flow
and is conserved. Pools should form on the outer banks of bends and bars on the inner banks.
Rivers should build deltas where they enter lakes, and straight channels should keep a smooth
floor without the grid-aligned grooves. Phase 2 shapes the bed once; Phase 4 lets the flow
evolve it.

### What the references actually contain

- **terracraft.** `step_swe_flux` is a Bates / de Almeida local-inertial flux with Manning
  friction and θ = 0.85, as documented. `step_swe_continuity_and_exner` is **not** Meyer-Peter
  Müller (MPM) or Talmon. It relaxes the bed toward its 4-neighbour mean
  (`0.20 · (zb_avg − zb_c)`) and cuts by a clamped `(u/1.35)² − 1` anomaly. MPM, the Talmon
  slope deflection and the helical term appear only as equations in `riverbed.typ` stage 4.
  The helical intensity `I_s` is computed in `main.rs` on the CPU and only skews the depth
  prior; it is never coupled to transport. **Port the equations from `riverbed.typ`, not the
  shader.** The helical advection loop in `main.rs` (lines 505–590) is the one piece worth
  taking line by line.
- **bevy_fft** (`~/git/bevy_fft`, also on Bevy 0.20). `assets/shallow_water/simulator.wesl` is a
  staggered CMF10 solver (Chentanez and Müller):
  - MacCormack advection of the face velocities (`macU_*`, `macW_*`);
  - upwind depth flux with `h_avgmax` limiting (`integrateHeight`);
  - pressure from η (`integrateVelocityU/W`);
  - reflective wet/dry faces;
  - linear friction;
  - an overshoot limiter;
  - PML and ocean-band boundaries;
  - cell-centred velocity reconstruction.

  It also has a water rendering stack: transmission with scene depth, caustics, foam advection
  and breaking (`src/water/*`), fed through `src/shallow_water/lod_write.wesl`.

### Decision: flow solver (4a)

| Option | Momentum advection | Face velocities for Exner | Cost per step | Notes |
|---|---|---|---|---|
| Keep virtual pipes, add Manning | no | derived from flux | 5 dispatches (now) | cheapest; jets die at lake mouths |
| Bates local-inertial (terracraft) | no (θ-diffusion only) | yes, signed per face | about 5 | built for steady, friction-dominated flow |
| **CMF10 from bevy_fft** | yes (MacCormack) | yes, staggered u and w | about 12 | your solver, already validated; carries jets into lakes and around bends |

**Decision: import bevy_fft's CMF10 solver as a library.** Do not copy its kernels.
- Deltas and mouth bars come from momentum carrying a jet into still water, and only CMF10
  advects momentum.
- Its staggered u and w are exactly the face velocities that a conservative Exner step needs.
- One solver serves both crates, so fixes and Manning friction land once.

Using it as it is fails on four points, so bevy_fft needs some work first (4a.0). Bates stays
the fallback if CMF10 costs too much at 512²; its flux kernel is about 70 lines.

### 4a.0. Prerequisite in bevy_fft: make the solver importable (done, `6494249`)

These changes go in `~/git/bevy_fft/src/shallow_water/` and `assets/shallow_water/simulator.wesl`.
Ship them as a bevy_fft release (or a git dependency while iterating).

1. **Embed `simulator.wesl`.**
   - Problem: `render.rs:106` loads it from bevy_fft's `assets/` folder with
     `asset_server.load`, so the load fails in any other crate.
   - Fix: register it with `embedded_asset!` (as `surface.wesl` and `lod_write.wesl` already
     are) and load it from `embedded://bevy_fft/...`.
2. **Split the solver into a core and a plugin.**
   - Core: a render-world solver object that owns the pipelines, the textures (`bed_water`,
     `flow_x`, `flow_y`, MacCormack temps, `velocity`) and the bind groups. It exposes:
     - `encode_step(&self, pass: &mut ComputePass)`: one step, from MacCormack through the
       overshoot limiter;
     - `encode_sync_bed`;
     - `encode_reconstruct_velocity`.
   - Plugin: `ShallowWaterPlugin` keeps its current behaviour on top of the core, with the
     presets, particles, PML, ocean forcing and LOD write. The examples stay unchanged.
3. **Bed that changes over time** (`bed_live`).
   - A `syncBed` kernel copies the bed height from an `R32Float` bed texture into
     `bed_water.x` and leaves the water depth `h` alone, so the bed can change under the water.
   - Today `loadPreset` reads `bed_source` only on reset.
4. **Source texture.**
   - An optional `Rgba32Float` cell texture, applied in `integrateHeight`:
     - `.x` = depth added per second (negative removes water);
     - `.y` = sink weight (1 = drain to dry).
   - Add a uniform evaporation rate. With all four borders set to `Drain`, this covers this
     crate's springs, rain, brush, evaporation, ocean sink and draining map border.
   - `Rg32Float` is unreliable on Metal (bevy_fft's own note), hence `Rgba32Float`.
5. **Manning friction.** Add an implicit Manning option next to the linear `friction_factor`:
   `u ← u / (1 + g Δt n² |u| / h^(4/3))`. Steady flow speed then scales with h^(2/3)·S^(1/2),
   which the Shields stress in 4b depends on.
6. Optional: a cargo feature that builds `shallow_water` without the FFT ocean.

Done when:
- bevy_fft's `shallow_water` and `beach` examples behave as before;
- a test app outside the crate drives `encode_step` against a bed that changes every frame.

### 4a. Flow through the imported solver (done)

bevy_erosion depends on bevy_fft as a git dependency on `water-foundations`, pinned in
`Cargo.lock`. `WaterControl::solver` picks `Swe` (default) or `Pipes`; the panel shows it as
"Momentum Flow (SWE)". Below it are two sliders:
- "Bed Friction (Manning n)", 0.002–0.1;
- "Gravity (cells/s²)", 250–8000.

What differs from the plan below:
- **Kernels.** `swe_export`, `swe_gather` and `swe_import` live in `water.wgsl`, sharing its
  `source_depth`. They do not get their own shader. Water group 1 grows to the render texture
  plus four solver bindings (bed, sources, `bed_water`, velocity).
- **Map border.** All solver borders stay `Wall`, and `swe_export` sets the sink weight to 1 on
  the outer ring of cells and in the ocean. bevy_fft's `Drain` border is a fixed 3 cells per
  second outflow wherever the bed is below 1.0, meant for metre-scale scenes.
- **Per step:** `swe_export`, `encode_sync_bed`, `encode_step`, `encode_reconstruct_velocity`,
  `swe_gather`, then the existing `water_erode`, `water_advect` and `water_copy`. The bed is
  exported every step because `water_erode` changes it every step.
- **Hand-over:** `swe_import` hands this crate's depth to the solver when it takes over from the
  pipes and after Shape Beds.
- **Settings:**
  - gravity defaults to 2000 cells/s², with heights × 100 in cells;
  - Manning n defaults to 0.01;
  - the overshoot limiter is off, because it does not conserve water.
- **What sets the speed:**
  - Manning n sets how fast water runs. Friction-limited speed is h^(2/3) S^(1/2) / n and does
    not depend on g. At 0.05, a 0.3-cell sheet on a 0.1 slope moved about 3 texels/s, which
    looked stalled on a 512² map.
  - Gravity sets how fast waves travel and lakes level.
  - The solver's depth limiter (β·dx / (g·Δt), about 0.5 cells at these settings) did not turn
    out to be the brake. In a frictionless dam break the front reaches 38 texels at g = 2000.

Verified headlessly with a spring of rate 0.007 and radius 12 at the map centre, with
evaporation and erosion off:
- the volume matches the inflow at 30 and 120 fps (6.331 and 6.331 against 6.333 after 6 s);
- no NaNs;
- with evaporation on, the volume after 20 s is the same at both frame rates.

Probe pitfall: a probe that keeps calling `app.update()` while it waits for a readback also
advances the water. Pause `SimControl` during readbacks, or the extra frames look like a mass
gain proportional to the frame time. A headless probe also needs a GPU poll each frame;
otherwise frames queue up faster than the GPU runs them.

Cost:
- the probe: about 10 ms per frame at 8 steps, against about 6 ms for the pipes;
- the demo's `--simple` mode: 7.4 ms against 7.2 ms.

In the demo a spring now forms a pool that spills down a channel. The pipes instead throw the
water off the slope at their 1 texel per step speed cap.

GPU checks in `tests/water_swe_gpu.rs`:
- **Setup.** Each test draws an analytic bed through the terrain edit layer, with the KTT
  erosion off, and checks the bed survives to within 0.001.
- **Running them.** They are ignored by default:
  `cargo test --release --test water_swe_gpu -- --ignored --test-threads=1 --nocapture`.
  `TEST_GRAVITY` and `TEST_MANNING` override the defaults.

Results at the defaults:
- **Dam break.** A mound poured on a flat bed spreads to 25 texels (pour radius 16) in 0.5 s. It
  stays mirror and diagonal symmetric to within 0.08% of its peak, with volume 8.039 of 8.042.
- **Manning flume.** Steady rain on a 0.1 slope gives 0.977 of the Manning speed and 0.993 of the
  steady Manning depth mid-plane.
- **Lake jet.** A channel runs into a filled basin. Two channel widths into the lake, SWE keeps
  about 8–12 texels/s (70–100% of the channel speed); the pipes keep 2.8–5.3 (5–10%). The still
  lake beside the jet moves at about 2 texels/s.
- **Bed under a lake.** Raising a bowl's floor by up to 0.02 re-runs the cascade under a settled
  lake:
  - the volume stays (785.10 → 785.08);
  - the surface rises 0.0010 for 0.0010 expected;
  - it settles flat (RMS 0.0003–0.0004).
  A seiche of a few hundredths of a cell lingers for several seconds at this low friction.

Known in bevy_fft:
- `integrateHeight` and `overshootReduce` update depth in place while reading their neighbours'
  depth. That is a data race, but it made no measurable difference here.
- A fix that stages depth in `pml_state.z` and limits each cell's outflow (three more dispatches
  per step) was tried and set aside.


Ownership: this crate keeps the bed (the padded atomic heights) and all source logic.
bevy_fft owns the water state.

- **Setup.**
  - Create the core solver at the base map size with all borders `Drain`, `bed_live` on, and
    PML, particles and ocean forcing off.
  - Turn on the GPU features bevy_fft needs: read-write `Rgba32Float` storage textures, and
    `FLOAT32_FILTERABLE` as its examples do.
- **Per frame, in this crate's water pass**, keeping the fixed-step accumulator in
  `plan_erosion_frame`:
  1. `water_bed_export` (new, in `water.wgsl`) writes the heights to the `R32Float` bed texture.
  2. `water_sources` rasterises the springs, brush and rain into the source texture's `.x`, and
     the ocean (terrain below `sea_level`) into `.y`. This replaces `source_depth`.
  3. For each step: `encode_sync_bed`, `encode_step`, then this crate's Exner kernels (4b),
     which bind bevy_fft's `flow_x` / `flow_y` handles directly in their own layout.
  4. `encode_reconstruct_velocity`, then `water_gather` (new), which:
     - reads `bed_water.y` and `velocity`;
     - converts the velocity to texels per step (`v · Δt / dx`);
     - fills this crate's `depth` and `velocity` buffers;
     - writes the KTT outflow `masks.y`;
     - writes the `Rgba16Float` render texture (depth, velocity, sediment).

  The renderer, ripples, foam, picking and bed shaping are then unchanged.
- **Removed:** `water_flux`, `water_depth` and the `flux` buffer. `water_erode` (thin-water
  capacity model and slumping), `water_advect` and `water_copy` stay.
- **Units:**
  - dx = 1 texel, with η in height units (vertical exaggeration as now).
  - Pick g so the wave speed `sqrt(g·h)·Δt / dx` stays under 0.5 at a depth of 0.08.
  - The step is CFL-bound, so `WATER_STEP` can probably grow and fewer steps run per frame.
- **Validation:** `WaterControl::solver: Pipes | Swe` stays until the probes pass, then the
  pipe kernels are deleted.

Verify by rerunning the Phase 1.5 probes:
- spring volume at 30 fps against 120 fps;
- flat lake surface;
- ocean drain.

Add four checks:
- a dam break stays symmetric;
- a tilted flume reaches Manning's speed within 10%;
- a river entering a lake keeps a visible jet a few widths past the mouth (velocity readback);
- Raise or Lower under a lake moves the bed without resetting the water.

### 4b. Bed load and Exner

New shader `assets/water_morpho.wgsl`, run after each flow step (or every k steps). It uses
non-dimensional versions of `riverbed.typ` stage 4:
- **Shields:** `θ = n² |u|² / (R · h^(1/3))`, with the grain and density ratio folded into
  `R` (UI: Grain).
- **MPM:** `|q_b| = K · max(θ − θ_c, 0)^1.5` (UI: Bed Load K, Critical Shields θ_c).
- **Direction:** along û, rotated by the helical angle δ (4c), with Talmon's slope term
  `− |q_b| / (A_s · sqrt(θ)) · ∇z_b` (UI: Slope Effect A_s). The slope term is what keeps
  channels from growing grooves; it acts as anisotropic diffusion.
- **Exner on faces:** compute q_b on the same staggered faces as u and w, with the face velocity
  and face bed slope, then `Δz = −mor_fac · Δt · div(q_b)`. Material is conserved exactly
  apart from the outflow through the border and ocean.
- **Height quantization (must fix):** heights are `u32` at 1e-6, and `u32(new · SCALE)`
  truncates, so small per-step changes vanish or drift downward.
  - Accumulate Δz in a float `bed_delta` buffer.
  - Flush whole quanta, rounded to nearest, into the atomic heights.
  - Also switch `water_erode` to round to nearest.
- **Coupling with the existing model:**
  - Cells deeper than `FULL_DEPTH` use Exner only.
  - Thin sheet flow and drying cells keep the capacity model, so hillslope wash still works.
  - Wet slumping stays as the angle of repose.
  - Suspended `sediment` stays for the silt tint.
- **Bed stability:** clamp `|Δz|` per step to a fraction of the local depth, so a large
  `mor_fac` cannot dry a cell in one step.

Verify:
- the bed volume change equals the border and ocean outflow of bed load (within quantization);
- a straight tilted flume lowers uniformly with no grid grooves;
- no NaNs;
- a dry map is unchanged.

### 4c. Helical flow in bends

The CPU code in terracraft, ported to the GPU in `water_morpho.wgsl`:
- **Curvature** at cells from the velocity: `κ = (u × (u·∇)u) / |u|³`, clamped.
- **Intensity** `I_s`: semi-Lagrangian advection along the velocity (as `water_advect`), with
  relaxation `(I_eq − I_s) · |u| / λ_s` toward `I_eq = h·κ`, plus a little diffusion.
  - Ping-pong pair of f32 buffers.
  - λ_s is an adaptation length in texels; terracraft uses 42 at 1 m/px, start near 20 here.
- **Deflection:** rotate the q_b direction by `δ = atan(A_h · I_s)` (UI: Helical Strength).
  Bed load moves toward the inner bank, the outer bank scours, and the point bar builds inside.

Verify on an S-bend carved with terrain edits: the deepest point of each bend's cross-section
sits on the outer bank, and the skew changes sign between the two bends.

### 4d. Deltas

- **Physical:** with momentum (4a) and Exner (4b), bed load drops where the jet slows in still
  water, so a mouth bar and delta should grow without special code. Check this first.
- **Shaped (Phase 2 path, only if Exner deltas are too slow or too blobby):**
  - In `bed_prior`, find river mouths: wet cells where the solver speed crosses from
    `RIVER_SPEED` to `STILL_SPEED`.
  - Dilate a distance from them into the lake.
  - Blend the depth from the mouth thalweg depth to the lake bowl with the quintic S5 over a
    Foreset Length (terracraft stage 2's foreset ramp).

### 4e. Run modes and UI

- **Refine Beds** (button, M in the demo, since R is taken): runs N steps (default 400) of flow
  plus Exner with `mor_fac` (default about 20), spread over a few frames. Springs keep flowing
  and the KTT cascade restarts once at the end, as Shape Beds does.
- **Live Bed Load** checkbox: Exner runs every water step instead of only on request, with
  `mor_fac` = 1.
- **New panel controls:** Manning n, Grain, Bed Load K, Critical Shields, Slope Effect, Helical
  Strength, Morph. Factor, Refine Steps.

### Order and risks

- **Order:** 4a.0 (bevy_fft), then 4a, 4b and 4c, with 4d last.
  - 4a stands on its own and is worth having even if Exner slips.
  - 4b–4d only need the face velocities, so they can be prototyped on the pipe solver's flux
    while 4a.0 is in progress.
- **Cross-crate coupling:**
  - Both crates must use the same Bevy release.
  - A bevy_fft API change breaks this crate.
  - Pin a git revision while iterating, and a crates.io version once 4a.0 ships.
- **Binding budget:** bevy_fft's kernel is at Metal's eight storage textures, so coupling fields
  live in this crate's layouts (bed export, sources, gather, Exner), never in bevy_fft's.
- **Cost:** 512² with about 12 flow, 2 sync and 4 morpho dispatches per step. Measure in 4a
  before committing to running bed load live.
- **Quantization** (see 4b) is a correctness bug in the current `water_erode` too.

## Rendering cost (done)

The demo now freezes the atmosphere environment map, as flex does
(`freeze_atmosphere_env_map`):
- it bakes for 6 render-ready frames;
- then `GeneratedEnvironmentMapLight` comes off the camera, which stops Bevy's per-frame cubemap
  downsample and convolution;
- it bakes again when the sun direction moves by more than about half a degree.

Measured in the demo with a spring running, on a GPU shared with other sessions:
- re-filtering the map every frame cost 4–11 ms;
- the water shading in `shade_water` costs nothing measurable: the frame is 15.6–16.0 ms with it
  and without it;
- the rest of the PBR frame against `--simple` (about 15.6 against 7.3 ms) is the atmosphere,
  shadows, bloom and auto exposure.

`--simple` (or `EROSION_SIMPLE=1`, preview key 9) draws Lambert terrain with a flat water tint,
turns off vsync and logs the mean frame time. The bench variables are documented at the top of
`examples/demo.rs`.

## Phase 5 (optional): render through bevy_fft's water stack

This would replace `shade_water` with bevy_fft's transmissive `WaterSurfaceMaterial`
(caustics, scene-depth optics, advected foam, breaking). After 4a the water state already lives
in bevy_fft's solver, so its `lod_write` pass can feed the shared water LOD data directly:
depth in the foam texture's green channel, and velocity plus the surface rise rate in the flow
texture. Before 4a, the same blend would have to read this crate's water texture. It requires
HDR, a depth prepass, `Msaa::Off` and one water viewer.

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
