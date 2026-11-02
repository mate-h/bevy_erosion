# Validation against the literature

The water and bed model (Phases 4 and 6 of `WATER_PLAN.md`) checked against primary sources.

**How to reproduce:**
- Each sweep is an ignored GPU test in `tests/water_swe_gpu.rs` named `validate_*`. It writes a
  CSV to `target/validation` (or `TEST_DATA_DIR`):
  `cargo test --release --test water_swe_gpu -- --ignored --test-threads=1 --nocapture validate_`.
- Papers live in `.dev/papers` (not committed) and were converted to markdown with MinerU 4
  (`~/git/papers/convert.sh`).
- Units are the solver's: depths in cells (height × 100), speeds in cells per second,
  g = 2000 cells/s², Manning n = 0.01 unless stated.

## Summary

| Mechanism | Literature | Model | Verdict |
|---|---|---|---|
| Uniform flow | Manning u = h^(2/3) S^(1/2) / n | speed 0.98, depth 0.99 of Manning (`flume_follows_manning`) | agrees |
| Bed load relation | Wong & Parker 2006: q* = 3.97 (θ − 0.0495)^1.5 | q* = 4 (θ − 0.047)^1.5 | same form and coefficients by construction |
| Cyclic steps, onset | Parker & Izumi 2000; Balmforth & Vakil 2012: Fr > 1 (slightly below 1 with fast erosion) | none at Fr 0.70 and 0.92; steps at Fr ≥ 1.2 | agrees |
| Cyclic steps, wavelength | set by eddy viscosity in shallow-water theory (k = √(3/ν)); recurrent isolated steps in flumes (Taki & Parker 2005) | 13–31 texels, 30–300 depths, 0.2–1.2 cells relief | plausible; set by numerical diffusion |
| Cyclic steps, migration | upstream at every wavenumber once Fr > 1 (Parker & Izumi 2000; Sun & Parker 2005) | upstream at 0.04–0.28 texels/s | agrees |
| Roll waves, onset | Jeffreys 1925 / Vedernikov: Fr > 1.5 for Manning (2 for Chézy); convective, so hard to excite in short flumes | first-order flux: grows only from Fr ≈ 2.7; second-order flux: from Fr ≈ 2.2 | onset 1.5–2× too high (numerical damping) |
| Alternate bars, onset | β = W/2h above β_cr ≈ 3–8 (Redolfi et al. 2020, Table 1) | no bars at β 6; bars from β ≈ 10 | agrees |
| Alternate bars, wavelength | L = 5–12 W; shorter (< 5 W) diagonal bars at high flow (Redolfi et al. 2020) | 1–3 W, a criss-cross of oblique bars | too short (diagonal or braid bars) |
| Hydraulic geometry, depth | Leopold & Maddock 1953: depth ∝ Q^0.4 | deepest point ∝ Q^0.41–0.52 | agrees |
| Hydraulic geometry, width | width ∝ Q^0.5 downstream (0.26 at a station) | channel width ∝ Q^0.9–1.1 | too fast: no bank strength, short runs |
| Transverse slope effect | tan γ = −(r/√θ) ∂η/∂n, r = 0.3–0.6 (Redolfi: 0.40) | r = 1/A_s = 1 by default | stronger than measured; kept (see below) |
| Bends | outer-bank pools, inner point bars | scour 3.8–4.5 texels toward the outer bank, switching sides | agrees qualitatively |
| Deltas | bar where the jet slows (Edmonds & Slingerland 2007) | mouth +0.03, far basin unchanged | agrees qualitatively |

## Roll waves (`validate_roll_wave_onset`)

**Setup:** a 24-texel fixed-bed flume at slope 0.2 (in cells), fed by a spring, with Manning n
swept to vary the Froude number.

**Metric:** the depth's detrended coefficient of variation along the centreline, over the
upstream and downstream halves of the reach. An instability grows downstream; an inlet
disturbance decays.

First-order upwind depth flux (the default):

| Fr | 0.87 | 1.14 | 1.55 | 1.82 | 2.15 | 2.67 | 3.30 | 4.16 | 5.24 |
|---|---|---|---|---|---|---|---|---|---|
| CV upstream | 0.004 | 0.014 | 0.032 | 0.049 | 0.059 | 0.108 | 0.050 | 0.073 | 0.078 |
| CV downstream | 0.000 | 0.002 | 0.009 | 0.017 | 0.021 | 0.147 | 0.028 | 0.024 | 0.102 |

Second-order depth flux (`TEST_SECOND_ORDER=1`, the Second-Order Depth Flux toggle):

| Fr | 0.87 | 1.14 | 1.55 | 1.81 | 2.19 | 2.64 | 3.32 | 4.17 | 5.24 |
|---|---|---|---|---|---|---|---|---|---|
| CV upstream | 0.006 | 0.016 | 0.047 | 0.049 | 0.086 | 0.058 | 0.052 | 0.169 | 0.199 |
| CV downstream | 0.001 | 0.003 | 0.013 | 0.019 | 0.141 | 0.199 | 0.033 | 0.076 | 0.101 |

- Theory puts roll-wave instability at Fr > 1.5 for Manning friction. In the model, unevenness
  decays downstream up to Fr 2.15 and grows only near Fr 5 (2.67 is close to even).
- Balmforth & Vakil (2012) show that eddy viscosity cuts off roll-wave growth. The solver's
  first-order upwind depth flux acts as a numerical viscosity of about u·Δx/2. This is why roll
  waves show only on steep slopes in the demo.
- Roll waves are convective, so a 256-texel flume with a quiet inlet also limits them. Taki &
  Parker's flume saw none.
- bevy_fft's `second_order_depth` setting replaces the upwind depth flux with a flux-limited
  Lax-Wendroff one (van Leer limiter, Sweby 1984). Growth then starts near Fr 2.2 instead of
  2.7, and the decay at Fr 1.5–1.8 is weaker, but it still decays. The remaining damping
  likely comes from the momentum update (semi-Lagrangian advection, the depth limiter and the
  velocity cap) and from the short reach, since the flux limiter falls back to first order at
  every crest and trough.
- The same change brings back an outflow limit on each cell and staged depth writes. Every
  cell now reads its neighbours' depth from before the step, and the depth no longer goes
  negative and gets clamped, which created water. Without second order the results are
  unchanged (the first table).

## Cyclic steps (`validate_cyclic_steps`)

**Setup:** a 6-texel erodible flume with a strong spring and live bed load for 25 s.

| Slope (cells) | 0.1 | 0.2 | 0.4 | 0.6 | 0.9 | 1.3 |
|---|---|---|---|---|---|---|
| Fr | 0.70 | 0.92 | 1.20 | 1.38 | 1.36 | 1.22 |
| Mean depth (cells) | 0.95 | 0.64 | 0.44 | 0.34 | 0.19 | 0.10 |
| Step wavelength (texels) | – | – | 13.1 | 16.7 | 20.4 | 30.7 |
| Relief (cells) | 0.02 | 0.04 | 0.20 | 0.49 | 0.96 | 1.22 |

- Relief jumps once the flow is supercritical, as the Parker & Izumi instability requires
  (Fr > 1).
- In shallow-water theory the most unstable wavelength is set by the eddy viscosity. Without
  one, the short waves grow fastest. Ours has none beyond numerical diffusion, so the 13–31
  texel spacing is a property of the grid and scheme, not of the sediment.
- Taki & Parker observed recurrent steps migrating upstream; `validate_step_migration` below
  measures it.

## Step migration and mean slope (`validate_step_migration`)

**Setup:** the stepped flumes above. After 25 s of live bed load, the detrended centreline bed
is cross-correlated between snapshots 0.5 s apart over 4 s.

| Slope (cells) | 0.6 | 0.9 | 1.3 |
|---|---|---|---|
| Wavelength, first order (texels) | 16.7 | 20.4 | 30.7 |
| Migration, first order (texels/s) | −0.28 | −0.14 | −0.04 |
| S_e / S_n, first order | 0.974 | 0.998 | 0.997 |
| Wavelength, second order (texels) | 10.2 | 13.1 | 14.2 |
| Migration, second order (texels/s) | −0.08 | −0.07 | −0.06 |
| S_e / S_n, second order | 0.981 | 0.997 | 1.000 |

- Negative speed is upstream. Steps migrate upstream at every slope, as Parker & Izumi (2000)
  find for all wavenumbers once Fr > 1 (their eq. 7.6a) and as Sun & Parker (2005) find for
  transportational steps.
- Sun & Parker predict that steps usually lower the mean slope needed for the same transport
  (S_e / S_n < 1 on their major branch). The mean slope falls by up to 2.6 %, but 25 s is far
  from equilibrium, so this is the sign only.
- With the second-order flux the spacing tightens to 10–14 texels, another sign that the grid
  and scheme set it (see above).
- Sawai's flumes (Parker & Izumi, table 1) had wavelengths 7–28 times the deepest flow. Ours are
  30–300 times the mean depth, and the ratio grows with slope as the sheet thins. The steps are
  far too long for their depth, as expected when the scheme rather than the flow sets the
  spacing.

## Alternate bars (`validate_alternate_bars`)

**Setup:** flumes 6–52 texels wide at slope 0.3 (cells), one discharge, 30 s of live bed load.
Some runs also add 6000 refine steps.

| W (texels) | 6 | 10 | 16 | 24 | 36 | 52 |
|---|---|---|---|---|---|---|
| β = W/2h | 6 | 10 | 19 | 33 | 56 | 112 |
| Thalweg wander (texels) | 0.7 | 1.2 | 3.2 | 5.7 | 7.4 | 13 |
| Wavelength / W | – | 2.9 | 1.8 | 2.8 | ≥ 2.8 | ≥ 3.8 |

- Bars appear between β 6 and 10, within Redolfi's β_cr of 3.2–7.9.
- The bed then shows a criss-cross of oblique bars about 2 W long, like Redolfi's diagonal
  bars or braid bars. Classical alternate bars are 5–12 W long.
- Neither a lower Slope Effect (r = 0.4), a gentler slope (lower Shields) nor 6000 refine
  steps lengthened them.

Likely causes:
- the channels are only 5–8 cells from bank to centre;
- the reach holds only about 10 widths;
- the spring feeds clear water with no sediment;
- Shields numbers are 0.15–0.6, against 0.04–0.15 in the flume.

## Hydraulic geometry (`validate_hydraulic_geometry`)

**Setup:** springs of 0.25–4 times the default discharge carve their own channel in an erodible
plane (slope 0.2 in cells), with 3000 refine steps and 6 s of live bed load. Values are means
over x = 96–200.

Fitted exponents against Q:
- deepest point 0.45–0.52 and mean depth 0.20–0.23, against Leopold & Maddock's 0.4;
- channel width (water deeper than half the deepest) 0.9–1.1, against 0.5.

**Cause:** the channels cut only 0.003–0.008, so the flow is still a sheet whose width the plane
sets. Natural channels hold their width with cohesive or vegetated banks. A trial "bank strength"
(a higher critical Shields number next to dry ground) changed the exponent by under 0.05 in runs
this short, so it was not kept.

## Transverse slope effect and bends

Our Talmon term matches the form tan γ = −(r/√θ) ∂η/∂n with r = 1/A_s. Flume calibrations put r
at 0.3–0.6 (Redolfi et al. 2020 used 0.40), so the default A_s = 1 (r = 1) is strong.

With r = 0.4, the S-bend check (`bends_scour_their_outer_banks`) lost its consistent helical
signal: the outward shift was −0.4 to −1.7 texels at Helical Strength 8–40 over 25 s. The bend
cross-sections are only 16 texels wide, and with weak gravitational smoothing other bedforms
dominate them. The default stays at r = 1, and the slider covers the literature range.

## River brush (`inspect_river_brush`)

**Setup:** a river placed with the brush, carved, then run for 20 s with live bed load and full
armour. Two terrains: the default eroded terrain from its highest corner (283 texels, 11 %
slope), and a winding valley at 0.8 % (`TEST_RIVER_TERRAIN=valley`, 341 texels). The test
writes the long profile, cross-sections and heights; steps are counted lip to pool on the
detrended carved bed.

| Measure | Literature | Model |
|---|---|---|
| Step height H, spacing L, slope S | H/L ≈ 1–2 × S (Abrahams et al. 1995) | H/L/S 1.30 (H 1.4 cells, L 9.8 texels) |
| Step spacing | 1–4 channel widths, closer when steeper (Chin 1999) | 1.0 widths at 11 % slope |
| Riffle-pool spacing | 5–7 channel widths (Leopold et al. 1964) | 73 texels, about 6.5 widths |
| Pool depth | about twice the riffles' | 1.8 × |
| Flow over steps and pools | supercritical over steps, subcritical in pools | Froude 0.3 in pools to 1.7 over steps |
| Lowland flow | Froude about 0.1–0.4 | 0.1 mean, 0.37 max |
| Flow kept in the channel | at or below bankfull | 1–2 % of wet cells outside the channel |
| Bends | cut bank outside, point bar inside | thalweg toward the outer bank, gently sloping bar inside |

- The first version was too smooth for its slope (H/L/S 0.84, steps 2.3 widths apart) and
  carved twice the flow's depth, so the banks stood far above the water. Step height now
  follows the channel depth and spacing follows H / (1.5 S); the lips sit 0.7 of the channel
  depth below the banks, so the channel stays continuous.
- The spring feeds half the design (bankfull) discharge. At the full discharge, a third of the
  wet cells were outside the channel: the water spilled over the lips and ran down beside it.
- Routes on the default terrain have sinuosity 1.01–1.2, typical of channels this steep; it
  slopes 5–25 % everywhere, so every river there is a step-pool torrent.
- With armour 1 the bed is unchanged after the flow; with armour 0, live bed load and water
  erosion take its step relief from 0.46 to 0.19 cells in 15 s (`river_brush_keeps_its_steps`).

## Not yet validated

- **Point bars at equilibrium** (Struiksma et al. 1985; Talmon et al. 1995): the transverse bed
  slope against h/R.
- **Mouth bar distance against jet momentum** (Edmonds & Slingerland 2007).
- **Step wavelength** against Taki & Parker's measurements (Part 1 is still unread).
- **Bar counts against Crosato & Mosselman (2009).**

The open-access copies of these papers could not be downloaded:

| Paper | Link |
|---|---|
| Taki & Parker 2005 | https://doi.org/10.1080/00221680509500147 |
| Colombini et al. 1987 | https://doi.org/10.1017/S0022112087002064 |
| Talmon et al. 1995 | https://doi.org/10.1080/00221689509498657 |
| Struiksma et al. 1985 | https://doi.org/10.1080/00221688509499377 |
| Crosato & Mosselman 2009 | https://doi.org/10.1029/2008WR007242 |
| Edmonds & Slingerland 2007 | https://doi.org/10.1029/2006JF000574 |
| Ikeda 1984 | https://doi.org/10.1061/(ASCE)0733-9429(1984)110:4(371) |

## Sources read

- N. J. Balmforth & A. Vakil (2012), *Cyclic steps and roll waves in shallow water flow over an
  erodible bed*, J. Fluid Mech. 695, 35–62
  ([author copy](https://personal.math.ubc.ca/~njb/Research/step.pdf)).
- L. B. Leopold & T. Maddock (1953), *The hydraulic geometry of stream channels and some
  physiographic implications*, USGS Professional Paper 252
  ([scan](https://www.arlis.org/docs/vol1/AK_Heritage/USGS/ProfPaper/2/ProfPaper-252.pdf)).
- M. Redolfi et al. (2020), *Morphometric properties of alternate bars and water discharge: a
  laboratory investigation*, Earth Surf. Dynam. 8, 789–808
  ([open access](https://esurf.copernicus.org/articles/8/789/2020/esurf-8-789-2020.pdf)).
- M. Wong & G. Parker (2006), *Reanalysis and correction of bed-load relation of Meyer-Peter and
  Müller using their own database*, J. Hydraul. Eng. 132(11)
  ([copy](https://documents.epfl.ch/users/b/bl/blanckae/www/HydrauliqueFluviale/Papers/2006_JHE_Wong_Parker.pdf)).
- A. D. Abrahams, G. Li & J. F. Atkinson (1995), *Step-pool streams: adjustment to maximum flow
  resistance*, Water Resour. Res. 31(10), 2593–2602 (cited, not read here).
- A. Chin (1999), *The morphologic structure of step-pools in mountain streams*,
  Geomorphology 27, 191–204 (cited, not read here).
- L. B. Leopold, M. G. Wolman & J. P. Miller (1964), *Fluvial Processes in Geomorphology*
  (cited, not read here).
- G. Parker & N. Izumi (2000), *Purely erosional cyclic and solitary steps created by flow over
  a cohesive bed*, J. Fluid Mech. 419, 203–238 (https://doi.org/10.1017/S0022112000001403).
- T. Sun & G. Parker (2005), *Transportational cyclic steps created by flow over an erodible
  bed. Part 2. Theory and numerical simulation*, J. Hydraul. Res. 43(5)
  (https://doi.org/10.1080/00221680509500148).
- Downloaded but not yet read: *Channel confinement mollifies roll wave instabilities*
  ([arXiv 2609.23108](https://arxiv.org/pdf/2609.23108)).
