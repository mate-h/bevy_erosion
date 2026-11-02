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
| Cyclic steps, wavelength | set by eddy viscosity in shallow-water theory (k = √(3/ν)); recurrent isolated steps in flumes (Taki & Parker 2005) | 10–15 texels, 34–115 depths, 0.7–1.2 cells relief | plausible; set by numerical diffusion |
| Roll waves, onset | Jeffreys 1925 / Vedernikov: Fr > 1.5 for Manning (2 for Chézy); convective, so hard to excite in short flumes | depth unevenness decays downstream up to Fr 2.15; grows only at Fr ≈ 5 | onset 2–3× too high (numerical damping) |
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

| Fr | 0.86 | 1.14 | 1.55 | 1.82 | 2.15 | 2.67 | 3.30 | 4.16 | 5.24 |
|---|---|---|---|---|---|---|---|---|---|
| CV upstream | 0.004 | 0.014 | 0.032 | 0.049 | 0.059 | 0.108 | 0.050 | 0.073 | 0.078 |
| CV downstream | 0.000 | 0.002 | 0.009 | 0.017 | 0.021 | 0.091 | 0.023 | 0.024 | 0.102 |

- Theory puts roll-wave instability at Fr > 1.5 for Manning friction. In the model, unevenness
  decays downstream up to Fr 2.15 and grows only near Fr 5 (2.67 is close to even).
- Balmforth & Vakil (2012) show that eddy viscosity cuts off roll-wave growth. The solver's
  first-order upwind depth flux acts as a numerical viscosity of about u·Δx/2. This is why roll
  waves show only on steep slopes in the demo.
- Roll waves are convective, so a 256-texel flume with a quiet inlet also limits them. Taki &
  Parker's flume saw none.
- Lowering the onset would need a higher-order depth flux in bevy_fft's solver. That trades
  against the shock-capturing robustness the water needs.

## Cyclic steps (`validate_cyclic_steps`)

**Setup:** a 6-texel erodible flume with a strong spring and live bed load for 25 s.

| Slope (cells) | 0.1 | 0.2 | 0.4 | 0.6 | 0.9 | 1.3 |
|---|---|---|---|---|---|---|
| Fr | 0.70 | 0.92 | 1.20 | 1.30 | 1.18 | 1.44 |
| Step wavelength (texels) | – | – | – | 9.7 | 14.2 | 15.3 |
| Relief (cells) | 0.02 | 0.04 | 0.24 | 0.71 | 1.06 | 1.22 |

- Relief jumps once the flow is supercritical, as the Parker & Izumi instability requires
  (Fr > 1).
- In shallow-water theory the most unstable wavelength is set by the eddy viscosity. Without
  one, the short waves grow fastest. Ours has none beyond numerical diffusion, so the 10–15
  texel spacing is a property of the grid and scheme, not of the sediment.
- Taki & Parker observed recurrent steps migrating upstream; migration is not measured yet.

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

## Not yet validated

- **Point bars at equilibrium** (Struiksma et al. 1985; Talmon et al. 1995): the transverse bed
  slope against h/R.
- **Mouth bar distance against jet momentum** (Edmonds & Slingerland 2007).
- **Step migration speed**, and step wavelength against Taki & Parker's measurements.
- **Bar counts against Crosato & Mosselman (2009).**

The open-access copies of these papers could not be downloaded:

| Paper | Link |
|---|---|
| Parker & Izumi 2000 | [JFM 419](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/purely-erosional-cyclic-and-solitary-steps-created-by-flow-over-a-cohesive-bed/F9A9D022A58B15539D7432ED5D11A489) |
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
- Downloaded but not yet read: *Channel confinement mollifies roll wave instabilities*
  ([arXiv 2609.23108](https://arxiv.org/pdf/2609.23108)).
