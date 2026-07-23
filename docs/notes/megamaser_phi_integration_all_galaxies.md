# All-galaxy GPU validation of peak-partition integration

Date: 2026-07-22  
Hardware: Glamdring `cmbgpu`, NVIDIA RTX 3090, one GPU per validation or
batching job  
Scope: all six configured megamaser galaxies; circular and
eccentric-plus-quadratic-warp objectives; complete conditional-`r_ang`,
marginalized-`phi` DE likelihood

This is the all-galaxy follow-up to
`docs/notes/megamaser_phi_integration_research.md`.  That report derives the
value-only peak-partition algorithm and records the first 62 research loops.
The present campaign tests transfer, pathological proposals, non-circular
motion, and GPU execution policy.  All accuracy numbers below compare the
complete production likelihood against the same independent float64
full-support `(r, phi)` reference ladder.

## Questions and decision rules

The campaign was designed to answer four separate questions.

1. Does the final v6 peak-partition method transfer to every configured
   galaxy, rather than only NGC6323, NGC6264, and NGC4258?
2. Does it remain finite and bounded-cost at very poor proposals for which an
   independent tensor reference itself cannot converge?
3. Does the numerical value-only construction survive simultaneous
   eccentricity and quadratic warp, where the circular harmonic root bound
   and any apparent `phi -> -phi` symmetry are unavailable?
4. Which static scan, candidate-wave, and spot-batch sizes maximize sustained
   RTX 3090 throughput without spending validated accuracy margin?

The hard production gates are absolute total log-likelihood error <= 0.1,
worst spot <= 0.01, p99 spot <= 0.005, RMS spot <= 0.001, identical finite
masks, zero ranking inversions, and zero root-capacity overflows.  A reference
is called converged only when its required final consecutive transitions pass
total <= 0.01, worst spot <= 0.001, and RMS <= 0.0001.  Ordinary reference
levels are `5001x2501`, `10001x5001`, and `20001x10001` in `(r, phi)`;
NGC4258 uses radial levels `20001`, `40001`, `80001`, and `160001` at
`50001` phi nodes, with the same final-two-transition requirement.

Broad Sobol proposals are always evaluated completely.  A proposal can be
excluded from an accuracy verdict only after both production and reference
scores place it below the calibrated posterior-deficit gate and its geometry
rails.  It is still required to produce finite values.  Anchor and local-cloud
points are never excused.  This distinction is essential: an unconverged
tensor reference cannot be used to declare either production method accurate
or inaccurate.

## Iterative physics and GPU plan

The follow-up used the following decision loops.  Each row represents a
distinct hypothesis test; galaxies within a row were submitted as independent
GPU jobs when possible.

| Loop | Experiment | Purpose | Decision at this stage |
|---:|---|---|---|
| 1 | Strict current-policy circular run, all six galaxies | Transfer and pathological Sobol baseline | All production values finite; five ordinary cases pass; NGC4258 requires its separately converged relevant-fit interpretation |
| 2 | 97/49 scan, five ordinary galaxies | First cheaper angular frontier | Faster everywhere; retain only where error margins are unchanged |
| 3 | 65/33 scan, five ordinary galaxies | Aggressive angular frontier | Reject globally: NGC5765b and NGC6323 fail; useful only as a per-galaxy bound |
| 4 | Candidate wave 4 versus production wave 8 | GPU candidate parallelism | Wave 8 wins on all six galaxies with identical numerical policy |
| 5 | Sustained 64/128-candidate spot-batch suites | GPU occupancy, fusion, and memory | All-spots is best for most float32 galaxies; NGC6264 and NGC4258 need separate decisions |
| 6 | Eccentric plus quadratic warp, all six galaxies | Remove reliance on circular formulas | Five ordinary broad suites pass; all values finite, maximum observed roots three, zero overflows |
| 7 | 97/49 non-circular scan, plus NGC4258 HV 49 | Transfer the scan frontier | Ordinary values stable; NGC4258 HV reduction does not move its systemic residual |
| 8 | Nonzero-eccentricity local clouds | Probe neighbourhoods rather than only zero-e anchors | Finds radial-conditioning cases in NGC5765b and UGC3789 and a systemic NGC4258 case |
| 9 | 65/33 non-circular local clouds | Establish the low scan boundary | CGCG passes; NGC6264 loses substantial margin; UGC failure is unchanged and therefore radial |
| 10 | Scan-envelope width, radial refinement, 384 local nodes, forced float64 | Separate coverage, centring, resolution, and precision | Float64 and angular work are not remedies; value-only radial hypotheses are tested independently |
| 11 | NGC4258 all-class refinement and systemic 513 scan | Find one circular/eccentric-safe NGC4258 profile | Relevant Pesce/checkpoint tests only; unconverged catastrophic local needles are not tuning targets |
| 12 | Anchor-stable exact local reruns | Remove local-cloud seed-order confounding | Local seeds are now keyed by anchor name, independent of omitted anchors |
| 13 | Radial parity/density and envelope frontier | Check 257/384/512 nodes and width 25/50/100 | Retain only a setting that improves the exact failing point and preserves anchors |
| 14 | Width-by-resolution factorial combination | Test support and resolution jointly | Final radial stress decision |
| 15 | NGC4258 513-node systemic scan plus all-class radial refinement | Test the two individually insufficient changes jointly | Passes the converged eccentric Pesce point |
| 16 | NGC4258 circular Pesce plus independent checkpoint | Guard against improving eccentricity by moving the circular objective | Both pass with the selected profile and batch 32 |
| 17 | First final circular/eccentric matrix, all six galaxies | Exercise config, Pesce, two local clouds, and four broad Sobol points | Exposes an additional converged circular NGC5765b radial stress point |
| 18 | Sustained legacy fixed-grid controls | Preserve a matched GPU denominator before changing config | Deterministic seed-123 Sobol rates recorded for all galaxies |
| 19 | NGC5765b 320-node two-variant audit | Test the cheapest eccentric-passing radial profile | Reject: circular local-Pesce worst spot is 0.0208 |
| 20 | NGC5765b 384/512-node audit | Bracket the circular radial failure | 384 narrowly fails; 512 passes peak partition |
| 21 | UGC3789 512-node width-50 audit | Ask whether local density fixes the second local cloud | Reject: the same systemic spot is unchanged |
| 22 | NGC5765b 416/448 intermediate grids | Find the density threshold | 416 fails non-monotonically; 448 passes, identifying node resonance |
| 23 | NGC5765b centred 289/321/385 grids | Put an exact node at the fitted radial centre | 289 and 385 fail; 321 passes both orbit classes |
| 24 | One-pass value-only radial refinement | Seek a cheaper replacement for 321 local nodes | Reject: it degrades circular Pesce and local-Pesce anchors |
| 25 | UGC3789 40001x20001 reference extension | Resolve the nearly converged local-Pesce reference | Reference converges; both production methods share a 0.034 systemic miss |
| 26 | UGC3789 256-node global-radius discovery | Test for an unresolved disconnected radial mode | Reject: the converged outlier is unchanged |
| 27 | NGC5765b selected 321-node full matrix | Recheck broad pathological proposals at the chosen shape | All production values finite; all converged config/Pesce/local points pass |
| 28 | Checked-in profile and full regression suite | Verify config merge, objective policy, parser, and kernels | Pass |
| 29 | Matched sustained selected-profile benchmark | Measure production GPU speed and memory versus legacy fixed grid | 7.47--23.46x faster, lower measured JAX peak memory for all six |

The predecessor report's 62 loops and these 29 follow-up loops give 91
explicit research/decision loops.  This count excludes pure cache reruns,
formatting tests, and duplicated timing repetitions.

## Circular all-galaxy baseline

The current 129/65 ordinary kernel and NGC4258's 385/65 kernel were compared
with the full legacy fixed grids before any new tuning.

| Galaxy | Fixed throughput | Peak throughput | Six-point block speedup | Largest relevant peak total error | Largest relevant peak worst-spot error | Verdict |
|---|---:|---:|---:|---:|---:|---|
| CGCG074-064 | 46.67/s | 341.91/s | 7.33x | 0.01455 | 0.000281 | pass |
| NGC5765b | 41.16/s | 323.62/s | 7.86x | 0.02483 | 0.006696 | pass |
| NGC6264 | 90.82/s | 792.95/s | 8.73x | 0.005432 | 0.000617 | pass |
| NGC6323 | 88.65/s | 728.33/s | 8.22x | 0.01134 | 0.000700 | pass |
| UGC3789 | 36.59/s | 351.46/s | 9.60x | 0.01492 | 0.000415 | pass |
| NGC4258 | 0.640/s | 17.872/s | 27.94x | 0.03833 | 0.002760 | relevant-fit pass; broad/config reference caveat |

The NGC4258 row uses its converged Pesce/Reid point for the accuracy columns.
Its configured geometry and broad Sobol points are needle-like enough that the
three-level tensor ladder does not converge; both production methods remain
finite there.  A separate final-transition audit and the independent DE
checkpoint establish the relevant-fit result, as documented in the
predecessor report.

Across all 36 circular baseline candidates, every peak value is finite, no
root buffer overflows, and the maximum observed stationary-root count in any
half-plane is three.  Four pathological Sobol points in each ordinary case are
evaluated and then classified posterior-irrelevant where appropriate.  The
NGC4258 tail-three report deliberately remains a formal failure because it has
no converged anchor under that stronger historical-transition requirement;
the tail-two relevant-fit rerun passes.

## Angular scan frontier

The scan sweep deliberately changes only the number of value samples used to
find extrema.  Root refinement, peak/tail quadrature, radial nodes, precision,
and GPU candidate wave remain fixed.

- 97/49 improves ordinary circular throughput by about 4--12%.  CGCG,
  NGC5765b, NGC6264, and UGC3789 retain their anchors.  NGC6323's converged
  bad Sobol point loses margin, from 0.000700 to 0.003236 worst-spot error, so
  NGC6323 keeps 129/65.
- 65/33 is not a safe global setting.  NGC5765b fails at 0.01048 worst-spot
  error and NGC6323 fails catastrophically at 0.4228.  CGCG passes but its
  circular config worst error rises from 0.000281 to 0.001007.  NGC6264's
  non-circular local-config worst error rises from 0.000452 at 129/65 to
  0.006325 at 65/33.
- NGC4258's eccentric Pesce result is unchanged when the HV scan is reduced
  from 65 to 49.  Its residual is in systemic spots, so spending or removing
  HV nodes cannot address it.

These results also demonstrate why a single globally minimal scan is the
wrong optimisation target.  Static per-galaxy scans preserve XLA efficiency
without forcing NGC6323's failure boundary onto every galaxy.

## Non-circular transfer and pathological behaviour

The combined eccentric-plus-quadratic-warp variant exercises the residual-
stable eccentric velocity path, radius-dependent disk orientation, and root
capacity eight.  No analytic circular root formula is used.

| Galaxy | Fixed throughput | Peak throughput | Speedup | Broad-suite result |
|---|---:|---:|---:|---|
| CGCG074-064 | 17.81/s | 92.98/s | 5.22x | pass |
| NGC5765b | 15.56/s | 90.59/s | 5.82x | pass |
| NGC6264 | 34.57/s | 243.18/s | 7.04x | pass |
| NGC6323 | 34.25/s | 232.71/s | 6.79x | pass |
| UGC3789 | 14.13/s | 103.83/s | 7.35x | pass |
| NGC4258 | 0.112/s | 3.320/s | 29.63x | finite; relevant eccentric Pesce systemic residual under refinement |

For the five ordinary galaxies every broad candidate is finite, no root or
node buffer overflows, and the maximum observed root count remains three even
though capacity eight is available.  The bad broad Sobol reference in each
case is unconverged and far below the anchors; this is reported, not hidden.

The first nonzero-e local-cloud pass found two useful radial stress cases.
NGC5765b's eccentric local-Pesce point is about 5,875 nats below Pesce and has
a converged reference.  Fixed grid misses one systemic spot by 0.0548; the
original peak profile is much closer but narrowly fails its distributional
gates.  A later two-variant matrix finds a different, also-converged circular
local-Pesce outlier.  This turns out to be a radial-node resonance: 320 passes
eccentric but fails circular, 384 narrowly fails circular at 0.01024 worst
spot, 416 fails badly, and 448/512 pass.  Centred odd grids establish a much
cheaper boundary: 289 and 385 fail, while 321 passes every circular and
eccentric config/Pesce/local candidate.  Its limiting eccentric local-Pesce
errors are 0.00207 total, 0.00482 worst, 0.00434 p99, and 0.000789 RMS.
Forced float64 and a one-pass value-only centre refinement do not help, so the
static centred 321-node profile is selected.

UGC3789's eccentric local-config point is about 677 nats below Pesce.  Fixed
grid misses one systemic spot by 0.0325.  The combination of 384 local nodes
and a scan-derived width at `Delta logL=50` reduces peak partition's error to
0.00689 total and 0.00741 worst spot, so peak passes where fixed fails.  A
second local-Pesce point initially has an almost-converged reference.  Adding
a `40001x20001` level makes the last transition converge at `8.23e-7` worst
spot and confirms a common fixed/peak miss of approximately 0.034 in one
systemic spot.  Neither 512 local nodes, odd parity, nor 256 global discovery
nodes moves it.  It is a genuine bad-fit limitation of the shared
conditional-radius construction, not a peak-partition or angular-root
regression; both methods remain finite and differ by only about `5e-5` at the
outlying spot.

At NGC4258, the eccentric-plus-warp Pesce point has `e=0.009` and a converged
tail-two reference.  The former 385/65, HV-only-refined peak profile misses
systemic spots 109/111, with 0.03156 worst-spot error.  Neither a 513-node
systemic scan nor all-class refinement alone suffices.  Their combination,
with the existing three refinement passes and eight-step width solve, passes:
0.03448 total, 0.00176 worst spot, 0.000093 p99, and 0.000130 RMS.  The same
513/65 profile passes the circular Pesce and independent-DE checkpoint at
about 0.0327 total and `9.4e-5` worst spot.  A catastrophic local neighbour
reaches roughly -978,000 in production log likelihood; its finest reference
transition still changes by 1,822 total and 269 at the worst spot.  Both
production methods nevertheless return finite values and peak partition
records zero overflow nodes.

## GPU execution frontier

All production kernels have static XLA shapes.  Peak partition evaluates
eight DE candidates per GPU wave; fixed grid uses one.  Reducing peak's wave
to four is slower on every galaxy:

| Galaxy | Wave 8 throughput | Wave 4 throughput | Wave-8 gain |
|---|---:|---:|---:|
| CGCG074-064 | 341.91/s | 322.96/s | 5.9% |
| NGC5765b | 323.62/s | 296.56/s | 9.1% |
| NGC6264 | 792.95/s | 674.78/s | 17.5% |
| NGC6323 | 728.33/s | 649.93/s | 12.1% |
| UGC3789 | 351.46/s | 330.43/s | 6.4% |
| NGC4258 | 17.87/s | 16.95/s | 5.4% |

The NGC4258 sustained 64-candidate suite gives the clearest batching/memory
frontier:

| Spot batch | Sustained throughput | JAX peak allocation |
|---:|---:|---:|
| 16 | 23.90/s | 1.196 GiB |
| 32 | 24.35/s | 1.176 GiB |
| 64 | 23.95/s | 2.061 GiB |
| 96 | 24.07/s | 3.081 GiB |
| 128 | 19.87/s | 4.182 GiB |
| all | 24.87/s | 6.299 GiB |

Batch 32 is preferred to the marginally faster all-spot result: the latter's
first compile took 111 s and uses over five times the measured JAX memory for
only about 2% more steady throughput.  Batch 32 is consequently safer for the
future eccentric kernel and for shared GPU nodes.  Across 16/32/64/96/128/all
tilings, finite masks are identical and maximum f64 objective differences are
at roundoff scale (up to `3.73e-9`).

For 128-candidate float32 peak workloads, all-spots is faster than batch 32 by
about 16--28% for CGCG074-064, NGC5765b, NGC6323, and UGC3789.  NGC6264 is the
exception: repeated batch-32 runs reach about 1,725 candidates/s versus 1,673/s
for all-spots, with identical objective hashes.  Different float32 tilings can
change pathological total log likelihoods through reduction order; repeated
identical tilings are bitwise stable, finite masks match, and the largest
observed absolute difference is 0.03125 on objectives as large as `1e8`.

Sampled `nvidia-smi` traces see all physical cards on `gpu07` and can include
other jobs.  Per-process JAX `peak_bytes_in_use` is therefore the memory number
used for settings decisions.  The batching benchmark now reports the actual
partition scan geometry rather than the unused legacy dense-phi arrays.

## Final converged-fit accuracy matrix

The table below takes the maximum over every circular and
eccentric-plus-quadratic-warp config, Pesce/Reid, local-config, and
local-Pesce candidate whose independent reference converged.  It therefore
does not convert unconverged broad-Sobol tensors into accuracy claims.

| Galaxy | Converged fit-neighbourhood candidates | Peak max total | Peak max worst spot | Peak max p99 | Peak max RMS | Result |
|---|---:|---:|---:|---:|---:|---|
| CGCG074-064 | 8 | 0.01883 | 0.000691 | 0.000361 | 0.000138 | pass |
| NGC5765b | 8 | 0.01148 | 0.004817 | 0.004337 | 0.000789 | pass |
| NGC6264 | 8 | 0.01334 | 0.002652 | 0.002243 | 0.000577 | pass |
| NGC6323 | 8 | 0.007858 | 0.000336 | 0.000288 | 0.000127 | pass |
| UGC3789 | 7 | 0.007711 | 0.007408 | 0.000367 | 0.000599 | pass; separate shared-radial exception below |
| NGC4258 | 3 relevant anchors | 0.03448 | 0.001756 | 0.000093 | 0.000130 | pass |

The UGC3789 count excludes only the fine-reference local-Pesce point described
above.  On that point the reference is now proven converged but both production
methods fail: fixed has 0.03080 total, 0.033948 worst, 0.005602 p99, and
0.002887 RMS; peak has 0.03827, 0.033994, 0.005640, and 0.002891.  The nearly
identical outlier is retained as a documented conditional-radius limitation.
Peak partition does not turn it into a non-finite score and does not introduce
an angular or root-capacity failure.

Every production value in the selected-profile matrices is finite.  The
maximum observed stationary-root count is three, circular capacity is four,
eccentric capacity is eight, and the total number of overflow spots and
overflow nodes is zero.  These statements include all broad Sobol proposals,
not only candidates admitted to the accuracy verdict.

Several broad proposals remain deliberately formal failures because their
`20001x10001` reference is not converged.  Some rail strongly enough and lie
far enough below the calibrated posterior gate to be marked irrelevant; some
do not meet both conditions and remain unclassified failures.  In both cases
production is finite and bounded-cost.  The reports preserve the raw numbers,
reference transitions, railing fractions, and classifications.

## Checked-in production settings

`config_maser.toml` now selects peak partition globally.  The legacy fixed
grid remains available through `--phi-integration fixed-grid` for diagnostics.

| Galaxy | Precision | Systemic/HV scan | Local/global r nodes | Extra radial policy | DE spot batch | Candidate wave |
|---|---|---:|---:|---|---:|---:|
| CGCG074-064 | f32 | 97/49 | 256/128 | none | all | 8 |
| NGC5765b | f32 | 97/49 | 321/128 | centred odd local grid | all | 8 |
| NGC6264 | f32 | 129/65 | 256/128 | none | 32 | 8 |
| NGC6323 | f32 | 129/65 | 256/128 | none | all | 8 |
| UGC3789 | f32 | 97/49 | 384/128 | scan-width drop 50 | all | 8 |
| NGC4258 | f64 | 513/65 | 256/128 | 3x7-point refinement for all classes; 8-step width solve | 32 | 8 |

All profiles retain full-physical-support global discovery, independent
left/right local spans, four/eight circular/eccentric root capacity, root
refinement order seven with four value-only iterations, 24-point cores,
8-point tails, and the finite scan-trapezoid overflow fallback.  No selected
profile assumes a circular orbit or uses analytic peak locations.

The scan choices are intentionally per galaxy.  A single 65/33 policy is
cheaper but fails NGC5765b and NGC6323.  NGC6323 in particular retains 129/65
because 97/49 degrades a converged poor proposal.  Static per-galaxy shapes
still compile efficiently and avoid data-dependent GPU control flow.

## Matched sustained GPU speed and memory

The final benchmark uses deterministic scrambled Sobol coordinates with seed
123 for both methods.  The five f32 cases use 128 candidates; NGC4258 uses 64
f64 candidates.  Each method/tiling runs in a fresh child process with three
warmups and three timed repeats on one RTX 3090.  Fixed grid uses one candidate
per GPU wave; selected peak partition uses eight.  The rate is the best
repeated tiling's median warmed throughput.

| Galaxy | Legacy fixed | Selected peak | Speedup | Fixed JAX peak | Peak JAX peak |
|---|---:|---:|---:|---:|---:|
| CGCG074-064 | 70.40/s | 666.89/s | 9.47x | 0.679 GiB | 0.251 GiB |
| NGC5765b | 61.99/s | 463.27/s | 7.47x | 0.755 GiB | 0.251 GiB |
| NGC6264 | 139.32/s | 1707.73/s | 12.26x | 0.306 GiB | 0.126 GiB |
| NGC6323 | 135.16/s | 1603.14/s | 11.86x | 0.345 GiB | 0.126 GiB |
| UGC3789 | 56.62/s | 544.02/s | 9.61x | 1.001 GiB | 0.305 GiB |
| NGC4258 | 0.8155/s | 19.132/s | 23.46x | 1.844 GiB | 1.509 GiB |

The geometric-mean speedup is 11.51x across all six galaxies and 9.98x over
the five ordinary f32 galaxies.  Measured JAX peak allocation is lower for
peak partition in every row.  The NGC4258 selected profile is slower than its
earlier 385/65 HV-only research profile because the extra systemic scan and
all-class refinement are required by the eccentric stress point; it still
delivers 23.46x over the old fixed configuration.

Both repeated tilings produce identical objective SHA-256 digests for every
galaxy, zero finite-mask mismatches, and zero cross-tiling numerical
difference.  All 704 benchmark objective values are finite, including poor
Sobol proposals.  Candidate sources explicitly report the same seed-123 Sobol
construction; incompatible historical peak checkpoints fall back rather than
mixing stored fitness from an old numerical policy.

Cold compilation is not included in the speedup.  It is shape- and persistent-
cache-dependent and is amortized over a DE run.  The final peak first warmup is
about 7--12 s for the f32 cases.  NGC4258's first fresh suite child takes about
90 s, while the second child using the persistent cache takes about 23 s;
warmed 64-candidate passes take 3.35 s.  This is why batch 32 is retained even
though an all-spot NGC4258 experiment was about 2% faster: all-spots used
6.30 GiB and had a 111 s first compile, versus the safer selected profile's
1.51 GiB measured peak.

## GPU design conclusions

The speedup is not only a reduction in scalar node count.  The production
shape is designed around the accelerator:

1. Two systemic half-planes and each HV half-plane are scanned in fixed-size
   batches; no candidate-dependent array shapes are created.
2. Eight DE candidates are fused per device wave.  Wave eight is 5--18%
   faster than wave four on the six-galaxy frontier.
3. Radius-only physics is hoisted, all global radii are scanned concurrently,
   and their phi marginals are reused exactly in the final local/global union.
4. Root, radial-centre, and width refinements use fixed-count value-only JAX
   loops.  They remain valid for eccentric velocities and quadratic warps.
5. Float32 targets use all spots where measured fastest.  NGC6264 and the f64
   NGC4258 target use 32-spot tiles because that wins or nearly ties throughput
   with much lower compilation/memory risk.
6. Capacity overflow reuses the already-computed scan trapezoid, so a bad DE
   proposal remains finite without an extra physics call.  The overflow is
   still surfaced to the validator.

The batching benchmark was extended to select the integrator and candidate
wave explicitly, reject illegal fixed-grid waves, validate checkpoint policy,
return objective values, compare finite masks and values across tilings, and
report method-aware memory geometry.  Peak systemic memory geometry includes
both half-planes; local and global radial scans are reported as separate
passes rather than as a nonexistent dense-phi union.

## Software verification

The focused test command completed with exit code zero:

```bash
PYTHONPATH=$PWD PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_PLATFORMS=cpu \
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
venv_candel/bin/python -m pytest -q \
  tests/test_megamaser_phi_partition.py \
  tests/test_megamaser_phi_validation.py \
  tests/test_megamaser_de_lshade.py
```

The test coverage includes peak integration and overflow fallback, radial
support/refinement controls, circular/eccentric root capacities, validator
parsing/cache identity/stable local seeds, benchmark method/wave/checkpoint and
memory geometry, DE candidate-wave and spot-batch policy, and the checked-in
per-galaxy settings.  Python byte-compilation and `git diff --check` also pass.

## Reproducibility and artifacts

The principal machine-readable reports are:

- `results/Megamaser/convergence/all_galaxies_20260722/loop20_final_CGCG074-064/validation.json`
- `results/Megamaser/convergence/all_galaxies_20260722/loop28_final_NGC5765b_r321/validation.json`
- `results/Megamaser/convergence/all_galaxies_20260722/loop20_final_NGC6264/validation.json`
- `results/Megamaser/convergence/all_galaxies_20260722/loop20_final_NGC6323/validation.json`
- `results/Megamaser/convergence/all_galaxies_20260722/loop20_final_UGC3789/validation.json`
- `results/Megamaser/convergence/all_galaxies_20260722/loop24_UGC3789_ecc_local_fine_reference/validation.json`
- `results/Megamaser/convergence/all_galaxies_20260722/loop16_ngc4258_ecc_sys513_refine_all/validation.json`
- `results/Megamaser/convergence/all_galaxies_20260722/loop19_ngc4258_circular_selected_anchors/validation.json`

The strict baseline jobs were 794279--794284; final ordinary matrices were
794388--794393 plus selected NGC5765b job 794440.  NGC4258 selected-profile
jobs were 794382 and 794387.  The extended UGC3789 reference was job 794416.
The rejected 256-node UGC3789 global-discovery check was job 794441.
Legacy fixed sustained controls were jobs 794350 and 794359--794363; selected
peak sustained benchmarks were jobs 794442--794447.  Slurm stdout retains the
full benchmark JSON, candidate source, digests, JAX memory, warmups, and timed
repeats.

## Final recommendation

Use the checked-in peak-partition profiles and candidate wave eight for DE.
Keep fixed grid only as an explicit diagnostic.  These settings preserve the
validated circular and eccentric-plus-warp fit neighbourhood, improve one
converged UGC3789 bad-fit point that fixed grid misses, stay finite on every
tested pathological proposal, reduce measured GPU memory, and provide
7.47--23.46x sustained speedup over the old fixed-grid configuration.

The remaining numerical limitation is explicit rather than hidden: one
converged UGC3789 local-Pesce point is missed by the shared conditional-radius
construction in both methods, and extremely poor broad/NGC4258 needles can
defeat even the independent tensor reference.  Neither is evidence for a
peak-partition angular failure.  Future work on those cases should target a
multi-window or error-estimating radial marginal, not denser phi scans or
analytic circular peak finding.
