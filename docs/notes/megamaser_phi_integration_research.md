# Megamaser DE integration research: robust peak partition v6

Date: 2026-07-21/22  
Hardware: Glamdring `cmbgpu`, one RTX 3090 per validation job and three for the final DE reconnaissance  
Scope: complete conditional-`r_ang`, marginalized-`phi` DE objective

## Executive conclusion

The peak-partition design remains the best production architecture. Its
largest observed failure was not a `phi` quadrature failure: the old
data-conditioned global-radius window could exclude a disconnected radial
mode at a terrible DE proposal. Scanning the same 128 log-radius nodes across
the full physical support fixed the only converged broad-box failure for
NGC6323: total error fell from **1316.5** to **0.0114** with no extra radius
nodes. Applying the same correction to fixed grid reduced its error from
**10.10** to **0.00683**. Independent left/right local-radius spans prevent a
seed close to one physical boundary from collapsing the other half of the
local grid.

For the circular cheap-galaxy workload, the accepted kernel uses 129 systemic
and 65 high-velocity scan nodes, four value-only root-refinement passes of
order seven, 24/8-point Gauss--Legendre core/tail rules, capacity four, and
eight DE candidates per GPU wave. The accepted no-stencil kernel reached
**770.9 candidates/s** on NGC6323: **8.74x** the contemporaneous
full fixed-grid objective and **2.38x** the original peak-partition baseline.
NGC6264 reached **735.6 candidates/s** and **9.25x** its full fixed-grid
objective. Peak memory did not increase relative to the fixed method in these
final runs (about 332--355 MiB as reported by JAX, depending on the run).

The method does not assume circular orbits. An eccentric and an
eccentric-plus-quadratic-warp NGC6323 test retained the value-only search and
an eight-root capacity. All converged config, Pesce/Reid, local-cloud, and one
converged Sobol comparison passed, with zero root overflows; the pre-stencil
eccentric throughput was **6.89--7.03x** fixed grid.

At NGC4258's deliberately poor configured point, a 129/65 scout was 43.5x
faster than its 60,001/20,001-point fixed grids; 257/129 and 513/257 agreed
with one another, isolating the remaining method difference as radial. The
independent dense reference was pushed to 20,001 x 200,001 nodes, but its last
refinement still changed by 0.0336 total, 0.0199 at the worst spot, and
0.00194 RMS, so it correctly did **not** certify. Both production methods miss
the same disconnected systemic needles there. This is a reference/irrelevant-
fit limitation, not evidence for selecting either production value.

The first radial-refinement family (4x7, 6x7, 5x9, odd local grids, and a
three-point curvature width) was deliberately rejected: it was non-monotone,
slowed NGC6323, and worsened the converged eccentric stress point.  Relevant-
fit population diagnostics then isolated the problem to a few NGC4258 red
spots.  Increasing the high-velocity scan from 65 through 257 changed their
errors by zero at reported precision, while shrinking the radial bracket made
them worse.  The failure was an ill-conditioned local curvature width, not an
angular peak or missing radial mode.

The accepted NGC4258 correction keeps three 7-point value-only narrowing
passes, obtains a guarded parabolic interpolation from the already evaluated
final triplet, and replaces the fragile curvature by a fixed-count two-sided
solve for `Delta logL = K_sigma^2/2`.  The measured feasibility boundary is
sharp: two width bisections fail with worst-spot errors of 0.0688 and 0.0855;
three pass but are pre-asymptotic; six through sixteen are on the stable
plateau.  The production choice uses eight steps and the cheaper 385-node
systemic scan.  Pesce/Reid then has **0.03833 total / 0.00276 worst /
0.000260 RMS** error and the independent DE checkpoint has **0.02953 /
0.000541 / 0.000119**.  Both pass every declared gate, red and blue errors
fall to approximately `9e-5` per spot, zero root buffers overflow, and warmed
batched throughput is **27.78x fixed grid** (**5.835 versus 0.210 candidates/s**).
The two methods reported the same 1.992 GB JAX peak allocation in that run.

To avoid judging NGC4258 at a negligible-likelihood geometry, a three-GPU,
unseeded L-SHADE reconnaissance was run with the then-current 129/65
objective. It converged at generation 440 to logP **-5685.51**, substantially above the
independently scored Pesce/Reid diagnostic (**-8458.80**), at a plausible disk
geometry rather than a support-railing proposal. The full independent
good-fit validation is recorded below.

At both the Pesce/Reid point and the independent DE checkpoint, the final
three-level dense-reference transition passes; an additional fourth level
shows that the last **two** transitions pass.  The fourth-level totals are
`-8396.47301414717` and `-5634.359962753176`, respectively.  The three arrays
shared between the pre- and post-vertex source hashes are bitwise identical,
so the fourth level is a valid extension of the final-source ladder rather
than a cross-code numerical comparison.  The final NGC4258 production
frontier and its strict errors are reported in the dedicated section below.

## Physics-first plan

The investigation was organised around the actual likelihood geometry, not
around a catalogue of generic quadrature rules:

1. Establish the unmodified fixed-grid/peak-partition accuracy and speed on
   NGC6323 and NGC6264 using production GPU execution.
2. Write the fixed-radius circular log-integrand in its trigonometric basis;
   identify which symmetries are real, and which apparent mirror symmetries
   are broken by an individual spot's measured position, velocity, and
   acceleration.
3. Build an independent float64 full-support `(r, phi)` reference ladder and
   require convergence of the reference itself before judging production.
4. Separate angular quadrature error, radial mode-discovery error, radial
   centre/width error, and ordinary floating-point/overflow failure with
   one-variable-at-a-time GPU experiments.
5. Preserve static shapes and bounded work so the routine remains suitable
   for DE candidate waves on a GPU.
6. Retain only value-based operations in the production search.  Circular
   harmonic structure may set a safe storage capacity, but it must not become
   an analytic root formula that fails for eccentric velocity fields.
7. Stress the method on eccentric and quadratic-warp variants and on
   posterior-irrelevant, extremely poor fits; a bad point should receive a
   finite bounded-cost score and be rejected by DE rather than crash or poison
   the population.
8. Run an unseeded NGC4258 DE reconnaissance, validate both its checkpoint and
   the Pesce/Reid point, and tune only against independently converged
   relevant-fit references.
9. Iterate until both accuracy gates and warmed throughput are satisfied, then
   rerun the chosen settings alone on `cmbgpu` and preserve every rejected
   branch in the report.

## The integral and the usable symmetry

For each observed spot the production target is, schematically,

\[
  \log L_i(\theta) = \log \int_{r_{\min}}^{r_{\max}} dr
      \int_{\Phi_i} d\phi\;\exp[-\chi_i^2(r,\phi;\theta)/2]
      + \log N_i(\theta).
\]

At fixed radius for a circular disk, predicted position is affine in
`sin(phi)` and `cos(phi)`, line-of-sight velocity is affine in `sin(phi)`, and
acceleration is affine in `cos(phi)`. Squaring Gaussian residuals makes the
log-integrand a degree-two trigonometric polynomial,

\[
  \ell(\phi)=c_0+c_s\sin\phi+c_c\cos\phi+c_{ss}\sin^2\phi
  +c_{cc}\cos^2\phi+c_{sc}\sin\phi\cos\phi.
\]

Its derivative is also degree two and therefore has at most four isolated
stationary roots on a complete circle (unless it is identically constant).
That justifies a structural capacity of four for circular half-plane scans;
the tested models used at most three. This is a capacity argument only: roots
are still found numerically from neighbouring values, as requested.

There is no general `phi -> -phi` folding symmetry for an observed spot. The
linear and mixed terms depend on its measured `(x,y,v,a)` and on the warped
orientation, so the two sides normally have different height and curvature.
Systemic support is therefore split into two independently searched
half-planes, but their values are not identified or mirrored.

For eccentric motion the velocity contains factors such as
`(1 + e_x cos(phi) + e_y sin(phi))**(-1/2)` together with the
radius-dependent periapsis rotation and relativistic terms. The degree-two
root bound and any mirror symmetry disappear. The same numerical value-only
algorithm is retained with capacity eight; no circular formula is substituted.

## Final algorithm

1. Build the same 256-node per-spot local radial grid and 128-node shared
   global radial grid used by the complete DE likelihood.
2. Spread the global nodes over the complete physical radius support. This
   does not add nodes or `phi` calls; it removes a dangerous assumption that
   data-derived circular seeds bracket every mode.
3. On each allowed `phi` half-plane, evaluate the 129/65 ordinary-galaxy scan,
   classify changes in adjacent value differences, and refine every bracket in
   parallel with static value-only stencils. NGC4258 uses 385 systemic and 65
   high-velocity nodes. Higher 513-node diagnostics established the systemic
   pass/fail frontier without becoming the production setting.
4. Partition at the numerical extrema. Integrate the peak-side core with a
   24-point Gauss--Legendre rule and its tail with an 8-point rule; find the
   `Delta log L=24` split with 12 fixed bisection steps.
5. Interpolate the coarse radial winner from three adjacent log-radius scan
   values.  Ordinary galaxies stop here.  NGC4258 applies three batched
   7-point value-only narrowing passes to red/blue groups only, then fits a
   guarded quadratic vertex to the final triplet.  This is numerical
   interpolation of evaluated likelihoods, not analytic peak finding.
6. For the NGC4258 red/blue groups, solve both local half-widths at
   `Delta logL=K_sigma^2/2` with a fixed-count, value-only bisection.  Use the
   wider side to guarantee coverage; retain the curvature/propagated width as
   a finite fallback.  The operation calls the same circular or eccentric
   marginal and has static XLA shapes.
7. Use independent left and right local-radius spans at physical boundaries,
   merge local and global nodes, and reuse the already evaluated global
   marginals in the final radial integral.
8. Evaluate eight DE candidates per GPU wave. All dimensions and loop counts
   remain static for XLA.
9. If the root buffer ever overflows, integrate the already available scan
   with trapezoidal weights. This costs no additional physics evaluations,
   keeps a bad DE proposal finite, and still exposes the overflow diagnostic;
   validation continues to require zero overflows.

The overflow fallback is intentionally a graceful degradation, not a claim
that a low-order scan is a high-accuracy reference. It prevents an
implementation capacity limit from manufacturing `-inf` and destroying DE
selection while preserving a hard diagnostic for posterior-relevant tests.
Together with the full-support radius scan, this gives a bad DE proposal a
finite, bounded-cost score and lets selection move away from it. It does not
pretend that any finite tensor grid can certify an isolated mode narrower than
its spacing: when the independent ladder itself fails to converge, the
validator reports that fact and withholds an accuracy verdict. The NGC4258
checkpoint experiment demonstrates the intended operational behaviour: the
optimiser escaped the pathological configured geometry and reached a smooth,
high-probability basin without a published-point seed.

## Validation protocol

The repository validator compares both complete production objectives against
the same independent float64 full-support two-dimensional reference. Default
cheap-galaxy reference levels are 5,001 x 2,501, 10,001 x 5,001, and
20,001 x 10,001 in `(r, phi)`. NGC4258 uses radial levels 20,001, 40,001,
80,001, and 160,001 at 50,001 phi nodes. Individual-spot tests of the former
offenders pass both final radial transitions after the paired ladder was shown
to over-resolve phi while remaining pre-asymptotic in radius; the full
all-spot GPU run remains the acceptance gate. The production acceptance
limits are:

- absolute total log-likelihood error <= 0.1;
- worst spot <= 0.01, p99 <= 0.005, RMS <= 0.001;
- identical finite masks, no ranking inversion, and no root overflow;
- the reference itself must converge to total <= 0.01, worst spot <= 0.001,
  and RMS <= 0.0001 across the required tail levels.

Broad Sobol points are retained to probe failure behaviour. A production
accuracy claim is made only when their independent dense reference converges.
An unconverged reference cannot decide which production value is closer.

## Task 1: baseline verification first

These were the unmodified production schemes submitted first on `cmbgpu`.

| Galaxy | SLURM job | Fixed throughput | Peak throughput | Steady speedup | Config total / worst error, fixed | Config total / worst error, peak | Pesce total / worst error, fixed | Pesce total / worst error, peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NGC6323 | 794000 | 86.96/s | 324.15/s | 3.728x | 0.00582 / 0.000157 | 0.000236 / 0.00316 | 0.00622 / 0.000167 | 0.00603 / 0.000206 |
| NGC6264 | 794001 | 92.45/s | 333.83/s | 3.611x | 0.00611 / 0.000158 | 0.00573 / 0.000405 | 0.00644 / 0.000386 | 0.00680 / 0.000742 |

Both baselines passed their validator verdicts. Their config and Pesce/Reid
anchors were accurate. All values were finite, neither had a root overflow,
and the observed maximum was three roots per half-plane. The NGC6323 broad
Sobol-0000 reference converged and exposed a real weakness: fixed-grid total
error 10.10 and peak-partition error 1316.51. This became the primary bad-fit
regression target.

Cold compilation was slower for peak partition in the original jobs
(40.83 s versus 22.73 s for NGC6323), so the speed claim is explicitly a
steady DE-throughput claim. A long DE run amortizes compilation; a one-off
single evaluation may not.

## Iterative research log

At least ten loops were requested; sixty-two numbered decision/confirmation
loops were executed.
Every speed decision below is from a GPU run, not a CPU extrapolation.

| Loop | Experiment | Main evidence | Decision |
|---:|---|---|---|
| 1 | Add local clouds; vary `K=5`, local `r=512` | Local anchor clouds mostly passed; narrower `K` worsened the converged bad point and 512 local nodes added cost without fixing it | The failure is not ordinary local resolution |
| 2 | `phi` scans 1025/513 and 257/129 | Numerical values were unchanged; 257/129 raised throughput to 494.9/s | The failure is radial; reduce scan |
| 3 | Root steps 8/10; circular capacity 4 | More steps did not help; capacity four had no overflow and reached 508.7/s | Use structural circular capacity four |
| 4 | 48/16 quadrature, global `r=256`, `K=20` | Higher quadrature and more global nodes did not solve the bad case; `K=20` helped some modes but hurt anchor margin | Reject brute-force node increases |
| 5 | `K=30,40` and local clouds | One bad mode improved monotonically, others plateaued; config error rose | Reject global widening as the remedy |
| 6 | Full-support global radius; asymmetric local grid; scan-envelope width | Same 128 nodes changed converged Sobol-0000 error 1316.5 -> 0.0109. Asymmetry improved other broad points. Envelope was slightly more accurate but ~10% slower | Accept full support + asymmetric spans; reject envelope cost |
| 7 | Scan frontier 193/97, 129/65, 97/49, 65/33, 33/17 | 129/65 preserved 0.0114 on converged bad point. 97/49 visibly degraded to 0.0172; 65/33 failed at 0.598 | Accept conservative 129/65 for ordinary galaxies |
| 8 | Refinement/quadrature frontier | 4x7 root refinement and 12 drop steps preserved values. 16/6 core/tail approached limits; 14/5 and 12/4 failed. Capacity-four + 24/8 reached 770.9/s | Accept cheap refinement, retain 24/8 quadrature |
| 9 | Cheaper uniform periodic/fixed alternatives | Quarter fixed grid passed anchors but failed converged bad point (0.358) and was still 2.65x slower than tuned peak; tenth/twentieth grids failed worse | Reject simple global node thinning |
| 10 | Candidate waves 1,2,4,8 | 260.4, 424.9, 524.7, 600.0 candidates/s with identical values | Keep wave eight |
| 11 | NGC6264 transfer + local clouds | Anchors and config-local clouds passed; one local-Pesce point ~19,400 nats below the anchor failed for both methods, and three broad references did not converge. Peak was 9.25x fixed | Transfer is good near anchors; report poor-fit limitation rather than hide it |
| 12 | NGC4258 129/65, 257/129, 513/257; full reference | 129/65 was 43.5x fixed at the terrible configured point. 257/129 and 513/257 agreed; their residual was radial. Dense reference still failed its own strict convergence gate | Isolate radial behaviour, then retest at a relevant fit before setting a galaxy-specific scan |
| 13 | Eccentric and eccentric+quadratic warp | Every converged anchor/local comparison and eccentric Sobol-0000 passed; zero overflows; 6.89--7.03x fixed | Keep value-only algorithm and capacity eight |
| 14 | Four-by-seven batched radial narrowing | NGC4258 peak-fixed changed from -5.36 to -7.26 while speed fell to 30.17x; NGC6323 fell to 539.7/s; eccentric Sobol error rose 0.00840 -> 0.01365 | Reject as a global default |
| 15 | Six-by-seven and five-by-nine radial narrowing | NGC4258 peak-fixed worsened to -28.80 and -29.28 | More stencil work is not monotone; reject |
| 16 | Odd 257-node local radial grids with 4x7 and 6x7 | Four steps gave -0.251 peak-fixed; six steps crossed to +15.05 | Reveals centre-alignment sensitivity, not controlled convergence |
| 17 | Curvature width with 4x7 stencil | Reproduced the 4x7 result exactly because non-concave probes correctly fell back | Fallback is safe but adds no value |
| 18 | Curvature width with 6x7, even/odd local grids | Both gave about +0.273 peak-fixed and 28.9--29.0x speed; both remained about 60 from an unconverged dense reference | Keep stencil opt-in and disabled |
| 19 | Three-GPU DE reconnaissance plus checkpoint/Pesce dense validation | DE converged without Pesce seeding to logP -5685.51 versus Pesce -8458.80; both dense ladders' final transition passed | Base the NGC4258 decision on relevant-fit evidence |
| 20 | NGC4258 Pesce scan frontier at 257/129 and 513/257 | Peak errors were 0.1631/0.4984 and 0.1607/0.4984 total/worst: extra phi nodes did not move the offending spot | Treat the remaining error as radial |
| 21 | Final-source NGC6323 and NGC6264 confirmation | Both pass; NGC6323 Sobol-0000 remains 0.0113 total / 0.000700 worst and NGC6264 anchors remain accurate | No regression in the final source; preserve dedicated solo jobs for headline timing |
| 22 | NGC4258 final-pair convergence gate | Pesce and checkpoint final transitions passed; a fourth level changed both totals by 2.81e-6 with worst/RMS below 8.4e-9 | The independent reference is decisively converged in the asymptotic regime |
| 23--25 | One, two, and three 7-point radial narrowing passes | Total Pesce errors were 1075, 5.90, and 10.60; applying the refinement to systemic spots was unstable | Restrict NGC4258 refinement to high-velocity groups |
| 26--28 | HV-only radial depth 4, 5, 6 with systemic scan 513 | Best was depth 4: 0.0769 total, 0.01725 worst, 24.83x fixed; deeper was not monotone | A local centre/width error remains |
| 29--30 | Four-level Pesce and checkpoint references | Both final pairs passed at the 1e-6--1e-8 scale; the production failures were therefore real | Stop attributing the residual to the reference |
| 31--35 | Cache identity audit, odd/local-513 grids | Prefix arrays matched bitwise; 257/513 local nodes did not remove the same worst errors and cost throughput | Node parity and ordinary local density are not the cause |
| 36 | Guarded quadratic vertex from the already evaluated final stencil | Pesce passed: 0.0481 total, 0.00627 worst, 0.000461 RMS at 24.94x fixed | Accept the zero-extra-evaluation vertex |
| 37 | Quadratic vertex at the independent DE checkpoint | Error improved but failed in red spots: 0.567 total, 0.273 worst | Keep the checkpoint as a hard stress candidate |
| 38--41 | Radial depth and systemic-scan frontier | Depth 3 improved Pesce to 0.0376/0.00314; systemic 321 barely passed, 289 failed at 0.0125 worst | Use depth 3 and retain at least 385 systemic nodes |
| 42--43 | Checkpoint depth 5 and 6 | Errors worsened to 1.02 and 2.00 total | The three-point curvature becomes ill-conditioned as the bracket shrinks |
| 44--45 | Checkpoint HV scans 129 and 257 | Both reproduced the 65-node result exactly at reported precision | Angular resolution is conclusively not the residual |
| 46--48 | Systemic 289; checkpoint depth 3; global radius 256 | Systemic 289 failed; depth 3 improved checkpoint to 0.160/0.0472; doubling global radii worsened both candidates and slowed to 19.1x | Replace the fragile curvature width, not the scan or global grid |
| 49--51 | Fixed-count value-only width solve at 16, 12, and 14 steps | Both relevant candidates passed with stable errors; speed was 24.02--24.15x fixed at systemic 513 | The explicit width solve removes the ill-conditioned curvature failure |
| 52 | Systemic 385 plus 16-step width solve | Both candidates passed; 27.51x fixed, with Pesce/checkpoint totals 0.0383/0.0295 | Accept the cheaper systemic scan |
| 53--54 | Width steps 8 and 10 at systemic 513 | Both reproduced the accuracy plateau and reached 24.66x/24.48x | Eight steps are sufficient for saturation |
| 55 | Eccentric NGC6323 with the width path enabled | Both converged anchors passed, zero overflows, 4.89x fixed | The new solve is empirically eccentric-safe, not a circular special case |
| 56--57 | Width steps 4 and 6 | Both passed; six matched the 8--16 plateau more closely | Continue downward to locate the boundary |
| 58--59 | Width steps 2 and 3 | Two failed (worst 0.0688/0.0855); three passed but with worse 0.00320/0.00190 worst errors | Choose a setting comfortably above the sharp boundary |
| 60 | Final 385/65, depth-3 high-velocity refinement, width 8; five timing repeats | Both candidates passed at 0.03833/0.02953 total error; 5.835/s versus 0.210/s, **27.78x** | Enable this exact NGC4258 policy |
| 61--62 | Post-config NGC6323/NGC6264 confirmations | Both passed; 8.76x and 9.22x fixed in these partially filled-wave jobs, with the same anchor errors and zero overflows | Confirm the NGC4258 override is isolated and the checked-in config is sound |

## Accepted ordinary-galaxy accuracy frontier

For NGC6323 with full-support/asymmetric radius handling and the accepted
circular kernel:

| Candidate | Reference status | Fixed total / worst | Peak total / worst |
|---|---|---:|---:|
| Config | converged | 0.00615 / 0.000156 | 0.00639 / 0.000336 |
| Pesce/Reid | converged | 0.00589 / 0.000185 | 0.00609 / 0.000188 |
| Sobol-0000 bad fit | converged | 0.00683 / 0.000426 | 0.01134 / 0.000700 |

The fixed column in this table also uses the new full-support radius scan. It
shows that the radial correction improves both integrators and costs no extra
nodes. The original fixed value for Sobol-0000 was 10.10.

Other NGC6323 broad points improved substantially (peak totals approximately
25.1, 125.1, and 17.35 rather than 212.6, 777.1, and 1765.3), but their dense
references did not converge. These numbers demonstrate stability and finite
behaviour, not certified absolute errors.

For NGC6264, config and Pesce/Reid peak totals were 0.00538 and 0.00543, and
the two config-local totals were 0.0208 and 0.00657. A perturbed Pesce point at
log likelihood -20,399 had peak total/worst error 0.0613/0.00876 and failed
the p99/RMS gate, as did fixed grid at 0.0577/0.00754. The validator's overall
FAIL is therefore recorded. That point is vastly below the approximately
-996 anchor even though the automatic irrelevance classifier did not excuse
it because its support-railing fraction was small.

## Eccentricity and warp stress test

The NGC6323 eccentric case passed overall after excusing one unconverged,
railed broad point. Peak errors were:

- config: total 0.00638, worst 0.000336;
- Pesce/Reid: 0.00606, 0.000196;
- local config: 0.0169, 0.000710;
- local Pesce/Reid: 0.0251, 0.00145;
- converged Sobol-0000: 0.00840, 0.00446.

The eccentric-plus-quadratic-warp anchors and local clouds also passed. Its
broad Sobol references did not converge, so the case-level verdict remained
FAIL rather than being waived. Both variants had zero root overflows. These
are the important design facts: the implementation did not fold `phi`, did
not use the circular polynomial evaluator, and did not lower eccentric root
capacity.

After introducing the drop-width solve, a separate eccentric NGC6323 run
enabled the complete depth-3/high-velocity/10-step path.  The config and
Pesce/Reid candidates both passed: peak total/worst errors were
0.00625/0.000336 and 0.00624/0.000143, respectively.  It had zero overflows
and ran at 160.5 versus 32.8 candidates/s, or 4.89x fixed grid.  This is a
direct execution test of the new width code through the eccentric velocity
field; it is stronger evidence than merely noting that the function contains
no circular formula.  It does not by itself select future eccentric NGC4258
scan counts, which should be revalidated when that science model is enabled.

## NGC4258 evidence and limitation

NGC4258 has 358 spots and extremely narrow angular likelihoods. With one
config candidate and candidate-wave padding to eight, the pre-stencil results
were:

| Peak scan | Peak throughput | Speedup over 60,001/20,001 fixed | Peak - fixed total |
|---:|---:|---:|---:|
| 129/65 | 4.51/s | 43.47x | -17.376 |
| 257/129 | 4.09/s | 38.48x | -5.361 |
| 513/257 | 2.50/s | 23.89x | -5.353 |

The equality of 257/129 and 513/257 isolates the residual from scan density.
At 257/129 the largest differences were red spots 277 (-1.871), 290 (-1.478),
276 (-0.963), and 248 (-0.906). The fixed-grid values for these spots matched
the finest available dense reference closely, pointing to the peak method's
coarse radial centre. Conversely, systemic spots 136 and 141 differed from
the dense reference by roughly 33 and 22 under *both* production methods.
Those are disconnected needle modes at a very poor configured geometry, not
evidence that one `phi` rule is superior.

The full dense reference progressed as follows:

| Consecutive grids | Absolute total change | Worst spot | RMS |
|---|---:|---:|---:|
| 5,001x50,001 -> 10,001x100,001 | 0.802 | 0.926 | 0.0868 |
| 10,001x100,001 -> 20,001x200,001 | 0.0336 | 0.0199 | 0.00194 |

This is strong convergence progress but still outside the declared reference
thresholds. No production error relative to that final array is labelled
certified. The user's suggestion to use cheaper galaxies for routine
validation was correct.

The follow-up radial frontier used the same reduced 1,001x10,001 through
4,001x40,001 reference cache; that reference was also unconverged, so the
useful comparison is peak minus fixed and the runtime cost:

| Radial experiment | Peak - fixed total | Peak throughput | Speedup |
|---|---:|---:|---:|
| 4x7, 256 local nodes | -7.260 | 3.149/s | 30.17x |
| 6x7, 256 local nodes | -28.796 | 3.046/s | 29.33x |
| 5x9, 256 local nodes | -29.278 | 3.079/s | 29.10x |
| 4x7, 257 local nodes | -0.251 | 3.137/s | 30.09x |
| 6x7, 257 local nodes | +15.047 | 3.096/s | 29.08x |
| 6x7 plus curvature width, 256 nodes | +0.274 | 3.014/s | 28.89x |
| 6x7 plus curvature width, 257 nodes | +0.273 | 3.081/s | 28.98x |

This non-monotone centre-alignment behaviour showed that a raw stencil
midpoint or a shrinking three-point curvature could not be accepted as a
production accuracy fix.  It did not justify abandoning radial refinement:
the experiment was performed at a geometry whose reference was itself
unconverged.  The subsequent relevant-fit work kept the robust full-support
discovery step, restricted refinement to the affected high-velocity groups,
used the final stencil values for a guarded numerical vertex, and replaced
the fragile curvature width by an explicit fixed-count drop-width solve.

### Relevant-fit NGC4258 checkpoint

The reconnaissance used three RTX 3090 GPUs, a 2,048-point scrambled Sobol
prescreen, a 256-member population reducing to 64, and no Pesce/Reid seed.
The first 50 generations reached -9835.48. A compatible resume converged at
generation 440 after 100 stale generations, at -5685.51 and 35,158 DE
candidate evaluations. Aggregate steady device throughput was approximately
97.6 candidates/s (32.55/s per GPU). Its fitted globals were
`D_A=7.3610 Mpc`, `Omega0=88.301 deg`, `i0=93.487 deg`,
`dOmega/dr=2.093 deg/mas`, and `di/dr=-2.259 deg/mas`; this is qualitatively
the expected disk geometry and is not a support-railing proposal.

The independent relevant-fit reference ladders are converged, rather than
merely dense:

| Candidate | 5,001x50,001 -> 10,001x100,001 total / worst / RMS | 10,001x100,001 -> 20,001x200,001 | 20,001x200,001 -> 40,001x400,001 |
|---|---:|---:|---:|
| Pesce/Reid | 0.175 / 0.162 / 0.0134 | 2.94e-4 / 1.39e-4 / 1.00e-5 | 2.81e-6 / 8.39e-9 / 7.85e-9 |
| DE checkpoint | 0.338 / 0.0886 / 0.00869 | 3.90e-5 / 3.03e-4 / 2.17e-5 | 2.81e-6 / 7.99e-9 / 7.84e-9 |

The first transition illustrates why a single coarse/fine comparison would
have been misleading.  The final two transitions both pass all reference
gates.  The current-source three-level arrays equal the corresponding prefix
of the four-level arrays bit for bit (maximum absolute difference exactly
zero), so the fourth-level evidence remains applicable after the production-
only vertex/width changes.

The relevant-fit production frontier is:

| Peak settings | Pesce total / worst | Checkpoint total / worst | Steady speedup over fixed | Verdict |
|---|---:|---:|---:|---|
| 129/65, no radial stencil | 11.07 / 1.649 | 0.672 / 0.378 | 38.6x | fail: radial |
| 513/65, depth-3 vertex + curvature | 0.0376 / 0.00314 | 0.160 / 0.0472 | 24.9x | checkpoint fail |
| 513/65, depth-3 + 2-step drop width | 0.0612 / 0.0688 | 0.0240 / 0.0855 | 24.76x | fail: width under-resolved |
| 513/65, depth-3 + 3-step drop width | 0.0324 / 0.00320 | 0.0275 / 0.00190 | 24.71x | pass, pre-asymptotic |
| 513/65, depth-3 + 4-step drop width | 0.0330 / 0.00146 | 0.0290 / 0.000541 | 24.54x | pass |
| 513/65, depth-3 + 6-step drop width | 0.0325 / 0.00146 | 0.0295 / 0.000541 | 24.68x | pass, plateau |
| 513/65, depth-3 + 8-step drop width | 0.0325 / 0.00146 | 0.0296 / 0.000541 | 24.66x | pass, plateau |
| 513/65, depth-3 vertex + 16-step drop width | 0.0325 / 0.00146 | 0.0296 / 0.000541 | 24.02x | **pass** |
| 385/65, depth-3 vertex + 16-step drop width | 0.0383 / 0.00276 | 0.0295 / 0.000541 | **27.51x** | **pass** |
| **385/65, depth-3 + 8-step drop width** | **0.03833 / 0.00276** | **0.02953 / 0.000541** | **27.78x** | **production pass** |

The final row is the production recommendation.  Eight steps are deliberately
above the three-step feasibility boundary and already on the six-through-
sixteen plateau.  Its high-velocity errors have collapsed to the approximately
`9e-5`-per-spot floor also seen in fixed grid; the slightly larger Pesce worst
spot is systemic and remains 3.6 times below the acceptance limit.  Both
candidates have identical finite masks, zero capacity overflows, and no ranking
issue.  Five matched timing repeats gave 5.835 peak candidates/s versus 0.210
fixed candidates/s; cold compile-plus-first-evaluation remained slower (103.2
versus 20.9 s), so the speedup applies to the long DE workload.

## Why the alternatives were rejected

- **Uniform periodic trapezoid:** for analytic periodic functions it can
  converge geometrically, but the constant depends on the distance to nearby
  complex singularities and, operationally here, a node must first resolve
  the very narrow real peak. Empirical uniform-grid thinning failed the
  converged bad point before it caught peak partition's speed.
- **Clenshaw--Curtis:** it offers convergence comparable to Gauss rules in many
  smooth problems and nested nodes are attractive for adaptivity. It does not
  by itself locate a narrow interior mode; mapping it separately over the
  numerically found partitions recreates the present design with no measured
  advantage over Gauss--Legendre.
- **Adaptive Gauss--Kronrod/QUADPACK:** excellent serial general-purpose error
  control, but data-dependent subdivision gives different work and shapes per
  `(candidate, spot, radius)`. It can also accept a region before sampling an
  isolated needle. That is a poor fit to the static batched GPU target.
- **Tanh--sinh/double exponential:** designed primarily to neutralize endpoint
  singularities. These integrands are smooth with sharp interior likelihood
  modes, so it concentrates work in the wrong place unless preceded by the
  same peak partition.
- **Laplace/Gauss--Hermite around one maximum:** cheapest locally, but loses a
  second mode, finite tails, boundary truncation, and non-Gaussian eccentric
  structure. Those are precisely the geometries probed here.
- **Analytic stationary roots or exact generalized-Bessel sums:** circular
  algebra permits special treatment, but it duplicates the physics formula,
  has degeneracy/stability issues, and fails immediately for eccentric
  velocity and periapsis warp. It also conflicts with the explicit requirement
  not to rely on analytic peak finding.
- **Simply widen the radial local grid or add global nodes:** loops 4--5 showed
  higher cost, reduced anchor margin, and incomplete recovery.  At the
  relevant NGC4258 points, doubling global nodes to 256 worsened Pesce to
  0.0822/0.0436 and the checkpoint to 0.1649/0.0659 total/worst while slowing
  the method to 19.1x fixed.  A 513-node local grid likewise retained the same
  bad spots.  Changing where the existing global nodes look was the effective
  mode-discovery intervention; width accuracy required a different tool.

The fixed-shape choice is also aligned with JAX execution: shapes/dtypes affect
compilation, while `lax.fori_loop`, `scan`, and other structured primitives
represent static staged control flow. The final algorithm uses those
properties rather than host-driven adaptive recursion.

## Code changes

- `candel/model/model_H0_maser.py`
  - full-support global radius option and asymmetric local grids;
  - cheaper scan defaults and circular/eccentric root capacities;
  - optional value-only radial narrowing, a guarded no-extra-call quadratic
    vertex, and a fixed-count two-sided drop-width solve for narrow radial
    likelihoods;
  - finite trapezoid fallback on root-capacity overflow;
  - no `-inf` poisoning of cached/global/local radial reductions.
- `scripts/megamaser/config_maser.toml`
  - 129/65 ordinary-galaxy defaults;
  - full-support/asymmetric radius defaults;
  - optional radial-stencil and drop-width controls, disabled globally and
    enabled only in the validated NGC4258 block.
- `scripts/megamaser/run_de_map.py`
  - peak policy v6 and radial policy v2 encode the numerical objective and
    reject stale checkpoint values;
  - the final conditional-radius diagnostic is JIT staged, avoiding thousands
    of tiny post-optimisation GPU dispatches.
- `scripts/megamaser/convergence/validate_phi_partition.py`
  - typed scheme overrides for the new experiments;
  - effective root-capacity reporting;
  - compute-node operation when `git` is absent;
  - corrected candidate-ID report formatting;
  - strict loading of an actual DE checkpoint candidate for independent
    validation.
- Focused tests cover fallback finiteness, radial stencil narrowing, exact
  quadratic-vertex recovery, the value-only drop-width solve, systemic-skip
  scope, asymmetric support, full-radius vectorization/cache reuse, setting
  overrides, checkpoint-candidate validation, and compute-node metadata.

The global default remains `fixed-grid`; changing every untested galaxy was
outside the evidence gathered here.  The NGC4258 galaxy block now selects the
validated 385/65 peak policy explicitly.  That choice is based on the
relevant-fit cross-check, not on the unconverged configured-point stress
reference.

## Test status

Completed locally under an explicit CPU JAX backend:

- `tests/test_megamaser_phi_partition.py` plus precision kernels: 13 passed;
- `tests/test_megamaser_phi_validation.py`: 32 passed;
- `tests/test_megamaser_de_lshade.py`: 54 passed, two existing JAX deprecation
  warnings;
- combined result: 99 passed;
- Python compilation and `git diff --check`: passed.

## Reproduction

Baseline submission:

```bash
./scripts/megamaser/convergence/validate_phi_partition.sh -q cmbgpu \
  --galaxies NGC6323
```

Final ordinary circular settings:

```bash
./scripts/megamaser/convergence/validate_phi_partition.sh -q cmbgpu \
  --galaxies NGC6323 \
  --scheme-setting fixed-grid.global_r_full_support=true \
  --scheme-setting fixed-grid.asymmetric_r_local=true \
  --scheme-setting peak-partition.global_r_full_support=true \
  --scheme-setting peak-partition.asymmetric_r_local=true \
  --scheme-setting peak-partition.n_phi_partition_sys=129 \
  --scheme-setting peak-partition.n_phi_partition_hv=65 \
  --scheme-setting peak-partition.root_capacity=4 \
  --scheme-setting peak-partition.root_steps=4 \
  --scheme-setting peak-partition.root_order=7 \
  --scheme-setting peak-partition.drop_steps=12 \
  --scheme-setting peak-partition.peak_r_refine_steps=0
```

Eccentric variants automatically use root capacity eight and otherwise retain
the same value-only operations.  The NGC4258 production settings are:

```bash
./scripts/megamaser/convergence/validate_phi_partition.sh -q cmbgpu \
  --galaxies NGC4258 --sobol-candidates 0 --no-config-point \
  --reference-tail-levels 2 --timing-repeats 5 \
  --checkpoint-candidate \
  results/Megamaser/de_checkpoints/NGC4258/\
de_ckpt_rmap_peakpartition_lshade_nopesce.npz \
  --allow-checkpoint-policy-mismatch \
  --scheme-setting peak-partition.n_phi_partition_sys=385 \
  --scheme-setting peak-partition.n_phi_partition_hv=65 \
  --scheme-setting peak-partition.peak_r_refine_steps=3 \
  --scheme-setting peak-partition.peak_r_refine_hv_only=true \
  --scheme-setting peak-partition.peak_r_width_steps=8
```

Historical NGC4258 reconnaissance and its then-compatible resume:

```bash
./scripts/megamaser/submit.sh -q cmbgpu --galaxy NGC4258 \
  --sampler de --gpu-count 3 --cpus 1 \
  --phi-integration peak-partition --peak-candidates-per-wave 8 -- \
  --log2-N 11 --pop-size 256 --min-pop-size 64 \
  --population-reduction-evaluations 12800 \
  --max-generations 50 --patience 50

./scripts/megamaser/submit.sh -q cmbgpu --galaxy NGC4258 \
  --sampler de --gpu-count 3 --cpus 1 \
  --phi-integration peak-partition --peak-candidates-per-wave 8 -- \
  --resume --log2-N 11 --pop-size 256 --min-pop-size 64 \
  --population-reduction-evaluations 12800 \
  --max-generations 500 --patience 100
```

Those commands document how the independent point was obtained.  Do not
resume that v5 checkpoint into v6: the objective-policy guard rejects it.  A
new production NGC4258 DE should start fresh under the config's v6 policy.

Independent checkpoint validation:

```bash
./scripts/megamaser/convergence/validate_phi_partition.sh -q cmbgpu \
  --galaxies NGC4258 --sobol-candidates 0 --no-config-point \
  --checkpoint-candidate \
  results/Megamaser/de_checkpoints/NGC4258/\
de_ckpt_rmap_peakpartition_lshade_nopesce.npz \
  --allow-checkpoint-policy-mismatch
```

## Result artifacts

- Baseline NGC6323: `results/Megamaser/convergence/phi_partition_20260721T220446Z/validation.md`
- Baseline NGC6264: `results/Megamaser/convergence/phi_partition_20260721T220510Z/validation.md`
- Accepted pre-stencil NGC6323 kernel: `results/Megamaser/convergence/agentic_20260721/loop08_final_capacity4/validation.md`
- Full-support fixed comparison: `results/Megamaser/convergence/agentic_20260721/loop09_fixed_full_support/validation.md`
- NGC6264 transfer: `results/Megamaser/convergence/agentic_20260721/loop11_ngc6264_final_local/validation.md`
- Full NGC4258 reference: `results/Megamaser/convergence/agentic_20260721/loop12_ngc4258_full/validation.md`
- Eccentric/warp test: `results/Megamaser/convergence/agentic_20260721/loop13_eccentric/validation.md`
- Radial-stencil frontier: `results/Megamaser/convergence/agentic_20260721/loop14_ngc4258_radial_refine/validation.md` through `loop18_ngc4258_curvature6_odd/validation.md`
- Relevant-fit NGC4258 validation: `results/Megamaser/convergence/agentic_20260721/loop19_ngc4258_goodfit/validation.md`
- Four-level reference audits: `results/Megamaser/convergence/agentic_20260721/loop29_ngc4258_pesce_four_level/validation.md` and `loop30_ngc4258_checkpoint_four_level/validation.md`
- First drop-width pass: `results/Megamaser/convergence/agentic_20260721/loop49_ngc4258_drop_width16/validation.md`
- Systemic-scan transfer: `results/Megamaser/convergence/agentic_20260721/loop52_ngc4258_final_sys385_width16/validation.md`
- Eccentric drop-width test: `results/Megamaser/convergence/agentic_20260721/loop55_ngc6323_eccentric_drop_width10/validation.md`
- Width feasibility boundary: `results/Megamaser/convergence/agentic_20260721/loop58_ngc4258_drop_width2/validation.md` and `loop59_ngc4258_drop_width3/validation.md`
- **Final selected NGC4258 benchmark:** `results/Megamaser/convergence/agentic_20260721/loop60_ngc4258_final_sys385_width8/validation.md`
- Post-config transfers: `results/Megamaser/convergence/agentic_20260721/loop61_ngc6323_postconfig/validation.md` and `loop62_ngc6264_postconfig/validation.md`

## Numerical-method references

- L. N. Trefethen and J. A. C. Weideman, [The Exponentially Convergent
  Trapezoidal Rule](https://epubs.siam.org/doi/10.1137/130932132), *SIAM
  Review* 56 (2014), 385--458.
- L. N. Trefethen, [Is Gauss Quadrature Better than
  Clenshaw--Curtis?](https://epubs.siam.org/doi/10.1137/060659831), *SIAM
  Review* 50 (2008), 67--87.
- R. Piessens et al., [QUADPACK: A Subroutine Package for Automatic
  Integration](https://link.springer.com/book/10.1007/978-3-642-61786-7),
  Springer, 1983.
- M. Mori, [Discovery of the double exponential transformation and its
  developments](https://www.sciencedirect.com/science/article/pii/S037704270000501X),
  *Journal of Computational and Applied Mathematics* 127 (2001), 287--296.
- [JAX control-flow documentation](https://docs.jax.dev/en/latest/control-flow.html).
