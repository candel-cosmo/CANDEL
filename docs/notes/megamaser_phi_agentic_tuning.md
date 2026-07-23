# Agentic megamaser integration tuning

## Purpose

This note defines a small, reproducible optimisation loop around
`validate_phi_partition.sh`. The agent varies production numerical settings,
holds the physical problem and candidate points fixed, rejects any setting that
fails the existing accuracy gates, and ranks the feasible settings by warmed
production throughput. The expensive float64 reference arrays are shared
between trials through the 48-hour cache.

This is an integration-hyperparameter pilot. It does not yet establish that a
full DE run reaches the same optimum or that the CPU timing order transfers to
the production GPU.

## Agent-facing interface

Repeat `--scheme-setting METHOD.KEY=VALUE` to change numerical settings without
editing `config_maser.toml`. For example:

```bash
./scripts/megamaser/convergence/validate_phi_partition.sh --local \
    --galaxies NGC6264 --timing-repeats 5 \
    --scheme-setting peak-partition.n_phi_partition_sys=65 \
    --scheme-setting peak-partition.n_phi_partition_hv=17 \
    --output-dir results/Megamaser/convergence/TRIAL_NAME
```

The whitelisted controls are:

| Method | Method-specific controls | Shared conditional-radius controls |
|---|---|---|
| `fixed-grid` | `n_phi_sys`, `n_phi_hv_high`, `n_phi_hv_low` | `n_r_local`, `n_r_global`, `K_sigma`, `n_refine_steps`, `refine_r_center` |
| `peak-partition` | `n_phi_partition_sys`, `n_phi_partition_hv` | `n_r_local`, `n_r_global`, `K_sigma`, `n_refine_steps`, `refine_r_center` |

For every trial, the agent should:

1. Keep the galaxy, variant, seed, candidates, reference grids, and tolerances
   fixed.
2. Require a zero exit code and top-level `passed: true` in `validation.json`.
3. Reject relevant-candidate reference failures, finite-mask mismatches,
   ranking inversions, or root-capacity overflows rather than trading them for
   speed.
4. Rank feasible trials using
   `cases[].timing[METHOD].throughput_candidates_per_second`.
5. Repeat the apparent winner in a fresh process, then confirm it on the
   production GPU and in full optimiser runs.

The validator still compiles, times, and evaluates both production methods on
every invocation. Only the independent dense ladder is cached. A hit is
printed explicitly as `Reference cache HIT ... dense ladder skipped`.

## NGC6264 circular pilot

The pilot ran locally on the CPU backend with JAX/JAXLIB 0.9.2. Every trial
used the configured and Pesce/Reid anchors plus four scrambled Sobol points
with seed 44. The independent float64 reference ladder was
`5001 x 2501 -> 10001 x 5001 -> 20001 x 10001` in radial-by-phi nodes. Both
anchors converged at every level. The four broad Sobol points had
posterior-irrelevant railed geometry; their real reference results remain in
the reports, but their failures are excused by the existing policy.

`Worst` statistics below are over the two relevant anchors. Throughput is the
local warmed peak-partition throughput, so it is suitable for this CPU pilot
only.

| Systemic scan | HV scan | Verdict | candidates/s | Peak/fixed speedup | Worst total error | Worst spot | Worst p99 | Worst RMS |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 513 | 257 | PASS | 12.54 | 2.04x | 0.00688 | 0.000752 | 0.000687 | 0.000214 |
| 257 | 129 | PASS | 18.87 | 3.04x | 0.00689 | 0.000752 | 0.000687 | 0.000214 |
| 129 | 65 | PASS | 27.20 | 4.38x | 0.00688 | 0.000752 | 0.000687 | 0.000214 |
| 65 | 33 | PASS | 32.30 | 5.22x | 0.00902 | 0.00228 | 0.00106 | 0.000314 |
| 65 | 17 | PASS | 33.85 | 5.49x | 0.00904 | 0.00228 | 0.00106 | 0.000314 |
| 65 | 17, repeat | PASS | 33.30 | 5.39x | 0.00904 | 0.00228 | 0.00106 | 0.000314 |
| 57 | 17 | FAIL | 34.09 | 5.50x | 0.02095 | 0.00582 | 0.00518 | 0.00107 |
| 49 | 17 | FAIL | 35.24 | 5.74x | 0.03894 | 0.01242 | 0.01050 | 0.00209 |
| 33 | 17 | FAIL | 35.90 | 5.78x | 0.07571 | 0.01441 | 0.01408 | 0.00348 |

The `57/17` trial is a useful negative control: it is faster but crosses both
the p99 threshold of 0.005 and RMS threshold of 0.001. The hard feasibility
gate therefore prevents the search from selecting it. The limiting dimension
for these NGC6264 anchors is the systemic scan; reducing the HV scan from 33
to 17 did not materially change the measured errors.

The two `65/17` timings give a median 33.58 candidates/s, 2.68 times the
configured `513/257` peak-partition baseline on this CPU. There were no root
overflows or ranking inversions in either run. `129/65` is the conservative
follow-up candidate because its measured errors are indistinguishable from the
baseline, while `65/17` is the current speed frontier for broader testing.
Neither setting should be promoted to production from this single-galaxy CPU
pilot alone.

The complete reports are under
`results/Megamaser/convergence/agentic_ngc6264_20260721/`.

## Next validation rung

Before changing production defaults:

1. Repeat `129/65` and `65/17` on the production GPU with more local Sobol
   points around both anchors and at least two seeds.
2. Run the same acceptance sweep for all galaxies and enabled model variants;
   use the worst-galaxy feasible setting rather than an NGC6264-only optimum.
3. Run matched DE jobs from identical initial populations and seeds. Compare
   best objective value, recovered parameters, convergence history, evaluation
   count, wall time, and checkpoint/resume behaviour against the baseline.
4. Only then vary shared radial controls or optimiser hyperparameters. Each
   added dimension should retain the same cached-reference feasibility gate
   and a matched optimiser-level confirmation.
