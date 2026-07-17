# Megamaser DE GPU-memory calibration

Measured 2026-07-17 with JAX/jaxlib 0.10.1 on one NVIDIA GeForce RTX
2080 Ti (11,264 MiB, driver 610.43.02).  The benchmark used the production
conditional-r evaluator, the configured integration grids, one candidate per
GPU wave, the fixed eight-candidate device block, and 32 checkpoint candidates.
Candidates inside the block are evaluated sequentially, so the values below
are per GPU and do not grow with the DE population size.

No integration resolution was changed for these tests.

## What was measured

Job 790912 repeated the float32 UGC3789 benchmark in fresh processes for each
spot batch.  `JAX peak` is `peak_bytes_in_use` from JAX device memory stats.
`NVML resident` is the maximum process memory sampled every 50 ms with
`nvidia-smi`.  The rate is for exact population evaluation after three warm-up
passes.  `Fixed score` includes fresh-process compilation and one exact score.

| Spot batch | JAX peak (MB) | NVML resident (MiB) | Fixed score (s) | Candidates/s |
|---:|---:|---:|---:|---:|
| 1 | 135.272 | 332 | 67.35 | 22.90 |
| 4 | 68.163 | 252 | 24.30 | 33.85 |
| 8 | 34.608 | 214 | 17.01 | 38.45 |
| 16 | 34.608 | 214 | 13.32 | 37.66 |
| 34 | 5.248 | 184 | 10.73 | 34.43 |
| 68 | 5.248 | 182 | 8.59 | 40.90 |
| all, run 1 | 34.608 | 212 | 12.69 | 40.99 |
| all, run 2 | 34.608 | 212 | 13.09 | 41.01 |

Every float32 case returned the same bitwise population digest and the same
fixed-point exact log probability, `-2446.855224609375`.  On this evaluator,
small scan batches are not memory-saving: their scan-loop state costs more than
the fused all-spots executable.  `all` is effectively tied with batch 68 for
the best throughput and uses only 212 MiB process-resident memory.

Job 790948 provided a matched float64 calibration because float64 changes the
compiled GPU plan by much more than the nominal twofold element size:

| UGC3789 path | Raw largest block (GB) | JAX peak (GB) | NVML resident (GiB) | Candidates/s |
|---|---:|---:|---:|---:|
| float64, batch 16 | 0.737 | 0.741 | 1.324 | 3.24 |
| float64, all | 3.134 | 3.139 | 4.320 | 3.48 |

The fixed-point float64 score was `-2446.8551513624966` in both cases at the
printed precision.  Their full-population byte digests differ because batching
changes floating-point reduction order.  Production UGC3789 remains float32;
these float64 measurements exist only to calibrate the forced-float64 NGC4258
estimate.

## Geometry model

For spot class `s`, define the raw bytes for one array and one candidate as

```text
G_s = N_s,live * (n_r_local + n_r_global) * n_phi,s * dtype_bytes
G   = max_s G_s
```

Here `N_s,live` is all spots in class `s` for an all-spots executable, or
`min(N_s, spot_batch)` for a scanned executable.  The high-velocity azimuthal
length is the concatenated grid
`n_phi_hv_high + 2 * n_phi_hv_low`.  All galaxies use
`n_r_local + n_r_global = 256 + 128 = 384` and 32 refinement steps.

The extrapolation scales measured memory only between matching precision and
execution paths:

```text
JAX_peak(g) = JAX_peak(reference) * G(g) / G(reference)
NVML(g)     = baseline + (NVML(reference) - baseline) * G(g) / G(reference)
```

The measured RTX-2080-Ti baselines were 176 MiB for float32 and 312 MiB for
float64.  The float32/all reference is UGC3789 all-spots.  NGC4258 batch 16 and
hypothetical NGC4258 all-spots use the corresponding float64 UGC3789 control.

This is a shape-calibrated engineering estimate, not an XLA allocation
guarantee.  XLA fusion makes the float32 peak highly non-linear and architecture
dependent.  The raw geometry is therefore useful as a relative size variable,
not as the amount actually allocated.  A fresh benchmark on a new GPU model is
the final authority.

The runner's diagnostic planner deliberately assumes eight simultaneously live
arrays and therefore prints `8 * G`.  For example, it reports 12.54 GB for
float32 UGC3789 all-spots, while the measured current executable used only
0.035 GB JAX-live and 0.207 GiB process-resident.  Keep the planner value as an
OOM-avoidance bound; do not interpret it as a calibrated prediction.

## Current-production estimates

`phi sys / HV` gives the actual concatenated azimuth lengths.  `Raw block` is
`G`, not the older conservative eight-live-buffer estimate.  UGC3789 is the
measurement anchor; the remaining rows are extrapolations.

| Galaxy | Spots sys/red/blue | phi sys / HV | Precision | Current spot batch | Raw block (GB) | Expected JAX peak (GB/GPU) | Expected NVML resident (GiB/GPU) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CGCG074-064 | 45/71/49 | 10001 / 10003 | f32 | all | 1.091 | 0.024 | 0.196 |
| NGC5765b | 40/73/79 | 10001 / 10003 | f32 | all | 1.214 | 0.027 | 0.199 |
| NGC6264 | 11/23/32 | 10001 / 10003 | f32 | all | 0.492 | 0.011 | 0.183 |
| NGC6323 | 11/36/21 | 10001 / 10003 | f32 | all | 0.553 | 0.012 | 0.184 |
| UGC3789 | 42/46/68 | 10001 / 15003 | f32 | all | 1.567 | 0.035 measured | 0.207 measured |
| NGC4258 | 187/139/32 | 60001 / 30003 | f64 | 16 | 2.949 | 2.962 | 4.382 |

Practical per-GPU allowances are 0.5 GiB for any of the five float32 galaxies
and 6 GiB for production NGC4258 at `spot_batch=16`.  These deliberately round
above the estimates to cover compiler, driver, and GPU-architecture variation.
Thus all five float32 all-spots paths should fit even on an 8 GiB GPU, and the
current NGC4258 batch-16 path should also fit with materially less headroom.

For comparison, forcing all 187 NGC4258 systemic spots into the float64
all-spots executable gives:

| NGC4258 mode | Raw block (GB) | Expected JAX peak (GB/GPU) | Expected NVML resident (GiB/GPU) | Suggested allowance |
|---|---:|---:|---:|---:|
| production batch 16 | 2.949 | 2.962 | 4.382 | 6 GiB |
| hypothetical all spots | 34.468 | 34.519 | 44.468 | at least 56 GiB |

The hypothetical all-spots NGC4258 path is appropriate only for a large-memory
device such as an 80 GiB H100.  It is not safe for an 8/12/24 GiB GPU, and a
40/48 GiB device has insufficient or uncomfortably small operational headroom.
The configured `conditional_spot_batch=16` should remain in place.

## Multi-GPU interpretation

Each GPU receives different candidates and compiles the same one-candidate
evaluator.  Memory is replicated per GPU; adding GPUs does not divide one
candidate's memory.  With enough population members, total throughput should
scale close to the number of equally fast GPUs, while each GPU stays near the
per-GPU values above.  The round-robin/weighted dispatcher corrects modest
device-speed imbalance.  Population size and L-SHADE archive size live mainly
on the host and do not multiply the quadrature working set on a GPU.

The old `eval_chunk` control no longer exists.  Exact candidate evaluation is
fixed at one candidate per GPU wave; `spot_batch` is the only exact-likelihood
memory batching control.
