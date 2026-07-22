# TRGBH0 mock bias test (fiducial model, no sky exposure)

Set up 2026-07-17 for the "Mock validation" appendix of the TRGBH0 paper
(requested in the `\rs{}` note at `main.tex:731`; the "Mock results"
subsection is a placeholder to be populated with these results).

The test draws synthetic TRGB catalogues from the forward model with known
parameters, re-infers them with the standard NUTS pipeline, and records the
standardised bias and posterior truth percentile per parameter. A correctly
specified implementation gives mean standardised biases consistent with
zero and truth percentiles consistent with uniform.

## Configuration

Fiducial model WITHOUT the angular sky-exposure term:

- single Manticore-Local COLA PCS field (density and velocity),
- double power-law source-density bias,
- 4 Mpc/h Gaussian density smoothing (velocity unsmoothed),
- Galactic-plane mask |b| >= 10 deg,
- soft TRGB-magnitude window (edge and width inferred in recovery),
- Student-t redshift likelihood,
- beta fixed to 1 in generation and recovery.

Recovery uses the same priors as the corresponding fixed-beta
`TRGBH0_main` paper model, including the H0, Vext, sigma_int, and
selection-width priors; beta has the same delta prior at unity.

Every component is applied consistently in BOTH mock generation and
recovery: the mock samples hosts from the smoothed, unit-normalised density
via the model's bias law (inhomogeneous Malmquist), rejects |b| < b_min,
draws cz noise from the location-scale Student-t matching
`student_t_logpdf_var`, and interpolates host LOS arrays from the same
smoothed field; the recovery volume integral gets the same field index,
smoothing, supersampling, and `store_rhat` (needed for the b_min mask).

## Injected truths

Paper-motivated truth point held in `FIDUCIAL_MANTICORE_DEFAULTS` in
`scripts/mocks/mock_TRGB.py`:

| Parameter | Value |
|---|---|
| H0 | 72.2 |
| M_TRGB | -4.03 |
| sigma_int | 0.10 |
| sigma_v | 66 km/s |
| nu_cz | 2.33 |
| beta | 1 (fixed) |
| Vext | 332 km/s towards (l, b) = (285, -4) deg |
| mag window | 22.1 < m < 24.06, width 0.94 |
| alpha_low | 2.25 |
| alpha_high_frac | 0.76 (= 1.70 / 2.25) |
| log_rho_t | 0.54 |
| log_rho_width | 0.71 |
| nsamples | 400 hosts (real sample: 394) |
| rmax | 69.3 Mpc (= 50 Mpc/h at h = 0.722) |

The effective sampling radius is set by the selection tail (~37 Mpc at
these truths), not by rmax, so the magnitude window bounds the survey
consistently on both sides.

## How injected parameters are chosen

`--default-manticore` applies the preset via `parser.set_defaults()`, so
any explicitly passed flag except beta overrides it (e.g. `--H0 70`); beta
remains fixed to unity. The Manticore
field is picked deterministically as `master seed % 80` in batch mode
(`seed % 80` in `--single` mode) unless `--field-index` is given;
`mock_TRGB.sh` resolves the index once before GPU sharding so all shards
of a batch share one field. Anything not in the preset falls back to
`DEFAULT_TRUE_PARAMS` / `DEFAULT_ANCHORS` in `candel/mock/TRGB_mock.py`.

This is a fixed-truth calibration test at the fiducial point, not
prior-drawn simulation-based calibration.

## Running

Field-based runs require the PCS products and warmed field cache configured by
`local_config.toml`; production batches should run on glamdring.

```bash
cd scripts/mocks

# smoke test (one mock, field = seed % 80)
./submit_TRGBH0_mock_bias.sh --local --single --seed 42

# production batch (GPU shards + merge job; field = master seed % 80)
./submit_TRGBH0_mock_bias.sh -q gpulong --gpu --n-mocks 100 --master-seed 3
```

`submit_TRGBH0_mock_bias.sh` is a thin wrapper for
`mock_TRGB.sh --default-manticore`; all submission options are forwarded.
Outputs land in `results/mocks_TRGB/mock_TRGB_biases_<mode_tag>.npz` with
bias/percentile summary plots alongside; the mode tag encodes field name +
index, likelihood, b_min, and smoothing, e.g.
`TRGB_magnitude_field_ManticoreLocalCOLA3_infersel_student_t_bmin10_smooth4`.

## Local verification (2026-07-17)

A nofield Student-t closure ran on the Mac (150 hosts, 200/200 NUTS):
all 13 tracked parameters within ~1.2 sigma of truth, including
nu_cz = 2.06 +/- 0.49 against injected 2.33 and the inferred selection
edge/width. A PCS field-42 plot-only smoke also completed the full field path:
smoothed recovery volume, masked density-biased host sampling, and smoothed
host LOS interpolation.
