# Running Megamaser Disk Jobs

The megamaser inference path is BlackJAX collapsed Gibbs.

## Entry Points

- `scripts/megamaser/run_maser.py`: unified runner. It defaults to the production Gibbs sampler; use `--sampler de` for the deterministic DE MAP initialiser and diagnostic optimiser.
- `scripts/megamaser/submit.sh`: cluster submission helper for `--sampler gibbs` or `--sampler de`.
- `scripts/megamaser/toy_joint_H0.py`: toy combiner for saved per-galaxy distance posteriors.
- `scripts/megamaser/convergence/*`: numerical quadrature diagnostics for checking the `phi` integrals and conditional-`r` grid accuracy.

## Local Runs

```bash
python scripts/megamaser/run_maser.py NGC5765b
python scripts/megamaser/run_maser.py NGC6264 \
    --num-warmup 100 --num-samples 100 --n-sys 6 --n-red 7 --n-blue 7
```

Use `run_maser.py --help` for the default Gibbs options and
`run_maser.py --sampler de --help` for DE options.

## Cluster Submission

```bash
bash scripts/megamaser/submit.sh -q cmbgpu --galaxy NGC5765b --sampler gibbs
bash scripts/megamaser/submit.sh -q cmbgpu --galaxy NGC5765b,NGC6264 --sampler gibbs
bash scripts/megamaser/submit.sh -q cmbgpu --galaxy NGC5765b --sampler de
```

Useful forwarded options:

| Flag | Meaning |
|------|---------|
| `--num-warmup N` | BlackJAX warmup steps |
| `--num-samples N` | Gibbs samples to save |
| `--n-inner N` | radial Metropolis sweeps per global update |
| `--seed N` | RNG seed |
| `--spot-batch N` | spot-axis chunk size for large phi grids |
| `--f64` | enable JAX float64 |
| `--no-progress` | disable progress bars |
| `--no-ecc`, `--add-ecc` | toggle eccentricity for the selected galaxy |
| `--no-quadratic-warp`, `--add-quadratic-warp` | toggle quadratic warp terms |

DE jobs also accept `--resume` and `--checkpoint-interval-minutes M`.

## Configuration

All galaxies are configured in `scripts/megamaser/config_maser.toml`.

The production sampler marginalises `phi` numerically and samples each `r_ang` directly. These keys control the `phi` grids:

```toml
phi_hv_inner_deg = 30.0
phi_hv_outer_deg = 90.0
n_phi_hv_high = 5001
n_phi_hv_low = 2501
phi_sys_ranges_deg = [[-45, 45], [135, 225]]
n_phi_sys = 5001
```

These keys support the DE initialiser and numerical diagnostics:

```toml
n_r_local = 256
n_r_global = 128
K_sigma = 10.0
refine_r_center = true
n_refine_steps = 32
conditional_spot_batch = 16
```

The `[convergence.fixed_r_reference]` and `[convergence.fixed_r_gradient_reference]` blocks are used only by the quadrature test scripts.

## Numerical Diagnostics

Keep these scripts for checking the `phi` integrals and related gradients:

```bash
python scripts/megamaser/convergence/convergence_phi_marginal.py --galaxies NGC6264 --no-grad
python scripts/megamaser/convergence/check_conditional_r_grad_vs_numerical.py --galaxy NGC6264
```

`r_ang_posteriors.py` and `check_conditional_r_delta.py` are plotting/debugging helpers for the same quadrature machinery.

## Outputs

Production samples are written under `results/Megamaser`, unless `--output` is supplied. The HDF5 files contain sampled global parameters, `r_ang`, derived `D_A`/`M_BH`, and sampler diagnostics.
