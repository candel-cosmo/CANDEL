"""Post-hoc stacking of the TRGBH0 single-field posteriors.

Each model variant is run once per Manticore realisation. The field
marginalisation is then the evidence-weighted mixture of the per-field
posteriors, and the equal-weight stack is the conservative alternative that
carries the field-to-field scatter into the quoted uncertainty. Both are
computed here so the table writer and the figure scripts share one
implementation.

Weights are returned per sample rather than resampled, so the summaries and the
KDEs are deterministic.
"""
import glob
import re

import h5py
import numpy as np

LN10 = np.log(10.0)
FIELD_RE = re.compile(r"_field(\d+)_")


def field_paths(pattern):
    """Chains matching a glob, ordered by field index."""
    paths = sorted(glob.glob(str(pattern)),
                   key=lambda path: int(FIELD_RE.search(path).group(1)))
    if not paths:
        raise FileNotFoundError(f"No single-field chains match {pattern}")
    return paths


def load_fields(pattern, keys=("H0",)):
    """Read per-field samples and log-evidences for chains matching a glob.

    Returns the field indices, a {key: [array per field]} dict, and the
    natural-log evidences. A key absent from a chain yields None for that
    field.
    """
    paths = field_paths(pattern)
    index = np.asarray([int(FIELD_RE.search(path).group(1)) for path in paths])
    samples = {key: [] for key in keys}
    lnz = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            lnz.append(float(handle["gof/lnZ_harmonic"][()]))
            for key in keys:
                node = f"samples/{key}"
                samples[key].append(
                    np.asarray(handle[node], dtype=float).reshape(-1)
                    if node in handle else None)
    return index, samples, np.asarray(lnz, dtype=float)


def evidence_weights(lnz):
    """Per-field weights, their effective sample size, and log10 of Z_ens.

    Z_ens is the ensemble evidence, the mean of the per-field evidences, which
    is the marginal evidence of the variant over the realisation ensemble.
    Non-finite entries are dropped, which happens when the harmonic estimator
    fails on a single field.
    """
    finite = np.isfinite(lnz)
    if not finite.any():
        raise ValueError("No finite log-evidences in the ensemble.")
    peak = lnz[finite].max()
    weights = np.where(finite, np.exp(lnz - peak), 0.0)
    weights = weights / weights.sum()
    n_eff = float(1.0 / np.sum(weights**2))
    log_z_ens = float(
        (peak + np.log(np.mean(np.exp(lnz[finite] - peak)))) / LN10)
    return weights, n_eff, log_z_ens


def stacked_samples(per_field, weights=None):
    """Pool per-field samples into (values, per-sample weights).

    With weights None the pooling is equal-weight; otherwise each field's
    samples share that field's weight.
    """
    per_field = [np.asarray(x).reshape(-1) for x in per_field]
    values = np.concatenate(per_field)
    if weights is None:
        return values, np.ones(values.size)
    sample_weights = np.concatenate(
        [np.full(x.size, w / x.size) for x, w in zip(per_field, weights)])
    return values, sample_weights


def weighted_summary(values, sample_weights):
    """Weighted median and standard deviation of a stacked posterior."""
    order = np.argsort(values)
    x, w = values[order], sample_weights[order]
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    median = float(np.interp(0.5, cdf, x))
    mean = np.sum(w * x) / w.sum()
    std = float(np.sqrt(np.sum(w * (x - mean)**2) / w.sum()))
    return median, std


def field_median_scatter(per_field):
    """Standard deviation of the per-field posterior medians."""
    return float(np.std([np.median(x) for x in per_field], ddof=1))


def matched_field_dlogz(index_a, lnz_a, index_b, lnz_b):
    """Mean and scatter of the per-field log10-evidence difference A minus B.

    The comparison is paired on the field index, which is the more stable
    model-comparison statistic when N_eff is unity.
    """
    a = dict(zip(index_a.tolist(), lnz_a.tolist()))
    b = dict(zip(index_b.tolist(), lnz_b.tolist()))
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError("The two ensembles share no field indices.")
    diff = np.asarray([a[i] - b[i] for i in shared]) / LN10
    finite = np.isfinite(diff)
    return float(diff[finite].mean()), float(diff[finite].std(ddof=1))


def demo():
    """Self-check on synthetic fields with a known dominant realisation."""
    rng = np.random.default_rng(0)
    per_field = [rng.normal(loc, 1.0, 4000) for loc in (0.0, 5.0, 10.0)]
    lnz = np.log(10.0) * np.asarray([-100.0, -90.0, -110.0])

    weights, n_eff, log_z_ens = evidence_weights(lnz)
    assert np.isclose(weights[1], 1.0, atol=1e-9), weights
    assert np.isclose(n_eff, 1.0, atol=1e-6), n_eff
    # Z_ens is the mean of the per-field evidences, so it sits log10(3) below
    # the dominant field when that field carries all the weight.
    assert np.isclose(log_z_ens, -90.0 - np.log10(3.0), atol=1e-6), log_z_ens

    values, sample_weights = stacked_samples(per_field, weights)
    median, std = weighted_summary(values, sample_weights)
    assert abs(median - 5.0) < 0.1, median
    assert abs(std - 1.0) < 0.1, std

    values, sample_weights = stacked_samples(per_field)
    median, std = weighted_summary(values, sample_weights)
    assert abs(median - 5.0) < 0.2, median
    assert std > 4.0, std  # equal weighting carries the field-to-field spread

    index = np.arange(3)
    mean, scatter = matched_field_dlogz(index, lnz, index, lnz - np.log(10.0))
    assert np.isclose(mean, 1.0) and np.isclose(scatter, 0.0), (mean, scatter)

    lnz_gap = np.array([np.nan, lnz[1], lnz[2]])
    weights, n_eff, _ = evidence_weights(lnz_gap)
    assert weights[0] == 0.0 and np.isclose(n_eff, 1.0, atol=1e-6)

    print("stack_fields demo OK")


if __name__ == "__main__":
    demo()
