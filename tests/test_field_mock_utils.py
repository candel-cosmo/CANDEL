"""Sky sampling and rejection envelopes used by the field-based mocks."""
import numpy as np

from candel.field.mock_utils import (_density_max_within_radius,
                                     _sample_galactic_masked_xyz)


def test_field_pool_galactic_mask_sampler_obeys_b_min():
    gen = np.random.default_rng(123)

    xyz = _sample_galactic_masked_xyz(
        gen, r_sphere=50.0, pool_size=2000, b_min=10.0, rmin_h=0.1)

    r = np.linalg.norm(xyz, axis=1)
    b = np.rad2deg(np.arcsin(xyz[:, 2] / r))
    assert len(xyz) == 2000
    assert np.min(r) >= 0.1
    assert np.max(r) <= 50.0
    assert np.all(np.abs(b) >= 10.0 - 1.0e-6)


def test_ppc_density_envelope_uses_radius_limited_field_max():
    density = np.ones((4, 4, 4), dtype=float)
    density[0, 0, 0] = 100.0
    density[1, 1, 1] = 5.0

    rho_max = _density_max_within_radius(
        density, boxsize=4.0, observer_pos=np.array([2.0, 2.0, 2.0]),
        radius=1.0)

    assert rho_max == 5.0
