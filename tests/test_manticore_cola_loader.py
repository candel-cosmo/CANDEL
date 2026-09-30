import numpy as np
from h5py import File

from candel.field.loader import (ManticoreLocalCOLA_FieldLoader,
                                 ManticoreLocalSWIFT_FieldLoader,
                                 name2field_loader)


def test_manticore_local_cola_loader_reads_overdensity_and_velocity(tmp_path):
    fpath_root = tmp_path
    mas_root = fpath_root / "CIC"
    mas_root.mkdir()
    fname = mas_root / "mcmc_0.hdf5"

    overdensity = np.arange(8, dtype=np.float32).reshape(2, 2, 2) / 10
    velocity = np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3)

    with File(fname, "w") as f:
        f.attrs["boxsize"] = 123.0
        f.attrs["Omega_m"] = 0.25
        f.attrs["grid_shape"] = np.array([2, 2, 2])
        f.attrs["frame"] = "icrs"
        f.create_dataset("overdensity", data=overdensity)
        f.create_dataset("velocity", data=velocity)

    loader_cls = name2field_loader("ManticoreLocalCOLA")
    assert loader_cls is ManticoreLocalCOLA_FieldLoader

    loader = loader_cls(nsim=0, fpath_root=str(fpath_root))

    np.testing.assert_allclose(loader.load_density(), 1 + overdensity)
    np.testing.assert_allclose(
        loader.load_velocity(), np.moveaxis(velocity, -1, 0))
    np.testing.assert_allclose(
        loader.load_velocity_component(1), velocity[..., 1])
    assert loader.coordinate_frame == "icrs"
    assert loader.boxsize == 123.0
    assert loader.Omega_m == 0.25
    assert loader.ngrid == 2


def test_manticore_local_swift_loader_reads_density_and_momenta(tmp_path):
    fpath_root = tmp_path
    fname = fpath_root / "mcmc_0.hdf5"

    density = np.full((2, 2, 2), 2.0, dtype=np.float32)
    p0 = density * 10
    p1 = density * 20
    p2 = density * 30

    with File(fname, "w") as f:
        f.create_dataset("density", data=density)
        f.create_dataset("p0", data=p0)
        f.create_dataset("p1", data=p1)
        f.create_dataset("p2", data=p2)

    loader_cls = name2field_loader("ManticoreLocalSWIFT")
    assert loader_cls is ManticoreLocalSWIFT_FieldLoader

    loader = loader_cls(nsim=0, fpath_root=str(fpath_root))

    expected_density = density / (loader.boxsize * 1e3 / 2)**3
    np.testing.assert_allclose(loader.load_density(), expected_density)
    np.testing.assert_allclose(
        loader.load_velocity(),
        np.stack([np.full_like(density, 10.0),
                  np.full_like(density, 20.0),
                  np.full_like(density, 30.0)]))
    np.testing.assert_allclose(
        loader.load_velocity_component(1), np.full_like(density, 20.0))
    assert loader.boxsize == 681.1
    assert loader.Omega_m == 0.306
    assert loader.coordinate_frame == "icrs"
    assert loader.fname == str(fname)
