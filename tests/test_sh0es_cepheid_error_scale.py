"""SH0ES Cepheid error rescaling and host-subset selection (JWST forecast)."""
import os

import numpy as np
import pytest

from candel.pvdata.catalogues import load_SH0ES_separated

ROOT = "data/SH0ES"
HOSTS = ["N3982", "N3254", "M1337"]


@pytest.fixture(scope="module")
def base():
    if not os.path.isdir(ROOT):
        pytest.skip("no SH0ES data")
    return load_SH0ES_separated(ROOT, 3300)


def test_error_scale_is_DCD(base):
    d = load_SH0ES_separated(ROOT, 3300, cepheid_error_scale=0.5,
                             cepheid_error_scale_hosts=HOSTS)
    names = np.array([h[3:] for h in base["host_names"]]
                     + ["N4258", "LMC", "M31"])
    host = names[base["L_Cepheid_host_dist"].argmax(1)]
    f = np.where(np.isin(host, HOSTS), 0.5, 1.0)
    np.testing.assert_allclose(d["C_Cepheid"],
                               base["C_Cepheid"] * np.outer(f, f))
    np.testing.assert_allclose(d["L_Cepheid"] @ d["L_Cepheid"].T,
                               d["C_Cepheid"], atol=1e-10)


def test_keep_hosts(base):
    d = load_SH0ES_separated(ROOT, 3300, keep_hosts=HOSTS)
    assert sorted(h[3:] for h in d["host_names"]) == sorted(HOSTS)
    assert d["num_hosts"] == len(HOSTS)
    assert d["L_Cepheid_host_dist"].shape[1] == len(HOSTS) + 3
    assert len(d["czcmb_cepheid_host"]) == len(HOSTS)


def test_unknown_host_raises(base):
    with pytest.raises(ValueError):
        load_SH0ES_separated(ROOT, 3300, cepheid_error_scale=0.5,
                             cepheid_error_scale_hosts=["N9999"])
