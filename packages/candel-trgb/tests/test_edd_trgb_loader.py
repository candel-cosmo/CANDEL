import numpy as np

from candel.util import SPEED_OF_LIGHT, data_path
from candel_trgb import load_EDD_TRGB

DATA_ROOT = data_path("data", "EDD_TRGB")


def test_edd_trgb_loader_matches_sample_summary():
    data = load_EDD_TRGB(str(DATA_ROOT))

    assert len(data["mag"]) == 401
    assert np.all(data["mag"] >= 22.1)
    assert np.all(np.isfinite(data["mag"]))
    assert np.all(np.isfinite(data["colour_dered"]))
    assert np.all(np.isfinite(data["e_colour_dered"]))
    assert np.all(np.isfinite(data["colour_606_814"])
                  | (data["e_colour_dered"] == 0))
    assert np.all(np.abs(data["zcmb"] * SPEED_OF_LIGHT) < 9999)
    assert "M_TRGB" not in data
    assert "DM_tip" not in data

    # First retained row, UGC12894: T814 - A_814 = 25.83 - 0.166.
    # This differs from the EDD colour-corrected DM_tip - 4.06 value.
    assert np.isclose(data["mag"][0], 25.83 - 0.166)
    assert not np.isclose(data["mag"][0], 29.78 - 4.06)
    assert np.isclose(data["colour_dered"][0], 0.98)
    assert np.isclose(data["colour_606_814"][0], 1.06)
    assert np.isclose(data["e_colour_dered"][0], (1.13 - 0.99) / 2)
