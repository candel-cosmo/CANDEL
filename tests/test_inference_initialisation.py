import numpy as np

from candel.inference import inference


def test_initialise_from_lbfgs_selects_best_start(monkeypatch):
    calls = []
    objectives = {44: 4.0, 45: 2.0, 46: np.inf, 47: 3.0}

    def fake_find_initial_point(*args, seed, **kwargs):
        calls.append(seed)
        if not np.isfinite(objectives[seed]):
            return None, {"x"}, np.inf
        return {"x": seed}, {"x"}, objectives[seed]

    monkeypatch.setattr(
        inference, "find_initial_point", fake_find_initial_point)
    params, site_names, _ = inference._initialise_from_lbfgs(
        None, {}, {"seed": 44, "init_num_starts": 4}, init_maxiter=10)

    assert calls == [44, 45, 46, 47]
    assert params == {"x": 45}
    assert site_names == {"x"}
