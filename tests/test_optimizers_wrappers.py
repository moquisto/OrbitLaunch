import numpy as np
import types
import pytest

import importlib
import main

# Ensure main exposes expected symbols for the optimizer modules.
if not hasattr(main, "MU_EARTH"):
    main.MU_EARTH = 1.0
if not hasattr(main, "R_EARTH"):
    main.R_EARTH = 1.0
if not hasattr(main, "orbital_elements_from_state"):
    from gravity import orbital_elements_from_state as _oes
    main.orbital_elements_from_state = _oes

mto = importlib.import_module("multithread_optimization")
ot = importlib.import_module("optimize_trajectory")


class DummyLog:
    def __init__(self, m0, r_vec, v_vec):
        self.m = [m0 - 10.0]
        self.r = [np.array(r_vec, dtype=float)]
        self.v = [np.array(v_vec, dtype=float)]


def test_multithread_run_simulation_wrapper_success(monkeypatch):
    target_r = mto.R_EARTH + mto.TARGET_ALT_M
    v_circ = np.sqrt(mto.MU_EARTH / target_r)
    dummy_log = DummyLog(m0=1000.0, r_vec=[target_r, 0, 0], v_vec=[0, v_circ, 0])

    state0 = types.SimpleNamespace(m=1000.0)
    monkeypatch.setattr(mto, "build_simulation", lambda: (types.SimpleNamespace(run=lambda *a, **k: dummy_log), state0, 0.0))
    monkeypatch.setattr(mto, "orbital_elements_from_state", lambda r, v, mu: (0.0, target_r, target_r))

    params = np.array([5.0, 200.0, 300.0, 0.8])
    result = mto.run_simulation_wrapper(params)
    assert result["status"] == "PERFECT"
    assert result["error"] < 1e-3


def test_optimize_trajectory_objective_function_penalizes_crash(monkeypatch):
    # Force orbital_elements_from_state to return crash
    def fake_build():
        state0 = types.SimpleNamespace(m=100.0)
        log = DummyLog(m0=100.0, r_vec=[ot.R_EARTH, 0, 0], v_vec=[0, 0, 0])
        sim = types.SimpleNamespace(run=lambda *a, **k: log)
        return sim, state0, 0.0

    monkeypatch.setattr(ot, "build_simulation", fake_build)
    monkeypatch.setattr(ot, "orbital_elements_from_state", lambda r, v, mu: (None, None, None))

    cost = ot.objective_function([5.0, 1000.0, 50000.0, 0.8])
    assert cost >= ot.PENALTY_CRASH
