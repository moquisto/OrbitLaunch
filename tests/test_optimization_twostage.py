import numpy as np
import types
import pytest

import optimization_twostage as opt
from config import CFG


class DummyLog:
    def __init__(self, m0, r_vec, v_vec):
        self.m = [m0 - 100.0]
        self.altitude = [1000.0]
        self.r = [np.array(r_vec, dtype=float)]
        self.v = [np.array(v_vec, dtype=float)]


class DummyGuidance:
    def __init__(self):
        self.pitch_program = None
        self.throttle_schedule = None


class DummyRocket:
    def __init__(self):
        self.booster_throttle_program = None


class DummySimulation:
    def __init__(self, log):
        self.guidance = DummyGuidance()
        self.rocket = DummyRocket()
        self._log = log

    def run(self, t0, duration, dt, state0, orbit_target_radius=None, exit_on_orbit=False):
        return self._log


class RecordingThrottleProgram:
    def __init__(self, schedule, apply_to_stage0=False):
        self.schedule = schedule
        self.apply_to_stage0 = apply_to_stage0

    def __call__(self, t, state):
        return 1.0


@pytest.fixture
def reset_cfg(monkeypatch):
    """Ensure CFG mutations in the wrapper don't leak across tests."""
    keys = [
        "pitch_guidance_mode",
        "pitch_guidance_function",
        "meco_mach",
        "separation_delay_s",
        "upper_ignition_delay_s",
        "upper_stage_throttle_program",
        "booster_throttle_program",
        "orbit_alt_tol",
        "exit_on_orbit",
    ]
    saved = {k: getattr(CFG, k) for k in keys}
    yield
    for k, v in saved.items():
        setattr(CFG, k, v)


def test_run_simulation_wrapper_success(monkeypatch, reset_cfg):
    target_r = opt.R_EARTH + opt.TARGET_ALT_M
    v_circ = np.sqrt(opt.MU_EARTH / target_r)
    dummy_log = DummyLog(m0=1000.0, r_vec=[target_r, 0, 0], v_vec=[0, v_circ, 0])

    state0 = types.SimpleNamespace(m=1000.0)
    monkeypatch.setattr(opt, "build_simulation", lambda: (DummySimulation(dummy_log), state0, 0.0))
    monkeypatch.setattr(opt, "orbital_elements_from_state", lambda r, v, mu: (0.0, target_r, target_r))

    captured = []

    def record_program(*args, **kwargs):
        prog = RecordingThrottleProgram(*args, **kwargs)
        captured.append(prog)
        return prog

    monkeypatch.setattr(opt, "ParameterizedThrottleProgram", record_program)

    params = np.array([
        5.0, 10, 80, 50, 60, 200, 40, 300, 20, 500, 5,
        100, 150,
        0.6, 0.7, 0.8, 0.9,
        0.2, 0.4, 0.8,
        0.6, 0.7, 0.8, 0.9,
        0.2, 0.4, 0.8
    ], dtype=float)

    results = opt.run_simulation_wrapper(params)

    assert results["status"] == "PERFECT"
    assert results["error"] < 1e-3
    assert results["fuel"] > 0
    # Booster program should be configured to apply on stage 0
    assert any(p.apply_to_stage0 for p in captured)
    # Pitch guidance should be switched to parameterized for safe import
    assert CFG.pitch_guidance_mode == "parameterized"


def test_run_simulation_wrapper_crash_penalty(monkeypatch, reset_cfg):
    state0 = types.SimpleNamespace(m=500.0)
    dummy_log = DummyLog(m0=500.0, r_vec=[opt.R_EARTH, 0, 0], v_vec=[0, 0, 0])

    monkeypatch.setattr(opt, "build_simulation", lambda: (DummySimulation(dummy_log), state0, 0.0))
    monkeypatch.setattr(opt, "orbital_elements_from_state", lambda r, v, mu: (None, None, None))

    params = np.ones(27, dtype=float)
    results = opt.run_simulation_wrapper(params)

    assert results["status"] == "CRASH"
    assert results["error"] > 1e6  # ensure we hit the large crash penalty path
