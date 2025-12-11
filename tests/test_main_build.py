import numpy as np

from main import build_simulation
from config import CFG


def test_build_simulation_with_function_pitch(monkeypatch):
    monkeypatch.setattr(CFG, "pitch_guidance_mode", "function")
    monkeypatch.setattr(CFG, "pitch_guidance_function", "custom_guidance.simple_pitch_program")

    sim, state0, _ = build_simulation()

    direction = sim.guidance.pitch_program(0.0, state0)
    assert np.isclose(np.linalg.norm(direction), 1.0)
    # Should align with radial direction for the default simple program
    r_hat = state0.r_eci / np.linalg.norm(state0.r_eci)
    np.testing.assert_allclose(direction, r_hat, atol=1e-6)
