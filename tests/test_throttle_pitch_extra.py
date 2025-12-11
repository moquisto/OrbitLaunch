import numpy as np
import pytest

from main import ParameterizedThrottleProgram
from custom_guidance import create_pitch_program_callable


def test_upper_stage_throttle_interpolation():
    schedule = [[0.0, 0.2], [10.0, 1.0]]
    program = ParameterizedThrottleProgram(schedule=schedule)

    state = type("S", (), {})()
    state.stage_index = 1
    state.upper_ignition_start_time = 100.0

    assert program(95.0, state) == pytest.approx(0.0)  # before ignition
    assert program(100.0, state) == pytest.approx(0.2)  # ignition: first schedule value
    assert program(105.0, state) == pytest.approx(0.6)  # halfway between 0.2 and 1.0
    assert program(112.0, state) == pytest.approx(0.0)  # beyond last point, right=0


def test_pitch_program_extrapolates_endpoints():
    # Ensure interpolation clamps to first/last angles outside the provided time range.
    points = [(0.0, 90.0), (10.0, 0.0)]
    program = create_pitch_program_callable(points)

    state = type("S", (), {})()
    state.r_eci = np.array([0, 0, 1.0])
    state.v_eci = np.array([0, 1.0, 0])

    # Before first time, should use left angle (90 deg -> vertical)
    np.testing.assert_allclose(program(-5.0, state), np.array([0.0, 0.0, 1.0]), atol=1e-6)
    # After last time, should use right angle (0 deg -> horizontal)
    horizontal = np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(program(20.0, state), horizontal, atol=1e-6)
