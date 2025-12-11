import numpy as np
import pytest
from unittest.mock import Mock, patch

from optimization_twostage import run_simulation_wrapper, soft_bounds_penalty, ObjectiveFunctionWrapper, PENALTY_CRASH, TARGET_TOLERANCE_M, Counter, PERIGEE_FLOOR_M, ECC_TOLERANCE
from config import Config
from gravity import orbital_elements_from_state
from custom_guidance import ParameterizedThrottleProgram

@pytest.fixture
def config():
    return Config()

@pytest.fixture
def mock_orbital_elements_success(config):
    """Mock orbital_elements_from_state for a successful orbit."""
    a = config.earth_radius_m + config.target_orbit_alt_m + 100
    rp = config.earth_radius_m + config.target_orbit_alt_m + 50
    ra = config.earth_radius_m + config.target_orbit_alt_m + 150
    return a, rp, ra

@pytest.fixture
def mock_orbital_elements_crash():
    """Mock orbital_elements_from_state for a crash."""
    return None, None, None

@pytest.fixture
def mock_orbital_elements_suborbital(config):
    """Mock orbital_elements_from_state for a suborbital flight."""
    a = config.earth_radius_m + 50_000.0
    rp = config.earth_radius_m - 1000.0
    ra = config.earth_radius_m + 100_000.0
    return a, rp, ra

@pytest.fixture
def mock_sim_run_result(config):
    """Mock result from sim.run for successful orbit."""
    r_final = np.array([config.earth_radius_m + 200000.0, 0, 0])
    v_final = np.array([0, 7600.0, 0])
    
    class MockLog:
        def __init__(self):
            self.m = [100000.0, 90000.0]
            self.r = [np.array([0,0,0]), r_final]
            self.v = [np.array([0,0,0]), v_final]
            self.altitude = [0, 100000.0, 200000.0]
            self.time = [0, 100, 200]

    return MockLog()

@pytest.fixture
def mock_sim_run_crash_result(config):
    """Mock result from sim.run for a crash."""
    class MockLog:
        def __init__(self):
            self.m = [100000.0, 99000.0]
            self.r = [np.array([0,0,0]), np.array([config.earth_radius_m - 10000, 0, 0])]
            self.v = [np.array([0,0,0]), np.array([100, 0, 0])]
            self.altitude = [0, 5000, 1000]

    return MockLog()

@pytest.fixture
def mock_sim_run_suborbital_result(config):
    """Mock result from sim.run for a suborbital flight (no crash, but no orbit)."""
    r_final_sub = np.array([config.earth_radius_m + 50000.0, 0, 0])
    v_final_sub = np.array([0, 1000.0, 0])
    
    class MockLog:
        def __init__(self):
            self.m = [100000.0, 95000.0]
            self.r = [np.array([0,0,0]), r_final_sub]
            self.v = [np.array([0,0,0]), v_final_sub]
            self.altitude = [0, 50000.0]
    return MockLog()


@pytest.fixture
def mock_build_simulation_base():
    """Base fixture for mocking build_simulation components."""
    with patch('optimization_twostage.build_simulation') as mock_build:
        mock_sim = Mock()
        mock_sim.guidance = Mock()
        mock_sim.rocket = Mock()
        with patch('optimization_twostage.create_pitch_program_callable') as mock_create_pitch:
            mock_create_pitch.return_value = Mock(name="mock_pitch_program_callable")
            mock_build.return_value = (mock_sim, Mock(m=100000.0), 0.0)
            yield mock_build, mock_sim, mock_create_pitch

@pytest.fixture
def mock_build_simulation_success(mock_build_simulation_base, mock_sim_run_result, mock_orbital_elements_success):
    mock_build, mock_sim, mock_create_pitch = mock_build_simulation_base
    mock_sim.run.return_value = mock_sim_run_result
    with patch('optimization_twostage.orbital_elements_from_state') as mock_oe:
        mock_oe.return_value = mock_orbital_elements_success
        yield mock_build, mock_sim, mock_create_pitch, mock_oe

@pytest.fixture
def mock_build_simulation_crash(mock_build_simulation_base, mock_sim_run_crash_result, mock_orbital_elements_crash):
    mock_build, mock_sim, mock_create_pitch = mock_build_simulation_base
    mock_sim.run.return_value = mock_sim_run_crash_result
    with patch('optimization_twostage.orbital_elements_from_state') as mock_oe:
        mock_oe.return_value = mock_orbital_elements_crash
        yield mock_build, mock_sim, mock_create_pitch, mock_oe

@pytest.fixture
def mock_build_simulation_suborbital(mock_build_simulation_base, mock_sim_run_suborbital_result, mock_orbital_elements_suborbital):
    mock_build, mock_sim, mock_create_pitch = mock_build_simulation_base
    mock_sim.run.return_value = mock_sim_run_suborbital_result
    with patch('optimization_twostage.orbital_elements_from_state') as mock_oe:
        mock_oe.return_value = mock_orbital_elements_suborbital
        yield mock_build, mock_sim, mock_create_pitch, mock_oe


def test_run_simulation_wrapper_successful_orbit(mock_build_simulation_success, config):
    mock_build, mock_sim, mock_create_pitch, mock_oe = mock_build_simulation_success
    scaled_params = [
        6.0, 10.0, 80.0, 50.0, 60.0, 100.0, 40.0, 150.0, 20.0, 200.0, 10.0,
        30.0, 180.0, 10.0, 90.0,
        0.0, 10.0, 60.0, 5.0, 180.0, 0.0,
        0.9, 0.9, 0.9, 0.9, 0.2, 0.5, 0.8, 0.9, 0.9, 0.9, 0.9, 0.2, 0.5, 0.8
    ]
    
    results = run_simulation_wrapper(scaled_params, config)
    
    mock_build.assert_called_once()
    assert mock_sim.run.call_count >= 1
    mock_oe.assert_called_once()
    assert results["status"] == "PERFECT"

def test_run_simulation_wrapper_crash_scenario(mock_build_simulation_crash, config):
    mock_build, mock_sim, mock_create_pitch, mock_oe = mock_build_simulation_crash
    scaled_params = [
        6.0, 10.0, 80.0, 50.0, 60.0, 100.0, 40.0, 150.0, 20.0, 200.0, 10.0,
        30.0, 180.0, 10.0, 90.0,
        0.0, 10.0, 60.0, 5.0, 180.0, 0.0,
        0.9, 0.9, 0.9, 0.9, 0.2, 0.5, 0.8, 0.9, 0.9, 0.9, 0.9, 0.2, 0.5, 0.8
    ]
    
    results = run_simulation_wrapper(scaled_params, config)
    
    assert results["status"] == "CRASH"

def test_objective_phase1_success(mock_build_simulation_success, config):
    with patch('optimization_twostage.run_simulation_wrapper') as mock_run_sim:
        mock_run_sim.return_value = {"error": 200.0, "fuel": 10000.0, "status": "PERFECT"}
        objective = ObjectiveFunctionWrapper(phase=1, config=config, bounds=[(-1e6, 1e6)] * 35)
        scaled_params = np.zeros(35)
        cost = objective(scaled_params)
        assert cost == pytest.approx(200.0)

def test_objective_phase2_success(mock_build_simulation_success, config):
    with patch('optimization_twostage.run_simulation_wrapper') as mock_run_sim:
        mock_run_sim.return_value = {"error": 200.0, "fuel": 10000.0, "status": "PERFECT"}
        objective = ObjectiveFunctionWrapper(phase=2, config=config, bounds=[(-1e6, 1e6)] * 35)
        scaled_params = np.zeros(35)
        cost = objective(scaled_params)
        assert cost == pytest.approx(10000.0)
