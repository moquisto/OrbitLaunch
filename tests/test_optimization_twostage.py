import numpy as np
import pytest
from unittest.mock import Mock, patch

from Analysis.config import AnalysisConfig, OptimizationParams # Import OptimizationParams as well
from Environment.config import EnvironmentConfig
from Hardware.config import HardwareConfig
from Software.config import SoftwareConfig
from Main.config import SimulationConfig
from Logging.config import LoggingConfig
from Analysis.optimization import ObjectiveFunctionWrapper, run_simulation_wrapper
from Analysis.cost_functions import PENALTY_CRASH, TARGET_TOLERANCE_M, evaluate_simulation_results

@pytest.fixture
def configs():
    env_config = EnvironmentConfig()
    hw_config = HardwareConfig()
    sw_config = SoftwareConfig()
    sim_config = SimulationConfig()
    log_config = LoggingConfig()
    analysis_config = AnalysisConfig()
    return env_config, hw_config, sw_config, sim_config, log_config, analysis_config

@pytest.fixture
def mock_orbital_elements_success(configs):
    env_config, hw_config, sw_config, sim_config, log_config, analysis_config = configs
    a = env_config.earth_radius_m + sim_config.target_orbit_alt_m + 100
    rp = env_config.earth_radius_m + sim_config.target_orbit_alt_m + 50
    ra = env_config.earth_radius_m + sim_config.target_orbit_alt_m + 150
    return a, rp, ra

@pytest.fixture
def mock_orbital_elements_crash(configs):
    env_config, hw_config, sw_config, sim_config, log_config, analysis_config = configs
    """Mock orbital_elements_from_state for a crash."""
    return None, None, None

@pytest.fixture
def mock_orbital_elements_suborbital(configs):
    env_config, hw_config, sw_config, sim_config, log_config, analysis_config = configs
    """Mock orbital_elements_from_state for a suborbital flight."""
    a = env_config.earth_radius_m + 50_000.0
    rp = env_config.earth_radius_m - 1000.0
    ra = env_config.earth_radius_m + 100_000.0
    return a, rp, ra

@pytest.fixture
def mock_sim_run_result(configs):
    env_config, hw_config, sw_config, sim_config, log_config, analysis_config = configs
    """Mock result from sim.run for successful orbit."""
    r_final = np.array([env_config.earth_radius_m + 200000.0, 0, 0])
    v_final = np.array([0, 7600.0, 0])
    
    class MockLog:
        def __init__(self):
            self.m = [100000.0, 90000.0]
            self.r = [np.array([0,0,0]), r_final]
            self.v = [np.array([0,0,0]), v_final]
            self.altitude = [0, 100000.0, 200000.0]
            self.t_sim = [0, 100, 200] # Added t_sim attribute
            self.orbit_achieved = False # Add this, as it is checked by evaluate_simulation_results

    return MockLog()

@pytest.fixture
def mock_sim_run_crash_result(configs):
    env_config, hw_config, sw_config, sim_config, log_config, analysis_config = configs
    """Mock result from sim.run for a crash."""
    class MockLog:
        def __init__(self):
            self.m = [100000.0, 99000.0]
            self.r = [np.array([0,0,0]), np.array([env_config.earth_radius_m - 10000, 0, 0])]
            self.v = [np.array([0,0,0]), np.array([100, 0, 0])]
            self.altitude = [0, 5000, 1000]
            self.t_sim = [0, 10, 20] # Added t_sim attribute
            self.orbit_achieved = False # Add this, as it is checked by evaluate_simulation_results

    return MockLog()

@pytest.fixture
def mock_sim_run_suborbital_result(configs):
    env_config, hw_config, sw_config, sim_config, log_config, analysis_config = configs
    """Mock result from sim.run for a suborbital flight (no crash, but no orbit)."""
    r_final_sub = np.array([env_config.earth_radius_m + 50000.0, 0, 0])
    v_final_sub = np.array([0, 1000.0, 0])
    
    class MockLog:
        def __init__(self):
            self.m = [100000.0, 95000.0]
            self.r = [np.array([0,0,0]), r_final_sub]
            self.v = [np.array([0,0,0]), v_final_sub]
            self.altitude = [0, 50000.0]
            self.t_sim = [0, 50, 100] # Added t_sim attribute
            self.orbit_achieved = False # Add this, as it is checked by evaluate_simulation_results
    return MockLog()


@pytest.fixture
def mock_build_simulation_base():
    """Base fixture for mocking build_simulation components."""
    with patch('Analysis.optimization.main_orchestrator') as mock_main_orchestrator: # Corrected patch target
        mock_sim = Mock()
        mock_sim.guidance = Mock()
        mock_sim.rocket = Mock()
        mock_sim_config = Mock(main_duration_s=10000.0) # Mock sim_config with main_duration_s
        with patch('Software.guidance.create_pitch_program_callable') as mock_create_pitch: # Corrected patch target
            mock_create_pitch.return_value = Mock(name="mock_pitch_program_callable")
            # main_orchestrator now returns 5 values: sim, state0, t0, log_config, analysis_config
            mock_main_orchestrator.return_value = (mock_sim, Mock(m=100000.0), 0.0, Mock(), mock_sim_config)
            yield mock_main_orchestrator, mock_sim, mock_create_pitch

@pytest.fixture
def mock_build_simulation_success(mock_build_simulation_base, mock_sim_run_result, mock_orbital_elements_success):
    mock_main_orchestrator, mock_sim, mock_create_pitch = mock_build_simulation_base
    mock_sim.run.return_value = mock_sim_run_result
    with patch('Analysis.cost_functions.orbital_elements_from_state') as mock_oe: # Corrected patch target
        mock_oe.return_value = mock_orbital_elements_success
        yield mock_main_orchestrator, mock_sim, mock_create_pitch, mock_oe

@pytest.fixture
def mock_build_simulation_crash(mock_build_simulation_base, mock_sim_run_crash_result, mock_orbital_elements_crash):
    mock_main_orchestrator, mock_sim, mock_create_pitch = mock_build_simulation_base
    mock_sim.run.return_value = mock_sim_run_crash_result
    with patch('Analysis.cost_functions.orbital_elements_from_state') as mock_oe: # Corrected patch target
        mock_oe.return_value = mock_orbital_elements_crash
        yield mock_main_orchestrator, mock_sim, mock_create_pitch, mock_oe

@pytest.fixture
def mock_build_simulation_suborbital(mock_build_simulation_base, mock_sim_run_suborbital_result, mock_orbital_elements_suborbital):
    mock_main_orchestrator, mock_sim, mock_create_pitch = mock_build_simulation_base
    mock_sim.run.return_value = mock_sim_run_suborbital_result
    with patch('Analysis.cost_functions.orbital_elements_from_state') as mock_oe: # Corrected patch target
        mock_oe.return_value = mock_orbital_elements_suborbital
        yield mock_main_orchestrator, mock_sim, mock_create_pitch, mock_oe


def test_run_simulation_wrapper_successful_orbit(mock_build_simulation_success, configs, mock_orbital_elements_success):
    from Analysis.optimization import run_simulation_wrapper # Dynamic import
    env_config, hw_config, sw_config, sim_config, log_config, analysis_config = configs
    mock_build, mock_sim, mock_create_pitch, mock_oe = mock_build_simulation_success # Unpack mock_oe
    scaled_params = [
        6.0, 10.0, 80.0, 50.0, 60.0, 100.0, 40.0, 150.0, 20.0, 200.0, 10.0,
        30.0, 180.0, 10.0, 90.0,
        0.0, 10.0, 60.0, 5.0, 180.0, 0.0,
        0.9, 0.9, 0.9, 0.9, 0.2, 0.5, 0.8, 0.9, 0.9, 0.9, 0.9, 0.2, 0.5, 0.8
    ]
    
    results = run_simulation_wrapper(scaled_params, env_config, hw_config, sw_config, sim_config, log_config, phase=1)
    
    mock_build.assert_called_once()
    assert mock_sim.run.call_count >= 1
    mock_oe.assert_called_once() # Assert on the mock from the fixture
    assert results["status"] == "PERFECT"

def test_run_simulation_wrapper_crash_scenario(mock_build_simulation_crash, configs, mock_orbital_elements_crash):
    from Analysis.optimization import run_simulation_wrapper # Dynamic import
    env_config, hw_config, sw_config, sim_config, log_config, analysis_config = configs
    mock_build, mock_sim, mock_create_pitch, mock_oe = mock_build_simulation_crash # Unpack mock_oe
    scaled_params = [
        6.0, 10.0, 80.0, 50.0, 60.0, 100.0, 40.0, 150.0, 20.0, 200.0, 10.0,
        30.0, 180.0, 10.0, 90.0,
        0.0, 10.0, 60.0, 5.0, 180.0, 0.0,
        0.9, 0.9, 0.9, 0.9, 0.2, 0.5, 0.8, 0.9, 0.9, 0.9, 0.9, 0.2, 0.5, 0.8
    ]
    
    results = run_simulation_wrapper(scaled_params, env_config, hw_config, sw_config, sim_config, log_config, phase=1)
    
    mock_build.assert_called_once()
    assert mock_sim.run.call_count >= 1
    mock_oe.assert_called_once() # Assert on the mock from the fixture
    assert results["status"] == "CRASH"

def test_objective_phase1_success(mock_build_simulation_success, configs):
    env_config, hw_config, sw_config, sim_config, log_config, analysis_config = configs
    with patch('Analysis.optimization.run_simulation_wrapper') as mock_run_sim:
        mock_run_sim.return_value = {"cost": 200.0, "orbital_error": 200.0, "fuel": 10000.0, "status": "PERFECT"}
        objective = ObjectiveFunctionWrapper(
            phase=1, 
            env_config=env_config, 
            hw_config=hw_config, 
            sw_config=sw_config, 
            sim_config=sim_config, 
            log_config=log_config, 
            analysis_config=analysis_config, 
            bounds=[(-1e6, 1e6)] * 35
        )
        scaled_params = np.zeros(35)
        cost = objective(scaled_params)
        assert cost == pytest.approx(200.0)

def test_objective_phase2_success(mock_build_simulation_success, configs):
    env_config, hw_config, sw_config, sim_config, log_config, analysis_config = configs
    with patch('Analysis.optimization.run_simulation_wrapper') as mock_run_sim:
        mock_run_sim.return_value = {"cost": 10000.0, "orbital_error": 0.0, "fuel": 10000.0, "status": "PERFECT"}
        objective = ObjectiveFunctionWrapper(
            phase=2, 
            env_config=env_config, 
            hw_config=hw_config, 
            sw_config=sw_config, 
            sim_config=sim_config, 
            log_config=log_config, 
            analysis_config=analysis_config, 
            bounds=[(-1e6, 1e6)] * 35
        )
        scaled_params = np.zeros(35)
        cost = objective(scaled_params)
        assert cost == pytest.approx(10000.0)


def test_fuel_metric_ignores_stage_mass_drop(configs, mock_orbital_elements_success):
    env_config, hw_config, sw_config, sim_config, log_config, analysis_config = configs

    class MockLog:
        def __init__(self):
            self.t_sim = [0.0, 1.0, 2.0]
            self.mdot = [-10.0, -10.0, -10.0]  # 10 kg/s burn over 2 s => 20 kg burned
            self.m = [100.0, 90.0, 30.0]  # includes a big discrete mass drop
            self.r = [np.zeros(3), np.zeros(3), np.array([env_config.earth_radius_m + 420_000.0, 0.0, 0.0])]
            self.v = [np.zeros(3), np.zeros(3), np.array([0.0, 7_700.0, 0.0])]

    log = MockLog()
    with patch("Analysis.cost_functions.orbital_elements_from_state") as mock_oe:
        mock_oe.return_value = mock_orbital_elements_success
        results = evaluate_simulation_results(
            log=log,
            initial_mass=100.0,
            cfg_env=env_config,
            sim_config=sim_config,
            max_altitude=420_000.0,
            phase=2,
        )

    assert results["fuel"] == pytest.approx(20.0)
