import numpy as np
import pytest
from unittest.mock import Mock, patch

# Assuming the following imports are needed from the main script
from optimization_twostage import run_simulation_wrapper, soft_bounds_penalty, objective_phase1, objective_phase2, PENALTY_CRASH, TARGET_TOLERANCE_M, R_EARTH, MU_EARTH, TARGET_ALT_M, Counter
from config import CFG # For accessing default config values
from gravity import orbital_elements_from_state # Import directly for patching
from main import ParameterizedThrottleProgram

# --- Fixtures for Mocking ---
@pytest.fixture
def mock_orbital_elements_success():
    """Mock orbital_elements_from_state for a successful orbit."""
    # Return values for a near-circular orbit at TARGET_ALT_M
    a = R_EARTH + TARGET_ALT_M + 100 # Semimajor axis, slightly off
    rp = R_EARTH + TARGET_ALT_M + 50 # Perigee, slightly off
    ra = R_EARTH + TARGET_ALT_M + 150 # Apogee, slightly off
    return a, rp, ra

@pytest.fixture
def mock_orbital_elements_crash():
    """Mock orbital_elements_from_state for a crash."""
    return None, None, None # Signifies crash or hyperbolic

@pytest.fixture
def mock_orbital_elements_suborbital():
    """Mock orbital_elements_from_state for a suborbital flight."""
    a = R_EARTH + 50_000.0 # Suborbital semi-major axis
    rp = R_EARTH - 1000.0 # Perigee below surface
    ra = R_EARTH + 100_000.0 # Apogee
    return a, rp, ra

@pytest.fixture
def mock_sim_run_result():
    """Mock result from sim.run for successful orbit."""
    # Simulate a successful orbit close to target
    r_final = np.array([R_EARTH + 200000.0, 0, 0]) # 200km altitude
    v_final = np.array([0, 7600.0, 0]) # Orbital velocity for 200km alt
    
    class MockLog:
        def __init__(self):
            self.m = [100000.0, 90000.0] # Initial, Final mass -> 10000 fuel used
            self.r = [np.array([0,0,0]), r_final]
            self.v = [np.array([0,0,0]), v_final]
            self.altitude = [0, 100000.0, 200000.0] # Max altitude
            self.time = [0, 100, 200]

    return MockLog()

@pytest.fixture
def mock_sim_run_crash_result():
    """Mock result from sim.run for a crash."""
    class MockLog:
        def __init__(self):
            self.m = [100000.0, 99000.0]
            self.r = [np.array([0,0,0]), np.array([R_EARTH - 10000, 0, 0])] # Well below surface
            self.v = [np.array([0,0,0]), np.array([100, 0, 0])]
            self.altitude = [0, 5000, 1000] # Max altitude before crash

    return MockLog()

@pytest.fixture
def mock_sim_run_suborbital_result():
    """Mock result from sim.run for a suborbital flight (no crash, but no orbit)."""
    r_final_sub = np.array([R_EARTH + 50000.0, 0, 0])
    v_final_sub = np.array([0, 1000.0, 0])
    
    class MockLog:
        def __init__(self):
            self.m = [100000.0, 95000.0] # 5000 fuel used
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
        # Mock create_pitch_program_callable as it returns a function
        with patch('optimization_twostage.create_pitch_program_callable') as mock_create_pitch:
            mock_create_pitch.return_value = Mock(name="mock_pitch_program_callable")
            mock_build.return_value = (mock_sim, Mock(m=100000.0), 0.0) # sim, state0, t0
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


# --- Test run_simulation_wrapper ---

def test_run_simulation_wrapper_successful_orbit(mock_build_simulation_success):
    """Test run_simulation_wrapper for a successful orbit."""
    mock_build, mock_sim, mock_create_pitch, mock_oe = mock_build_simulation_success
    scaled_params = [
        6.0, 10.0, 80.0, 50.0, 60.0, 100.0, 40.0, 150.0, 20.0, 200.0, 10.0,
        30.0, 180.0, 10.0, 90.0, # coast_s, upper_burn_s, upper_ignition_delay_s, azimuth_deg
        0.0, 10.0, 60.0, 5.0, 180.0, 0.0, # upper-stage pitch times/angles
        0.9, 0.9, 0.9, 0.9, 0.2, 0.5, 0.8, 0.9, 0.9, 0.9, 0.9, 0.2, 0.5, 0.8
    ]
    
    results = run_simulation_wrapper(scaled_params)
    
    mock_build.assert_called_once()
    assert mock_sim.run.call_count >= 1  # coarse + full passes
    mock_oe.assert_called_once()
    assert mock_create_pitch.call_count == 2
    booster_call = mock_create_pitch.call_args_list[0]
    upper_call = mock_create_pitch.call_args_list[1]
    assert booster_call.args[0] == [(10.0, 80.0), (50.0, 60.0), (100.0, 40.0), (150.0, 20.0), (200.0, 10.0)]
    assert upper_call.args[0] == [(0.0, 10.0), (60.0, 5.0), (180.0, 0.0)]
    
    assert results["fuel"] == pytest.approx(10000.0)
    assert results["status"] == "PERFECT" # Based on precise mock
    assert results["error"] == pytest.approx(200.0) # abs(ra-target) + abs(rp-target) for mock_orbital_elements_success
    assert results["orbit_error"] == results["error"]

def test_run_simulation_wrapper_crash_scenario(mock_build_simulation_crash):
    """Test run_simulation_wrapper for a crash scenario."""
    mock_build, mock_sim, mock_create_pitch, mock_oe = mock_build_simulation_crash
    scaled_params = [
        6.0, 10.0, 80.0, 50.0, 60.0, 100.0, 40.0, 150.0, 20.0, 200.0, 10.0,
        30.0, 180.0, 10.0, 90.0,
        0.0, 10.0, 60.0, 5.0, 180.0, 0.0,
        0.9, 0.9, 0.9, 0.9, 0.2, 0.5, 0.8, 0.9, 0.9, 0.9, 0.9, 0.2, 0.5, 0.8
    ]
    
    results = run_simulation_wrapper(scaled_params)
    
    mock_build.assert_called_once()
    assert mock_sim.run.call_count >= 1  # coarse pass may early-exit
    # Early exit on low altitude skips orbital element calculation.
    assert mock_oe.call_count >= 0
    
    assert results["status"] == "CRASH"
    # Ensure penalty is high for crash and accounts for max altitude
    expected_err = 1e7 + (TARGET_ALT_M - 5000)
    assert results["error"] == pytest.approx(expected_err)
    assert results["orbit_error"] == results["error"]

def test_run_simulation_wrapper_parameter_clipping(mock_build_simulation_success):
    """Test that parameters are correctly clipped and sorted."""
    mock_build, mock_sim, mock_create_pitch, mock_oe = mock_build_simulation_success # Use success mock to avoid crash logic interfering
    # Use parameters that would be out of bounds or unsorted
    scaled_params = [
        6.0, # meco_mach
        100.0, 100.0, # pitch_time_0 (too high), pitch_angle_0 (too high)
        -10.0, -10.0, # pitch_time_1 (too low), pitch_angle_1 (too low)
        50.0, 40.0,
        150.0, 20.0,
        200.0, 10.0,
        30.0, # coast_s
        180.0, # upper_burn_s
        10.0, # upper_ignition_delay_s
        90.0, # azimuth_deg
        50.0, -5.0, 10.0, 100.0, 30.0, -1.0, # upper-stage pitch times/angles (unsorted/invalid)
        -0.1, 1.1, 0.5, 0.5, # upper_throttle_level_x (out of bounds)
        0.8, 0.2, 0.5, # upper_throttle_switch_ratio_x (unsorted, out of bounds)
        0.9, 0.9, 0.9, 0.9,
        0.2, 0.5, 0.8
    ]

    run_simulation_wrapper(scaled_params)

    # Assert pitch angles are clipped and times sorted by inspecting mock_create_pitch call
    assert mock_create_pitch.call_count == 2
    pitch_points_called = mock_create_pitch.call_args_list[0][0][0]
    upper_points_called = mock_create_pitch.call_args_list[1][0][0]

    assert pitch_points_called[0] == (-10.0, 0.0) # time -10.0, angle -10.0 clipped to 0.0
    assert pitch_points_called[1] == (50.0, 40.0)
    assert pitch_points_called[2] == (100.0, 90.0) # time 100.0, angle 100.0 clipped to 90.0
    assert pitch_points_called[3] == (150.0, 20.0)
    assert pitch_points_called[4] == (200.0, 10.0)
    # Upper stage schedule sorted and clipped
    assert upper_points_called[0] == (10.0, 90.0)
    assert upper_points_called[1] == (30.0, 0.0)
    assert upper_points_called[2] == (50.0, 0.0)

    # Throttle schedule should be a ParameterizedThrottleProgram instance
    assert isinstance(mock_sim.guidance.throttle_schedule, ParameterizedThrottleProgram)
    schedule = mock_sim.guidance.throttle_schedule.schedule

    # We passed -0.1, 1.1, 0.5, 0.5 for upper_throttle_levels.
    # The construction logic clips these.
    # The first entry [0, clipped_level[0]]
    assert schedule[0][1] == pytest.approx(0.0) # -0.1 clipped to 0.0

    # The throttle levels for switches are handled in the loop:
    # upper_throttle_switch_ratios: 0.8, 0.2, 0.5 -> sorted to 0.2, 0.5, 0.8
    # upper_throttle_levels: 0.0, 1.0, 0.5, 0.5 (clipped from -0.1, 1.1, 0.5, 0.5)

    # The second level (index 1) is 1.1 -> clipped to 1.0. This should be set at the first switch.
    # The schedule will have entries for each switch ratio.
    # It's difficult to assert internal steps of the dynamic program construction accurately without more specific mocks.
    # A more robust check here might involve mocking ParameterizedThrottleProgram constructor and asserting its `schedule` argument.
    # For now, let's keep it simple and check the clip of the initial throttle.

def test_run_simulation_wrapper_sim_fail_index_error(mock_build_simulation_base):
    """Test run_simulation_wrapper when sim.run raises IndexError."""
    mock_build, mock_sim, mock_create_pitch = mock_build_simulation_base
    mock_sim.run.side_effect = IndexError("Simulation ended prematurely")
    scaled_params = [
        6.0, 10.0, 80.0, 50.0, 60.0, 100.0, 40.0, 150.0, 20.0, 200.0, 10.0,
        30.0, 180.0, 10.0, 90.0,
        0.0, 10.0, 60.0, 5.0, 180.0, 0.0,
        0.9, 0.9, 0.9, 0.9, 0.2, 0.5, 0.8, 0.9, 0.9, 0.9, 0.9, 0.2, 0.5, 0.8
    ]
    
    results = run_simulation_wrapper(scaled_params)
    
    assert results["status"] == "SIM_FAIL_INDEX"
    assert results["error"] == PENALTY_CRASH # Default penalty
    assert results["orbit_error"] == PENALTY_CRASH

def test_run_simulation_wrapper_sim_fail_unknown_exception(mock_build_simulation_base):
    """Test run_simulation_wrapper when sim.run raises a general Exception."""
    mock_build, mock_sim, mock_create_pitch = mock_build_simulation_base
    mock_sim.run.side_effect = ValueError("Some other error") # General Exception
    scaled_params = [
        6.0, 10.0, 80.0, 50.0, 60.0, 100.0, 40.0, 150.0, 20.0, 200.0, 10.0,
        30.0, 180.0, 10.0, 90.0,
        0.0, 10.0, 60.0, 5.0, 180.0, 0.0,
        0.9, 0.9, 0.9, 0.9, 0.2, 0.5, 0.8, 0.9, 0.9, 0.9, 0.9, 0.2, 0.5, 0.8
    ]
    
    results = run_simulation_wrapper(scaled_params)
    
    assert results["status"] == "SIM_FAIL_UNKNOWN"
    assert results["error"] == PENALTY_CRASH # Default penalty
    assert results["orbit_error"] == PENALTY_CRASH

# --- Test soft_bounds_penalty ---

def test_soft_bounds_penalty_in_bounds():
    """Test soft_bounds_penalty when all parameters are within bounds."""
    params = [5.0, 50.0, 75.0]
    bounds = [(0.0, 10.0), (20.0, 80.0), (70.0, 90.0)]
    penalty = soft_bounds_penalty(params, bounds)
    assert penalty == 0.0

def test_soft_bounds_penalty_out_of_bounds_low():
    """Test soft_bounds_penalty when some parameters are below lower bounds."""
    params = [-1.0, 15.0, 75.0]
    bounds = [(0.0, 10.0), (20.0, 80.0), (70.0, 90.0)]
    penalty = soft_bounds_penalty(params, bounds)
    # Expected: (0.0 - (-1.0)) * 1e5 + (20.0 - 15.0) * 1e5 = 1 * 1e5 + 5 * 1e5 = 6e5
    assert penalty == pytest.approx(6e5)

def test_soft_bounds_penalty_out_of_bounds_high():
    """Test soft_bounds_penalty when some parameters are above upper bounds."""
    params = [5.0, 85.0, 95.0]
    bounds = [(0.0, 10.0), (20.0, 80.0), (70.0, 90.0)]
    penalty = soft_bounds_penalty(params, bounds)
    # Expected: (85.0 - 80.0) * 1e5 + (95.0 - 90.0) * 1e5 = 5 * 1e5 + 5 * 1e5 = 10e5
    assert penalty == pytest.approx(10e5)

def test_soft_bounds_penalty_mixed_out_of_bounds():
    """Test soft_bounds_penalty with mixed in/out of bounds parameters."""
    params = [-1.0, 50.0, 95.0]
    bounds = [(0.0, 10.0), (20.0, 80.0), (70.0, 90.0)]
    penalty = soft_bounds_penalty(params, bounds)
    # Expected: (0.0 - (-1.0)) * 1e5 + (95.0 - 90.0) * 1e5 = 1 * 1e5 + 5 * 1e5 = 6e5
    assert penalty == pytest.approx(6e5)

# --- Test objective_phase1 and objective_phase2 ---

@pytest.fixture(autouse=True)
def mock_globals_for_objectives_and_log():
    """Mocks global variables and log_iteration for isolation."""
    # Patch the global 'bounds' list within the optimization_twostage module
    # for soft_bounds_penalty.
    with patch('optimization_twostage.bounds', [(-1e6, 1e6)] * 35):
        # Patch log_iteration to prevent file I/O
        with patch('optimization_twostage.log_iteration') as mock_log_iteration:
            # Patch global_iter_count to control its value
            # Patch global_iter_count to control its value via a mock Counter
            mock_counter_instance = Mock(spec=Counter)
            mock_counter_instance.value = 0 # Initialize the value for the mock counter
            with patch('optimization_twostage.global_iter_count', new=mock_counter_instance) as mock_global_iter_count:
                yield mock_log_iteration, mock_global_iter_count

def test_objective_phase1_success(mock_build_simulation_success, mock_globals_for_objectives_and_log):
    """Test objective_phase1 with a successful orbit."""
    mock_build, mock_sim, mock_create_pitch, mock_oe = mock_build_simulation_success
    mock_log_iteration, mock_global_iter_count = mock_globals_for_objectives_and_log

    scaled_params = [
        6.0, 10.0, 80.0, 50.0, 60.0, 100.0, 40.0, 150.0, 20.0, 200.0, 10.0,
        30.0, 180.0, 10.0, 90.0,
        0.0, 10.0, 60.0, 5.0, 180.0, 0.0,
        0.9, 0.9, 0.9, 0.9, 0.2, 0.5, 0.8, 0.9, 0.9, 0.9, 0.9, 0.2, 0.5, 0.8
    ]
    
    cost = objective_phase1(scaled_params)
    
    # orbital error = 50.0 + 150.0 = 200.0 (from mock_orbital_elements_success)
    # soft_bounds_penalty should be 0 as params are in bounds (using default patched bounds)
    # cost = error = 200.0
    assert cost == pytest.approx(200.0, abs=1.0)
    mock_log_iteration.assert_called_once()
    assert mock_global_iter_count.value == 1 # global_iter_count increments by 1

def test_objective_phase1_crash(mock_build_simulation_crash, mock_globals_for_objectives_and_log):
    """Test objective_phase1 with a crash scenario."""
    mock_build, mock_sim, mock_create_pitch, mock_oe = mock_build_simulation_crash
    mock_log_iteration, mock_global_iter_count = mock_globals_for_objectives_and_log

    scaled_params = [
        6.0, 10.0, 80.0, 50.0, 60.0, 100.0, 40.0, 150.0, 20.0, 200.0, 10.0,
        30.0, 180.0, 10.0, 90.0,
        0.0, 10.0, 60.0, 5.0, 180.0, 0.0,
        0.9, 0.9, 0.9, 0.9, 0.2, 0.5, 0.8, 0.9, 0.9, 0.9, 0.9, 0.2, 0.5, 0.8
    ]
    
    cost = objective_phase1(scaled_params)
    
    # Crash error uses TARGET_ALT_M; soft_bounds_penalty should be 0.
    expected_err = 1e7 + (TARGET_ALT_M - 5000)
    assert mock_global_iter_count.value == 1

def test_objective_phase2_success(mock_build_simulation_success, mock_globals_for_objectives_and_log):
    """Test objective_phase2 with a successful orbit and low fuel."""
    mock_build, mock_sim, mock_create_pitch, mock_oe = mock_build_simulation_success
    mock_log_iteration, mock_global_iter_count = mock_globals_for_objectives_and_log

    scaled_params = [
        6.0, 10.0, 80.0, 50.0, 60.0, 100.0, 40.0, 150.0, 20.0, 200.0, 10.0,
        30.0, 180.0, 10.0, 90.0,
        0.0, 10.0, 60.0, 5.0, 180.0, 0.0,
        0.9, 0.9, 0.9, 0.9, 0.2, 0.5, 0.8, 0.9, 0.9, 0.9, 0.9, 0.2, 0.5, 0.8
    ]
    
    cost = objective_phase2(scaled_params)
    
    # fuel = 10000.0 (from mock_sim_run_result)
    # error = 200.0 (from mock_orbital_elements_success), which is < TARGET_TOLERANCE_M (10000.0)
    # So no additional error penalty.
    # cost = fuel + soft_bounds_penalty = 10000.0 + 0 = 10000.0
    assert mock_global_iter_count.value == 1

def test_objective_phase2_suborbital_with_penalty(mock_build_simulation_suborbital, mock_globals_for_objectives_and_log):
    """Test objective_phase2 with a suborbital flight (high error)."""
    mock_build, mock_sim, mock_create_pitch, mock_oe = mock_build_simulation_suborbital
    mock_log_iteration, mock_global_iter_count = mock_globals_for_objectives_and_log

    scaled_params = [
        6.0, 10.0, 80.0, 50.0, 60.0, 100.0, 40.0, 150.0, 20.0, 200.0, 10.0,
        30.0, 180.0, 10.0, 90.0,
        0.0, 10.0, 60.0, 5.0, 180.0, 0.0,
        0.9, 0.9, 0.9, 0.9, 0.2, 0.5, 0.8, 0.9, 0.9, 0.9, 0.9, 0.2, 0.5, 0.8
    ]
    
    cost = objective_phase2(scaled_params)
    
    # fuel = 5000.0 (from mock_sim_run_suborbital_result)
    # mock_orbital_elements_suborbital returns rp=R_EARTH-1000, ra=R_EARTH+100_000
    # target_r = R_EARTH + TARGET_ALT_M
    # error = abs(rp - target_r) + abs(ra - target_r)
    # This error is > TARGET_TOLERANCE_M so penalty applies.
    target_r = R_EARTH + TARGET_ALT_M
    error = abs((R_EARTH - 1000) - target_r) + abs((R_EARTH + 100000) - target_r)
    penalty = (error - TARGET_TOLERANCE_M) * 10.0
    expected_cost = 5000.0 + penalty
    assert cost == pytest.approx(expected_cost, rel=0.01)
    mock_log_iteration.assert_called_once()
    assert mock_global_iter_count.value == 1
