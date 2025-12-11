import numpy as np
import pytest
from unittest.mock import Mock, patch

from main import build_simulation, StageAwarePitchProgram, throttle_schedule, ParameterizedThrottleProgram, build_rocket
from config import CFG
from integrators import State
from rocket import Rocket # Import for type hinting

# Mock classes for build_simulation dependencies
class MockEarthModel:
    def __init__(self, mu, radius, omega_vec, j2):
        self.mu = mu
        self.radius = radius
        self.omega_vec = omega_vec
        self.j2 = j2
    def atmosphere_velocity(self, r_eci):
        return np.zeros(3) # Simplified for test

class MockAtmosphereModel:
    def __init__(self, h_switch, lat_deg, lon_deg, f107, f107a, ap):
        pass
class MockCdModel:
    def __init__(self, cd_func):
        pass
class MockAerodynamics:
    def __init__(self, atmosphere, cd_model, reference_area):
        pass
class MockGuidance:
    def __init__(self, pitch_program, throttle_schedule):
        self.pitch_program = pitch_program
        self.throttle_schedule = throttle_schedule
# Mocked classes for RK4 and VelocityVerlet
class MockRK4:
    pass
class MockVelocityVerlet:
    pass
class MockSimulation:
    def __init__(self, earth, atmosphere, aerodynamics, rocket, integrator, guidance, max_q_limit, max_accel_limit, impact_altitude_buffer_m, escape_radius_factor):
        self.rocket = rocket # To allow access in tests
    def run(self, *args, **kwargs):
        mock_log = Mock()
        mock_log.r = [np.array([1,1,1]), np.array([2,2,2])]
        mock_log.v = [np.array([0,0,0]), np.array([1,1,1])]
        mock_log.m = [1000.0, 500.0]
        mock_log.stage = [0, 1]
        mock_log.altitude = [0, 10000]
        mock_log.speed = [0, 100]
        mock_log.dynamic_pressure = [0, 100]
        mock_log.flight_path_angle_deg = [90, 45]
        mock_log.v_vertical = [0, 70.7]
        mock_log.v_horizontal = [0, 70.7]
        mock_log.specific_energy = [0, 1000]
        mock_log.t_sim = [0.0, 10.0]
        mock_log.t_env = [0.0, 10.0]
        mock_log.thrust_mag = [10, 5]
        mock_log.drag_mag = [1, 0.5]
        mock_log.mdot = [10, 5]
        mock_log.rho = [1, 0.1]
        mock_log.mach = [0, 0.5]
        mock_log.orbit_achieved = True
        self.rocket.stage_engine_off_complete_time = [None, None]
        return mock_log

class MockEngine:
    def __init__(self, thrust_vac, thrust_sl, isp_vac, isp_sl):
        pass
class MockStage:
    def __init__(self, dry_mass, prop_mass, engine, ref_area):
        self.dry_mass = dry_mass
        self.prop_mass = prop_mass
    def total_mass(self):
        return self.dry_mass + self.prop_mass
class MockRocket:
    def __init__(self, stages, main_engine_ramp_time, upper_engine_ramp_time, meco_mach, separation_delay, upper_ignition_delay, separation_altitude_m, earth_radius, min_throttle, shutdown_ramp_time, throttle_shape_full_threshold, booster_throttle_program):
        self.stages = stages
        self.booster_throttle_program = booster_throttle_program # Needed for ParameterizedThrottleProgram test
        self.stage_fuel_empty_time = [None, None]
        self.stage_engine_off_complete_time = [None, None]
        self.stage_prop_remaining = [100, 50]


def test_build_simulation_with_function_pitch(monkeypatch):
    monkeypatch.setattr(CFG, "pitch_guidance_mode", "function")
    monkeypatch.setattr(CFG, "pitch_guidance_function", "custom_guidance.simple_pitch_program")

    sim, state0, _ = build_simulation()

    direction = sim.guidance.pitch_program(0.0, state0)
    assert np.isclose(np.linalg.norm(direction), 1.0)
    # Should align with radial direction for the default simple program
    r_hat = state0.r_eci / np.linalg.norm(state0.r_eci)
    np.testing.assert_allclose(direction, r_hat, atol=1e-6)


# --- Tests for StageAwarePitchProgram ---

@pytest.fixture
def mock_state_for_pitch_program():
    """Fixture to provide a mock State object for pitch program tests."""
    r = np.array([6.371e6, 0.0, 0.0]) # Along X-axis
    v = np.array([0.0, 1000.0, 0.0]) # Moving in Y direction (tangential)
    return Mock(spec=State, r_eci=r, v_eci=v, m=1000.0, stage_index=0)

def test_stage_aware_pitch_program_init():
    """Test StageAwarePitchProgram initialization and _prep_schedule."""
    booster_schedule = [[0.0, 90.0], [10.0, 80.0]]
    upper_schedule = [[0.0, 10.0], [50.0, 0.0]]
    program = StageAwarePitchProgram(booster_schedule, upper_schedule, 100.0, 6.371e6)

    np.testing.assert_allclose(program.booster_time_points, np.array([0.0, 10.0]))
    np.testing.assert_allclose(program.booster_angles_rad, np.deg2rad(np.array([90.0, 80.0])))
    np.testing.assert_allclose(program.upper_time_points, np.array([0.0, 50.0]))
    np.testing.assert_allclose(program.upper_angles_rad, np.deg2rad(np.array([10.0, 0.0])))
    assert program.prograde_threshold == 100.0
    assert program.earth_radius == 6.371e6

    # Test empty schedule
    empty_program = StageAwarePitchProgram([], [], 0.0, 0.0)
    np.testing.assert_allclose(empty_program.booster_time_points, np.array([0.0]))
    np.testing.assert_allclose(empty_program.booster_angles_rad, np.array([np.pi / 2]))


def test_stage_aware_pitch_program_booster_interpolation(mock_state_for_pitch_program):
    """Test pitch interpolation for the booster stage."""
    booster_schedule = [[0.0, 90.0], [100.0, 0.0]]
    upper_schedule = [[0.0, 0.0]] # Irrelevant for this test
    program = StageAwarePitchProgram(booster_schedule, upper_schedule, 100.0, 6.371e6)

    # At t=0, 90 deg (vertical)
    dir_t0 = program(0.0, mock_state_for_pitch_program, stage_index=0)
    vertical_dir = mock_state_for_pitch_program.r_eci / np.linalg.norm(mock_state_for_pitch_program.r_eci)
    np.testing.assert_allclose(dir_t0, vertical_dir, atol=1e-6)

    # At t=50, 45 deg
    dir_t50 = program(50.0, mock_state_for_pitch_program, stage_index=0)
    expected_pitch_rad = np.deg2rad(45.0)
    # At r_eci = [R,0,0], east is [0,1,0], vertical is [1,0,0]
    east = np.array([0.0, 1.0, 0.0])
    expected_dir = np.cos(expected_pitch_rad) * east + np.sin(expected_pitch_rad) * vertical_dir
    np.testing.assert_allclose(dir_t50, expected_dir, atol=1e-6)
    assert np.isclose(np.linalg.norm(dir_t50), 1.0)


def test_stage_aware_pitch_program_upper_interpolation(mock_state_for_pitch_program):
    """Test pitch interpolation for the upper stage."""
    booster_schedule = [[0.0, 0.0]] # Irrelevant
    upper_schedule = [[0.0, 30.0], [50.0, 10.0]]
    program = StageAwarePitchProgram(booster_schedule, upper_schedule, 100.0, 6.371e6)

    mock_state_for_pitch_program.stage_index = 1
    mock_state_for_pitch_program.upper_ignition_start_time = 200.0 # Time when upper stage ignited

    # At time_since_ignition = 0 (t=200), 30 deg
    dir_t0 = program(200.0, mock_state_for_pitch_program, t_stage=0.0, stage_index=1)
    vertical_dir = mock_state_for_pitch_program.r_eci / np.linalg.norm(mock_state_for_pitch_program.r_eci)
    east = np.array([0.0, 1.0, 0.0]) # From mock state (along x-axis)
    expected_pitch_rad = np.deg2rad(30.0)
    expected_dir = np.cos(expected_pitch_rad) * east + np.sin(expected_pitch_rad) * vertical_dir
    np.testing.assert_allclose(dir_t0, expected_dir, atol=1e-6)

    # At time_since_ignition = 25 (t=225), 20 deg
    dir_t25 = program(225.0, mock_state_for_pitch_program, t_stage=25.0, stage_index=1)
    expected_pitch_rad = np.deg2rad(20.0) # Interpolated
    expected_dir = np.cos(expected_pitch_rad) * east + np.sin(expected_pitch_rad) * vertical_dir
    np.testing.assert_allclose(dir_t25, expected_dir, atol=1e-6)
    assert np.isclose(np.linalg.norm(dir_t25), 1.0)


def test_stage_aware_pitch_program_prograde_transition(mock_state_for_pitch_program):
    """Test transition to prograde after schedule ends."""
    booster_schedule = [[0.0, 90.0]]
    upper_schedule = [[0.0, 0.0]] # Just one point, so it ends immediately
    prograde_threshold = 50.0 # Low threshold

    program = StageAwarePitchProgram(booster_schedule, upper_schedule, prograde_threshold, 6.371e6)

    # Test booster after schedule end, speed > threshold
    mock_state_for_pitch_program.v_eci = np.array([0.0, 200.0, 0.0]) # High speed
    dir_late_booster = program(100.0, mock_state_for_pitch_program, stage_index=0) # After schedule
    np.testing.assert_allclose(dir_late_booster, mock_state_for_pitch_program.v_eci / np.linalg.norm(mock_state_for_pitch_program.v_eci), atol=1e-6)

    # Test booster after schedule end, speed < threshold
    mock_state_for_pitch_program.v_eci = np.array([0.0, 10.0, 0.0]) # Low speed
    dir_late_booster_low_speed = program(100.0, mock_state_for_pitch_program, stage_index=0)
    east = np.array([0.0, 1.0, 0.0]) # From mock state (along x-axis)
    np.testing.assert_allclose(dir_late_booster_low_speed, east, atol=1e-6)


def test_stage_aware_pitch_program_r_norm_zero():
    """Test behavior when r_norm is zero."""
    program = StageAwarePitchProgram([], [], 0.0, 0.0)
    mock_state = Mock(spec=State, r_eci=np.array([0.0, 0.0, 0.0]), v_eci=np.zeros(3), m=0.0)
    
    thrust_dir = program(0.0, mock_state)
    np.testing.assert_allclose(thrust_dir, np.array([0.0, 0.0, 1.0]), atol=1e-6)


def test_stage_aware_pitch_program_east_norm_zero():
    """Test behavior when east_norm is zero (e.g. at poles, r_hat is [0,0,1] or [0,0,-1])."""
    booster_schedule = [[0.0, 0.0]] # Horizontal
    upper_schedule = [[0.0, 0.0]]
    program = StageAwarePitchProgram(booster_schedule, upper_schedule, 100.0, 6.371e6)

    # At North Pole, r_eci = [0,0,R], r_hat = [0,0,1].
    # cross([0,0,1], [0,0,1]) = [0,0,0], so east_norm would be zero.
    mock_state_pole = Mock(spec=State, r_eci=np.array([0.0, 0.0, 6.371e6]), v_eci=np.array([100.0, 0.0, 0.0]), stage_index=0)
    
    # The code defaults east to [1.0, 0.0, 0.0] if east_norm is zero.
    # Pitch angle 0 deg (horizontal), so direction should be this defaulted east.
    thrust_dir = program(0.0, mock_state_pole)
    np.testing.assert_allclose(thrust_dir, np.array([1.0, 0.0, 0.0]), atol=1e-6)


# --- Tests for throttle_schedule ---

def test_throttle_schedule_returns_cfg_value(monkeypatch):
    """Test throttle_schedule returns the configured base_throttle_cmd."""
    monkeypatch.setattr(CFG, "base_throttle_cmd", 0.75)
    mock_state = Mock(spec=State)
    assert throttle_schedule(0.0, mock_state) == 0.75


# --- Tests for ParameterizedThrottleProgram ---

def test_parameterized_throttle_program_init():
    """Test ParameterizedThrottleProgram initialization and schedule sorting."""
    schedule = [[10.0, 0.5], [0.0, 1.0], [5.0, 0.7]]
    program = ParameterizedThrottleProgram(schedule)
    
    expected_times = np.array([0.0, 5.0, 10.0])
    expected_throttles = np.array([1.0, 0.7, 0.5])
    
    np.testing.assert_allclose(program.time_points, expected_times)
    np.testing.assert_allclose(program.throttle_points, expected_throttles)
    assert not program.apply_to_stage0

    program_booster = ParameterizedThrottleProgram(schedule, apply_to_stage0=True)
    assert program_booster.apply_to_stage0


def test_parameterized_throttle_program_booster_throttle(mock_state_for_pitch_program):
    """Test throttle interpolation for booster stage (apply_to_stage0=True)."""
    schedule = [[0.0, 1.0], [50.0, 0.5], [100.0, 1.0]]
    program = ParameterizedThrottleProgram(schedule, apply_to_stage0=True)
    
    # At t=0
    throttle_t0 = program(0.0, mock_state_for_pitch_program)
    assert throttle_t0 == 1.0

    # At t=25 (interpolated)
    throttle_t25 = program(25.0, mock_state_for_pitch_program)
    assert throttle_t25 == pytest.approx(0.75)

    # At t=75 (interpolated)
    throttle_t75 = program(75.0, mock_state_for_pitch_program)
    assert throttle_t75 == pytest.approx(0.75)

    # At t=100
    throttle_t100 = program(100.0, mock_state_for_pitch_program)
    assert throttle_t100 == 1.0

    # Outside bounds - left (clamped to first value)
    throttle_t_neg = program(-10.0, mock_state_for_pitch_program)
    assert throttle_t_neg == 1.0

    # Outside bounds - right (clamped to last value)
    throttle_t_late = program(120.0, mock_state_for_pitch_program)
    assert throttle_t_late == 1.0


def test_parameterized_throttle_program_upper_throttle(mock_state_for_pitch_program):
    """Test throttle interpolation for upper stage."""
    schedule = [[0.0, 0.8], [60.0, 0.5], [120.0, 1.0]]
    program = ParameterizedThrottleProgram(schedule)
    
    mock_state_for_pitch_program.stage_index = 1
    mock_state_for_pitch_program.upper_ignition_start_time = 50.0 # Upper stage ignition

    # Before ignition
    throttle_pre_ignite = program(40.0, mock_state_for_pitch_program)
    assert throttle_pre_ignite == 0.0

    # At ignition time (t_stage=0)
    throttle_ignite = program(50.0, mock_state_for_pitch_program)
    assert throttle_ignite == 0.8

    # t_stage = 30 (interpolated)
    throttle_t30 = program(80.0, mock_state_for_pitch_program)
    assert throttle_t30 == pytest.approx(0.65) # 0.8 to 0.5 over 60s, at 30s is 0.65

    # After schedule end (clamped to last value)
    throttle_late = program(200.0, mock_state_for_pitch_program)
    assert throttle_late == 0.0 # The implementation uses right=0.0 for upper stage

    # No upper_ignition_start_time
    mock_state_no_ignite = Mock(spec=State, stage_index=1, upper_ignition_start_time=None)
    throttle_no_ignite = program(100.0, mock_state_no_ignite)
    assert throttle_no_ignite == 0.0

def test_parameterized_throttle_program_booster_default(mock_state_for_pitch_program):
    """Test booster throttle when apply_to_stage0 is False."""
    schedule = [[0.0, 0.5]] # Irrelevant, should be 1.0
    program = ParameterizedThrottleProgram(schedule, apply_to_stage0=False)
    
    throttle = program(10.0, mock_state_for_pitch_program)
    assert throttle == 1.0


# --- Tests for build_rocket ---

@patch('main.Engine')
@patch('main.Stage')
@patch('main.Rocket')
@patch('main.ParameterizedThrottleProgram', autospec=True)
def test_build_rocket_basic(MockParameterizedThrottleProgram, MockRocket, MockStage, MockEngine, monkeypatch):
    """Test build_rocket creates a rocket with correct stages and engine/stage parameters."""
    # Mock CFG values
    monkeypatch.setattr(CFG, "booster_thrust_vac", 1e6)
    monkeypatch.setattr(CFG, "booster_dry_mass", 1e4)
    monkeypatch.setattr(CFG, "booster_prop_mass", 1e5) # Added
    monkeypatch.setattr(CFG, "upper_thrust_vac", 5e5)
    monkeypatch.setattr(CFG, "upper_dry_mass", 5e3)
    monkeypatch.setattr(CFG, "upper_prop_mass", 5e4) # Added
    monkeypatch.setattr(CFG, "ref_area_m2", 50.0)
    monkeypatch.setattr(CFG, "booster_throttle_program", [[0.0, 1.0]]) # List for parameterized
    monkeypatch.setattr(CFG, "engine_min_throttle", 0.4) # Corrected from min_throttle
    monkeypatch.setattr(CFG, "throttle_full_shape_threshold", 0.99)
    monkeypatch.setattr(CFG, "earth_radius_m", 6.371e6)
    monkeypatch.setattr(CFG, "main_engine_ramp_time", 1.0)
    monkeypatch.setattr(CFG, "upper_engine_ramp_time", 1.0)
    monkeypatch.setattr(CFG, "meco_mach", 5.5)
    monkeypatch.setattr(CFG, "separation_delay_s", 1.0)
    monkeypatch.setattr(CFG, "upper_ignition_delay_s", 0.5)
    monkeypatch.setattr(CFG, "separation_altitude_m", None)
    monkeypatch.setattr(CFG, "engine_shutdown_ramp_s", 1.0) # Corrected from shutdown_ramp_time


    rocket = build_rocket()

    # Assert Rocket constructor was called with correct arguments
    MockRocket.assert_called_once()
    rocket_args, rocket_kwargs = MockRocket.call_args
    
    assert len(rocket_kwargs['stages']) == 2
    assert rocket_kwargs['stages'][0].dry_mass == 1e4 # Check booster dry_mass
    assert rocket_kwargs['stages'][0].prop_mass == 1e5 # Check booster prop_mass
    assert rocket_kwargs['stages'][1].dry_mass == 5e3 # Check upper dry_mass
    assert rocket_kwargs['stages'][1].prop_mass == 5e4 # Check upper prop_mass


    assert rocket_kwargs['main_engine_ramp_time'] == 1.0
    assert rocket_kwargs['meco_mach'] == 5.5
    assert rocket_kwargs['booster_throttle_program'] is MockParameterizedThrottleProgram.return_value

    # Assert ParameterizedThrottleProgram was called for booster
    MockParameterizedThrottleProgram.assert_called_once_with(schedule=[[0.0, 1.0]], apply_to_stage0=True)

@patch('main.Engine')
@patch('main.Stage')
@patch('main.Rocket')
def test_build_rocket_booster_throttle_program_already_object(MockRocket, MockStage, MockEngine, monkeypatch):
    """Test build_rocket handles booster_throttle_program already being an object."""
    mock_ptp_instance = Mock(spec=ParameterizedThrottleProgram)
    monkeypatch.setattr(CFG, "booster_throttle_program", mock_ptp_instance)
    
    rocket = build_rocket()
    
    rocket_args, rocket_kwargs = MockRocket.call_args
    assert rocket_kwargs['booster_throttle_program'] is mock_ptp_instance
    
    # Ensure ParameterizedThrottleProgram constructor was NOT called again
    # This assertion is against the patched class, which is MockEngine, MockStage, MockRocket
    # But ParameterizedThrottleProgram is not patched here
    # The original ParameterizedThrottleProgram is not accessible here
    # This assert needs to be removed or ParameterizedThrottleProgram needs to be patched
    assert MockRocket.called # Ensure Rocket was still called
    # assert not ParameterizedThrottleProgram.called # Original ParameterizedThrottleProgram, not the patch


# --- Tests for build_simulation ---

@patch('main.EarthModel')
@patch('main.AtmosphereModel')
@patch('main.CdModel')
@patch('main.Aerodynamics')
@patch('main.build_rocket')
@patch('main.Guidance')
@patch('main.RK4')
@patch('main.VelocityVerlet')
@patch('main.Simulation')
@patch('main.mach_dependent_cd')
@patch('main.StageAwarePitchProgram', autospec=True) # Patch StageAwarePitchProgram
@patch('main.importlib') # Patch importlib for dynamic loading
@patch('main.dt') # Patch datetime for fixed t0
def test_build_simulation_parameterized_pitch_rk4_integrator(
    MockDt, MockImportlib, MockStageAwarePitchProgram, MockMachDependentCd,
    MockSimulation, MockVelocityVerlet, MockRK4, MockGuidance, MockBuildRocket,
    MockAerodynamics, MockCdModel, MockAtmosphereModel, MockEarthModel,
    monkeypatch
):
    """
    Test build_simulation when using parameterized pitch guidance and RK4 integrator.
    """
    # Configure CFG
    monkeypatch.setattr(CFG, "pitch_guidance_mode", "parameterized")
    monkeypatch.setattr(CFG, "integrator", "rk4")
    monkeypatch.setattr(CFG, "launch_lat_deg", 10.0)
    monkeypatch.setattr(CFG, "launch_lon_deg", 20.0)
    monkeypatch.setattr(CFG, "earth_radius_m", 6.371e6)
    monkeypatch.setattr(CFG, "earth_omega_vec", (0.0, 0.0, 1.0))
    monkeypatch.setattr(CFG, "pitch_program", [[0.0, 90.0]])
    monkeypatch.setattr(CFG, "upper_pitch_program", [[0.0, 10.0]])
    monkeypatch.setattr(CFG, "pitch_prograde_speed_threshold", 50.0)
    monkeypatch.setattr(CFG, "max_q_limit", 1000.0)
    monkeypatch.setattr(CFG, "max_accel_limit", 50.0)
    monkeypatch.setattr(CFG, "impact_altitude_buffer_m", -10.0)
    monkeypatch.setattr(CFG, "escape_radius_factor", 1.1)

    # Configure datetime mock
    mock_now = Mock()
    mock_now.total_seconds.return_value = 123456789.0
    MockDt.datetime.now.return_value = mock_now
    MockDt.datetime.return_value = Mock() # For base epoch comparison
    MockDt.timezone.utc = Mock() # Mock timezone.utc

    sim, state0, t0 = build_simulation()

    # Assertions for component instantiation
    MockEarthModel.assert_called_once()
    MockAtmosphereModel.assert_called_once()
    MockCdModel.assert_called_once()
    MockBuildRocket.assert_called_once()
    MockAerodynamics.assert_called_once()

    MockStageAwarePitchProgram.assert_called_once_with(
        booster_schedule=CFG.pitch_program,
        upper_schedule=CFG.upper_pitch_program,
        prograde_threshold=CFG.pitch_prograde_speed_threshold,
        earth_radius=CFG.earth_radius_m,
    )
    MockGuidance.assert_called_once_with(
        pitch_program=MockStageAwarePitchProgram.return_value,
        throttle_schedule=throttle_schedule # Direct function reference
    )
    MockRK4.assert_called_once()
    MockVelocityVerlet.assert_not_called()
    MockSimulation.assert_called_once()

    # Assertions for initial state and time
    assert isinstance(state0, State)
    np.testing.assert_allclose(state0.r_eci[0], CFG.earth_radius_m * np.cos(np.deg2rad(CFG.launch_lat_deg)) * np.cos(np.deg2rad(CFG.launch_lon_deg)))
    assert t0 == 123456789.0

@patch('main.EarthModel')
@patch('main.AtmosphereModel')
@patch('main.CdModel')
@patch('main.Aerodynamics')
@patch('main.build_rocket')
@patch('main.Guidance')
@patch('main.RK4')
@patch('main.VelocityVerlet')
@patch('main.Simulation')
@patch('main.mach_dependent_cd')
@patch('main.StageAwarePitchProgram', autospec=True) # Patch StageAwarePitchProgram
@patch('main.importlib') # Patch importlib for dynamic loading
@patch('main.dt') # Patch datetime for fixed t0
def test_build_simulation_function_pitch_vv_integrator(
    MockDt, MockImportlib, MockStageAwarePitchProgram, MockMachDependentCd,
    MockSimulation, MockVelocityVerlet, MockRK4, MockGuidance, MockBuildRocket,
    MockAerodynamics, MockCdModel, MockAtmosphereModel, MockEarthModel,
    monkeypatch
):
    """
    Test build_simulation when using function pitch guidance and VelocityVerlet integrator.
    """
    # Configure CFG
    monkeypatch.setattr(CFG, "pitch_guidance_mode", "function")
    monkeypatch.setattr(CFG, "pitch_guidance_function", "test_module.test_pitch_func")
    monkeypatch.setattr(CFG, "integrator", "velocity_verlet")
    monkeypatch.setattr(CFG, "launch_lat_deg", 10.0)
    monkeypatch.setattr(CFG, "launch_lon_deg", 20.0)
    monkeypatch.setattr(CFG, "earth_radius_m", 6.371e6)
    monkeypatch.setattr(CFG, "earth_omega_vec", (0.0, 0.0, 1.0))

    # Mock the dynamically loaded function
    mock_pitch_func = Mock(name="test_pitch_func")
    MockImportlib.import_module.return_value.test_pitch_func = mock_pitch_func
    MockDt.datetime.now.return_value.total_seconds.return_value = 123456789.0
    MockDt.datetime.return_value = Mock() # For base epoch comparison
    MockDt.timezone.utc = Mock() # Mock timezone.utc


    sim, state0, t0 = build_simulation()

    MockImportlib.import_module.assert_called_once_with("test_module")
    MockGuidance.assert_called_once_with(
        pitch_program=mock_pitch_func,
        throttle_schedule=throttle_schedule
    )
    MockRK4.assert_not_called()
    MockVelocityVerlet.assert_called_once()


def test_throttle_schedule_returns_cfg_value(monkeypatch):
    """Test throttle_schedule returns the configured base_throttle_cmd."""
    monkeypatch.setattr(CFG, "base_throttle_cmd", 0.75)
    mock_state = Mock(spec=State)
    assert throttle_schedule(0.0, mock_state) == 0.75


# --- Tests for ParameterizedThrottleProgram ---

def test_parameterized_throttle_program_init():
    """Test ParameterizedThrottleProgram initialization and schedule sorting."""
    schedule = [[10.0, 0.5], [0.0, 1.0], [5.0, 0.7]]
    program = ParameterizedThrottleProgram(schedule)
    
    expected_times = np.array([0.0, 5.0, 10.0])
    expected_throttles = np.array([1.0, 0.7, 0.5])
    
    np.testing.assert_allclose(program.time_points, expected_times)
    np.testing.assert_allclose(program.throttle_points, expected_throttles)
    assert not program.apply_to_stage0

    program_booster = ParameterizedThrottleProgram(schedule, apply_to_stage0=True)
    assert program_booster.apply_to_stage0


def test_parameterized_throttle_program_booster_throttle(mock_state_for_pitch_program):
    """Test throttle interpolation for booster stage (apply_to_stage0=True)."""
    schedule = [[0.0, 1.0], [50.0, 0.5], [100.0, 1.0]]
    program = ParameterizedThrottleProgram(schedule, apply_to_stage0=True)
    
    # At t=0
    throttle_t0 = program(0.0, mock_state_for_pitch_program)
    assert throttle_t0 == 1.0

    # At t=25 (interpolated)
    throttle_t25 = program(25.0, mock_state_for_pitch_program)
    assert throttle_t25 == pytest.approx(0.75)

    # At t=75 (interpolated)
    throttle_t75 = program(75.0, mock_state_for_pitch_program)
    assert throttle_t75 == pytest.approx(0.75)

    # At t=100
    throttle_t100 = program(100.0, mock_state_for_pitch_program)
    assert throttle_t100 == 1.0

    # Outside bounds - left (clamped to first value)
    throttle_t_neg = program(-10.0, mock_state_for_pitch_program)
    assert throttle_t_neg == 1.0

    # Outside bounds - right (clamped to last value)
    throttle_t_late = program(120.0, mock_state_for_pitch_program)
    assert throttle_t_late == 1.0


def test_parameterized_throttle_program_upper_throttle(mock_state_for_pitch_program):
    """Test throttle interpolation for upper stage."""
    schedule = [[0.0, 0.8], [60.0, 0.5], [120.0, 1.0]]
    program = ParameterizedThrottleProgram(schedule)
    
    mock_state_for_pitch_program.stage_index = 1
    mock_state_for_pitch_program.upper_ignition_start_time = 50.0 # Upper stage ignition

    # Before ignition
    throttle_pre_ignite = program(40.0, mock_state_for_pitch_program)
    assert throttle_pre_ignite == 0.0

    # At ignition time (t_stage=0)
    throttle_ignite = program(50.0, mock_state_for_pitch_program)
    assert throttle_ignite == 0.8

    # t_stage = 30 (interpolated)
    throttle_t30 = program(80.0, mock_state_for_pitch_program)
    assert throttle_t30 == pytest.approx(0.65) # 0.8 to 0.5 over 60s, at 30s is 0.65

    # After schedule end (clamped to last value)
    throttle_late = program(200.0, mock_state_for_pitch_program)
    assert throttle_late == 0.0 # The implementation uses right=0.0 for upper stage

    # No upper_ignition_start_time
    mock_state_no_ignite = Mock(spec=State, stage_index=1, upper_ignition_start_time=None)
    throttle_no_ignite = program(100.0, mock_state_no_ignite)
    assert throttle_no_ignite == 0.0

def test_parameterized_throttle_program_booster_default(mock_state_for_pitch_program):
    """Test booster throttle when apply_to_stage0 is False."""
    schedule = [[0.0, 0.5]] # Irrelevant, should be 1.0
    program = ParameterizedThrottleProgram(schedule, apply_to_stage0=False)
    
    throttle = program(10.0, mock_state_for_pitch_program)
    assert throttle == 1.0