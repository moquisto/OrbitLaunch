import numpy as np
import pytest
from unittest.mock import Mock

from Hardware.engine import Engine
from Hardware.stage import Stage
from Hardware.rocket import Rocket
from Software.guidance import ParameterizedThrottleProgram
from Main.state import State
from Environment.config import EnvironmentConfig
from Hardware.config import HardwareConfig
from Software.config import SoftwareConfig
from Main.config import SimulationConfig

# --- Fixtures ---
@pytest.fixture
def env_config():
    return EnvironmentConfig()

@pytest.fixture
def hw_config():
    return HardwareConfig()

@pytest.fixture
def sw_config():
    return SoftwareConfig()

@pytest.fixture
def sim_config():
    return SimulationConfig()

@pytest.fixture
def default_engine(hw_config, env_config):
    return Engine(
        thrust_vac=hw_config.upper_thrust_vac,  # Using upper stage for general engine test
        thrust_sl=hw_config.upper_thrust_sl,
        isp_vac=hw_config.upper_isp_vac,
        isp_sl=hw_config.upper_isp_sl,
        p_sl=env_config.P_SL
    )

@pytest.fixture
def default_stage(default_engine, hw_config):
    return Stage(
        dry_mass=hw_config.upper_dry_mass,
        prop_mass=hw_config.upper_prop_mass,
        engine=default_engine,
        ref_area=hw_config.ref_area_m2
    )

@pytest.fixture
def default_rocket(hw_config, env_config, default_stage):
    booster_engine = Engine(
        thrust_vac=hw_config.booster_thrust_vac,
        thrust_sl=hw_config.booster_thrust_sl,
        isp_vac=hw_config.booster_isp_vac,
        isp_sl=hw_config.booster_isp_sl,
        p_sl=env_config.P_SL
    )
    booster_stage = Stage(
        dry_mass=hw_config.booster_dry_mass,
        prop_mass=hw_config.booster_prop_mass,
        engine=booster_engine,
        ref_area=hw_config.ref_area_m2
    )
    
    return Rocket(
        stages=[booster_stage, default_stage],
        hw_config=hw_config,
        env_config=env_config,
    )

@pytest.fixture
def mock_state(env_config):
    return Mock(spec=State, r_eci=np.array([0, 0, env_config.earth_radius_m]),
                 v_eci=np.array([0, 0, 0]), m=100_000.0, stage_index=0)

# --- Engine Tests ---
def test_engine_thrust_and_isp_vacuum(default_engine):
    """Test thrust and Isp in vacuum conditions (p_amb = 0)."""
    thrust, isp = default_engine.thrust_and_isp(throttle=1.0, p_amb=0.0)
    assert thrust == pytest.approx(default_engine.thrust_vac)
    assert isp == pytest.approx(default_engine.isp_vac)
def test_engine_thrust_and_isp_sea_level(default_engine, env_config):
    """Test thrust and Isp at sea level conditions (p_amb = env_config.P_SL)."""
    thrust, isp = default_engine.thrust_and_isp(throttle=1.0, p_amb=env_config.P_SL)
    assert thrust == pytest.approx(default_engine.thrust_sl)
    assert isp == pytest.approx(default_engine.isp_sl)

def test_engine_thrust_and_isp_mid_pressure(default_engine, env_config):
    """Test thrust and Isp at an intermediate ambient pressure."""
    p_mid = env_config.P_SL / 2
    thrust, isp = default_engine.thrust_and_isp(throttle=1.0, p_amb=p_mid)
    
    expected_thrust = default_engine.thrust_sl + 0.5 * (default_engine.thrust_vac - default_engine.thrust_sl)
    expected_isp = default_engine.isp_sl + 0.5 * (default_engine.isp_vac - default_engine.isp_sl)
    
    assert thrust == pytest.approx(expected_thrust)
    assert isp == pytest.approx(expected_isp)

def test_engine_throttle_effect(default_engine, env_config):
    """Test thrust scaling with throttle."""
    throttle_val = 0.5
    thrust, isp = default_engine.thrust_and_isp(throttle=throttle_val, p_amb=env_config.P_SL)
    
    expected_thrust_full = default_engine.thrust_sl
    assert thrust == pytest.approx(expected_thrust_full * throttle_val)
    assert isp == pytest.approx(default_engine.isp_sl) # Isp should not change with throttle

def test_engine_throttle_clamping(default_engine, env_config):
    """Test that throttle is clamped between 0 and 1."""
    thrust_low, _ = default_engine.thrust_and_isp(throttle=-0.1, p_amb=env_config.P_SL)
    thrust_high, _ = default_engine.thrust_and_isp(throttle=1.1, p_amb=env_config.P_SL)
    
    thrust_zero, _ = default_engine.thrust_and_isp(throttle=0.0, p_amb=env_config.P_SL)
    thrust_one, _ = default_engine.thrust_and_isp(throttle=1.0, p_amb=env_config.P_SL)
    
    assert thrust_low == pytest.approx(thrust_zero)
    assert thrust_high == pytest.approx(thrust_one)

def test_engine_ambient_pressure_clamping(default_engine, env_config):
    """Test that ambient pressure is clamped for interpolation."""
    thrust_neg, _ = default_engine.thrust_and_isp(throttle=1.0, p_amb=-100.0)
    thrust_vac, _ = default_engine.thrust_and_isp(throttle=1.0, p_amb=0.0)
    assert thrust_neg == pytest.approx(thrust_vac)

    thrust_high, _ = default_engine.thrust_and_isp(throttle=1.0, p_amb=env_config.P_SL * 2)
    thrust_sl, _ = default_engine.thrust_and_isp(throttle=1.0, p_amb=env_config.P_SL)
    assert thrust_high == pytest.approx(thrust_sl)

# --- Rocket Thrust and Mass Flow Tests ---

def test_rocket_booster_ramp_up(default_rocket, mock_state, hw_config, env_config):
    """Test booster engine thrust ramp-up phase."""
    # config.main_engine_ramp_time = 3.0 # This is now on hw_config
    default_rocket.main_engine_ramp_time = hw_config.main_engine_ramp_time

    control = {"t": 0.0, "throttle": 1.0, "thrust_dir_eci": np.array([0, 0, 1])}
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, env_config.P_SL)
    assert np.linalg.norm(thrust_vec) == pytest.approx(0.0, abs=1e-6)
    assert dm_dt == pytest.approx(0.0)

    half_ramp_time = hw_config.main_engine_ramp_time / 2
    control["t"] = half_ramp_time
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, env_config.P_SL)
    expected_thrust_mag = default_rocket.stages[0].engine.thrust_sl * 0.5
    assert np.linalg.norm(thrust_vec) == pytest.approx(expected_thrust_mag, rel=0.01)

    control["t"] = hw_config.main_engine_ramp_time
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, env_config.P_SL)
    expected_thrust_mag = default_rocket.stages[0].engine.thrust_sl
    assert np.linalg.norm(thrust_vec) == pytest.approx(expected_thrust_mag, rel=0.01)

def test_rocket_booster_constant_thrust_no_program(default_rocket, mock_state, config):
    """Test booster constant thrust phase without a throttle program."""
    # Advance time past ramp-up
    t_after_ramp = default_rocket.main_engine_ramp_time + 10.0
    control = {"t": t_after_ramp, "throttle": 1.0, "thrust_dir_eci": np.array([0, 0, 1])}
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, config.P_SL)
    
    expected_thrust_mag = default_rocket.stages[0].engine.thrust_sl
    assert np.linalg.norm(thrust_vec) == pytest.approx(expected_thrust_mag, rel=0.01)
    # Check mdot is negative (consuming propellant)
    assert dm_dt < 0

def test_rocket_booster_meco_mach_trigger(default_rocket, mock_state, config):
    """Test that MECO is triggered by Mach number."""
    meco_mach_val = 2.0
    config.meco_mach = meco_mach_val
    default_rocket.meco_mach = meco_mach_val
    
    altitude_for_test = np.linalg.norm(mock_state.r_eci) - default_rocket.earth_radius
    local_speed_of_sound = config.get_speed_of_sound(altitude_for_test)

    mock_state.v_eci = np.array([0, 0, (meco_mach_val - 0.5) * local_speed_of_sound])
    control = {"t": 10.0, "throttle": 1.0, "thrust_dir_eci": np.array([0, 0, 1])}
    default_rocket.thrust_and_mass_flow(control, mock_state, config.P_SL)
    assert default_rocket.meco_time is None

    mock_state.v_eci = np.array([0, 0, (meco_mach_val + 0.1) * local_speed_of_sound])
    control = {"t": 11.0, "throttle": 1.0, "thrust_dir_eci": np.array([0, 0, 1])}
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, config.P_SL)
    
    assert default_rocket.meco_time == pytest.approx(11.0)
    assert np.linalg.norm(thrust_vec) == pytest.approx(0.0, abs=1e-6)
    assert dm_dt == pytest.approx(0.0)

    mock_state.v_eci = np.array([0, 0, (meco_mach_val - 1.0) * local_speed_of_sound])
    control = {"t": 12.0, "throttle": 1.0, "thrust_dir_eci": np.array([0, 0, 1])}
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, config.P_SL)
    assert np.linalg.norm(thrust_vec) == pytest.approx(0.0, abs=1e-6)

def test_rocket_booster_fuel_depletion_ramp_down(default_rocket, mock_state, config):
    """Test booster engine fuel depletion and ramp-down phase over two steps."""
    config.main_engine_ramp_time = 0.0
    default_rocket.main_engine_ramp_time = 0.0

    thrust_sl, isp_sl = default_rocket.stages[0].engine.thrust_and_isp(1.0, config.P_SL)
    expected_dm_dt_per_sec = -thrust_sl / (isp_sl * config.G0)

    t1 = 10.0
    dt1 = 0.1
    
    prop_for_next_step = 0.0001 * -expected_dm_dt_per_sec
    initial_prop_mass = prop_for_next_step + (dt1 * -expected_dm_dt_per_sec)
    
    default_rocket.stage_prop_remaining[0] = initial_prop_mass
    default_rocket.stage_fuel_empty_time[0] = None
    default_rocket._last_time = t1 - dt1

    control = {"t": t1, "throttle": 1.0, "thrust_dir_eci": np.array([0, 0, 1])}
    
    thrust_vec1, dm_dt1 = default_rocket.thrust_and_mass_flow(control, mock_state, config.P_SL)
    
    assert default_rocket.stage_prop_remaining[0] > 0.0
    assert default_rocket.stage_fuel_empty_time[0] is None
    assert np.linalg.norm(thrust_vec1) == pytest.approx(default_rocket.stages[0].engine.thrust_sl, rel=0.01)
    assert dm_dt1 < 0

    t2 = t1 + dt1
    default_rocket._last_time = t1

    control["t"] = t2
    thrust_vec2, dm_dt2 = default_rocket.thrust_and_mass_flow(control, mock_state, config.P_SL)
    assert default_rocket.stage_prop_remaining[0] <= 0.0
    assert default_rocket.stage_fuel_empty_time[0] == pytest.approx(t2)
    assert np.linalg.norm(thrust_vec2) == pytest.approx(0.0, abs=1e-6)
    assert dm_dt2 == pytest.approx(0.0)




def test_rocket_upper_stage_ignition_and_burn(default_rocket, mock_state):
    """Test upper stage ignition and full thrust phase."""
    # Simulate after booster separation and upper stage ignition time
    t_ignition_start = 100.0
    default_rocket.upper_ignition_start_time = t_ignition_start
    mock_state.stage_index = 1
    mock_state.m = default_rocket.stages[1].total_mass() # Set mass for upper stage

    # Before ignition, thrust should be zero
    control = {"t": t_ignition_start - 1.0, "throttle": 1.0, "thrust_dir_eci": np.array([0, 0, 1])}
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, 0.0) # Vacuum
    assert np.linalg.norm(thrust_vec) == pytest.approx(0.0, abs=1e-6)

    # During ramp-up
    ramp_up_time = default_rocket.upper_engine_ramp_time / 2
    control["t"] = t_ignition_start + ramp_up_time
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, 0.0)
    expected_thrust_mag = default_rocket.stages[1].engine.thrust_vac * 0.5
    assert np.linalg.norm(thrust_vec) == pytest.approx(expected_thrust_mag, rel=0.01)
    assert dm_dt < 0

    # After ramp-up, full thrust
    control["t"] = t_ignition_start + default_rocket.upper_engine_ramp_time + 5.0
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, 0.0)
    expected_thrust_mag = default_rocket.stages[1].engine.thrust_vac
    assert np.linalg.norm(thrust_vec) == pytest.approx(expected_thrust_mag, rel=0.01)
    assert dm_dt < 0

def test_rocket_booster_throttle_program_interaction(default_rocket, mock_state, config):
    """Test interaction with booster_throttle_program."""
    # Define a simple booster throttle program
    def custom_booster_throttle(t, state):
        if t < 5.0: return 0.5
        elif t < 10.0: return 0.8
        else: return 1.0
    
    default_rocket.booster_throttle_program = custom_booster_throttle
    default_rocket.main_engine_ramp_time = 0.0 # Disable ramp for simpler testing

    # At t=2.5s, throttle should be 0.5
    control = {"t": 2.5, "throttle": 1.0, "thrust_dir_eci": np.array([0, 0, 1])}
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, config.P_SL)
    expected_thrust_mag = default_rocket.stages[0].engine.thrust_sl * 0.5
    assert np.linalg.norm(thrust_vec) == pytest.approx(expected_thrust_mag, rel=0.01)

    # At t=7.5s, throttle should be 0.8
    control["t"] = 7.5
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, config.P_SL)
    expected_thrust_mag = default_rocket.stages[0].engine.thrust_sl * 0.8
    assert np.linalg.norm(thrust_vec) == pytest.approx(expected_thrust_mag, rel=0.01)

    # At t=12.5s, throttle should be 1.0
    control["t"] = 12.5
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, config.P_SL)
    expected_thrust_mag = default_rocket.stages[0].engine.thrust_sl * 1.0
    assert np.linalg.norm(thrust_vec) == pytest.approx(expected_thrust_mag, rel=0.01)


def test_parameterized_throttle_program_booster_support(config):
    """Ensure ParameterizedThrottleProgram can drive the booster when requested."""
    schedule = [[0.0, 0.4], [10.0, 1.0]]
    program = ParameterizedThrottleProgram(schedule=schedule, apply_to_stage0=True)
    state = State(r_eci=np.array([0, 0, config.earth_radius_m]), v_eci=np.zeros(3), m=1.0, stage_index=0)

    assert program(0.0, state) == pytest.approx(0.4)
    assert program(5.0, state) == pytest.approx(0.7)
    assert program(25.0, state) == pytest.approx(1.0)


def test_build_rocket_converts_booster_schedule(config):
    """build_rocket should turn a schedule list into a callable throttle program for the booster."""
    schedule = [[0.0, 0.5], [5.0, 0.7], [10.0, 1.0]]
    config.booster_throttle_program = schedule
    config.main_engine_ramp_time = 0.0
    config.engine_min_throttle = 0.0
    config.throttle_full_shape_threshold = 1.0

    rocket = build_rocket(config)
    state = State(r_eci=np.array([0, 0, config.earth_radius_m]), v_eci=np.zeros(3), m=rocket.stages[0].total_mass(), stage_index=0)
    control = {"t": 0.0, "throttle": 1.0, "thrust_dir_eci": np.array([0, 0, 1])}

    thrust_vec0, _ = rocket.thrust_and_mass_flow(control, state, config.P_SL)
    assert np.linalg.norm(thrust_vec0) == pytest.approx(rocket.stages[0].engine.thrust_sl * 0.5, rel=0.01)

    control["t"] = 7.5
    thrust_vec_mid, _ = rocket.thrust_and_mass_flow(control, state, config.P_SL)
    assert np.linalg.norm(thrust_vec_mid) == pytest.approx(rocket.stages[0].engine.thrust_sl * 0.85, rel=0.01)

def test_rocket_min_throttle_enforcement(default_rocket, mock_state, config):
    """Test min_throttle enforcement."""
    config.min_throttle = 0.5
    default_rocket.min_throttle = 0.5
    default_rocket.throttle_shape_full_threshold = 0.99
    default_rocket.main_engine_ramp_time = 0.0

    control = {"t": 10.0, "throttle": 0.3, "thrust_dir_eci": np.array([0, 0, 1])}
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, config.P_SL)
    expected_thrust_mag = default_rocket.stages[0].engine.thrust_sl * default_rocket.min_throttle
    assert np.linalg.norm(thrust_vec) == pytest.approx(expected_thrust_mag, rel=0.01)

    control = {"t": 10.0, "throttle": 0.7, "thrust_dir_eci": np.array([0, 0, 1])}
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, config.P_SL)
    expected_thrust_mag = default_rocket.stages[0].engine.thrust_sl * 0.7
    assert np.linalg.norm(thrust_vec) == pytest.approx(expected_thrust_mag, rel=0.01)

    control = {"t": 10.0, "throttle": 0.0, "thrust_dir_eci": np.array([0, 0, 1])}
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, config.P_SL)
    assert np.linalg.norm(thrust_vec) == pytest.approx(0.0, abs=1e-6)

    control = {"t": 0.5, "throttle": 0.1, "thrust_dir_eci": np.array([0, 0, 1])}
    default_rocket.main_engine_ramp_time = 1.0
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, config.P_SL)
    expected_thrust_mag = default_rocket.stages[0].engine.thrust_sl * (0.1 * 0.5)
    assert np.linalg.norm(thrust_vec) == pytest.approx(expected_thrust_mag, rel=0.01)
