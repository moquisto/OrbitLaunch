import numpy as np
import pytest
from unittest.mock import Mock

from rocket import Engine, Stage, Rocket
from main import ParameterizedThrottleProgram, build_rocket
from integrators import State
from config import CFG

# --- Fixtures ---
@pytest.fixture
def default_engine():
    return Engine(
        thrust_vac=1.5e7,  # N
        thrust_sl=6.78e6,  # N
        isp_vac=380.0,     # s
        isp_sl=330.0       # s
    )

@pytest.fixture
def default_stage(default_engine):
    return Stage(
        dry_mass=10_000.0, # kg
        prop_mass=100_000.0, # kg
        engine=default_engine,
        ref_area=10.0 # m^2
    )

@pytest.fixture
def default_rocket(default_stage):
    # Need two stages for the Rocket class constructor
    booster_engine = Engine(thrust_vac=7.6e7, thrust_sl=7.6e7, isp_vac=380.0, isp_sl=330.0)
    booster_stage = Stage(dry_mass=1.8e5, prop_mass=3.4e6, engine=booster_engine, ref_area=50.0)
    
    return Rocket(
        stages=[booster_stage, default_stage], # Using default_stage as the upper stage
        main_engine_ramp_time=3.0,
        upper_engine_ramp_time=3.0,
        meco_mach=6.0,
        separation_delay=5.0,
        upper_ignition_delay=2.0,
        earth_radius=CFG.earth_radius_m,
        min_throttle=0.4,
        shutdown_ramp_time=1.0
    )

@pytest.fixture
def mock_state():
    # A basic mock state for testing, can be customized per test
    return Mock(spec=State, r_eci=np.array([0, 0, CFG.earth_radius_m]),
                 v_eci=np.array([0, 0, 0]), m=100_000.0, stage_index=0)

# --- Engine Tests ---
def test_engine_thrust_and_isp_vacuum(default_engine):
    """Test thrust and Isp in vacuum conditions (p_amb = 0)."""
    thrust, isp = default_engine.thrust_and_isp(throttle=1.0, p_amb=0.0)
    assert thrust == pytest.approx(default_engine.thrust_vac)
    assert isp == pytest.approx(default_engine.isp_vac)

def test_engine_thrust_and_isp_sea_level(default_engine):
    """Test thrust and Isp at sea level conditions (p_amb = CFG.P_SL)."""
    thrust, isp = default_engine.thrust_and_isp(throttle=1.0, p_amb=CFG.P_SL)
    assert thrust == pytest.approx(default_engine.thrust_sl)
    assert isp == pytest.approx(default_engine.isp_sl)

def test_engine_thrust_and_isp_mid_pressure(default_engine):
    """Test thrust and Isp at an intermediate ambient pressure."""
    p_mid = CFG.P_SL / 2
    thrust, isp = default_engine.thrust_and_isp(throttle=1.0, p_amb=p_mid)
    
    expected_thrust = default_engine.thrust_sl + 0.5 * (default_engine.thrust_vac - default_engine.thrust_sl)
    expected_isp = default_engine.isp_sl + 0.5 * (default_engine.isp_vac - default_engine.isp_sl)
    
    assert thrust == pytest.approx(expected_thrust)
    assert isp == pytest.approx(expected_isp)

def test_engine_throttle_effect(default_engine):
    """Test thrust scaling with throttle."""
    throttle_val = 0.5
    thrust, isp = default_engine.thrust_and_isp(throttle=throttle_val, p_amb=CFG.P_SL)
    
    expected_thrust_full = default_engine.thrust_sl
    assert thrust == pytest.approx(expected_thrust_full * throttle_val)
    assert isp == pytest.approx(default_engine.isp_sl) # Isp should not change with throttle

def test_engine_throttle_clamping(default_engine):
    """Test that throttle is clamped between 0 and 1."""
    thrust_low, _ = default_engine.thrust_and_isp(throttle=-0.1, p_amb=CFG.P_SL)
    thrust_high, _ = default_engine.thrust_and_isp(throttle=1.1, p_amb=CFG.P_SL)
    
    thrust_zero, _ = default_engine.thrust_and_isp(throttle=0.0, p_amb=CFG.P_SL)
    thrust_one, _ = default_engine.thrust_and_isp(throttle=1.0, p_amb=CFG.P_SL)
    
    assert thrust_low == pytest.approx(thrust_zero)
    assert thrust_high == pytest.approx(thrust_one)

def test_engine_ambient_pressure_clamping(default_engine):
    """Test that ambient pressure is clamped for interpolation."""
    thrust_neg, _ = default_engine.thrust_and_isp(throttle=1.0, p_amb=-100.0)
    thrust_vac, _ = default_engine.thrust_and_isp(throttle=1.0, p_amb=0.0)
    assert thrust_neg == pytest.approx(thrust_vac)

    thrust_high, _ = default_engine.thrust_and_isp(throttle=1.0, p_amb=CFG.P_SL * 2)
    thrust_sl, _ = default_engine.thrust_and_isp(throttle=1.0, p_amb=CFG.P_SL)
    assert thrust_high == pytest.approx(thrust_sl)

# --- Rocket Thrust and Mass Flow Tests ---

def test_rocket_booster_ramp_up(default_rocket, mock_state):
    """Test booster engine thrust ramp-up phase."""
    # Ensure ramp time is set in fixture or config
    CFG.main_engine_ramp_time = 3.0
    default_rocket.main_engine_ramp_time = 3.0 # Update fixture too if needed

    # At t=0, thrust should be 0 (or very low if min_throttle applies instantly)
    control = {"t": 0.0, "throttle": 1.0, "thrust_dir_eci": np.array([0, 0, 1])}
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, CFG.P_SL)
    assert np.linalg.norm(thrust_vec) == pytest.approx(0.0, abs=1e-6)
    assert dm_dt == pytest.approx(0.0)

    # At half ramp time, thrust should be half of sea level thrust
    half_ramp_time = CFG.main_engine_ramp_time / 2
    control["t"] = half_ramp_time
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, CFG.P_SL)
    expected_thrust_mag = default_rocket.stages[0].engine.thrust_sl * 0.5
    assert np.linalg.norm(thrust_vec) == pytest.approx(expected_thrust_mag, rel=0.01)

    # At full ramp time and beyond, thrust should be full sea level thrust
    control["t"] = CFG.main_engine_ramp_time
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, CFG.P_SL)
    expected_thrust_mag = default_rocket.stages[0].engine.thrust_sl
    assert np.linalg.norm(thrust_vec) == pytest.approx(expected_thrust_mag, rel=0.01)

def test_rocket_booster_constant_thrust_no_program(default_rocket, mock_state):
    """Test booster constant thrust phase without a throttle program."""
    # Advance time past ramp-up
    t_after_ramp = default_rocket.main_engine_ramp_time + 10.0
    control = {"t": t_after_ramp, "throttle": 1.0, "thrust_dir_eci": np.array([0, 0, 1])}
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, CFG.P_SL)
    
    expected_thrust_mag = default_rocket.stages[0].engine.thrust_sl
    assert np.linalg.norm(thrust_vec) == pytest.approx(expected_thrust_mag, rel=0.01)
    # Check mdot is negative (consuming propellant)
    assert dm_dt < 0

def test_rocket_booster_meco_mach_trigger(default_rocket, mock_state):
    """Test that MECO is triggered by Mach number."""
    # Set up state to trigger MECO_MACH
    meco_mach_val = 2.0 # Use a lower value for easier testing
    CFG.meco_mach = meco_mach_val
    default_rocket.meco_mach = meco_mach_val
    
    # Simulate current state with Mach < meco_mach
    mock_state.v_eci = np.array([0, 0, (meco_mach_val - 0.5) * CFG.mach_reference_speed])
    control = {"t": 10.0, "throttle": 1.0, "thrust_dir_eci": np.array([0, 0, 1])}
    default_rocket.thrust_and_mass_flow(control, mock_state, CFG.P_SL)
    assert default_rocket.meco_time is None

    # Simulate current state with Mach >= meco_mach
    mock_state.v_eci = np.array([0, 0, (meco_mach_val + 0.1) * CFG.mach_reference_speed])
    control = {"t": 11.0, "throttle": 1.0, "thrust_dir_eci": np.array([0, 0, 1])}
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, CFG.P_SL)
    
    assert default_rocket.meco_time == pytest.approx(11.0)
    # After MECO is triggered, engine should cut off
    assert np.linalg.norm(thrust_vec) == pytest.approx(0.0, abs=1e-6)
    assert dm_dt == pytest.approx(0.0)

    # Even if Mach goes down, MECO time should be preserved and thrust remain 0
    mock_state.v_eci = np.array([0, 0, (meco_mach_val - 1.0) * CFG.mach_reference_speed])
    control = {"t": 12.0, "throttle": 1.0, "thrust_dir_eci": np.array([0, 0, 1])}
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, CFG.P_SL)
    assert np.linalg.norm(thrust_vec) == pytest.approx(0.0, abs=1e-6)

def test_rocket_booster_fuel_depletion_ramp_down(default_rocket, mock_state):
    """Test booster engine fuel depletion and ramp-down phase over two steps."""
    # Temporarily adjust config for testing specific behavior
    original_main_engine_ramp_time = CFG.main_engine_ramp_time
    CFG.main_engine_ramp_time = 0.0 # Disable ramp-up for simpler calculations
    default_rocket.main_engine_ramp_time = 0.0

    # Calculate expected dm_dt for full thrust at sea level
    thrust_sl, isp_sl = default_rocket.stages[0].engine.thrust_and_isp(1.0, CFG.P_SL)
    expected_dm_dt_per_sec = -thrust_sl / (isp_sl * CFG.G0) # This is a negative value

    # --- Step 1: Consume most of the fuel ---
    t1 = 10.0
    dt1 = 0.1 # A small step
    
    # Set prop_mass such that it will be fully consumed in the *next* step
    # Let's say we have enough fuel for 0.0001s of burn remaining at the start of next step.
    prop_for_next_step = 0.0001 * -expected_dm_dt_per_sec # Remaining small positive fuel
    initial_prop_mass = prop_for_next_step + (dt1 * -expected_dm_dt_per_sec) # Fuel for current step + next step
    
    default_rocket.stage_prop_remaining[0] = initial_prop_mass
    default_rocket.stage_fuel_empty_time[0] = None # Ensure it's None initially
    default_rocket._last_time = t1 - dt1 # Set last_time for this step, so dt_internal in first call is dt1

    control = {"t": t1, "throttle": 1.0, "thrust_dir_eci": np.array([0, 0, 1])}
    
    thrust_vec1, dm_dt1 = default_rocket.thrust_and_mass_flow(control, mock_state, CFG.P_SL)
    
    # After step 1, fuel should still be positive, thrust should be full
    assert default_rocket.stage_prop_remaining[0] > 0.0
    assert default_rocket.stage_fuel_empty_time[0] is None
    assert np.linalg.norm(thrust_vec1) == pytest.approx(default_rocket.stages[0].engine.thrust_sl, rel=0.01)
    assert dm_dt1 < 0

    # --- Step 2: Fuel becomes depleted ---
    t2 = t1 + dt1 # Current time is now 10.1
    default_rocket._last_time = t1 # Update _last_time for this step, so dt_internal in second call is dt1

    control["t"] = t2
    thrust_vec2, dm_dt2 = default_rocket.thrust_and_mass_flow(control, mock_state, CFG.P_SL)
    # After step 2, fuel should be depleted, thrust should be zero
    assert default_rocket.stage_prop_remaining[0] <= 0.0
    assert default_rocket.stage_fuel_empty_time[0] == pytest.approx(t2)
    assert np.linalg.norm(thrust_vec2) == pytest.approx(0.0, abs=1e-6) # Thrust should be zero because fuel is empty
    assert dm_dt2 == pytest.approx(0.0) # Mass flow should be zero

    # Restore original config
    CFG.main_engine_ramp_time = original_main_engine_ramp_time
    default_rocket.main_engine_ramp_time = original_main_engine_ramp_time




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

def test_rocket_booster_throttle_program_interaction(default_rocket, mock_state):
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
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, CFG.P_SL)
    expected_thrust_mag = default_rocket.stages[0].engine.thrust_sl * 0.5
    assert np.linalg.norm(thrust_vec) == pytest.approx(expected_thrust_mag, rel=0.01)

    # At t=7.5s, throttle should be 0.8
    control["t"] = 7.5
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, CFG.P_SL)
    expected_thrust_mag = default_rocket.stages[0].engine.thrust_sl * 0.8
    assert np.linalg.norm(thrust_vec) == pytest.approx(expected_thrust_mag, rel=0.01)

    # At t=12.5s, throttle should be 1.0
    control["t"] = 12.5
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, CFG.P_SL)
    expected_thrust_mag = default_rocket.stages[0].engine.thrust_sl * 1.0
    assert np.linalg.norm(thrust_vec) == pytest.approx(expected_thrust_mag, rel=0.01)


def test_parameterized_throttle_program_booster_support():
    """Ensure ParameterizedThrottleProgram can drive the booster when requested."""
    schedule = [[0.0, 0.4], [10.0, 1.0]]
    program = ParameterizedThrottleProgram(schedule=schedule, apply_to_stage0=True)
    state = State(r_eci=np.array([0, 0, CFG.earth_radius_m]), v_eci=np.zeros(3), m=1.0, stage_index=0)

    assert program(0.0, state) == pytest.approx(0.4)
    assert program(5.0, state) == pytest.approx(0.7)
    assert program(25.0, state) == pytest.approx(1.0)


def test_build_rocket_converts_booster_schedule(monkeypatch):
    """build_rocket should turn a schedule list into a callable throttle program for the booster."""
    schedule = [[0.0, 0.5], [5.0, 0.7], [10.0, 1.0]]
    monkeypatch.setattr(CFG, "booster_throttle_program", schedule)
    monkeypatch.setattr(CFG, "main_engine_ramp_time", 0.0)
    monkeypatch.setattr(CFG, "engine_min_throttle", 0.0)
    monkeypatch.setattr(CFG, "throttle_full_shape_threshold", 1.0)

    rocket = build_rocket()
    state = State(r_eci=np.array([0, 0, CFG.earth_radius_m]), v_eci=np.zeros(3), m=rocket.stages[0].total_mass(), stage_index=0)
    control = {"t": 0.0, "throttle": 1.0, "thrust_dir_eci": np.array([0, 0, 1])}

    thrust_vec0, _ = rocket.thrust_and_mass_flow(control, state, CFG.P_SL)
    assert np.linalg.norm(thrust_vec0) == pytest.approx(rocket.stages[0].engine.thrust_sl * 0.5, rel=0.01)

    control["t"] = 7.5
    thrust_vec_mid, _ = rocket.thrust_and_mass_flow(control, state, CFG.P_SL)
    assert np.linalg.norm(thrust_vec_mid) == pytest.approx(rocket.stages[0].engine.thrust_sl * 0.85, rel=0.01)

def test_rocket_min_throttle_enforcement(default_rocket, mock_state):
    """Test min_throttle enforcement."""
    CFG.min_throttle = 0.5 # Set a clear min throttle
    default_rocket.min_throttle = 0.5
    default_rocket.throttle_shape_full_threshold = 0.99
    default_rocket.main_engine_ramp_time = 0.0 # Disable ramp for simpler testing

    # Commanded throttle below min_throttle, but shape is 1 (full)
    control = {"t": 10.0, "throttle": 0.3, "thrust_dir_eci": np.array([0, 0, 1])}
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, CFG.P_SL)
    expected_thrust_mag = default_rocket.stages[0].engine.thrust_sl * default_rocket.min_throttle
    assert np.linalg.norm(thrust_vec) == pytest.approx(expected_thrust_mag, rel=0.01)

    # Commanded throttle above min_throttle, should use commanded
    control = {"t": 10.0, "throttle": 0.7, "thrust_dir_eci": np.array([0, 0, 1])}
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, CFG.P_SL)
    expected_thrust_mag = default_rocket.stages[0].engine.thrust_sl * 0.7
    assert np.linalg.norm(thrust_vec) == pytest.approx(expected_thrust_mag, rel=0.01)

    # Commanded throttle is 0, min_throttle should not be enforced (engine off)
    control = {"t": 10.0, "throttle": 0.0, "thrust_dir_eci": np.array([0, 0, 1])}
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, CFG.P_SL)
    assert np.linalg.norm(thrust_vec) == pytest.approx(0.0, abs=1e-6)

    # Shape not yet "full", min_throttle should not be enforced
    control = {"t": 0.5, "throttle": 0.1, "thrust_dir_eci": np.array([0, 0, 1])}
    default_rocket.main_engine_ramp_time = 1.0 # Enable ramp-up
    thrust_vec, dm_dt = default_rocket.thrust_and_mass_flow(control, mock_state, CFG.P_SL)
    # Shape at t=0.5 should be 0.5. Effective throttle = cmd_throttle * shape = 0.1 * 0.5 = 0.05
    expected_thrust_mag = default_rocket.stages[0].engine.thrust_sl * (0.1 * 0.5)
    assert np.linalg.norm(thrust_vec) == pytest.approx(expected_thrust_mag, rel=0.01)
