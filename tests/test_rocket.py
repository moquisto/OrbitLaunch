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


