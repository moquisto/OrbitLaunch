import numpy as np
import pytest
import types # Added import for types

from Main.simulation import Simulation, ControlCommand
from Environment.config import EnvironmentConfig
from Hardware.config import HardwareConfig
from Software.config import SoftwareConfig
from Main.config import SimulationConfig
from Logging.config import LoggingConfig
from Environment.atmosphere import AtmosphereModel
from Software.guidance import Guidance, StageAwarePitchProgram, ParameterizedThrottleProgram # Import Guidance related classes
from Hardware.stage import Stage # For dummy rocket_stages_info


class DummyProps:
    def __init__(self, rho=1.0, T=300.0, p=0.0):
        self.rho = rho
        self.T = T
        self.p = p


class DummyAtmosphere:
    def __init__(self, props, env_config=None):
        self.props = props
        self.env_config = env_config or EnvironmentConfig()

    def properties(self, alt, t_env):
        return self.props
    
    def get_speed_of_sound(self, alt, t_env):
        gamma = self.env_config.air_gamma
        R_air = self.env_config.air_gas_constant
        return np.sqrt(max(gamma * R_air * self.props.T, 0.0))


class DummyEarth:
    def __init__(self, radius=1000.0, mu=1.0, omega_vec=np.zeros(3), j2=None):
        self.radius = radius
        self.mu = mu
        self.omega_vec = omega_vec
        self.j2 = j2

    def gravity_accel(self, r):
        return np.zeros(3)

    def atmosphere_velocity(self, r):
        return np.zeros(3)


def test_rhs_scales_thrust_by_max_accel(monkeypatch):
    env_config = EnvironmentConfig()
    hw_config = HardwareConfig()
    sim_config = SimulationConfig()
    log_config = LoggingConfig()
    sw_config = SoftwareConfig()

    env_config.use_jet_stream_model = False # Disable wind for this test
    sim_config.max_accel_limit = 1.0

    props = DummyProps(rho=0.0, T=300.0, p=0.0)
    atmosphere = DummyAtmosphere(props, env_config)
    earth = DummyEarth(radius=1000.0, mu=0.0, omega_vec=env_config.earth_omega_vec)

    # Mock Aerodynamics
    mock_aero = types.SimpleNamespace()
    mock_aero.drag_force = lambda state, earth, t_env, rocket: np.zeros(3)
    # Ensure it has the get_wind_at_altitude method if use_jet_stream_model is True
    # For this test, use_jet_stream_model is False, so no need to mock get_wind_at_altitude
    mock_aero.env_config = env_config

    # Mock Rocket
    mock_rocket = types.SimpleNamespace()
    mock_rocket.thrust_and_mass_flow = lambda t, throttle_command, thrust_direction_eci, state, p_amb, current_prop_mass: (np.array([10.0, 0.0, 0.0]), 0.0)
    mock_rocket.env_config = env_config # Needs G0
    mock_rocket.stages = [types.SimpleNamespace(prop_mass=1.0, dry_mass=1.0), types.SimpleNamespace(prop_mass=1.0, dry_mass=1.0)] # Mock stages
    mock_rocket.hw_config = hw_config # Not directly used in thrust_and_mass_flow mock

    # Dummy Guidance components
    dummy_pitch_program = StageAwarePitchProgram(sw_config=sw_config, env_config=env_config)
    dummy_upper_throttle_program = ParameterizedThrottleProgram(schedule=[[0.0, 1.0]])
    dummy_booster_program = ParameterizedThrottleProgram(schedule=[[0.0, 1.0]])
    # Need a basic rocket_stages_info for Guidance
    dummy_rocket_stages_info = [
        types.SimpleNamespace(dry_mass=1.0, prop_mass=1.0),
        types.SimpleNamespace(dry_mass=1.0, prop_mass=1.0)
    ]
    dummy_guidance = Guidance(
        sw_config=sw_config,
        env_config=env_config,
        pitch_program=dummy_pitch_program,
        upper_throttle_program=dummy_upper_throttle_program,
        booster_throttle_program=dummy_booster_program,
        rocket_stages_info=dummy_rocket_stages_info
    )

    sim = Simulation(
        earth=earth,
        atmosphere=atmosphere,
        aerodynamics=mock_aero,
        rocket=mock_rocket,
        sim_config=sim_config,
        env_config=env_config,
        log_config=log_config,
        guidance=dummy_guidance,
        sw_config=sw_config,
    )

    state = types.SimpleNamespace()
    state.r_eci = np.array([earth.radius + 1.0, 0.0, 0.0])
    state.v_eci = np.array([0.0, 0.0, 0.0])
    state.m = 1.0
    state.stage_index = 0

    control = ControlCommand(throttle=1.0, thrust_direction_eci=np.array([1.0, 0.0, 0.0])) # Unit vector for thrust direction

    _, dv_dt, _, _ = sim._rhs(t_env=0.0, t_sim=0.0, state=state, control=control)
    assert np.linalg.norm(dv_dt) <= 1.0 + 1e-6


def test_rhs_scales_throttle_by_max_q(monkeypatch):
    env_config = EnvironmentConfig()
    hw_config = HardwareConfig()
    sim_config = SimulationConfig()
    log_config = LoggingConfig()
    sw_config = SoftwareConfig()

    env_config.use_jet_stream_model = False # Disable wind for this test
    sim_config.max_q_limit = 10.0

    props = DummyProps(rho=1.0, T=300.0, p=0.0)
    atmosphere = DummyAtmosphere(props, env_config)
    earth = DummyEarth(radius=1000.0, mu=0.0, omega_vec=env_config.earth_omega_vec)

    # Mock Aerodynamics
    mock_aero = types.SimpleNamespace()
    mock_aero.drag_force = lambda state, earth, t_env, rocket: np.zeros(3) # Drag not relevant for this test, but needed
    mock_aero.env_config = env_config # For use_jet_stream_model

    # Mock Rocket
    mock_rocket = types.SimpleNamespace()
    # Intercept the control dict to check throttle
    mock_rocket.last_throttle = None
    def mock_thrust_and_mass_flow(t, throttle_command, thrust_direction_eci, state, p_amb, current_prop_mass):
        mock_rocket.last_throttle = throttle_command
        return np.array([1.0, 0.0, 0.0]), 0.0 # Return some thrust, 0 mass flow
    mock_rocket.thrust_and_mass_flow = mock_thrust_and_mass_flow
    mock_rocket.env_config = env_config # Needs G0
    mock_rocket.stages = [types.SimpleNamespace(prop_mass=1.0, dry_mass=1.0), types.SimpleNamespace(prop_mass=1.0, dry_mass=1.0)] # Mock stages
    mock_rocket.hw_config = hw_config # Not directly used in thrust_and_mass_flow mock

    # Dummy Guidance components
    dummy_pitch_program = StageAwarePitchProgram(sw_config=sw_config, env_config=env_config)
    dummy_upper_throttle_program = ParameterizedThrottleProgram(schedule=[[0.0, 1.0]])
    dummy_booster_program = ParameterizedThrottleProgram(schedule=[[0.0, 1.0]])
    # Need a basic rocket_stages_info for Guidance
    dummy_rocket_stages_info = [
        types.SimpleNamespace(dry_mass=1.0, prop_mass=1.0),
        types.SimpleNamespace(dry_mass=1.0, prop_mass=1.0)
    ]
    dummy_guidance = Guidance(
        sw_config=sw_config,
        env_config=env_config,
        pitch_program=dummy_pitch_program,
        upper_throttle_program=dummy_upper_throttle_program,
        booster_throttle_program=dummy_booster_program,
        rocket_stages_info=dummy_rocket_stages_info
    )

    sim = Simulation(
        earth=earth,
        atmosphere=atmosphere,
        aerodynamics=mock_aero,
        rocket=mock_rocket,
        sim_config=sim_config,
        env_config=env_config,
        log_config=log_config,
        guidance=dummy_guidance,
        sw_config=sw_config,
    )

    state = types.SimpleNamespace()
    state.r_eci = np.array([earth.radius + 1.0, 0.0, 0.0])
    state.v_eci = np.array([100.0, 0.0, 0.0]) # High speed to create dynamic pressure
    state.m = 1.0
    state.stage_index = 0

    control = ControlCommand(throttle=1.0, thrust_direction_eci=np.array([1.0, 0.0, 0.0]))

    sim._rhs(t_env=0.0, t_sim=0.0, state=state, control=control)

    # Dynamic pressure q = 0.5 * rho * v_rel_mag^2 = 0.5 * 1.0 * 100^2 = 5000.0
    # max_q_limit = 10.0
    # Expected throttle = 1.0 * (max_q_limit / q) = 1.0 * (10.0 / 5000.0)
    assert mock_rocket.last_throttle == pytest.approx(10.0 / 5000.0)

