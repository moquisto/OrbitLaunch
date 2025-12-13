import numpy as np
import types
import pytest
from unittest.mock import Mock, patch

from Main.simulation import Simulation, Guidance
from Main.state import State
from Main.integrators import RK4
from Environment.config import EnvironmentConfig
from Hardware.config import HardwareConfig
from Software.config import SoftwareConfig
from Main.config import SimulationConfig
from Logging.config import LoggingConfig
from Environment.gravity import EarthModel
from Environment.atmosphere import AtmosphereModel
from Environment.aerodynamics import Aerodynamics
from Hardware.rocket import Rocket
from Hardware.engine import Engine # Corrected import
from Hardware.stage import Stage # Corrected import
from Software.guidance import GuidanceCommand, StageAwarePitchProgram, ParameterizedThrottleProgram # Import for dummy guidance components
from typing import Optional


class DummyProps:
    def __init__(self):
        self.p = 0.0
        self.rho = 0.0
        self.T = 300.0


class DummyEarth:
    def __init__(self, mu=1e5, radius=1000.0, omega_vec=np.zeros(3), j2=None):
        self.mu = mu
        self.radius = radius
        self.omega_vec = omega_vec
        self.j2 = j2

    def gravity_accel(self, r):
        r = np.asarray(r, dtype=float)
        r_norm = np.linalg.norm(r)
        if r_norm == 0:
            return np.zeros(3)
        a_central = -self.mu * r / (r_norm ** 3)
        if self.j2 is None or self.j2 == 0.0:
            return a_central
        # Simplified J2 perturbation for testing
        x, y, z = r
        r2 = r_norm * r_norm
        z2 = z * z
        factor = 1.5 * self.j2 * self.mu * (self.radius**2) / (r_norm**5)
        k = 5.0 * z2 / r2
        a_j2_x = factor * x * (1.0 - k)
        a_j2_y = factor * y * (1.0 - k)
        a_j2_z = factor * z * (3.0 - k)
        a_j2 = np.array([a_j2_x, a_j2_y, a_j2_z], dtype=float)
        return a_central + a_j2

    def atmosphere_velocity(self, r):
        return np.zeros(3)


class DummyAtmosphere:
    def __init__(self, env_config=None):
        self._rho = 1.0
        self._T = 300.0
        self._p = 0.0
        self.env_config = env_config or EnvironmentConfig()

    def properties(self, alt, t_env):
        return type("Props", (), {"rho": self._rho, "T": self._T, "p": self._p})
    
    def get_speed_of_sound(self, alt, t_env):
        return 340.0


class DummyAero:
    def __init__(self, atmosphere=None, cd_model=None, env_config=None):
        self.atmosphere = atmosphere or DummyAtmosphere()
        self.cd_model = cd_model or type("CdModel", (), {"cd": lambda mach: 2.0})()
        self.env_config = env_config or EnvironmentConfig()

    def drag_force(self, state, earth, t_env, rocket):
        return np.zeros(3)

    # Add mock for _get_wind_at_altitude
    def _get_wind_at_altitude(self, altitude: float, r_eci: Optional[np.ndarray] = None) -> np.ndarray:
        return np.zeros(3)


def build_dummy_rocket(hw_config: HardwareConfig, env_config: EnvironmentConfig) -> Rocket:
    booster_engine = Engine(
        thrust_vac=hw_config.booster_thrust_vac,
        thrust_sl=hw_config.booster_thrust_sl,
        isp_vac=hw_config.booster_isp_vac,
        isp_sl=hw_config.booster_isp_sl,
        p_sl=env_config.P_SL,
    )
    upper_engine = Engine(
        thrust_vac=hw_config.upper_thrust_vac,
        thrust_sl=hw_config.upper_thrust_sl,
        isp_vac=hw_config.upper_isp_vac,
        isp_sl=hw_config.upper_isp_sl,
        p_sl=env_config.P_SL,
    )

    booster_stage = Stage(
        dry_mass=hw_config.booster_dry_mass,
        prop_mass=hw_config.booster_prop_mass,
        engine=booster_engine,
        ref_area=hw_config.ref_area_m2,
    )
    upper_stage = Stage(
        dry_mass=hw_config.upper_dry_mass,
        prop_mass=hw_config.upper_prop_mass,
        engine=upper_engine,
        ref_area=hw_config.ref_area_m2,
    )
    return Rocket(
        stages=[booster_stage, upper_stage],
        hw_config=hw_config,
        env_config=env_config,
    )


def test_simulation_orbit_exit():
    # Instantiate configs
    env_config = EnvironmentConfig()
    hw_config = HardwareConfig()
    sim_config = SimulationConfig(
        target_orbit_alt_m=1.0,  # Target is surface orbit for this test
        orbit_alt_tol=10.0,      # Small tolerance for altitude
        orbit_speed_tol=10.0,    # Small tolerance for speed
        orbit_radial_tol=10.0,   # Small tolerance for radial velocity
        exit_on_orbit=True       # Exit once orbit is achieved
    )
    log_config = LoggingConfig()
    sw_config = SoftwareConfig() # Needed for Simulation constructor

    earth = DummyEarth(mu=env_config.earth_mu, radius=env_config.earth_radius_m, omega_vec=env_config.earth_omega_vec)
    atmosphere = DummyAtmosphere(env_config)
    aero = DummyAero(atmosphere=atmosphere, env_config=env_config)
    rocket = build_dummy_rocket(hw_config, env_config)
    # Dummy Guidance components
    dummy_pitch_program = Mock(spec=StageAwarePitchProgram)
    dummy_pitch_program.booster_time_points = np.array([0.0])
    dummy_pitch_program.upper_time_points = np.array([0.0])
    dummy_pitch_program.booster_angles_rad = np.array([0.0])
    dummy_pitch_program.upper_angles_rad = np.array([0.0])
    dummy_pitch_program.prograde_threshold = 0.0
    dummy_pitch_program.earth_radius = env_config.earth_radius_m
    dummy_pitch_program.return_value = np.array([0,0,1]) # Mock __call__ method

    dummy_upper_throttle_program = Mock(spec=ParameterizedThrottleProgram)
    dummy_upper_throttle_program.schedule = [[0.0, 1.0]]
    dummy_upper_throttle_program.return_value = 1.0 # Mock __call__ method

    dummy_booster_program = ParameterizedThrottleProgram(schedule=[[0.0, 1.0]])
    dummy_rocket_stages_info = [
        types.SimpleNamespace(dry_mass=1.0, prop_mass=1.0),
        types.SimpleNamespace(dry_mass=1.0, prop_mass=1.0)
    ]
    guidance = Guidance(
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
        aerodynamics=aero,
        rocket=rocket,
        sim_config=sim_config,
        env_config=env_config,
        log_config=log_config,
        integrator=RK4(), # Fixed: directly use RK4
        guidance=guidance,
    )

    r0 = np.array([earth.radius, 0.0, 0.0])
    v_circ = np.sqrt(earth.mu / earth.radius)
    state0 = State(r_eci=r0, v_eci=np.array([0.0, v_circ, 0.0]), m=1.0, stage_index=0)

    log = sim.run(
        t_env_start=0.0,
        duration=100.0,
        dt=1.0,
        state0=state0,
        # These parameters are now taken from sim_config
    )

    assert log.orbit_achieved is True
    assert log.cutoff_reason == "orbit_target_met"
    assert len(log.t_sim) < 5


def test_simulation_impact_terminates():
    # Instantiate configs
    env_config = EnvironmentConfig()
    hw_config = HardwareConfig()
    sim_config = SimulationConfig()
    log_config = LoggingConfig()
    sw_config = SoftwareConfig()

    earth = DummyEarth(mu=env_config.earth_mu, radius=env_config.earth_radius_m, omega_vec=env_config.earth_omega_vec)
    atmosphere = DummyAtmosphere(env_config)
    aero = DummyAero(atmosphere=atmosphere, env_config=env_config)
    rocket = build_dummy_rocket(hw_config, env_config)
    # Dummy Guidance components
    dummy_pitch_program = Mock(spec=StageAwarePitchProgram)
    dummy_pitch_program.booster_time_points = np.array([0.0])
    dummy_pitch_program.upper_time_points = np.array([0.0])
    dummy_pitch_program.booster_angles_rad = np.array([0.0])
    dummy_pitch_program.upper_angles_rad = np.array([0.0])
    dummy_pitch_program.prograde_threshold = 0.0
    dummy_pitch_program.earth_radius = env_config.earth_radius_m
    dummy_pitch_program.return_value = np.array([0,0,1]) # Mock __call__ method

    dummy_upper_throttle_program = Mock(spec=ParameterizedThrottleProgram)
    dummy_upper_throttle_program.schedule = [[0.0, 1.0]]
    dummy_upper_throttle_program.return_value = 1.0 # Mock __call__ method

    dummy_booster_program = ParameterizedThrottleProgram(schedule=[[0.0, 1.0]])
    dummy_rocket_stages_info = [
        types.SimpleNamespace(dry_mass=1.0, prop_mass=1.0),
        types.SimpleNamespace(dry_mass=1.0, prop_mass=1.0)
    ]
    guidance = Guidance(
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
        aerodynamics=aero,
        rocket=rocket,
        sim_config=sim_config,
        env_config=env_config,
        log_config=log_config,
        integrator=RK4(),
        guidance=guidance,
    )

    r0 = np.array([earth.radius - 200.0, 0.0, 0.0])
    state0 = State(r_eci=r0, v_eci=np.zeros(3), m=1.0, stage_index=0)

    log = sim.run(
        t_env_start=0.0,
        duration=10.0,
        dt=1.0,
        state0=state0,
    )

    assert log.orbit_achieved is False
    assert log.cutoff_reason == "impact"
    assert len(log.t_sim) == 1


def test_simulation_stage_separation_and_mass_drop():
    # Instantiate configs
    env_config = EnvironmentConfig()
    hw_config = HardwareConfig()
    sim_config = SimulationConfig()
    log_config = LoggingConfig()
    sw_config = SoftwareConfig()
    sw_config.separation_delay_s = 0.0 # Set separation delay to 0 for immediate separation in test

    earth = DummyEarth(mu=env_config.earth_mu, radius=env_config.earth_radius_m, omega_vec=env_config.earth_omega_vec)
    atmosphere = DummyAtmosphere(env_config)
    aero = DummyAero(atmosphere=atmosphere, env_config=env_config)
    rocket = build_dummy_rocket(hw_config, env_config)
    # Dummy Guidance components (we'll patch compute_command below).
    dummy_pitch_program = Mock(spec=StageAwarePitchProgram)
    dummy_pitch_program.return_value = np.array([0, 0, 1])
    dummy_upper_throttle_program = Mock(spec=ParameterizedThrottleProgram)
    dummy_upper_throttle_program.return_value = 0.0
    dummy_booster_program = Mock(spec=ParameterizedThrottleProgram)
    dummy_booster_program.return_value = 0.0
    dummy_rocket_stages_info = [
        types.SimpleNamespace(dry_mass=hw_config.booster_dry_mass, prop_mass=rocket.stages[0].prop_mass),
        types.SimpleNamespace(dry_mass=hw_config.upper_dry_mass, prop_mass=rocket.stages[1].prop_mass),
    ]
    guidance = Guidance(
        sw_config=sw_config,
        env_config=env_config,
        pitch_program=dummy_pitch_program,
        upper_throttle_program=dummy_upper_throttle_program,
        booster_throttle_program=dummy_booster_program,
        rocket_stages_info=dummy_rocket_stages_info,
    )

    sim = Simulation(
        earth=earth,
        atmosphere=atmosphere,
        aerodynamics=aero,
        rocket=rocket,
        sim_config=sim_config,
        env_config=env_config,
        log_config=log_config,
        integrator=RK4(),
        guidance=guidance,
    )

    r0 = np.array([earth.radius + 10.0, 0.0, 0.0])
    # Initial mass should be the sum of all dry and prop masses
    initial_total_mass = sum(s.dry_mass + s.prop_mass for s in rocket.stages)
    state0 = State(r_eci=r0, v_eci=np.zeros(3), m=initial_total_mass, stage_index=0)

    with patch.object(guidance, "compute_command") as mock_compute_command:
        mock_compute_command.return_value = GuidanceCommand(
            throttle=0.0,
            thrust_direction_eci=np.array([0.0, 0.0, 1.0]),
            initiate_stage_separation=True,
            new_stage_index=1,
            dry_mass_to_drop=hw_config.booster_dry_mass + rocket.stages[0].prop_mass,
        )
        log = sim.run(
            t_env_start=0.0,
            duration=2.0,
            dt=1.0,
            state0=state0,
        )

    assert max(log.stage) == 1
    # Expected remaining mass after booster separation (booster dry mass + prop, upper stage dry + prop)
    # Booster dry mass (10.0) + remaining prop (5.0) are dropped.
    # So initial_total_mass - (booster dry + booster prop) = (upper dry + upper prop)
    expected_remaining_mass = rocket.stages[1].dry_mass + rocket.stages[1].prop_mass
    assert np.isclose(min(log.m), expected_remaining_mass, atol=1e-6)
