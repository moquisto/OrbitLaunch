import numpy as np
from unittest.mock import patch

from Main.simulation import Simulation
from Main.state import State
from Main.integrators import RK4
from Environment.config import EnvironmentConfig
from Hardware.config import HardwareConfig
from Software.config import SoftwareConfig
from Main.config import SimulationConfig
from Logging.config import LoggingConfig
from Software.guidance import Guidance, GuidanceCommand, StageAwarePitchProgram, ParameterizedThrottleProgram
from Hardware.rocket import Rocket


def test_stage_separation_mass_drop_propagates_when_triggered_on_k4():
    """
    Regression: if stage separation is triggered on an RK4 substep (e.g. k4 at t+dt),
    the mass drop must propagate into the integrated State.
    """
    env_config = EnvironmentConfig()
    hw_config = HardwareConfig()
    sw_config = SoftwareConfig()
    sim_config = SimulationConfig()
    log_config = LoggingConfig()

    earth = env_config.create_earth_model()
    atmosphere = env_config.create_atmosphere_model()
    aero = env_config.create_aerodynamics_model(atmosphere=atmosphere, reference_area=hw_config.ref_area_m2)
    booster_engine = hw_config.create_booster_engine(env_config)
    upper_engine = hw_config.create_upper_engine(env_config)
    booster_stage = hw_config.create_booster_stage(booster_engine)
    upper_stage = hw_config.create_upper_stage(upper_engine)
    rocket = Rocket(stages=[booster_stage, upper_stage], hw_config=hw_config, env_config=env_config)

    pitch_prog = StageAwarePitchProgram(sw_config, env_config)
    upper_throttle = ParameterizedThrottleProgram([[0.0, 0.0]])
    booster_throttle = ParameterizedThrottleProgram([[0.0, 0.0]])
    rocket_stages_info = rocket.stages
    guidance = Guidance(
        sw_config=sw_config,
        env_config=env_config,
        pitch_program=pitch_prog,
        upper_throttle_program=upper_throttle,
        booster_throttle_program=booster_throttle,
        rocket_stages_info=rocket_stages_info,
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

    # Start on the pad with full mass.
    lat = np.deg2rad(env_config.launch_lat_deg)
    lon = np.deg2rad(env_config.launch_lon_deg)
    r0 = env_config.earth_radius_m * np.array(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        dtype=float,
    )
    state0 = State(r_eci=r0, v_eci=np.zeros(3), m=sum(s.total_mass() for s in rocket.stages), stage_index=0)

    dry_mass_to_drop = rocket.stages[0].dry_mass + rocket.stages[0].prop_mass
    expected_remaining_mass = rocket.stages[1].dry_mass + rocket.stages[1].prop_mass

    def cmd_fn(tau: float, state: State, *_args, **_kwargs) -> GuidanceCommand:
        # Trigger separation only at tau >= 0.75 so that RK4's k4 evaluation (tau=1.0)
        # sees it, but k1/k2/k3 do not.
        separate_now = bool(tau >= 0.75 and int(getattr(state, "stage_index", 0)) == 0)
        return GuidanceCommand(
            throttle=0.0,
            thrust_direction_eci=np.array([0.0, 0.0, 1.0]),
            initiate_stage_separation=separate_now,
            new_stage_index=1 if separate_now else None,
            dry_mass_to_drop=dry_mass_to_drop if separate_now else None,
        )

    with patch.object(guidance, "compute_command", side_effect=cmd_fn):
        log = sim.run(t_env_start=0.0, duration=2.0, dt=1.0, state0=state0)

    assert max(log.stage) == 1
    assert np.isclose(min(log.m), expected_remaining_mass, atol=1e-6)
