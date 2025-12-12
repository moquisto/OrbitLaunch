"""
Entry point to run a simple end-to-end launch simulation.

This builds the Earth/atmosphere/aero/rocket stack, runs a fixed-step
integration, and prints a short summary. Parameters are loosely inspired by
SpaceX's BFR/Starship system but reduced to two stages with simplified thrust
and mass numbers for a lightweight demo.
"""

from __future__ import annotations

import datetime as dt
import importlib
import numpy as np
from typing import Tuple

from Environment.config import EnvironmentConfig
from Environment.gravity import orbital_elements_from_state

from Hardware.rocket import Rocket
from Hardware.config import HardwareConfig

from Software.guidance import Guidance, ParameterizedThrottleProgram
from Software.config import SoftwareConfig

from Main.integrators import RK4, VelocityVerlet
from Main.state import State
from Main.simulation import Simulation
from Main.config import SimulationConfig

from Logging.config import LoggingConfig
from Analysis.config import AnalysisConfig

# Main orchestrator function (will be implemented next)
def main_orchestrator(
    env_config: Optional[EnvironmentConfig] = None,
    hw_config: Optional[HardwareConfig] = None,
    sw_config: Optional[SoftwareConfig] = None,
    sim_config: Optional[SimulationConfig] = None,
    log_config: Optional[LoggingConfig] = None,
    analysis_config: Optional[AnalysisConfig] = None,
    # Optional guidance program instances for optimization
    pitch_program_instance: Optional[Any] = None,
    upper_throttle_program_instance: Optional[Any] = None,
    booster_throttle_schedule_list: Optional[List[List[float]]] = None,
):
    # 1. Instantiate all config objects if not provided
    env_config = env_config or EnvironmentConfig()
    hw_config = hw_config or HardwareConfig()
    sw_config = sw_config or SoftwareConfig()
    sim_config = sim_config or SimulationConfig()
    log_config = log_config or LoggingConfig()
    analysis_config = analysis_config or AnalysisConfig()

    # 2. Instantiate core components using factory methods from configs
    # Environment
    earth = env_config.create_earth_model()
    atmosphere = env_config.create_atmosphere_model()
    aero = env_config.create_aerodynamics_model(atmosphere=atmosphere)

    # Hardware
    booster_engine = hw_config.create_booster_engine(env_config)
    upper_engine = hw_config.create_upper_engine(env_config)

    booster_stage = hw_config.create_booster_stage(booster_engine)
    upper_stage = hw_config.create_upper_stage(upper_engine)

    rocket_stages = [booster_stage, upper_stage]

    # Software (Guidance)
    # The target_radius and mu are needed for the throttle program if it's function-based.
    # These are derived from sim_config and env_config.
    target_orbit_radius = env_config.earth_radius_m + sim_config.target_orbit_alt_m
    
    pitch_program = pitch_program_instance or sw_config.create_pitch_program(env_config)
    throttle_controller = upper_throttle_program_instance or sw_config.create_throttle_program(target_orbit_radius, env_config.earth_mu)
    
    guidance = sw_config.create_guidance(
        pitch_program=pitch_program,
        throttle_schedule=throttle_controller,
        booster_throttle_schedule=booster_throttle_schedule_list or sw_config.booster_throttle_program,
        rocket_stages_info=rocket_stages, # Pass stages info to Guidance
    )

    # Rocket instance
    rocket = Rocket(
        stages=rocket_stages,
        hw_config=hw_config,
        env_config=env_config,
    )

    # Integrator
    integrator_name = str(getattr(sim_config, "integrator", "rk4")).lower()
    if integrator_name in ("rk4", "runge-kutta", "rk"):
        integrator = RK4()
    elif integrator_name in ("velocity_verlet", "verlet", "vv"):
        integrator = VelocityVerlet()
    else:
        raise ValueError(f"Unknown integrator '{sim_config.integrator}'. Expected 'rk4' or 'velocity_verlet'.")

    # Simulation
    sim = Simulation(
        earth=earth,
        atmosphere=atmosphere,
        aerodynamics=aero,
        rocket=rocket,
        integrator=integrator,
        guidance=guidance,
        sim_config=sim_config,
        env_config=env_config,
        log_config=log_config,
        sw_config=sw_config # Pass sw_config to Simulation for potential internal use
    )

    # Initial state: surface at launch site, co-rotating atmosphere
    lat = np.deg2rad(env_config.launch_lat_deg)
    lon = np.deg2rad(env_config.launch_lon_deg)
    r0 = env_config.earth_radius_m * np.array(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        dtype=float,
    )
    v0 = earth.atmosphere_velocity(r0)  # start with Earth's rotation speed
    m0 = sum(stage.total_mass() for stage in rocket_stages)
    state0 = State(r_eci=r0, v_eci=v0, m=m0, stage_index=0)

    # Use a constant start time for deterministic runs.
    t0 = 0.0

    return sim, state0, t0, log_config, analysis_config


def run_simulation_and_get_log(sim: Simulation, state0: State, t_env0: float):
    """Runs the simulation and returns the log."""
    duration = sim.sim_config.main_duration_s
    dt = sim.sim_config.main_dt_s

    # The throttle controller is already set up in main_orchestrator
    # as part of the Guidance object passed to the Simulation.
    # No need to re-select or re-instantiate here.

    log = sim.run(
        t_env0,
        duration,
        dt,
        state0,
    )
    return log

def main():
    sim, state0, t_env0, log_config, analysis_config = main_orchestrator()
    log = run_simulation_and_get_log(sim, state0, t_env0)

def print_summary(log, sim, log_config, analysis_config):
    """Prints a summary of the simulation results."""
    def format_dist(val_km):
        """Helper to format distance values that could be None or infinity."""
        if val_km is None:
            return "n/a"
        if np.isinf(val_km):
            return "inf"
        return f"{val_km:.2f}"

    earth_radius = sim.env_config.earth_radius_m
    final_alt_km = (np.linalg.norm(log.r[-1]) - earth_radius) / 1000.0
    final_speed = np.linalg.norm(log.v[-1])
    final_mass = log.m[-1]
    final_stage = log.stage[-1]
    max_alt_km = max(log.altitude) / 1000.0 if log.altitude else 0.0
    max_speed = max(log.speed) if log.speed else 0.0
    max_q = max(log.dynamic_pressure) if log.dynamic_pressure else 0.0
    stage_switch_times = [log.t_sim[i] for i in range(1, len(log.stage)) if log.stage[i] != log.stage[i - 1]]
    
    # Basic orbital diagnostics from final state
    a, rp, ra = orbital_elements_from_state(log.r[-1], log.v[-1], sim.env_config.earth_mu)
    rp_alt_km = (rp - earth_radius) / 1000.0 if rp is not None else None
    ra_alt_km = (ra - earth_radius) / 1000.0 if ra is not None else None
    
    # Key event indices
    idx_max_alt = int(np.argmax(log.altitude)) if log.altitude else 0
    idx_max_speed = int(np.argmax(log.speed)) if log.speed else 0
    idx_upper_off = np.argmin(np.abs(np.array(log.t_sim) - (sim.rocket.stage_engine_off_complete_time[1] or log.t_sim[-1]))) if log.t_sim else 0

    def print_state(label: str, idx: int):
        if not log.t_sim or idx >= len(log.t_sim):
            print(f"{label}: Log data not available.")
            return
            
        a_i, rp_i, ra_i = orbital_elements_from_state(log.r[idx], log.v[idx], sim.env_config.earth_mu)
        rp_alt_i = (rp_i - earth_radius) / 1000.0 if rp_i is not None else None
        ra_alt_i = (ra_i - earth_radius) / 1000.0 if ra_i is not None else None
        
        print(
            f"{label} @ t={log.t_sim[idx]:.1f}s: "
            f"alt={log.altitude[idx]/1000:.2f} km, "
            f"speed={log.speed[idx]:.1f} m/s, "
            f"fpa={log.flight_path_angle_deg[idx]:.2f} deg, "
            f"q={log.dynamic_pressure[idx]:.0f} Pa, "
            f"rp={format_dist(rp_alt_i)} km, ra={format_dist(ra_alt_i)} km"
        )

    print("\n=== Simulation summary ===")
    print(f"Cutoff reason: {log.cutoff_reason}")
    print(f"Steps: {len(log.t_sim)}")
    if log.t_sim:
        print(f"Final sim time  : {log.t_sim[-1]:.1f} s")
        print(f"Final altitude  : {final_alt_km:.2f} km")
        print(f"Final speed     : {final_speed:.1f} m/s")
        print(f"Final mass      : {final_mass:.1f} kg")
        print(f"Final stage idx : {final_stage}")
    print(f"Max altitude    : {max_alt_km:.2f} km")
    print(f"Max speed       : {max_speed:.1f} m/s")
    print(f"Max q           : {max_q:.1f} Pa")
    print(f"Stage switches  : {stage_switch_times}")
    if a is not None and not np.isinf(a):
        print(f"Semi-major axis : {a/1000:.2f} km")
    print(f"Perigee altitude: {format_dist(rp_alt_km)} km")
    print(f"Apoapsis altitude: {format_dist(ra_alt_km)} km")
    if log.orbit_achieved:
        print("Orbit target met within tolerances.")
    else:
        print("Orbit target NOT met.")
        
    # Stage fuel/engine timing diagnostics
    booster_empty = sim.rocket.stage_fuel_empty_time[0]
    upper_empty = sim.rocket.stage_fuel_empty_time[1]
    booster_off = sim.rocket.stage_engine_off_complete_time[0]
    upper_off = sim.rocket.stage_engine_off_complete_time[1]
    if booster_empty is not None:
        print(f"Booster fuel empty at t = {booster_empty:.1f} s")
    if booster_off is not None:
        print(f"Booster engine off at t = {booster_off:.1f} s")
    if upper_empty is not None:
        print(f"Upper fuel empty at t = {upper_empty:.1f} s")
    if upper_off is not None:
        print(f"Upper engine off at t = {upper_off:.1f} s")
    print(f"Remaining prop (booster, upper): {[f'{p:.1f}' for p in sim.rocket.stage_prop_remaining]}")
    
    print_state("Max altitude", idx_max_alt)
    print_state("Max speed", idx_max_speed)
    print_state("Upper engine off", idx_upper_off)

def main():
    sim, state0, t_env0, log_config, analysis_config = main_orchestrator()
    log = run_simulation_and_get_log(sim, state0, t_env0)
    print_summary(log, sim, log_config, analysis_config)

    save_log_to_txt(log, log_config.log_filename)
    # Enable static trajectory plot; keep animation disabled for headless use.
    if log_config.plot_trajectory:
        plot_trajectory_3d(log, sim.env_config.earth_radius_m)
    if log_config.animate_trajectory:
        animate_trajectory(log, sim.env_config.earth_radius_m)

from Analysis.plotting import plot_trajectory_3d, animate_trajectory


from Logging.generate_logs import save_log_to_txt


if __name__ == "__main__":
    main()
