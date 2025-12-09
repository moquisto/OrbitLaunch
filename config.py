"""
Central configuration for the launch simulation and optimizer.
Tweak values here instead of scattered through the code.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # Launch site
    launch_lat_deg: float = 28.60839
    launch_lon_deg: float = -80.60433

    # Target orbit (circular)
    target_orbit_alt_m: float = 420_000.0

    # Vehicle (BFR-inspired)
    booster_thrust_vac: float = 7.35e7
    booster_thrust_sl: float = 7.0e7
    booster_isp_vac: float = 347.0
    booster_isp_sl: float = 327.0
    booster_dry_mass: float = 2.7e5
    booster_prop_mass: float = 3.4e6

    upper_thrust_vac: float = 1.5e7
    upper_thrust_sl: float = 1.2e7
    upper_isp_vac: float = 380.0
    upper_isp_sl: float = 330.0
    upper_dry_mass: float = 1.3e5
    upper_prop_mass: float = 1.2e6

    ref_area_m2: float = 3.14159265359 * (4.5 ** 2)  # ~9 m dia

    # Staging/ramps
    main_engine_ramp_time: float = 1.0
    upper_engine_ramp_time: float = 1.0
    separation_delay_s: float = 30.0
    upper_ignition_delay_s: float = 30.0

    # Simulation timing
    main_duration_s: float = 3000.0
    main_dt_s: float = 1.0

    # Orbit tolerances
    orbit_speed_tol: float = 50.0
    orbit_radial_tol: float = 50.0
    orbit_alt_tol: float = 500.0
    orbit_ecc_tol: float = 0.01

    # Optional path constraints
    max_q_limit: float | None = None  # set to a Pa value to penalize exceeding
    max_accel_limit: float | None = None  # set to m/s^2 to penalize exceeding


CFG = Config()


if __name__ == "__main__":
    # Running this file directly executes the main simulation with current CFG values.
    from main import main as run_main

    run_main()
