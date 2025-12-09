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

    # Vehicle (BFR/Starship-inspired, closer to public estimates)
    booster_thrust_vac: float = 7.4e7   # ~74 MN (33 Raptors total)
    booster_thrust_sl: float = 7.2e7
    booster_isp_vac: float = 356.0
    booster_isp_sl: float = 330.0
    booster_dry_mass: float = 2.4e5     # ~240 t dry
    booster_prop_mass: float = 3.4e6    # ~3400 t prop

    upper_thrust_vac: float = 1.39e7    # ~13.9 MN (6 Raptors)
    upper_thrust_sl: float = 1.29e7
    upper_isp_vac: float = 380.0
    upper_isp_sl: float = 330.0
    upper_dry_mass: float = 1.2e5       # ~120 t dry incl. payload attach
    upper_prop_mass: float = 8.0e5      # ~800 t prop (two-burn demo)

    ref_area_m2: float = 3.14159265359 * (4.5 ** 2)  # ~9 m dia

    # Staging/ramps
    main_engine_ramp_time: float = 3.0
    upper_engine_ramp_time: float = 3.0
    separation_delay_s: float = 5.0     # coast after booster cutoff
    upper_ignition_delay_s: float = 2.0 # settle delay before upper ignition

    # Simulation timing
    main_duration_s: float = 10000
    main_dt_s: float = 1.0

    # Orbit tolerances
    orbit_speed_tol: float = 50.0
    orbit_radial_tol: float = 50.0
    orbit_alt_tol: float = 500.0
    orbit_ecc_tol: float = 0.01

    # Optional path constraints
    max_q_limit: float | None = 3e5  # set to a Pa value to penalize exceeding
    max_accel_limit: float | None = 40.0  # set to m/s^2 to penalize exceeding


CFG = Config()


if __name__ == "__main__":
    # Running this file directly executes the main simulation with current CFG values.
    from main import main as run_main

    run_main()
