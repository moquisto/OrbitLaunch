"""
Central configuration for the launch simulation and optimizer.
Tweak values here instead of scattered through the code.
"""

import dataclasses
from dataclasses import dataclass


@dataclass
class Config:
    # Launch site
    launch_lat_deg: float = 28.60839
    launch_lon_deg: float = -80.60433

    # Target orbit (circular)
    target_orbit_alt_m: float = 420_000.0

    # Vehicle (BFR/Starship-inspired, based on recent public estimates)
    booster_thrust_vac: float = 7.4e7
    booster_thrust_sl: float = 7.35e7
    booster_isp_vac: float = 350.0
    booster_isp_sl: float = 327.0
    booster_dry_mass: float = 2.75e5    # ~275 t dry
    booster_prop_mass: float = 3.4e6    # ~3400 t prop

    upper_thrust_vac: float = 1.47e7
    upper_thrust_sl: float = 1.35e7
    upper_isp_vac: float = 380.0
    upper_isp_sl: float = 327.0
    upper_dry_mass: float = 1.0e5       # ~100 t dry
    upper_prop_mass: float = 1.2e6      # ~1200 t prop

    ref_area_m2: float = 3.14159265359 * (4.5 ** 2)  # ~9 m dia
    cd_constant: float = 0.35
    engine_min_throttle: float = 0.4  # Raptor throttle floor (fraction of full thrust)
    use_j2: bool = True
    j2_coeff: float = 1.08262668e-3

    # Staging/ramps
    main_engine_ramp_time: float = 3.0
    upper_engine_ramp_time: float = 3.0
    separation_delay_s: float = 5.0     # coast after booster cutoff
    upper_ignition_delay_s: float = 2.0 # settle delay before upper ignition
    meco_mach: float = 6.0
    separation_altitude_m: float | None = None

    # Simulation timing
    main_duration_s: float = 100000
    main_dt_s: float = 1.0

    # Orbit tolerances
    orbit_speed_tol: float = 50.0
    orbit_radial_tol: float = 50.0
    orbit_alt_tol: float = 500.0
    orbit_ecc_tol: float = 0.01
    exit_on_orbit: bool = False
    post_orbit_coast_s: float = 0.0

    # Optional path constraints
    max_q_limit: float | None = 3e5  # set to a Pa value to penalize exceeding
    max_accel_limit: float | None = 40.0  # set to m/s^2 to penalize exceeding

    # Pitch program shape
    pitch_turn_start_m: float = 5_000.0
    pitch_turn_end_m: float = 60_000.0

    # Atmosphere
    atmosphere_switch_alt_m: float = 86_000.0
    use_jet_stream_model: bool = True



CFG = Config()


if __name__ == "__main__":
    # Running this file directly executes the main simulation with current CFG values.
    from main import main as run_main

    run_main()
