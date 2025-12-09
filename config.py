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
    main_duration_s: float = 800.0
    main_dt_s: float = 1.0

    # Optimizer timing
    opt_coarse_duration_s: float = 800.0
    opt_coarse_dt_s: float = 1.0
    opt_refine_duration_s: float = 1200.0
    opt_refine_dt_s: float = 0.2

    # Orbit tolerances
    orbit_speed_tol: float = 50.0
    orbit_radial_tol: float = 50.0
    orbit_alt_tol: float = 500.0
    orbit_ecc_tol: float = 0.01

    # Optional path constraints
    max_q_limit: float | None = None  # set to a Pa value to penalize exceeding
    max_accel_limit: float | None = None  # set to m/s^2 to penalize exceeding

    # Optimizer search parameters
    opt_n_random: int = 15
    opt_n_heuristic: int = 15
    opt_top_k: int = 3
    opt_nm_maxiter: int = 60
    opt_plot_each: bool = False
    opt_use_cma: bool = True
    opt_cma_maxiter: int = 50
    opt_cma_sigma_scale: float = 0.2  # fraction of each bound range for initial sigma

    # Optimizer parameter bounds
    prop1_bounds: tuple[float, float] = (2.5e6, 3.6e6)
    prop2_bounds: tuple[float, float] = (0.8e6, 1.4e6)
    throttle1_bounds: tuple[float, float] = (0.8, 1.0)
    throttle2_bounds: tuple[float, float] = (0.8, 1.0)
    pitch_start_alt_bounds: tuple[float, float] = (3_000.0, 8_000.0)
    pitch_end_alt_bounds: tuple[float, float] = (40_000.0, 120_000.0)

    # Expanded guidance bounds (multi-breakpoint pitch angles and throttle splits)
    pitch_alt_bounds: tuple[float, float] = (3_000.0, 120_000.0)
    pitch_angle_bounds_deg: tuple[float, float] = (0.0, 90.0)  # 90 = vertical, 0 = horizontal
    throttle_split_time_bounds: tuple[float, float] = (20.0, 400.0)

    # Coarse tolerances for feasibility search
    orbit_speed_tol_coarse: float = 150.0
    orbit_radial_tol_coarse: float = 150.0
    orbit_alt_tol_coarse: float = 5_000.0


CFG = Config()
