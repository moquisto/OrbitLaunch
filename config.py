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

    # Central body (can be changed to model other planets)
    earth_mu: float = 3.986_004_418e14          # [m^3/s^2]
    earth_radius_m: float = 6_371_000.0         # [m]
    earth_omega_vec: tuple[float, float, float] = (0.0, 0.0, 7.292_115_9e-5)  # [rad/s]

    # Target orbit (circular)
    target_orbit_alt_m: float = 420_000.0

    # Vehicle (BFR/Starship-inspired, based on recent public estimates)
    booster_thrust_vac: float = 7.6e7
    booster_thrust_sl: float = 7.6e7
    booster_isp_vac: float = 380.0
    booster_isp_sl: float = 330.0
    booster_dry_mass: float = 1.2e5    # ~275 t dry
    booster_prop_mass: float = 3.4e6    # ~3400 t prop

    upper_thrust_vac: float = 1.5e7
    upper_thrust_sl: float = 1.35e7
    upper_isp_vac: float = 380.0
    upper_isp_sl: float = 330.0
    upper_dry_mass: float = 0.8e5       # ~100 t dry
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
    engine_shutdown_ramp_s: float = 1.0  # burn-out ramp-down duration for each stage
    throttle_full_shape_threshold: float = 0.99  # shape value considered "full" for min throttle enforcement
    mach_reference_speed: float = 340.0  # [m/s] reference speed of sound for Mach estimates
    meco_mach: float = 6.0
    separation_altitude_m: float | None = None

    # Simulation timing
    main_duration_s: float = 100000
    main_dt_s: float = 0.1
    integrator: str = "rk4"  # options: "rk4", "velocity_verlet"

    # Orbit tolerances
    orbit_speed_tol: float = 50.0
    orbit_radial_tol: float = 50.0
    orbit_alt_tol: float = 500.0
    orbit_ecc_tol: float = 0.01
    exit_on_orbit: bool = False
    post_orbit_coast_s: float = 0.0

    # Optional path constraints
    max_q_limit: float | None = 1e6  # set to a Pa value to penalize exceeding
    max_accel_limit: float | None = 40.0  # set to m/s^2 to penalize exceeding

    # --- Guidance ---
    # The guidance for both pitch and throttle can be specified in one of two ways:
    # 1. 'parameterized': A simple, data-driven program defined by a list of points.
    #    This is useful for optimization.
    # 2. 'function': A custom Python function or class that contains more complex logic.
    #    The function/class is specified by an import string.

    # Pitch Guidance
    pitch_guidance_mode: str = "parameterized"  # 'parameterized' or 'function'
    pitch_guidance_function: str = "custom_guidance.simple_pitch_program"
    pitch_program: list = dataclasses.field(
        default_factory=lambda: [
            [0.0, 89.0],     # Initial vertical ascent (very steep)
            [10_000.0, 88.0],  # Still very steep to clear initial atmosphere
            [40_000.0, 80.0],  # More aggressive pitch down than previous iteration
            [80_000.0, 40.0],  # Much more aggressive pitch over
            [120_000.0, 10.0], # Approaching horizontal faster
            [150_000.0, 0.0],  # Full horizontal at 150km (lower altitude than previous to build speed earlier)
        ]
    )
    pitch_prograde_speed_threshold: float = 100.0  # [m/s] speed needed to align with velocity

    # Atmosphere
    atmosphere_switch_alt_m: float = 86_000.0
    atmosphere_f107: float | None = None
    atmosphere_f107a: float | None = None
    atmosphere_ap: float | None = None
    use_jet_stream_model: bool = True

    # Physics
    G0: float = 9.80665  # [m/s^2] standard gravity for Isp conversion
    P_SL: float = 101325.0  # [Pa] reference sea-level pressure
    air_gamma: float = 1.4
    air_gas_constant: float = 287.05  # J/(kg*K)

    # Aerodynamics
    wind_direction_vec: tuple[float, float, float] = (1.0, 0.0, 0.0) # From East
    wind_alt_points: list = dataclasses.field(default_factory=lambda: [8_000.0, 10_500.0, 13_000.0])
    wind_speed_points: list = dataclasses.field(default_factory=lambda: [0.0, 50.0, 0.0])
    mach_cd_map: list = dataclasses.field(
        default_factory=lambda: [
            [0.0, 0.15], # Subsonic
            [0.8, 0.3],  # Approaching transonic, starts increasing
            [1.0, 0.6],  # Transonic, before peak
            [1.1, 0.65], # Peak at M=1.1, more realistic value for rounded nose
            [1.2, 0.6],  # After peak, decreasing
            [1.8, 0.5],  # Supersonic, decreasing
            [3.0, 0.4],  # Supersonic, decreasing
            [5.0, 0.35], # Supersonic, decreasing
            [10.0, 0.3]  # Supersonic/Hypersonic, asymptoting
        ]
    )

    # Throttle Guidance
    throttle_guidance_mode: str = "parameterized"  # 'parameterized' or 'function'
    throttle_guidance_function_class: str = "custom_guidance.TwoPhaseUpperThrottle"
    booster_throttle_program: list = dataclasses.field(
        default_factory=lambda: [
            [0.0, 0.7],    # Start at 70% throttle
            [60.0, 0.9],   # Ramp to 90% at 60 seconds
            [120.0, 1.0],  # Full throttle at 120 seconds
            [1000.0, 1.0], # Hold full throttle
        ]
    )
    upper_stage_throttle_program: list = dataclasses.field(
        default_factory=lambda: [
            [0.0, 1.0],    # Full throttle from ignition
            [100.0, 1.0],  # Continue full throttle for 100s
            [101.0, 0.0],  # Cutoff
            [2500.0, 0.0], # Coast until 2500s (relative to upper stage ignition)
            [2501.0, 1.0], # Re-ignite for circularization
            [2600.0, 1.0], # Burn for 100s
            [2601.0, 0.0], # Final cutoff
        ]
    )
    # Tolerances for the original TwoPhaseUpperThrottle function
    upper_throttle_vr_tolerance: float = 2.0
    upper_throttle_alt_tolerance: float = 1000.0
    base_throttle_cmd: float = 1.0  # default throttle for simple schedule



    # Termination logic
    impact_altitude_buffer_m: float = -100.0  # stop after sinking below this altitude
    escape_radius_factor: float = 1.05        # end sim if r exceeds this * planet radius with positive energy

    # Output/logging
    log_filename: str = "simulation_log.txt"
    plot_trajectory: bool = True
    animate_trajectory: bool = False




CFG = Config()


if __name__ == "__main__":
    # Running this file directly executes the main simulation with current CFG values.
    from main import main as run_main

    run_main()
