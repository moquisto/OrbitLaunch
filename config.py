"""
Central configuration for the launch simulation and optimizer.
Refined for realistic Starship Block 2/3 parameters (Post-IFT-6 data).
"""

import dataclasses
from dataclasses import dataclass
from atmosphere import AtmosphereModel


@dataclass
class Config:
    # --- Launch Site ---
    # Boca Chica, Texas (approximate)
    launch_lat_deg: float = 26.0
    launch_lon_deg: float = -97.0

    # --- Central Body (Earth) ---
    earth_mu: float = 3.986_004_418e14          # [m^3/s^2]
    earth_radius_m: float = 6_371_000.0         # [m]
    earth_omega_vec: tuple[float, float, float] = (0.0, 0.0, 7.292_115_9e-5)  # [rad/s]

    # --- Target Orbit ---
    target_orbit_alt_m: float = 420_000.0       # Standard LEO / Parking Orbit

    # --- Vehicle Specifications (Starship Block 2 Estimates) ---
    
    # BOOSTER (Super Heavy)
    # Thrust: 33 Raptor 2s @ ~230tf = ~75-76 MN total
    booster_thrust_vac: float = 7.6e7
    booster_thrust_sl: float = 7.6e7
    
    # ISP: Booster uses sea-level optimized nozzles.
    # Real values: ~327s SL, ~350s Vacuum.
    booster_isp_vac: float = 350.0  
    booster_isp_sl: float = 330.0
    
    # Mass: 
    # Propellant: ~3400t (Consistent with public diagrams)
    # Dry Mass: ~250t (200t is the aspirational goal; 275t is current prototype. 250t is a fair balance).
    booster_dry_mass: float = 2.5e5     
    booster_prop_mass: float = 3.4e6    

    # UPPER STAGE (Starship)
    # Thrust: 3 Vac (RVac) + 3 SL Raptors. ~1500tf total.
    upper_thrust_vac: float = 1.5e7   
    upper_thrust_sl: float = 7.5e6 # Efficiency drops at SL
    
    # ISP: RVac engines pull the average up significantly in vacuum.
    upper_isp_vac: float = 380.0    
    upper_isp_sl: float = 330.0
    
    # Mass: ~120t dry + payload (Starlink v2 / HLS).
    upper_dry_mass: float = 1.2e5       
    upper_prop_mass: float = 1.2e6      

    # Aero / Dimensions
    ref_area_m2: float = 63.62  # A = pi * r^2 = pi * (4.5)^2 ≈ 63.62 m^2
    engine_min_throttle: float = 0.4  # Raptor deep throttle limit
    throttle_full_shape_threshold: float = 0.99  # shape value considered "full" for min throttle enforcement
    use_j2: bool = True
    j2_coeff: float = 1.08262668e-3

    # --- Staging & Timing ---
    main_engine_ramp_time: float = 1.0
    upper_engine_ramp_time: float = 1.0
    separation_delay_s: float = 1.0     
    upper_ignition_delay_s: float = 0.5 # Hot staging (ignite before separation)
    engine_shutdown_ramp_s: float = 1.0  
    
    # Staging Velocity trigger
    # Starship stages earlier than F9. ~1.6 km/s (Mach 5.5).
    meco_mach: float = 5.5
    separation_altitude_m: float | None = None

    # --- Simulation Config ---
    main_duration_s: float = 10000
    main_dt_s: float = 0.05  # 20Hz is a safer default for high-thrust/high-drag dynamics.
    integrator: str = "rk4"

    # --- Constraints ---
    max_q_limit: float | None = 60_000.0  # [Pa] Standard Max Q is ~30-40 kPa. 60kPa is a safe structural limit.
    max_accel_limit: float | None = 40.0  # ~4 Gs (Human/Cargo comfort limit)

    # --- Guidance Programs ---

    # Pitch Program (Gravity Turn) — time-based, separate booster/upper schedules
    pitch_guidance_mode: str = "parameterized"
    pitch_guidance_function: str = "custom_guidance.simple_pitch_program"
    # Booster pitch schedule (time from liftoff, deg from horizontal)
    # Realistic duration is ~145s, so points beyond that are not useful.
    pitch_program: list = dataclasses.field(
        default_factory=lambda: [
            [0.0, 89.8],    # Vertical clear of tower
            [10.0, 86.0],   # Initiate kick early
            [40.0, 72.0],   # Aggressive turn through lower atmosphere
            [80.0, 55.0],   # Punch through max Q
            [120.0, 40.0],  # Stratosphere transition
            [140.0, 25.0],  # Final moments before MECO
        ]
    )
    # Upper-stage pitch schedule (time from upper ignition, deg from horizontal)
    upper_pitch_program: list = dataclasses.field(
        default_factory=lambda: [
            [0.0, 10.0],   # Gentle initial pitch up to preserve altitude
            [60.0, 0.0],   # Transition to prograde during upper burn
            [180.0, 0.0],  # Hold prograde through main upper-stage burn
        ]
    )
    pitch_prograde_speed_threshold: float = 100.0

    # Throttle Program (Max-Q Bucket)
    throttle_guidance_mode: str = "parameterized"
    throttle_guidance_function_class: str = "custom_guidance.TwoPhaseUpperThrottle"
    
    # Booster: Liftoff 100%, Dip for Max Q, Back to 100%
    # Realistic duration is ~145s.
    booster_throttle_program: list = dataclasses.field(
        default_factory=lambda: [
            [0.0, 1.0],    # ALL ENGINES GO
            [50.0, 1.0],   # Approaching Max Q speed
            [60.0, 0.65],  # THROTTLE BUCKET: Reduce to 65% to save structure
            [75.0, 0.65],  # Hold bucket
            [85.0, 1.0],   # Max Q passed, power up
            [150.0, 1.0],  # Burn to depletion/MECO
        ]
    )
    
    # Upper Stage: Simple burn
    upper_stage_throttle_program: list = dataclasses.field(
        default_factory=lambda: [
            [0.0, 1.0],    # Full power
            [350.0, 1.0],  # Long burn to orbit
            [351.0, 0.0],  # SECO 1 (Main orbital insertion)
            [2500.0, 0.0], # Coast phase
            [2501.0, 1.0], # Circularization burn (if needed)
            [2520.0, 1.0], 
            [2521.0, 0.0], # Done
        ]
    )
    
    upper_throttle_vr_tolerance: float = 2.0
    upper_throttle_alt_tolerance: float = 1000.0
    base_throttle_cmd: float = 1.0

    # --- Atmosphere / Environment ---
    # Extended to 150km to prevent "physics shock" when drag instantly disappears
    atmosphere_switch_alt_m: float = 150_000.0 
    atmosphere_f107: float | None = None
    atmosphere_f107a: float | None = None
    atmosphere_ap: float | None = None
    use_jet_stream_model: bool = True

    # Physics Constants
    G0: float = 9.80665
    P_SL: float = 101325.0
    air_gamma: float = 1.4
    air_gas_constant: float = 287.05

    # Wind (Smoothed Shear)
    wind_direction_vec: tuple[float, float, float] = (1.0, 0.0, 0.0)
    wind_alt_points: list = dataclasses.field(default_factory=lambda: [8_000.0, 10_000.0, 14_000.0])
    wind_speed_points: list = dataclasses.field(default_factory=lambda: [0.0, 45.0, 0.0]) # 45 m/s is safer than 50
    
    # Drag Map (Transonic Rise)
    mach_cd_map: list = dataclasses.field(
        default_factory=lambda: [
            [0.0, 0.30],
            [0.8, 0.45], # Transonic drag spike
            [1.0, 0.60], # Max drag at Mach 1
            [1.2, 0.50],
            [2.0, 0.35],
            [5.0, 0.30], # Hypersonic
            [10.0, 0.25],
            [25.0, 0.20] # High hypersonic/Re-entry
        ]
    )

    # Tolerances
    orbit_speed_tol: float = 20.0
    orbit_radial_tol: float = 20.0
    orbit_alt_tol: float = 500.0
    orbit_ecc_tol: float = 0.01
    exit_on_orbit: bool = False
    post_orbit_coast_s: float = 0.0
    
    # Logging
    impact_altitude_buffer_m: float = -100.0
    escape_radius_factor: float = 1.05
    log_filename: str = "simulation_log.txt"
    plot_trajectory: bool = True
    animate_trajectory: bool = False

    # Optimizer (optional manual seed)
    optimizer_manual_seed: list | None = None

    def __post_init__(self):
        """Initialize atmosphere model after dataclass init."""
        self._atmosphere_model: AtmosphereModel = AtmosphereModel(
            h_switch=self.atmosphere_switch_alt_m,
            lat_deg=self.launch_lat_deg,
            lon_deg=self.launch_lon_deg,
            f107=self.atmosphere_f107,
            f107a=self.atmosphere_f107a,
            ap=self.atmosphere_ap,
        )

    def get_speed_of_sound(self, altitude: float, t: float | None = None) -> float:
        """Returns the local speed of sound from the atmosphere model."""
        return self._atmosphere_model.get_speed_of_sound(altitude, t)

CFG = Config()

if __name__ == "__main__":
    from main import main as run_main
    run_main()
