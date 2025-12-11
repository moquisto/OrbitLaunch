"""
Configuration for the main simulation loop and its termination conditions.
"""

import dataclasses
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class SimulationConfig:
    # --- Target Orbit ---
    target_orbit_alt_m: float = 420_000.0       # Standard LEO / Parking Orbit

    # --- Simulation Config ---
    main_duration_s: float = 10000
    main_dt_s: float = 0.05  # 20Hz is a safer default for high-thrust/high-drag dynamics.
    integrator: str = "rk4"

    # --- Constraints ---
    max_q_limit: float | None = 60_000.0  # [Pa] Standard Max Q is ~30-40 kPa. 60kPa is a safe structural limit.
    max_accel_limit: float | None = 40.0  # ~4 Gs (Human/Cargo comfort limit)

    # --- Tolerances ---
    orbit_speed_tol: float = 20.0
    orbit_radial_tol: float = 20.0
    orbit_alt_tol: float = 500.0
    orbit_ecc_tol: float = 0.01
    exit_on_orbit: bool = False
    post_orbit_coast_s: float = 0.0
