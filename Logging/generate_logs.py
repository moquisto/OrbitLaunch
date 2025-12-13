"""
Functions for generating and saving simulation logs.
"""
import numpy as np
from Main.telemetry import Logger
import csv
import os
from Analysis.config import OptimizationParams # Import OptimizationParams

LOG_FILENAME = "optimization_twostage_log.csv" # Define here for consistency

def save_log_to_txt(log: Logger, filename: str):
    """Write simulation states to a text (CSV-style) file for analysis."""
    with open(filename, "w") as f:
        f.write(
            "# t_sim_s,t_env_s,alt_m,speed_mps,mass_kg,stage,thrust_N,drag_N,mdot_kgps,q_Pa,rho_kgpm3,mach,fpa_deg,v_vertical_mps,v_horizontal_mps,specific_energy_Jpkg,pos_x_m,pos_y_m,pos_z_m,vel_x_mps,vel_y_mps,vel_z_mps\n"
        )
        n = len(log.t_sim)
        for i in range(n):
            r = log.r[i]
            v = log.v[i]
            f.write(
                f"{log.t_sim[i]:.3f},{log.t_env[i]:.3f},{log.altitude[i]:.3f},{log.speed[i]:.3f},{log.m[i]:.3f},{log.stage[i]},{log.thrust_mag[i]:.3f},{log.drag_mag[i]:.3f},{log.mdot[i]:.6f},{log.dynamic_pressure[i]:.3f},{log.rho[i]:.6e},{log.mach[i]:.3f},{log.flight_path_angle_deg[i]:.3f},{log.v_vertical[i]:.3f},{log.v_horizontal[i]:.3f},{log.specific_energy[i]:.3f},{r[0]:.3f},{r[1]:.3f},{r[2]:.3f},{v[0]:.3f},{v[1]:.3f},{v[2]:.3f}\n"
            )
    print(f"Saved simulation log to {filename}")

def log_iteration(phase: str, iteration: int, params: OptimizationParams, results: dict):
    """Helper to log a single optimizer iteration with de-scaled physics values."""
    if not isinstance(params, OptimizationParams):
        params = OptimizationParams(*params)
    orbit_error = results.get("orbital_error", results.get("orbit_error", results.get("error", 0.0)))
    with open(LOG_FILENAME, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            phase,
            iteration,
            f"{params.meco_mach:.4f}",
            f"{params.booster_pitch_time_0:.1f}", f"{params.booster_pitch_angle_0:.1f}",
            f"{params.booster_pitch_time_1:.1f}", f"{params.booster_pitch_angle_1:.1f}",
            f"{params.booster_pitch_time_2:.1f}", f"{params.booster_pitch_angle_2:.1f}",
            f"{params.booster_pitch_time_3:.1f}", f"{params.booster_pitch_angle_3:.1f}",
            f"{params.booster_pitch_time_4:.1f}", f"{params.booster_pitch_angle_4:.1f}",
            f"{params.coast_s:.1f}",
            f"{params.upper_burn_s:.1f}",
            f"{params.upper_ignition_delay_s:.1f}",
            f"{params.azimuth_deg:.1f}",
            f"{params.upper_pitch_time_0:.1f}", f"{params.upper_pitch_angle_0:.1f}",
            f"{params.upper_pitch_time_1:.1f}", f"{params.upper_pitch_angle_1:.1f}",
            f"{params.upper_pitch_time_2:.1f}", f"{params.upper_pitch_angle_2:.1f}",
            f"{params.upper_throttle_level_0:.2f}", f"{params.upper_throttle_level_1:.2f}", f"{params.upper_throttle_level_2:.2f}", f"{params.upper_throttle_level_3:.2f}",
            f"{params.upper_throttle_switch_ratio_0:.2f}", f"{params.upper_throttle_switch_ratio_1:.2f}", f"{params.upper_throttle_switch_ratio_2:.2f}",
            f"{params.booster_throttle_level_0:.2f}", f"{params.booster_throttle_level_1:.2f}", f"{params.booster_throttle_level_2:.2f}", f"{params.booster_throttle_level_3:.2f}",
            f"{params.booster_throttle_switch_ratio_0:.2f}", f"{params.booster_throttle_switch_ratio_1:.2f}", f"{params.booster_throttle_switch_ratio_2:.2f}",
            f"{results.get('cost', 0.0):.2f}",
            f"{results.get('fuel', 0.0):.2f}",
            f"{orbit_error:.2f}",
            f"{results.get('perigee_error_m', 0.0):.2f}",
            f"{results.get('apoapsis_error_m', 0.0):.2f}",
            results.get('status', 'UNKNOWN')
        ])

def ensure_log_header():
    """Create the CSV log file with a header if it's missing or empty."""
    if not os.path.exists(LOG_FILENAME) or os.path.getsize(LOG_FILENAME) == 0:
        with open(LOG_FILENAME, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "phase", "iteration", "meco_mach",
                "pitch_time_0", "pitch_angle_0",
                "pitch_time_1", "pitch_angle_1",
                "pitch_time_2", "pitch_angle_2",
                "pitch_time_3", "pitch_angle_3",
                "pitch_time_4", "pitch_angle_4",
                "coast_s", "upper_burn_s", "upper_ignition_delay_s",
                "azimuth_deg",
                "upper_pitch_time_0", "upper_pitch_angle_0",
                "upper_pitch_time_1", "upper_pitch_angle_1",
                "upper_pitch_time_2", "upper_pitch_angle_2",
                "upper_throttle_level_0", "upper_throttle_level_1", "upper_throttle_level_2", "upper_throttle_level_3",
                "upper_throttle_switch_ratio_0", "upper_throttle_switch_ratio_1", "upper_throttle_switch_ratio_2",
                "booster_throttle_level_0", "booster_throttle_level_1", "booster_throttle_level_2", "booster_throttle_level_3",
                "booster_throttle_switch_ratio_0", "booster_throttle_switch_ratio_1", "booster_throttle_switch_ratio_2",
                "cost", "fuel", "orbit_error", "perigee_error_m", "apoapsis_error_m", "status",
            ])
