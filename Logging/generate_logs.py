"""
Functions for generating and saving simulation logs.
"""
from __future__ import annotations

import numpy as np
from Main.telemetry import Logger
import csv
import datetime as dt
import json
import os
from pathlib import Path
from Analysis.config import OptimizationParams # Import OptimizationParams

LOG_FILENAME = "optimization_twostage_log.csv" # Define here for consistency

def _format_points(points: list[list[float]], *, t_digits: int = 3, v_digits: int = 3) -> str:
    formatted = [[round(float(t), t_digits), round(float(v), v_digits)] for t, v in points]
    return json.dumps(formatted, separators=(",", ":"))


def _desired_header() -> list[str]:
    return [
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
        "cost", "fuel", "orbit_error", "perigee_error_m", "apoapsis_error_m",
        "max_altitude_m", "cutoff_reason", "perigee_alt_m", "apoapsis_alt_m", "eccentricity",
        "booster_pitch_program_sorted", "upper_pitch_program_sorted",
        "booster_throttle_schedule", "upper_throttle_schedule",
        "status",
    ]


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

    # Provide human-readable, effective schedules (sorted/expanded) to reduce confusion.
    booster_pitch_sorted = sorted(
        [
            [params.booster_pitch_time_0, params.booster_pitch_angle_0],
            [params.booster_pitch_time_1, params.booster_pitch_angle_1],
            [params.booster_pitch_time_2, params.booster_pitch_angle_2],
            [params.booster_pitch_time_3, params.booster_pitch_angle_3],
            [params.booster_pitch_time_4, params.booster_pitch_angle_4],
        ],
        key=lambda p: float(p[0]),
    )
    upper_pitch_sorted = sorted(
        [
            [params.upper_pitch_time_0, params.upper_pitch_angle_0],
            [params.upper_pitch_time_1, params.upper_pitch_angle_1],
            [params.upper_pitch_time_2, params.upper_pitch_angle_2],
        ],
        key=lambda p: float(p[0]),
    )

    try:
        from Software.guidance import create_throttle_schedule_from_ratios

        upper_burn_duration = max(1.0, float(params.upper_burn_s))
        upper_throttle_levels = np.array(
            [
                params.upper_throttle_level_0,
                params.upper_throttle_level_1,
                params.upper_throttle_level_2,
                params.upper_throttle_level_3,
            ],
            dtype=float,
        )
        upper_switch_ratios = np.array(
            [
                params.upper_throttle_switch_ratio_0,
                params.upper_throttle_switch_ratio_1,
                params.upper_throttle_switch_ratio_2,
            ],
            dtype=float,
        )
        upper_throttle_schedule = create_throttle_schedule_from_ratios(
            burn_duration=upper_burn_duration,
            throttle_levels=upper_throttle_levels,
            switch_ratios=upper_switch_ratios,
            shutdown_after_burn=True,
        )

        booster_burn_duration_est = max(10.0, float(params.booster_pitch_time_4))
        booster_throttle_levels = np.array(
            [
                params.booster_throttle_level_0,
                params.booster_throttle_level_1,
                params.booster_throttle_level_2,
                params.booster_throttle_level_3,
            ],
            dtype=float,
        )
        booster_switch_ratios = np.array(
            [
                params.booster_throttle_switch_ratio_0,
                params.booster_throttle_switch_ratio_1,
                params.booster_throttle_switch_ratio_2,
            ],
            dtype=float,
        )
        booster_throttle_schedule = create_throttle_schedule_from_ratios(
            burn_duration=booster_burn_duration_est,
            throttle_levels=booster_throttle_levels,
            switch_ratios=booster_switch_ratios,
            shutdown_after_burn=False,
        )
    except Exception:
        # Keep logging robust even if schedule building fails for a pathological candidate.
        upper_throttle_schedule = []
        booster_throttle_schedule = []

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
            f"{results.get('max_altitude', 0.0):.2f}",
            str(results.get("cutoff_reason", "") or ""),
            f"{results.get('perigee_alt_m', 0.0):.2f}",
            f"{results.get('apoapsis_alt_m', 0.0):.2f}",
            f"{results.get('eccentricity', 0.0):.6f}",
            _format_points(booster_pitch_sorted, t_digits=1, v_digits=1),
            _format_points(upper_pitch_sorted, t_digits=1, v_digits=1),
            _format_points(booster_throttle_schedule, t_digits=3, v_digits=3),
            _format_points(upper_throttle_schedule, t_digits=3, v_digits=3),
            results.get('status', 'UNKNOWN')
        ])

def ensure_log_header():
    """Create the CSV log file with a header if it's missing or empty."""
    desired = _desired_header()
    log_path = Path(LOG_FILENAME)

    if not log_path.exists() or log_path.stat().st_size == 0:
        with log_path.open("w", newline="") as f:
            csv.writer(f).writerow(desired)
        return

    with log_path.open(newline="") as f:
        reader = csv.reader(f)
        existing = next(reader, None)

    if existing == desired:
        return

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = log_path.with_name(f"{log_path.stem}_legacy_{timestamp}{log_path.suffix}")
    os.replace(str(log_path), str(backup))
    with log_path.open("w", newline="") as f:
        csv.writer(f).writerow(desired)
    print(f"Log header changed; previous log moved to {backup}", flush=True)
