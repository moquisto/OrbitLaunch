"""
Functions for generating and saving simulation logs.
"""
import numpy as np
from Main.telemetry import Logger


def save_log_to_txt(log: Logger, filename: str):
    """Write simulation states to a text (CSV-style) file for analysis."""
    with open(filename, "w") as f:
        f.write(
            "# t_sim_s,t_env_s,alt_m,speed_mps,mass_kg,stage," "thrust_N,drag_N,mdot_kgps,q_Pa,rho_kgpm3,mach," "fpa_deg,v_vertical_mps,v_horizontal_mps,specific_energy_Jpkg," "pos_x_m,pos_y_m,pos_z_m,vel_x_mps,vel_y_mps,vel_z_mps\n"
        )
        n = len(log.t_sim)
        for i in range(n):
            r = log.r[i]
            v = log.v[i]
            f.write(
                f"{log.t_sim[i]:.3f},{log.t_env[i]:.3f}," "{log.altitude[i]:.3f},{log.speed[i]:.3f}," "{log.m[i]:.3f},{log.stage[i]}," "{log.thrust_mag[i]:.3f},{log.drag_mag[i]:.3f}," "{log.mdot[i]:.6f},{log.dynamic_pressure[i]:.3f}," "{log.rho[i]:.6e},{log.mach[i]:.3f}," "{log.flight_path_angle_deg[i]:.3f},{log.v_vertical[i]:.3f},{log.v_horizontal[i]:.3f},{log.specific_energy[i]:.3f}," "{r[0]:.3f},{r[1]:.3f},{r[2]:.3f}," "{v[0]:.3f},{v[1]:.3f},{v[2]:.3f}\n"
            )
    print(f"Saved simulation log to {filename}")
