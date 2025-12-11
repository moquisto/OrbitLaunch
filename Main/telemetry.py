"""
Minimal in-memory logger for trajectories.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any

import numpy as np

# from Main.state import State # Will be added later


class Logger:
    """
    Minimal in-memory logger for trajectories.
    """

    def __init__(self):
        self.t_sim = []
        self.t_env = []
        self.r = []
        self.v = []
        self.m = []
        self.stage = []
        self.altitude = []
        self.speed = []
        self.thrust_mag = []
        self.drag_mag = []
        self.mdot = []
        self.dynamic_pressure = []
        self.rho = []
        self.mach = []
        self.flight_path_angle_deg = []
        self.v_vertical = []
        self.v_horizontal = []
        self.specific_energy = []
        self.orbit_achieved = False
        self.cutoff_reason = ""

    def record(self, t_sim: float, t_env: float, state: Any, extras: Dict[str, Any]): # state: State will be replaced with Any for now
        self.t_sim.append(float(t_sim))
        self.t_env.append(float(t_env))
        self.r.append(np.asarray(state.r_eci, dtype=float).copy())
        self.v.append(np.asarray(state.v_eci, dtype=float).copy())
        self.m.append(float(state.m))
        self.stage.append(int(getattr(state, "stage_index", 0)))
        self.altitude.append(float(extras.get("altitude", 0.0)))
        self.speed.append(float(extras.get("speed", 0.0)))
        self.thrust_mag.append(float(extras.get("thrust_mag", 0.0)))
        self.drag_mag.append(float(extras.get("drag_mag", 0.0)))
        self.mdot.append(float(extras.get("mdot", 0.0)))
        self.dynamic_pressure.append(float(extras.get("dynamic_pressure", 0.0)))
        self.rho.append(float(extras.get("rho", 0.0)))
        self.mach.append(float(extras.get("mach", 0.0)))
        self.flight_path_angle_deg.append(float(extras.get("fpa_deg", 0.0)))
        self.v_vertical.append(float(extras.get("v_vertical", 0.0)))
        self.v_horizontal.append(float(extras.get("v_horizontal", 0.0)))
        self.specific_energy.append(float(extras.get("specific_energy", 0.0)))
