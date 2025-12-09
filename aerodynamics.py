"""
Aerodynamics module structure: drag-only interface.
Implementations are placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union, Any

import numpy as np
from atmosphere import AtmosphereModel


class CdModel:
    """
    Drag coefficient model. Accepts either a constant value or a callable
    returning Cd as a function of Mach number.
    """

    def __init__(self, value_or_callable: Union[float, Callable[[float], float]] = 2.0):
        self.value_or_callable = value_or_callable

    def cd(self, mach: float) -> float:
        """Return drag coefficient for a given Mach number.

        If value_or_callable is a constant, that value is returned.
        If it is a callable, it is evaluated as Cd(Mach).
        """
        if callable(self.value_or_callable):
            return float(self.value_or_callable(mach))
        return float(self.value_or_callable)


@dataclass
class Aerodynamics:
    atmosphere: AtmosphereModel
    cd_model: CdModel
    reference_area: float  # [m^2], reference/frontal area of the rocket

    def drag_force(self, state: Any, earth: Any, t: float) -> np.ndarray:
        """Compute aerodynamic drag force in the ECI frame.

        Assumptions
        -----------
        - The rocket's longitudinal axis is aligned with the air-relative
          velocity (zero angle of attack).
        - Drag acts purely opposite to the air-relative velocity vector.
        - The reference area is constant and represents the effective
          frontal area normal to the flow.
        - `state` provides `r_eci` and `v_eci` attributes (position and
          velocity in ECI, both as 3-vectors in meters / m/s).
        - `earth` provides `radius` and `atmosphere_velocity(r_eci)`.
        """
        # Extract position and velocity in ECI.
        r = np.asarray(state.r_eci, dtype=float)
        v = np.asarray(state.v_eci, dtype=float)

        r_norm = np.linalg.norm(r)
        if r_norm == 0.0:
            return np.zeros(3)

        # Altitude above mean surface.
        altitude = r_norm - float(earth.radius)
        if altitude < 0.0:
            altitude = 0.0

        # Atmospheric properties at this altitude and time.
        props = self.atmosphere.properties(altitude, t)
        rho = float(props.rho)
        T = float(props.T)

        if rho <= 0.0:
            return np.zeros(3)

        # Air-relative velocity: rocket velocity minus co-rotating atmosphere.
        v_atm = np.asarray(earth.atmosphere_velocity(r), dtype=float)
        v_rel = v - v_atm
        v_rel_mag = np.linalg.norm(v_rel)
        if v_rel_mag == 0.0:
            return np.zeros(3)

        # Speed of sound (ideal gas, dry air) and Mach number.
        gamma = 1.4
        R_air = 287.05  # J/(kg*K)
        a = np.sqrt(max(gamma * R_air * T, 0.0))
        mach = v_rel_mag / a if a > 0.0 else 0.0

        # Drag coefficient from model.
        cd = self.cd_model.cd(mach)

        # Drag magnitude: 0.5 * rho * |v_rel|^2 * Cd * A.
        A = float(self.reference_area)
        q = 0.5 * rho * v_rel_mag ** 2
        F_mag = q * cd * A

        # Direction opposite to air-relative velocity.
        F_vec = -F_mag * v_rel / v_rel_mag
        return F_vec

