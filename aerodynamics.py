"""
Aerodynamic helpers: drag model and force computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np


class CdModel:
    """
    Drag coefficient model. Accepts either a constant value or a callable
    returning Cd as a function of Mach number.
    """

    def __init__(self, value_or_callable: Union[float, Callable[[float], float]] = 2.0):
        self.value_or_callable = value_or_callable

    def cd(self, mach: float) -> float:
        if callable(self.value_or_callable):
            return max(0.0, float(self.value_or_callable(mach)))
        return max(0.0, float(self.value_or_callable))


@dataclass
class Aerodynamics:
    atmosphere: any
    cd_model: CdModel

    def drag_force(self, state, rocket, earth, t: float) -> np.ndarray:
        """
        Compute aerodynamic drag force (opposite to air-relative velocity).
        Lift and aero moments are intentionally neglected.
        """
        r = state.r_eci
        v = state.v_eci

        v_atm = earth.atmosphere_velocity(r)
        v_rel = v - v_atm
        speed_rel = np.linalg.norm(v_rel)
        if speed_rel == 0.0:
            return np.zeros(3)

        altitude = np.linalg.norm(r) - earth.radius
        props = self.atmosphere.properties(altitude, t)
        rho = props.get("rho", 0.0)

        # TODO: compute Mach number once speed of sound is modeled
        mach = 0.0
        cd = self.cd_model.cd(mach)
        area = rocket.reference_area(state)

        q = 0.5 * rho * speed_rel**2
        force_mag = q * cd * area
        return -force_mag * (v_rel / speed_rel)
