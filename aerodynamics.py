"""
Aerodynamics module structure: drag-only interface.
Implementations are placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union, Any

import numpy as np
from atmosphere import AtmosphereModel


from atmosphere import AtmosphereModel
from gravity import EarthModel, MU_EARTH, R_EARTH, OMEGA_EARTH


import matplotlib.pyplot as plt

from integrators import State, RK4


def speed_of_sound(T: float) -> float:
    """
    Speed of sound for dry air:
        a = sqrt(gamma * R * T)
    """
    gamma = 1.4
    R = 287.05
    return np.sqrt(gamma * R * T)


class CdModel:
    """
    Drag coefficient model. Accepts either a constant value or a callable
    returning Cd as a function of Mach number.
    """

    def __init__(self, value_or_callable: Union[float, Callable[[float], float]] = 2.0):
        self.value_or_callable = value_or_callable

    def cd(self, mach: float) -> float:
        raise NotImplementedError("Implement Cd lookup")


@dataclass
class Aerodynamics:
    atmosphere: AtmosphereModel
    cd_model: CdModel
    reference_area: float  # [m^2], reference/frontal area of the rocket

    def drag_force(self, state, rocket, earth, t: float) -> np.ndarray:
        """
        Compute aerodynamic drag force (opposite to air-relative velocity).
        """
        raise NotImplementedError("Implement drag force computation")
