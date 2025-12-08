"""
Aerodynamics module structure: drag-only interface.
Implementations are placeholders.
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
        raise NotImplementedError("Implement Cd lookup")


@dataclass
class Aerodynamics:
    atmosphere: any
    cd_model: CdModel

    def drag_force(self, state, rocket, earth, t: float) -> np.ndarray:
        """
        Compute aerodynamic drag force (opposite to air-relative velocity).
        """
        raise NotImplementedError("Implement drag force computation")
