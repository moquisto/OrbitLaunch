
"""
Vehicle model: outlines for engines, stages, and rocket behavior.
Implementations are placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Engine:
    thrust_vac: float
    thrust_sl: float
    isp_vac: float
    isp_sl: float

    def thrust_and_isp(self, throttle: float, p_amb: float) -> Tuple[float, float]:
        raise NotImplementedError("Implement thrust/Isp interpolation")


@dataclass
class Stage:
    dry_mass: float
    prop_mass: float
    engine: Engine
    ref_area: float

    def total_mass(self) -> float:
        raise NotImplementedError("Implement stage mass calculation")


class Rocket:
    def __init__(self, stages: List[Stage]):
        self.stages = stages

    def current_stage_index(self, state) -> int:
        raise NotImplementedError("Select active stage index")

    def current_stage(self, state) -> Stage:
        raise NotImplementedError("Return active stage")

    def reference_area(self, state) -> float:
        raise NotImplementedError("Return current reference area")

    def thrust_and_mass_flow(self, control, state, p_amb: float) -> Tuple[np.ndarray, float]:
        raise NotImplementedError("Compute thrust vector and mass flow")

    def stage_separation(self, state):
        raise NotImplementedError("Implement stage separation logic")
