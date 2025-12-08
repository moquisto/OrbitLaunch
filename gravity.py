"""
Earth model structure: central gravity and co-rotation velocity interfaces.
Implementations are left to be filled in later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EarthModel:
    mu: float
    radius: float
    omega_vec: np.ndarray

    def gravity_accel(self, r_eci: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Implement central gravity acceleration")

    def atmosphere_velocity(self, r_eci: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Implement co-rotating atmosphere velocity")


# Nominal constants for convenience (can be used by a real implementation)
MU_EARTH = 3.986_004_418e14  # m^3/s^2
R_EARTH = 6_371_000.0  # m
OMEGA_EARTH = np.array([0.0, 0.0, 7.292_115_9e-5])  # rad/s
