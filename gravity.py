"""
Earth gravity and rotation helpers.
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
        """Return central gravity acceleration in ECI."""
        r = np.linalg.norm(r_eci)
        if r == 0.0:
            return np.zeros(3)
        return -self.mu * r_eci / r**3

    def atmosphere_velocity(self, r_eci: np.ndarray) -> np.ndarray:
        """
        Return the local velocity of the co-rotating atmosphere at position r_eci.
        """
        return np.cross(self.omega_vec, r_eci)


MU_EARTH = 3.986_004_418e14  # m^3/s^2
R_EARTH = 6_371_000.0  # m
OMEGA_EARTH = np.array([0.0, 0.0, 7.292_115_9e-5])  # rad/s

DEFAULT_EARTH = EarthModel(mu=MU_EARTH, radius=R_EARTH, omega_vec=OMEGA_EARTH)
