
"""
Atmosphere models used by the simulation. These are lightweight placeholders
that match the intended structure (US Standard Atmosphere 1976 + NRLMSIS 2.1)
without depending on external datasets or network access.
"""

from __future__ import annotations

import math
from typing import Dict


class USStandardAtmosphere1976:
    """
    Simplified lower-atmosphere model.

    Uses an exponential density profile with a basic temperature lapse rate to
    keep the interface stable until a higher fidelity model is plugged in.
    """

    def __init__(
        self,
        rho0: float = 1.225,
        p0: float = 101_325.0,
        T0: float = 288.15,
        scale_height: float = 8_500.0,
        lapse_rate: float = 0.0065,
    ):
        self.rho0 = rho0
        self.p0 = p0
        self.T0 = T0
        self.scale_height = scale_height
        self.lapse_rate = lapse_rate

    def properties(self, altitude: float) -> Dict[str, float]:
        h = max(0.0, float(altitude))
        rho = self.rho0 * math.exp(-h / self.scale_height)
        p = self.p0 * math.exp(-h / self.scale_height)
        T = max(150.0, self.T0 - self.lapse_rate * h)
        return {"rho": rho, "p": p, "T": T}


class NRLMSIS21:
    """
    Simplified upper-atmosphere model.

    This placeholder follows the same interface as an NRLMSIS 2.1 wrapper so it
    can later be swapped with a real implementation. Altitude is expected in
    meters and time t is passed through for future use.
    """

    def __init__(
        self,
        rho_ref: float = 1e-8,
        T_ref: float = 800.0,
        p_ref: float | None = None,
        scale_height: float = 20_000.0,
        h_ref: float = 80_000.0,
    ):
        self.rho_ref = rho_ref
        self.T_ref = T_ref
        self.p_ref = p_ref
        self.scale_height = scale_height
        self.h_ref = h_ref

    def properties(self, altitude: float, t: float | None = None) -> Dict[str, float]:
        h = max(self.h_ref, float(altitude))
        rho = self.rho_ref * math.exp(-(h - self.h_ref) / self.scale_height)
        T = max(500.0, self.T_ref + 0.002 * (h - self.h_ref))
        gas_const = 287.05287
        p = self.p_ref if self.p_ref is not None else rho * gas_const * T
        return {"rho": rho, "p": p, "T": T}


class CombinedAtmosphere:
    """
    Dispatcher between low-altitude and high-altitude models.
    """

    def __init__(self, low_model: USStandardAtmosphere1976, high_model: NRLMSIS21, h_switch: float = 80_000.0):
        self.low_model = low_model
        self.high_model = high_model
        self.h_switch = h_switch

    def properties(self, altitude: float, t: float | None = None) -> Dict[str, float]:
        if altitude < self.h_switch:
            return self.low_model.properties(altitude)
        return self.high_model.properties(altitude, t)
