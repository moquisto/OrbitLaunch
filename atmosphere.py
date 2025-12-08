
"""
Atmosphere module structure: outlines interfaces for US76, NRLMSIS2.1, and a
combined dispatcher. Implementations are intentionally left as placeholders.
"""

from __future__ import annotations

from typing import Dict


class USStandardAtmosphere1976:
    """Lower-atmosphere model placeholder."""

    def properties(self, altitude: float) -> Dict[str, float]:
        raise NotImplementedError("Implement US76 properties lookup")


class NRLMSIS21:
    """Upper-atmosphere model placeholder."""

    def properties(self, altitude: float, t: float | None = None) -> Dict[str, float]:
        raise NotImplementedError("Implement NRLMSIS 2.1 properties lookup")


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
