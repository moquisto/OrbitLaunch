"""
State container for translational dynamics.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class State:
    r_eci: np.ndarray
    v_eci: np.ndarray
    m: float
    stage_index: int = 0
    upper_ignition_start_time: float | None = None

    def copy(self) -> "State":
        """
        Return a deep-ish copy of the state suitable for use in integrators.

        r_eci and v_eci are copied as new numpy arrays; m and stage_index are
        simple scalars.
        """
        return State(
            r_eci=self.r_eci.copy(),
            v_eci=self.v_eci.copy(),
            m=self.m,
            stage_index=self.stage_index,
            upper_ignition_start_time=self.upper_ignition_start_time,
        )
