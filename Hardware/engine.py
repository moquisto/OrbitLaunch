"""
Engine model: defines engine characteristics and thrust calculation.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Engine:
    thrust_vac: float
    thrust_sl: float
    isp_vac: float
    isp_sl: float
    p_sl: float

    def thrust_and_isp(self, throttle: float, p_amb: float) -> Tuple[float, float]:
        """
        Compute engine thrust and Isp for a given throttle and ambient pressure.

        Parameters
        ----------
        throttle : float
            Commanded throttle in [0, 1].
        p_amb : float
            Ambient static pressure [Pa]. Typically provided by the atmosphere
            model using the current altitude.

        Returns
        -------
        thrust : float
            Thrust magnitude [N] at the requested operating point.
        isp : float
            Specific impulse [s] at the requested operating point.

        Notes
        -----
        We interpolate linearly between sea-level (p = P_SL) and vacuum
        (p = 0) performance based on p_amb, then scale by throttle.
        """
        throttle = float(np.clip(throttle, 0.0, 1.0))

        # Clamp ambient pressure to [0, P_SL] for interpolation
        p = float(np.clip(p_amb, 0.0, self.p_sl))
        # Fraction of "vacuum-ness": 0 at sea level, 1 in vacuum
        f_vac = 1.0 - p / self.p_sl

        thrust_nominal = self.thrust_sl + f_vac * (self.thrust_vac - self.thrust_sl)
        isp_nominal = self.isp_sl + f_vac * (self.isp_vac - self.isp_sl)

        thrust = throttle * thrust_nominal
        return thrust, isp_nominal
