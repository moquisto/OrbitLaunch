"""
Configuration for the environment models (Earth, Atmosphere, Aerodynamics).
"""

import dataclasses
from dataclasses import dataclass
from typing import List, Tuple, Optional, TYPE_CHECKING

import numpy as np

from Environment.gravity import EarthModel
from Environment.atmosphere import AtmosphereModel

# Use TYPE_CHECKING to avoid circular imports for type hints
if TYPE_CHECKING:
    from Environment.aerodynamics import Aerodynamics


@dataclass
class EnvironmentConfig:
    # --- Launch Site ---
    # Boca Chica, Texas (approximate)
    launch_lat_deg: float = 26.0
    launch_lon_deg: float = -97.0

    # --- Central Body (Earth) ---
    earth_mu: float = 3.986_004_418e14          # [m^3/s^2]
    earth_radius_m: float = 6_371_000.0         # [m]
    earth_omega_vec: tuple[float, float, float] = (0.0, 0.0, 7.292_115_9e-5)  # [rad/s]
    use_j2: bool = True
    j2_coeff: float = 1.08262668e-3

    # --- Atmosphere / Environment ---
    # Extended to 150km to prevent "physics shock" when drag instantly disappears
    atmosphere_switch_alt_m: float = 150_000.0 
    atmosphere_f107: float | None = None
    atmosphere_f107a: float | None = None
    atmosphere_ap: float | None = None
    use_jet_stream_model: bool = True

    # Physics Constants
    G0: float = 9.80665
    P_SL: float = 101325.0
    air_gamma: float = 1.4
    air_gas_constant: float = 287.05

    # Wind (Smoothed Shear)
    wind_direction_vec: tuple[float, float, float] = (1.0, 0.0, 0.0)
    wind_alt_points: list = dataclasses.field(default_factory=lambda: [8_000.0, 10_000.0, 14_000.0])
    wind_speed_points: list = dataclasses.field(default_factory=lambda: [0.0, 45.0, 0.0])
    
    # Drag Map (Transonic Rise)
    mach_cd_map: list = dataclasses.field(
        default_factory=lambda: [
            [0.5, 0.245],
            [0.8, 0.281],
            [1.0, 0.508],
            [1.2, 0.733],
            [1.5, 0.645],
            [2.0, 0.568],
            [3.0, 0.513],
            [4.0, 0.495],
            [5.0, 0.478],
            [7.5, 0.428],
            [10.0, 0.341]
        ]
    )

    def create_earth_model(self) -> EarthModel:
        return EarthModel(
            mu=self.earth_mu,
            radius=self.earth_radius_m,
            omega_vec=np.array(self.earth_omega_vec, dtype=float),
            j2=self.j2_coeff if self.use_j2 else None,
        )

    def create_atmosphere_model(self) -> AtmosphereModel:
        return AtmosphereModel(self)

    def create_aerodynamics_model(self, atmosphere: AtmosphereModel, reference_area: Optional[float] = None) -> "Aerodynamics":
        # Local import for construction
        from Environment.aerodynamics import Aerodynamics, CdModel, mach_dependent_cd
        cd_model = CdModel(mach_dependent_cd, self)
        return Aerodynamics(
            atmosphere=atmosphere,
            cd_model=cd_model,
            env_config=self,
            reference_area=reference_area
        )
