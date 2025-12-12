"""
Configuration for vehicle hardware (engines, stages, rocket dimensions).
"""

import dataclasses
from dataclasses import dataclass
from typing import List, Tuple, Optional

from Hardware.engine import Engine
from Hardware.stage import Stage
from Environment.config import EnvironmentConfig


@dataclass
class HardwareConfig:
    # --- Vehicle Specifications (Starship Block 2 Estimates) ---
    
    # BOOSTER (Super Heavy)
    # Thrust: 33 Raptor 2s @ ~230tf = ~75-76 MN total
    booster_thrust_vac: float = 7.6e7
    booster_thrust_sl: float = 7.6e7
    
    # ISP: Booster uses sea-level optimized nozzles.
    # Real values: ~327s SL, ~350s Vacuum.
    booster_isp_vac: float = 350.0  
    booster_isp_sl: float = 330.0
    
    # Mass: 
    # Propellant: ~3400t (Consistent with public diagrams)
    # Dry Mass: ~250t (200t is the aspirational goal; 275t is current prototype. 250t is a fair balance).
    booster_dry_mass: float = 2.5e5     
    booster_prop_mass: float = 3.4e6    

    # UPPER STAGE (Starship)
    # Thrust: 3 Vac (RVac) + 3 SL Raptors. ~1500tf total.
    upper_thrust_vac: float = 1.5e7   
    upper_thrust_sl: float = 7.5e6 # Efficiency drops at SL
    
    # ISP: RVac engines pull the average up significantly in vacuum.
    upper_isp_vac: float = 380.0    
    upper_isp_sl: float = 330.0
    
    # Mass: ~120t dry + payload (Starlink v2 / HLS).
    upper_dry_mass: float = 1.2e5       
    upper_prop_mass: float = 1.2e6      

    # Aero / Dimensions
    ref_area_m2: float = 63.62  # A = pi * r^2 = pi * (4.5)^2 ≈ 63.62 m^2
    engine_min_throttle: float = 0.4  # Raptor deep throttle limit
    throttle_full_shape_threshold: float = 0.99  # shape value considered "full" for min throttle enforcement

    def create_booster_engine(self, env_config: EnvironmentConfig) -> Engine:
        return Engine(
            thrust_vac=self.booster_thrust_vac,
            thrust_sl=self.booster_thrust_sl,
            isp_vac=self.booster_isp_vac,
            isp_sl=self.booster_isp_sl,
            p_sl=env_config.P_SL,
        )

    def create_upper_engine(self, env_config: EnvironmentConfig) -> Engine:
        return Engine(
            thrust_vac=self.upper_thrust_vac,
            thrust_sl=self.upper_thrust_sl,
            isp_vac=self.upper_isp_vac,
            isp_sl=self.upper_isp_sl,
            p_sl=env_config.P_SL,
        )

    def create_booster_stage(self, engine: Engine) -> Stage:
        return Stage(
            dry_mass=self.booster_dry_mass,
            prop_mass=self.booster_prop_mass,
            engine=engine,
            ref_area=self.ref_area_m2,
        )

    def create_upper_stage(self, engine: Engine) -> Stage:
        return Stage(
            dry_mass=self.upper_dry_mass,
            prop_mass=self.upper_prop_mass,
            engine=engine,
            ref_area=self.ref_area_m2,
        )
