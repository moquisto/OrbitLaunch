"""
Stage model: defines stage characteristics and integrates with engine.
"""

from dataclasses import dataclass

from Hardware.engine import Engine


@dataclass
class Stage:
    dry_mass: float
    prop_mass: float
    engine: Engine
    ref_area: float

    def total_mass(self) -> float:
        """
        Return total stage mass (dry + propellant).

        The time evolution of the vehicle's total mass is handled via the
        State.m variable in the integrator; this value is mainly useful
        for constructing the initial mass budget or for stage-drop deltas.
        """
        return self.dry_mass + self.prop_mass
