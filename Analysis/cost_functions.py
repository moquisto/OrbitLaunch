# Analysis/cost_functions.py

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Main.telemetry import Logger
    from Environment.gravity import EarthModel

def calculate_orbital_insertion_cost(
    log: Logger,
    earth_model: EarthModel,
    target_altitude_m: float,
    target_inclination_deg: float = 0.0,
    dry_mass_remaining: float = 0.0,
) -> float:
    """
    Calculates a cost associated with orbital insertion,
    aiming for a target altitude and minimizing propellant usage.

    A lower cost indicates a better result.

    Parameters
    ----------
    log : Logger
        The simulation log containing the trajectory data.
    earth_model : EarthModel
        The Earth model used in the simulation.
    target_altitude_m : float
        The desired target altitude [m] for the orbit.
    target_inclination_deg : float, optional
        The desired target inclination [deg] for the orbit. Defaults to 0.0 (equatorial).
    dry_mass_remaining : float, optional
        The dry mass remaining at the end of the mission. Higher values are better.

    Returns
    -------
    float
        The calculated cost.
    """
    if not log.t_sim:
        return float('inf') # Infinite cost if no simulation data

    final_r_norm = np.linalg.norm(log.r[-1])
    final_v_norm = np.linalg.norm(log.v[-1])
    final_altitude = final_r_norm - earth_model.radius

    # Cost for altitude deviation
    altitude_deviation_cost = abs(final_altitude - target_altitude_m)

    # Cost for not achieving orbit or having a very elliptical orbit
    # A simple proxy: if specific energy is very negative, it's a sub-orbital trajectory.
    # If rp or ra are significantly different from target, it's not circular.
    from Environment.gravity import orbital_elements_from_state
    a, rp, ra = orbital_elements_from_state(log.r[-1], log.v[-1], earth_model.mu)
    
    perigee_altitude = (rp - earth_model.radius) if rp is not None else -float('inf')
    apoapsis_altitude = (ra - earth_model.radius) if ra is not None else -float('inf')

    # Penalize for crashing or not reaching minimum altitude for orbit
    if perigee_altitude < 0: # Crashed or sub-orbital
        perigee_penalty = abs(perigee_altitude) * 10 # Heavier penalty for impact
    else:
        perigee_penalty = abs(perigee_altitude - target_altitude_m)

    apoapsis_penalty = abs(apoapsis_altitude - target_altitude_m)

    # Penalize for high eccentricity (non-circular orbit)
    eccentricity_penalty = abs(apoapsis_altitude - perigee_altitude)

    # Simple cost: combine altitude deviation, perigee/apoapsis penalties, and penalize low dry mass
    # The weights here are arbitrary and would need tuning for actual optimization.
    cost = (
        altitude_deviation_cost * 0.1
        + perigee_penalty * 0.5
        + apoapsis_penalty * 0.5
        + eccentricity_penalty * 0.2
    )

    # Reward for remaining dry mass (e.g., payload capacity)
    if dry_mass_remaining > 0:
        cost -= dry_mass_remaining * 0.01 # Subtract cost for dry mass, effectively a reward

    # Massive penalty if orbit target not met and no orbit_achieved flag set
    if not log.orbit_achieved and (perigee_altitude < target_altitude_m * 0.5 or final_altitude < 0):
        cost += 1000000 # Large penalty for mission failure

    return cost

def calculate_reusability_cost(
    log: Logger,
    dry_mass_remaining_booster: float = 0.0,
    booster_landing_error_m: float = None, # Future: error for booster landing site
) -> float:
    """
    Calculates a cost related to booster reusability.
    A lower cost indicates a better result.
    """
    cost = 0.0
    # Reward for dry mass remaining in booster (indicates fuel for landing)
    cost -= dry_mass_remaining_booster * 0.1

    # Penalize large landing error for booster (future)
    if booster_landing_error_m is not None:
        cost += booster_landing_error_m * 0.01

    return cost
