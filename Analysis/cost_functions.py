# Analysis/cost_functions.py

import numpy as np
from typing import TYPE_CHECKING
from Main.telemetry import Logger
from Environment.gravity import EarthModel, orbital_elements_from_state
from Environment.config import EnvironmentConfig
from Main.config import SimulationConfig

# Constants for cost calculation (can be moved to Analysis/config.py later if needed)
PENALTY_CRASH = 1e9        # "Soft Wall" for failed orbits
PERIGEE_FLOOR_M = 120_000.0  # Minimum acceptable perigee altitude
ECC_TOLERANCE = 0.01         # Maximum eccentricity tolerated before penalty
TARGET_TOLERANCE_M = 10000.0

def calculate_orbital_insertion_cost(
    log: Logger,
    earth_model: EarthModel,
    target_altitude_m: float,
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

    # Cost for not achieving orbit or having a very elliptical orbit
    # A simple proxy: if specific energy is very negative, it's a sub-orbital trajectory.
    # If rp or ra are significantly different from target, it's not circular.
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

def evaluate_simulation_results(
    log: Logger,
    initial_mass: float,
    cfg_env: EnvironmentConfig,
    sim_config: SimulationConfig,
    max_altitude: float
) -> dict:
    """
    Evaluates the results of a simulation run to calculate error, fuel used, and status.

    Parameters
    ----------
    log : Logger
        The simulation log containing the trajectory data.
    initial_mass : float
        The initial mass of the rocket at the start of the simulation.
    cfg_env : EnvironmentConfig
        The environment configuration used in the simulation.
    sim_config : SimulationConfig
        The simulation configuration, including target orbit altitude.
    max_altitude : float
        The maximum altitude reached during the simulation.

    Returns
    -------
    dict
        A dictionary containing:
        - "fuel": Fuel consumed in kg.
        - "error": Orbital error in meters.
        - "status": A string indicating the outcome of the simulation ("CRASH", "SUBORBIT", "PERFECT", "GOOD", "OK", "SIM_FAIL").
        - "perigee_alt_m": Perigee altitude in meters.
        - "eccentricity": Eccentricity of the final orbit.
    """
    results = {"fuel": 0.0, "error": PENALTY_CRASH, "status": "INIT"}

    if not log.t_sim or len(log.t_sim) == 0:
        results["status"] = "SIM_FAIL_NO_DATA"
        results["error"] = PENALTY_CRASH
        return results

    results["fuel"] = initial_mass - log.m[-1]

    r, v = log.r[-1], log.v[-1]
    a, rp, ra = orbital_elements_from_state(r, v, cfg_env.earth_mu)

    if rp is None or ra is None or rp < (cfg_env.earth_radius_m - 5000): # -5000 to account for surface variation
        results["status"] = "CRASH"
        results["error"] = PENALTY_CRASH + (sim_config.target_orbit_alt_m - max_altitude)
        return results

    target_r = cfg_env.earth_radius_m + sim_config.target_orbit_alt_m
    results["error"] = abs(rp - target_r) + abs(ra - target_r)

    perigee_alt = rp - cfg_env.earth_radius_m
    ecc = abs((ra - rp) / (ra + rp)) if (ra + rp) != 0 else 0
    results["perigee_alt_m"] = perigee_alt
    results["eccentricity"] = ecc

    perigee_penalty = max(0.0, PERIGEE_FLOOR_M - perigee_alt) * 100.0
    ecc_penalty = max(0.0, (ecc or 0.0) - ECC_TOLERANCE) * target_r
    results["error"] += perigee_penalty + ecc_penalty

    if perigee_alt < PERIGEE_FLOOR_M:
        results["status"] = "SUBORBIT"
    elif results["error"] < TARGET_TOLERANCE_M * 0.5: # Half of target tolerance for "PERFECT"
        results["status"] = "PERFECT"
    elif results["error"] < TARGET_TOLERANCE_M * 2: # Twice of target tolerance for "GOOD"
        results["status"] = "GOOD"
    else:
        results["status"] = "OK"
    
    return results
