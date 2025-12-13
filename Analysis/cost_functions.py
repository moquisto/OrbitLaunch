# Analysis/cost_functions.py

from __future__ import annotations

import numpy as np
from Main.telemetry import Logger
from Environment.gravity import orbital_elements_from_state
from Environment.config import EnvironmentConfig
from Main.config import SimulationConfig

# Constants for cost calculation
PENALTY_CRASH = 1e9
PERIGEE_FLOOR_M = 120_000.0
ECC_TOLERANCE = 0.01
TARGET_TOLERANCE_M = 10_000.0

# When in phase 2, orbit accuracy must dominate fuel until within tolerance.
ORBIT_ERROR_WEIGHT = 200.0  # [kg per meter] in the penalty term


def _is_finite_number(x: object) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def _fuel_burned_from_log(log: Logger, fallback_mass_delta: float) -> float:
    """Estimate propellant burned by integrating -mdot over time.

    This intentionally ignores discrete mass drops from staging so the objective
    reflects propellant burned, not mass jettisoned.
    """
    t = getattr(log, "t_sim", None)
    mdot = getattr(log, "mdot", None)
    if not t or not mdot or len(t) < 2 or len(mdot) != len(t):
        return max(0.0, float(fallback_mass_delta))

    burned = 0.0
    for i in range(1, len(t)):
        dt = float(t[i]) - float(t[i - 1])
        if dt <= 0.0:
            continue
        mdot0 = min(0.0, float(mdot[i - 1]))
        mdot1 = min(0.0, float(mdot[i]))
        burned += -0.5 * (mdot0 + mdot1) * dt
    return max(0.0, burned)

def calculate_cost(
    results: dict,
    phase: int,
    target_orbit_alt_m: float,
    earth_radius_m: float
) -> float:
    """
    Calculates the cost for a given set of simulation results and optimization phase.

    Parameters
    ----------
    results : dict
        A dictionary of simulation results from `evaluate_simulation_results`.
    phase : int
        The current optimization phase (1 for targeting, 2 for optimization).
    target_orbit_alt_m : float
        The target orbital altitude in meters.
    earth_radius_m : float
        The radius of the Earth in meters.

    Returns
    -------
    float
        The calculated cost.
    """
    status = str(results.get("status", "INIT"))

    # Anything that is not a bound orbit should be catastrophically worse than any
    # valid LEO solution, but still provide a gradient (mainly via altitude).
    if status in {"CRASH", "ESCAPE", "SIM_FAIL_NO_DATA", "SIM_FAIL_INDEX", "SIM_FAIL_UNKNOWN"}:
        max_alt = float(results.get("max_altitude", 0.0) or 0.0)
        altitude_shortfall = max(0.0, float(target_orbit_alt_m) - max_alt)
        return float(PENALTY_CRASH + 1000.0 * altitude_shortfall)

    if phase == 1:
        # Phase 1: Achieve any stable orbit.
        # The goal is to get the perigee above the PERIGEE_FLOOR_M.
        perigee_alt = float(results.get("perigee_alt_m", -1.0))
        if perigee_alt >= PERIGEE_FLOOR_M:
            # Low cost for any stable orbit, with a small gradient based on how close to target
            target_r = float(earth_radius_m) + float(target_orbit_alt_m)
            rp = float(results.get("rp_m", 0.0) or 0.0)
            ra = float(results.get("ra_m", 0.0) or 0.0)
            if not _is_finite_number(rp) or not _is_finite_number(ra):
                return float(PENALTY_CRASH)
            return float(abs(rp - target_r) + abs(ra - target_r))
        else:
            # High penalty for sub-orbital trajectories, with a gradient to encourage higher perigee.
            perigee_shortfall = max(0.0, PERIGEE_FLOOR_M - perigee_alt)
            return float(PENALTY_CRASH + 1000.0 * perigee_shortfall)
    
    else: # Phase 2: Minimize fuel for a precise orbit.
        fuel_used = max(0.0, float(results.get("fuel", 0.0) or 0.0))
        orbital_error = results.get("orbital_error", PENALTY_CRASH)
        perigee_alt = float(results.get("perigee_alt_m", -1.0))

        # Enforce a minimum perigee even in phase 2 (otherwise "almost orbit" can
        # trade fuel for re-entry).
        if perigee_alt < PERIGEE_FLOOR_M:
            perigee_shortfall = max(0.0, PERIGEE_FLOOR_M - perigee_alt)
            return float(PENALTY_CRASH + 1000.0 * perigee_shortfall)

        if not _is_finite_number(orbital_error):
            return float(PENALTY_CRASH)
        orbital_error = float(orbital_error)
        
        # Penalize solutions that are not in a valid orbit
        if orbital_error > TARGET_TOLERANCE_M:
            # The penalty is proportional to the error, encouraging the optimizer to fix the orbit.
            penalty = (orbital_error - TARGET_TOLERANCE_M) * ORBIT_ERROR_WEIGHT
            return float(fuel_used + penalty)
        else:
            # Once the orbit is good enough, focus only on minimizing fuel.
            return float(fuel_used)

def evaluate_simulation_results(
    log: Logger,
    initial_mass: float,
    cfg_env: EnvironmentConfig,
    sim_config: SimulationConfig,
    max_altitude: float,
    phase: int
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
    phase : int
        The current optimization phase (1 for targeting, 2 for optimization).

    Returns
    -------
    dict
        A dictionary containing key performance indicators from the simulation.
    """
    results = {"fuel": 0.0, "status": "INIT", "max_altitude": float(max_altitude)}

    if not log.t_sim or len(log.t_sim) == 0:
        results["status"] = "SIM_FAIL_NO_DATA"
        results["perigee_error_m"] = 0.0
        results["apoapsis_error_m"] = 0.0
        results["orbital_error"] = PENALTY_CRASH
        results["cost"] = calculate_cost(results, phase, sim_config.target_orbit_alt_m, cfg_env.earth_radius_m)
        return results

    # Fuel used should mean propellant burned, not mass jettisoned at staging.
    results["fuel"] = _fuel_burned_from_log(log, fallback_mass_delta=float(initial_mass) - float(log.m[-1]))

    r, v = log.r[-1], log.v[-1]
    a, rp, ra = orbital_elements_from_state(r, v, cfg_env.earth_mu)
    results["rp_m"], results["ra_m"] = rp, ra

    # Reject anything unbound/non-finite: hyperbolic trajectories produce ra=inf and
    # break downstream error metrics.
    if rp is None or ra is None or a is None:
        results["status"] = "CRASH"
        results["perigee_error_m"] = 0.0
        results["apoapsis_error_m"] = 0.0
        results["orbital_error"] = PENALTY_CRASH
        results["cost"] = calculate_cost(results, phase, sim_config.target_orbit_alt_m, cfg_env.earth_radius_m)
        return results
    if not np.isfinite(float(a)) or not np.isfinite(float(rp)) or not np.isfinite(float(ra)) or float(a) <= 0.0:
        results["status"] = "ESCAPE"
        results["perigee_error_m"] = 0.0
        results["apoapsis_error_m"] = 0.0
        results["orbital_error"] = PENALTY_CRASH
        results["cost"] = calculate_cost(results, phase, sim_config.target_orbit_alt_m, cfg_env.earth_radius_m)
        return results
    if float(rp) <= float(cfg_env.earth_radius_m):
        results["status"] = "CRASH"
        results["perigee_error_m"] = 0.0
        results["apoapsis_error_m"] = 0.0
        results["orbital_error"] = PENALTY_CRASH
        results["cost"] = calculate_cost(results, phase, sim_config.target_orbit_alt_m, cfg_env.earth_radius_m)
        return results

    perigee_alt = float(rp) - float(cfg_env.earth_radius_m)
    apoapsis_alt = float(ra) - float(cfg_env.earth_radius_m)
    ecc = abs((float(ra) - float(rp)) / (float(ra) + float(rp))) if (float(ra) + float(rp)) != 0.0 else 0.0
    results["perigee_alt_m"] = perigee_alt
    results["apoapsis_alt_m"] = apoapsis_alt
    results["eccentricity"] = ecc

    target_r = float(cfg_env.earth_radius_m) + float(sim_config.target_orbit_alt_m)
    perigee_error = abs(float(rp) - target_r)
    apoapsis_error = abs(float(ra) - target_r)
    results["perigee_error_m"] = perigee_error
    results["apoapsis_error_m"] = apoapsis_error
    results["orbital_error"] = perigee_error + apoapsis_error

    # Determine status based on orbital parameters
    if perigee_alt < PERIGEE_FLOOR_M:
        results["status"] = "SUBORBIT"
    elif results["orbital_error"] < TARGET_TOLERANCE_M * 0.5:
        results["status"] = "PERFECT"
    elif results["orbital_error"] < TARGET_TOLERANCE_M * 2:
        results["status"] = "GOOD"
    else:
        results["status"] = "OK"

    results["cost"] = calculate_cost(results, phase, sim_config.target_orbit_alt_m, cfg_env.earth_radius_m)
    
    return results
