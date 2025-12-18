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
# When outside the tolerance band, discourage solutions that "nail" only one
# axis (e.g. perfect apoapsis but poor perigee) since the requirement is that
# both perigee and apoapsis be within tolerance.
ORBIT_BALANCE_WEIGHT = 50.0  # [kg per meter]
# Even within the tolerance band, prefer being closer to the exact target
# altitude (otherwise the minimum-fuel solution tends to sit on the lowest
# acceptable orbit).
ORBIT_ERROR_WEIGHT_IN_TOL = 0.2  # [kg per meter] in the within-tolerance term

# When a trajectory is hyperbolic, penalize *how* unbound it is using the
# specific orbital energy excess (epsilon > 0). This helps the optimizer learn
# how to back away from escape trajectories rather than only chasing altitude.
ENERGY_EXCESS_WEIGHT = 50.0  # [cost units per (J/kg)]

# In phase 1 we primarily need to get the perigee above the survivability
# threshold. Add an explicit perigee-floor penalty even before a valid orbit is
# reached so the search has a meaningful gradient.
PERIGEE_SHORTFALL_WEIGHT = 10.0  # [cost units per meter]


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

    max_alt = float(results.get("max_altitude", 0.0) or 0.0)
    target_r = float(earth_radius_m) + float(target_orbit_alt_m)

    def altitude_shortfall_penalty() -> float:
        altitude_shortfall = max(0.0, float(target_orbit_alt_m) - max_alt)
        return float(1000.0 * altitude_shortfall)

    # Generic failure modes (no data / simulation errors).
    if status in {"SIM_FAIL_NO_DATA", "SIM_FAIL_INDEX", "SIM_FAIL_UNKNOWN"}:
        return float(PENALTY_CRASH + altitude_shortfall_penalty())

    # Escape trajectories: keep the big crash penalty but add a gradient based on
    # how unbound the orbit is (epsilon > 0), otherwise the optimizer tends to
    # "sit" on escape solutions around the termination altitude.
    if status == "ESCAPE":
        eps = float(results.get("specific_energy_jpkg", 0.0) or 0.0)
        energy_excess = max(0.0, eps)
        return float(PENALTY_CRASH + altitude_shortfall_penalty() + ENERGY_EXCESS_WEIGHT * energy_excess)

    if phase == 1:
        # Phase 1: Find a bound trajectory and steer it toward the target orbit.
        # Use orbital-shape errors whenever (rp, ra) are available, even for
        # sub-orbital/perigee-inside-Earth cases. This gives a much better
        # gradient than using only max altitude for "crash" cases.
        perigee_alt = float(results.get("perigee_alt_m", -np.inf) or -np.inf)
        rp = results.get("rp_m", None)
        ra = results.get("ra_m", None)

        if _is_finite_number(rp) and _is_finite_number(ra):
            rp = float(rp)
            ra = float(ra)
            orbit_shape_error = abs(rp - target_r) + abs(ra - target_r)
        else:
            orbit_shape_error = float(PENALTY_CRASH)

        perigee_shortfall = max(0.0, PERIGEE_FLOOR_M - perigee_alt)

        # Anything that is not a survivable bound orbit must remain
        # catastrophically worse than any valid orbit solution.
        is_survivable_orbit = status in {"OK", "GOOD", "PERFECT"}
        base = 0.0 if is_survivable_orbit else float(PENALTY_CRASH)
        return float(base + orbit_shape_error + PERIGEE_SHORTFALL_WEIGHT * perigee_shortfall)
    
    else: # Phase 2: Minimize fuel for a precise orbit.
        fuel_used = max(0.0, float(results.get("fuel", 0.0) or 0.0))
        orbital_error = results.get("orbital_error", PENALTY_CRASH)
        perigee_error = results.get("perigee_error_m", orbital_error)
        apoapsis_error = results.get("apoapsis_error_m", orbital_error)
        perigee_alt = float(results.get("perigee_alt_m", -np.inf) or -np.inf)

        # Enforce a minimum perigee even in phase 2 (otherwise "almost orbit" can
        # trade fuel for re-entry).
        if perigee_alt < PERIGEE_FLOOR_M:
            perigee_shortfall = max(0.0, PERIGEE_FLOOR_M - perigee_alt)
            return float(PENALTY_CRASH + PERIGEE_SHORTFALL_WEIGHT * perigee_shortfall + altitude_shortfall_penalty())

        if not _is_finite_number(orbital_error):
            return float(PENALTY_CRASH)
        orbital_error = float(orbital_error)

        # Penalize both perigee and apoapsis errors (not just the worst-axis)
        # to encourage true circularization around the target altitude.
        if not _is_finite_number(perigee_error):
            perigee_error = orbital_error
        if not _is_finite_number(apoapsis_error):
            apoapsis_error = orbital_error
        perigee_error = float(perigee_error)
        apoapsis_error = float(apoapsis_error)
        orbit_shape_error = max(0.0, perigee_error) + max(0.0, apoapsis_error)

        # Always include a small orbit-accuracy term so the optimizer prefers the
        # exact target altitude, not just the edge of the tolerance band.
        cost = fuel_used + orbit_shape_error * ORBIT_ERROR_WEIGHT_IN_TOL

        # Outside the tolerance band, orbit accuracy must dominate fuel.
        #
        # Penalize the worst-axis error (requirement is both perigee and apoapsis
        # within tolerance) and add a balance term so the optimizer doesn't spend
        # effort perfecting one axis while leaving the other far off.
        if orbital_error > TARGET_TOLERANCE_M:
            worst_excess = float(max(0.0, orbital_error - TARGET_TOLERANCE_M))
            cost += worst_excess * ORBIT_ERROR_WEIGHT
            cost += abs(perigee_error - apoapsis_error) * ORBIT_BALANCE_WEIGHT

        return float(cost)

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
    results = {
        "fuel": 0.0,
        "status": "INIT",
        "max_altitude": float(max_altitude),
        "cutoff_reason": str(getattr(log, "cutoff_reason", "") or ""),
    }

    if not log.t_sim or len(log.t_sim) == 0:
        results["status"] = "SIM_FAIL_NO_DATA"
        results["perigee_error_m"] = 0.0
        results["apoapsis_error_m"] = 0.0
        results["orbital_error"] = PENALTY_CRASH
        results["cost"] = calculate_cost(results, phase, sim_config.target_orbit_alt_m, cfg_env.earth_radius_m)
        return results

    # Fuel used should mean propellant burned, not mass jettisoned at staging.
    results["fuel"] = _fuel_burned_from_log(log, fallback_mass_delta=float(initial_mass) - float(log.m[-1]))

    # Choose a representative state for orbit evaluation.
    #
    # - If the run ended in impact, evaluate at max altitude (final state is at/near
    #   the ground and produces misleading orbital elements).
    # - Otherwise, evaluate shortly after the last powered segment (end of burn).
    #
    # Evaluating at end-of-burn makes the objective largely independent of the
    # arbitrary simulation horizon (e.g., a long final plot run), avoiding
    # inconsistencies between "optimization complete" numbers and the final
    # trajectory evaluation.
    #
    # Be defensive about log array lengths: some unit tests use minimal mocks where
    # arrays can be different lengths.
    n_samples = min(len(log.t_sim), len(log.r), len(log.v), len(log.m))
    if n_samples <= 0:
        results["status"] = "SIM_FAIL_NO_DATA"
        results["perigee_error_m"] = 0.0
        results["apoapsis_error_m"] = 0.0
        results["orbital_error"] = PENALTY_CRASH
        results["cost"] = calculate_cost(results, phase, sim_config.target_orbit_alt_m, cfg_env.earth_radius_m)
        return results

    idx_eval: int
    if results["cutoff_reason"] == "impact" and getattr(log, "altitude", None):
        alt = list(log.altitude)
        n_alt = min(len(alt), n_samples)
        idx_eval = int(np.argmax(alt[:n_alt])) if n_alt > 0 else n_samples - 1
    else:
        burn_idx = None
        mdot = getattr(log, "mdot", None)
        if mdot:
            n = min(len(mdot), n_samples)
            for i in range(n - 1, -1, -1):
                try:
                    if float(mdot[i]) < -1e-6:
                        burn_idx = i
                        break
                except Exception:
                    continue

        if burn_idx is None:
            thrust = getattr(log, "thrust_mag", None)
            if thrust:
                n = min(len(thrust), n_samples)
                for i in range(n - 1, -1, -1):
                    try:
                        if float(thrust[i]) > 1e-3:
                            burn_idx = i
                            break
                    except Exception:
                        continue

        if burn_idx is None:
            idx_eval = n_samples - 1
        else:
            idx_eval = int(min(burn_idx + 1, n_samples - 1))

    results["eval_index"] = int(idx_eval)
    results["eval_t_sim_s"] = float(log.t_sim[idx_eval])

    r = np.asarray(log.r[idx_eval], dtype=float)
    v = np.asarray(log.v[idx_eval], dtype=float)
    r_norm = float(np.linalg.norm(r))
    v_norm = float(np.linalg.norm(v))
    r_hat = r / r_norm if r_norm > 0.0 else np.array([0.0, 0.0, 1.0], dtype=float)
    vr = float(np.dot(v, r_hat)) if r_norm > 0.0 else 0.0
    v_horizontal = float(np.sqrt(max(0.0, v_norm * v_norm - vr * vr)))
    fpa_deg = float(np.degrees(np.arctan2(vr, v_horizontal))) if (v_horizontal > 0.0 or vr != 0.0) else 0.0

    results["eval_altitude_m"] = float(r_norm - float(cfg_env.earth_radius_m))
    results["eval_speed_mps"] = float(v_norm)
    results["eval_vr_mps"] = float(vr)
    results["eval_fpa_deg"] = float(fpa_deg)

    # Prefer the simulation's recorded specific energy if present, otherwise
    # compute it directly.
    try:
        eps = float(getattr(log, "specific_energy", [])[idx_eval])
    except Exception:
        eps = 0.5 * v_norm * v_norm - float(cfg_env.earth_mu) / max(r_norm, 1e-6)
    results["specific_energy_jpkg"] = float(eps)

    a, rp, ra = orbital_elements_from_state(r, v, cfg_env.earth_mu)
    results["rp_m"], results["ra_m"] = rp, ra

    # Handle invalid orbital element computation.
    if rp is None or ra is None or a is None:
        results["status"] = "CRASH"
        results["perigee_error_m"] = 0.0
        results["apoapsis_error_m"] = 0.0
        results["orbital_error"] = PENALTY_CRASH
        results["cost"] = calculate_cost(results, phase, sim_config.target_orbit_alt_m, cfg_env.earth_radius_m)
        return results

    # Unbound / escape: ra=inf or a<=0.
    if (not np.isfinite(float(ra))) or (not np.isfinite(float(a))) or float(a) <= 0.0:
        results["status"] = "ESCAPE"
        results["perigee_error_m"] = 0.0
        results["apoapsis_error_m"] = 0.0
        results["orbital_error"] = PENALTY_CRASH
        results["cost"] = calculate_cost(results, phase, sim_config.target_orbit_alt_m, cfg_env.earth_radius_m)
        return results

    perigee_alt = float(rp) - float(cfg_env.earth_radius_m)
    apoapsis_alt = float(ra) - float(cfg_env.earth_radius_m)
    denom = float(ra) + float(rp)
    ecc = abs((float(ra) - float(rp)) / denom) if denom != 0.0 else 0.0
    results["perigee_alt_m"] = float(perigee_alt)
    results["apoapsis_alt_m"] = float(apoapsis_alt)
    results["eccentricity"] = float(ecc) if np.isfinite(float(ecc)) else 0.0

    target_r = float(cfg_env.earth_radius_m) + float(sim_config.target_orbit_alt_m)
    perigee_error = abs(float(rp) - target_r)
    apoapsis_error = abs(float(ra) - target_r)
    results["perigee_error_m"] = float(perigee_error)
    results["apoapsis_error_m"] = float(apoapsis_error)
    # Use the worst-axis error so `TARGET_TOLERANCE_M` directly corresponds to
    # "within ±tolerance of the target altitude" for *both* perigee and apoapsis.
    results["orbital_error"] = float(max(perigee_error, apoapsis_error))

    # Determine status based on outcome + orbital parameters.
    if results["cutoff_reason"] == "impact":
        results["status"] = "CRASH"
    elif perigee_alt < PERIGEE_FLOOR_M:
        results["status"] = "SUBORBIT"
    elif results["orbital_error"] < TARGET_TOLERANCE_M * 0.5:
        results["status"] = "PERFECT"
    elif results["orbital_error"] < TARGET_TOLERANCE_M * 2:
        results["status"] = "GOOD"
    else:
        results["status"] = "OK"

    results["cost"] = calculate_cost(results, phase, sim_config.target_orbit_alt_m, cfg_env.earth_radius_m)

    return results
