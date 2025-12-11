"""
multithread_optimization.py

Legacy single-phase optimizer (Differential Evolution). Kept for compatibility
with tests; the CMA-based optimizer in optimization_twostage.py is the primary
path. This module mirrors the old API so imports and wrappers continue to work.
"""

from __future__ import annotations

import csv
import time
import numpy as np
from scipy.optimize import differential_evolution

from main import build_simulation, orbital_elements_from_state
from gravity import MU_EARTH, R_EARTH
from config import CFG

# --- Configuration ---
TARGET_ALT_M = 200_000.0   # 200 km
LOG_FILENAME = "multithread_optimization_log.csv"
PENALTY_CRASH = 1e9        # "Soft Wall" for failed orbits

# Cost Function Weights
WEIGHT_FUEL = 1.0          # 1 kg fuel = 1 cost
WEIGHT_ERROR = 15.0        # 1 m of orbit error = 15 cost. Balances fuel vs accuracy.


def run_simulation_wrapper(scaled_params):
    """
    Runs the simulation with scaled input parameters and returns a results dictionary.

    Parameters
    ----------
    scaled_params : array-like
        [meco_mach, pitch_start_km, pitch_end_km, pitch_blend_exp]
        The latter three are retained for API compatibility; current guidance
        uses the config pitch schedules.
    """
    meco_mach, p_start_km, p_end_km, p_blend = scaled_params

    CFG.meco_mach = float(meco_mach)
    CFG.orbit_alt_tol = 1e6
    CFG.exit_on_orbit = True

    results = {"fuel": 0.0, "error": PENALTY_CRASH, "status": "INIT"}

    try:
        sim, state0, t0 = build_simulation()
        initial_mass = state0.m

        log = sim.run(
            t0,
            duration=2000.0,
            dt=1.0,
            state0=state0,
            orbit_target_radius=R_EARTH + TARGET_ALT_M,
            exit_on_orbit=True,
        )

        results["fuel"] = initial_mass - log.m[-1]
        r, v = log.r[-1], log.v[-1]
        a, rp, ra = orbital_elements_from_state(r, v, MU_EARTH)

        if rp is None or ra is None or rp < R_EARTH:
            results["status"] = "CRASH"
            return results

        target_r = R_EARTH + TARGET_ALT_M
        results["error"] = abs(rp - target_r) + abs(ra - target_r)

        if results["error"] < 5000:
            results["status"] = "PERFECT"
        elif results["error"] < 50000:
            results["status"] = "GOOD"
        else:
            results["status"] = "OK"

    except (IndexError, Exception):
        results["status"] = "SIM_FAIL"

    return results


def objective_function(scaled_params):
    """
    Single-phase objective function for the parallel optimizer.
    Calculates a combined cost from fuel usage and orbital error.
    """
    results = run_simulation_wrapper(scaled_params)
    cost = (results["fuel"] * WEIGHT_FUEL) + (results["error"] * WEIGHT_ERROR)
    return cost


def run_parallel_optimization():
    """
    Runs the optimization using Differential Evolution, which can distribute
    simulation runs across CPU cores via scipy's `workers` option.
    """
    bounds = [
        (2.0, 9.0),      # MECO Mach
        (0.5, 10.0),     # Pitch Start (km) - retained for compatibility
        (40.0, 150.0),   # Pitch End (km)
        (0.4, 1.5)       # Pitch Blend Exponent
    ]

    print("=" * 80)
    print("Starting parallel optimization using Differential Evolution.")
    print("=" * 80)

    start_time = time.time()

    result = differential_evolution(
        objective_function,
        bounds,
        strategy='best1bin',
        maxiter=200,
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=True,
        workers=-1
    )

    end_time = time.time()
    print(f"\nOptimization finished in {end_time - start_time:.2f} seconds.")

    print("\n" + "=" * 80)
    print("=== OPTIMIZATION COMPLETE ===")

    final_params = result.x
    final_results = run_simulation_wrapper(final_params)
    final_cost = (final_results["fuel"] * WEIGHT_FUEL) + (final_results["error"] * WEIGHT_ERROR)

    print(f"Final Fuel Used: {final_results['fuel']:.1f} kg")
    print(f"Final Orbit Error: {final_results['error']/1000:.1f} km")
    print(f"Final Cost: {final_cost:.2f}")
    print(f"Optimal SCALED Params: {final_params}")

    with open(LOG_FILENAME, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            f"{final_params[0]:.4f}",
            f"{final_params[1] * 1000.0:.1f}",
            f"{final_params[2] * 1000.0:.1f}",
            f"{final_params[3]:.3f}",
            f"{final_cost:.2f}",
            f"{final_results['fuel']:.2f}",
            f"{final_results['error']:.2f}",
            final_results['status']
        ])
    print(f"\nFinal results have been logged to {LOG_FILENAME}")


if __name__ == "__main__":
    run_parallel_optimization()
