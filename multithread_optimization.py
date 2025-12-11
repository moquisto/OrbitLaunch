"""
multithread_optimization.py

Solves the launch-to-orbit problem using a parallel, single-phase optimization
strategy with Differential Evolution. This approach leverages multiple CPU cores
to evaluate many candidate trajectories simultaneously, significantly speeding up
the search for an optimal solution.
"""
import csv
import numpy as np
import time
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
    (This function is identical to the one in optimization_twostage.py)
    Runs the simulation with scaled input parameters and returns a results dictionary.
    """
    meco_mach, p_start_km, p_end_km, p_blend = scaled_params
    p_start_m = p_start_km * 1000.0
    p_end_m   = p_end_km * 1000.0

    CFG.pitch_guidance_mode = 'function'
    CFG.meco_mach = float(meco_mach)
    CFG.pitch_turn_start_m = float(p_start_m)
    CFG.pitch_turn_end_m = float(p_end_m)
    CFG.pitch_blend_exp = float(p_blend)
    CFG.orbit_alt_tol = 1e6
    CFG.exit_on_orbit = True

    results = {"fuel": 0.0, "error": PENALTY_CRASH, "status": "INIT"}

    try:
        sim, state0, t0 = build_simulation()
        initial_mass = state0.m

        log = sim.run(t0, duration=2000.0, dt=1.0, state0=state0,
                      orbit_target_radius=R_EARTH + TARGET_ALT_M,
                      exit_on_orbit=True)

        results["fuel"] = initial_mass - log.m[-1]
        r, v = log.r[-1], log.v[-1]
        a, rp, ra = orbital_elements_from_state(r, v, MU_EARTH)

        if rp is None or ra is None or rp < R_EARTH:
            results["status"] = "CRASH"
            return results

        target_r = R_EARTH + TARGET_ALT_M
        results["error"] = abs(rp - target_r) + abs(ra - target_r)

        if results["error"] < 5000: results["status"] = "PERFECT"
        elif results["error"] < 50000: results["status"] = "GOOD"
        else: results["status"] = "OK"

    except (IndexError, Exception):
        results["status"] = "SIM_FAIL"
        
    return results

def objective_function(scaled_params):
    """
    Single-phase objective function for the parallel optimizer.
    Calculates a combined cost from fuel usage and orbital error.
    """
    results = run_simulation_wrapper(scaled_params)
    
    # The cost is a weighted sum of fuel used and the final orbit error.
    # This guides the optimizer to find a solution that is both fuel-efficient
    # and accurate.
    cost = (results["fuel"] * WEIGHT_FUEL) + (results["error"] * WEIGHT_ERROR)
    
    return cost

def run_parallel_optimization():
    """
    Runs the optimization using Differential Evolution, which distributes
    the simulation runs across all available CPU cores.
    """
    # Bounds are the same as before (min and max for each parameter)
    bounds = [
        (2.0, 9.0),      # MECO Mach
        (0.5, 10.0),     # Pitch Start (km)
        (40.0, 150.0),   # Pitch End (km)
        (0.4, 1.5)       # Pitch Blend Exponent
    ]

    print("="*80)
    print("Starting parallel optimization using Differential Evolution.")
    print("This will use all available CPU cores to speed up the process.")
    print("="*80)
    
    start_time = time.time()

    # --- Run Optimizer ---
    # differential_evolution is a global optimizer that is great for parallel execution.
    # `workers=-1` tells it to create a process pool using all available CPUs.
    result = differential_evolution(
        objective_function,
        bounds,
        strategy='best1bin',
        maxiter=200,          # More iterations as it's a global search
        popsize=15,           # Population size per generation
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=True,            # Display progress
        workers=-1            # THE KEY: Use all available CPU cores
    )
    
    end_time = time.time()
    print(f"\nOptimization finished in {end_time - start_time:.2f} seconds.")

    print("\n" + "="*80)
    print("=== OPTIMIZATION COMPLETE ===")
    
    final_params = result.x
    
    # Run a final simulation with the best parameters to get definitive results
    final_results = run_simulation_wrapper(final_params)
    final_cost = (final_results["fuel"] * WEIGHT_FUEL) + (final_results["error"] * WEIGHT_ERROR)

    print(f"Final Fuel Used: {final_results['fuel']:.1f} kg")
    print(f"Final Orbit Error: {final_results['error']/1000:.1f} km")
    print(f"Final Cost: {final_cost:.2f}")
    print(f"Optimal SCALED Params: {final_params}")

    # Log the final, best result to the CSV file
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
