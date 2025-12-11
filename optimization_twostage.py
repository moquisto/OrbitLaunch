"""
optimization_twostage.py

Solves the launch problem in two distinct phases for maximum efficiency:
Phase 1: "Targeting" - Find ANY parameters that hit the target orbit (Ignore fuel).
Phase 2: "Optimizing" - From that valid orbit, minimize fuel usage while staying in orbit.

This version incorporates input scaling and detailed logging for better performance and analysis.
"""
import csv
import numpy as np
import time
from scipy.optimize import minimize
from main import build_simulation, orbital_elements_from_state, ParameterizedThrottleProgram
from gravity import MU_EARTH, R_EARTH
from config import CFG

print("optimization_twostage.py script started.", flush=True)

# --- Configuration ---
TARGET_ALT_M = 200_000.0   # 200 km
# We need to be within 10km to consider it "Orbit" for Phase 2
TARGET_TOLERANCE_M = 10000.0
LOG_FILENAME = "optimization_twostage_log.csv"
PENALTY_CRASH = 1e9        # "Soft Wall" for failed orbits

# --- Initialize CSV ---
# with open(LOG_FILENAME, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow([
#         "phase", "iteration", "meco_mach", "pitch_start_m", "pitch_end_m",
#         "pitch_blend", "coast_s", "upper_burn_s", "cost", "fuel_used_kg",
#         "orbit_error_m", "status"
#     ])

global_iter_count = 0


def log_iteration(phase, iteration, params, results):
    """Helper to log a single optimizer iteration with de-scaled physics values."""
    meco_mach, p_start_km, p_end_km, p_blend, coast_s, upper_burn_s = params
    with open(LOG_FILENAME, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            phase,
            iteration,
            f"{meco_mach:.4f}",
            f"{p_start_km * 1000.0:.1f}",  # Log real value
            f"{p_end_km * 1000.0:.1f}",   # Log real value
            f"{p_blend:.3f}",
            f"{coast_s:.1f}",
            f"{upper_burn_s:.1f}",
            f"{results['cost']:.2f}",
            f"{results['fuel']:.2f}",
            f"{results['error']:.2f}",
            results['status']
        ])


def run_simulation_wrapper(scaled_params):
    """
    Runs the simulation with scaled input parameters.
    The optimizer works with small, normalized numbers (e.g., kilometers for altitude).
    This function de-scales them into real physics units (meters) for the simulation.
    Returns a dictionary with detailed results.
    """
    # 1. DE-SCALE PARAMETERS for physics simulation
    meco_mach, p_start_km, p_end_km, p_blend, coast_s, upper_burn_s = scaled_params
    p_start_m = p_start_km * 1000.0
    p_end_m = p_end_km * 1000.0

    # 2. UPDATE CONFIGURATION
    CFG.pitch_guidance_mode = 'function'  # CRITICAL: Use function, not table
    CFG.meco_mach = float(meco_mach)
    CFG.pitch_turn_start_m = float(p_start_m)
    CFG.pitch_turn_end_m = float(p_end_m)
    CFG.pitch_blend_exp = float(p_blend)

    # Second stage parameters
    CFG.separation_delay_s = float(coast_s)
    CFG.upper_ignition_delay_s = 2.0  # Keep a small settling time
    CFG.upper_stage_throttle_program = [
        [0.0, 1.0],
        [upper_burn_s, 1.0],
        [upper_burn_s + 1, 0.0]
    ]

    CFG.orbit_alt_tol = 1e6    # Loosen tolerance for the optimizer's internal check
    CFG.exit_on_orbit = False  # IMPORTANT: Simulate through second stage burn

    results = {
        "fuel": 0.0,
        "error": PENALTY_CRASH,  # Default to a high error
        "status": "INIT",
        "cost": PENALTY_CRASH
    }

    try:
        sim, state0, t0 = build_simulation()

        # Apply the parameterized throttle program for the upper stage
        throttle_controller = ParameterizedThrottleProgram(
            schedule=CFG.upper_stage_throttle_program)
        sim.guidance.throttle_schedule = throttle_controller

        initial_mass = state0.m

        # Run sim - longer duration to account for coast and second burn
        log = sim.run(t0, duration=3000.0, dt=1.0, state0=state0,
                      orbit_target_radius=R_EARTH + TARGET_ALT_M,
                      exit_on_orbit=False)

        results["fuel"] = initial_mass - log.m[-1]
        max_altitude = max(log.altitude) if log.altitude else 0.0

        r, v = log.r[-1], log.v[-1]
        a, rp, ra = orbital_elements_from_state(r, v, MU_EARTH)

        final_r_vec = log.r[-1]
        final_altitude = np.linalg.norm(final_r_vec) - R_EARTH

        # Crashed (with buffer)
        if rp is None or ra is None or rp < (R_EARTH - 5000):
            results["status"] = "CRASH"
            # New penalty function: Use max altitude to create a stronger gradient
            # The 1e7 constant ensures this is always worse than a non-crash scenario.
            # The second term encourages the optimizer to achieve higher altitudes.
            error = 1e7 + (TARGET_ALT_M - max_altitude)
            results["error"] = error
            return results

        target_r = R_EARTH + TARGET_ALT_M
        results["error"] = abs(rp - target_r) + abs(ra - target_r)

        if results["error"] < 5000:
            results["status"] = "PERFECT"
        elif results["error"] < 50000:
            results["status"] = "GOOD"
        else:
            results["status"] = "OK"

    except IndexError:
        # This can happen if the simulation ends prematurely (e.g., crash before MECO)
        results["status"] = "SIM_FAIL_INDEX"
    except Exception:
        results["status"] = "SIM_FAIL_UNKNOWN"

    return results


# New global variables for watchdog
last_params_phase1 = None
last_cost_phase1 = None
stuck_counter_phase1 = 0
last_params_phase2 = None
last_cost_phase2 = None
stuck_counter_phase2 = 0
STUCK_THRESHOLD = 1 # Number of stuck iterations before terminating
SOME_LARGE_ERROR_THRESHOLD = 1e7

def soft_bounds_penalty(scaled_params, bounds):
    penalty = 0.0
    for i, param in enumerate(scaled_params):
        lower, upper = bounds[i]
        if param < lower:
            penalty += (lower - param) * 1e5  # Large penalty for going below lower bound
        elif param > upper:
            penalty += (param - upper) * 1e5  # Large penalty for going above upper bound
    return penalty

# ...


def objective_phase1(scaled_params):
    global global_iter_count, last_params_phase1, last_cost_phase1, stuck_counter_phase1
    print(f"DEBUG: Entering objective_phase1, iteration {global_iter_count}", flush=True) # Added flush=True here too

    global_iter_count += 1

    results = run_simulation_wrapper(scaled_params)
    results['cost'] = results['error']  # Cost for phase 1 is just the error
    # Apply soft bounds penalty
    bound_penalty = soft_bounds_penalty(scaled_params, bounds)
    results['cost'] += bound_penalty

    log_iteration("Phase 1", global_iter_count, scaled_params, results)
    print(
        f"[Phase 1] Iter {global_iter_count:3d} | Error: {results['error']/1000:.1f} km | Status: {results['status']}", flush=True)

    # Watchdog check
    if last_params_phase1 is not None and np.allclose(scaled_params, last_params_phase1) and results['cost'] >= SOME_LARGE_ERROR_THRESHOLD:
        stuck_counter_phase1 += 1
        if stuck_counter_phase1 >= STUCK_THRESHOLD:
            raise RuntimeError(
                f"Phase 1 optimizer stuck at local minimum or flat region after {stuck_counter_phase1} iterations. Terminating.")
    else:
        stuck_counter_phase1 = 0  # Reset counter if progress is made or params change

    last_params_phase1 = np.copy(scaled_params)
    last_cost_phase1 = results['cost']

    return results['cost']


def objective_phase2(scaled_params):
    global global_iter_count, last_params_phase2, last_cost_phase2, stuck_counter_phase2

    global_iter_count += 1

    results = run_simulation_wrapper(scaled_params)
    fuel, error = results["fuel"], results["error"]

    # Penalize heavily if we stray too far from the target orbit
    if error > TARGET_TOLERANCE_M:
        penalty = (error - TARGET_TOLERANCE_M) * 10.0  # Heavy penalty
        cost = fuel + penalty
    else:
        cost = fuel
    # Apply soft bounds penalty
    bound_penalty = soft_bounds_penalty(scaled_params, bounds)
    cost += bound_penalty
    results['cost'] = cost

    log_iteration("Phase 2", global_iter_count, scaled_params, results)
    print(f"[Phase 2] Iter {global_iter_count:3d} | Fuel: {fuel:.0f} kg | Error: {error/1000:.1f} km | Cost: {cost:.0f} | Status: {results['status']}", flush=True)

    # Watchdog check
    if last_params_phase2 is not None and np.allclose(scaled_params, last_params_phase2) and results['cost'] >= SOME_LARGE_ERROR_THRESHOLD:
        stuck_counter_phase2 += 1
        if stuck_counter_phase2 >= STUCK_THRESHOLD:
            raise RuntimeError(
                f"Phase 2 optimizer stuck at local minimum or flat region after {stuck_counter_phase2} iterations. Terminating.")
    else:
        stuck_counter_phase2 = 0  # Reset counter if progress is made or params change

    last_params_phase2 = np.copy(scaled_params)
    last_cost_phase2 = results['cost']

    return results['cost']


def run_optimization():
    """Runs the two-phase optimization process."""
    global global_iter_count

    # --- Initial Guess & Bounds (SCALED) ---
    # Use km for altitudes to keep numbers in a similar magnitude
    # params: meco_mach, pitch_start_km, pitch_end_km, pitch_blend, coast_s, upper_burn_s
    x0 = [6.0, 0.5, 80.0, 0.85, 120.0, 180.0]
    global bounds # Make bounds global for soft_bounds_penalty
    bounds = [
        (2.0, 9.0),      # MECO Mach
        (0.5, 10.0),     # Pitch Start (km)
        (40.0, 150.0),   # Pitch End (km)
        (0.4, 1.5),      # Pitch Blend Exponent
        (10.0, 500.0),   # Coast duration after MECO (s)
        (50.0, 400.0)    # Upper stage burn duration (s)
    ]

    print(f"=== PHASE 1: TARGETING ORBIT (Logging to {LOG_FILENAME}) ===", flush=True)
    global_iter_count = 0
    print("DEBUG: About to call minimize for Phase 1", flush=True)
    res1 = minimize(
        objective_phase1,
        x0,
        method='Nelder-Mead',
        options={'maxiter': 150} # Nelder-Mead does not use bounds, ftol, gtol, eps
    )

    print(f"\n--- Phase 1 Complete ---", flush=True)
    print(f"Best Error: {res1.fun/1000:.1f} km", flush=True)
    final_params1 = res1.x
    print(
        f"Params: Mach={final_params1[0]:.2f}, Start={final_params1[1]:.1f}km, End={final_params1[2]:.1f}km, Blend={final_params1[3]:.2f}, Coast={final_params1[4]:.1f}s, Upper Burn={final_params1[5]:.1f}s", flush=True)

    if res1.fun > 50000:  # If we're still more than 50km off, it's probably not a good solution
        print("\nFailed to find a stable orbit in Phase 1. Stopping.", flush=True)
        return

    print("\n=== PHASE 2: MINIMIZING FUEL ===", flush=True)
    global_iter_count = 0  # Reset for phase 2 logging
    x0_phase2 = res1.x  # Start phase 2 from the best point of phase 1

    res2 = minimize(
        objective_phase2,
        x0_phase2,
        method='Nelder-Mead',
        options={'maxiter': 150} # Nelder-Mead does not use bounds, ftol, gtol, eps
    )

    print("\n=== OPTIMIZATION COMPLETE ===", flush=True)
    final_params2 = res2.x
    # Run one last time for final numbers
    final_results = run_simulation_wrapper(final_params2)

    print(f"Final Fuel Used: {final_results['fuel']:.1f} kg", flush=True)
    print(f"Final Orbit Error: {final_results['error']/1000:.1f} km", flush=True)
    print(
        f"Optimal Params: Mach={final_params2[0]:.2f}, Start={final_params2[1]*1000:.0f}m, End={final_params2[2]*1000:.0f}m, Blend={final_params2[3]:.2f}, Coast={final_params2[4]:.1f}s, Upper Burn={final_params2[5]:.1f}s", flush=True)


if __name__ == "__main__":
    try:
        run_optimization()
    except Exception as e:
        print(f"ERROR: An unhandled exception occurred during optimization: {e}", flush=True)
        import traceback
        traceback.print_exc()
