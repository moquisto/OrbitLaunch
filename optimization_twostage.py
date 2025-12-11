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
from scipy.optimize import minimize, differential_evolution
from main import build_simulation, ParameterizedThrottleProgram
from gravity import MU_EARTH, R_EARTH, orbital_elements_from_state
from config import CFG
from custom_guidance import create_pitch_program_callable

print("optimization_twostage.py script started.", flush=True)

# --- Configuration ---
TARGET_ALT_M = 200_000.0   # 200 km
# We need to be within 10km to consider it "Orbit" for Phase 2
TARGET_TOLERANCE_M = 10000.0
LOG_FILENAME = "optimization_twostage_log.csv"
PENALTY_CRASH = 1e9        # "Soft Wall" for failed orbits

# --- Initialize CSV ---
with open(LOG_FILENAME, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "phase", "iteration", "meco_mach",
        "pitch_time_0", "pitch_angle_0",
        "pitch_time_1", "pitch_angle_1",
        "pitch_time_2", "pitch_angle_2",
        "pitch_time_3", "pitch_angle_3",
        "pitch_time_4", "pitch_angle_4",
        "coast_s", "upper_burn_s",
        "upper_throttle_level_0", "upper_throttle_level_1", "upper_throttle_level_2", "upper_throttle_level_3",
        "upper_throttle_switch_ratio_0", "upper_throttle_switch_ratio_1", "upper_throttle_switch_ratio_2",
        "booster_throttle_level_0", "booster_throttle_level_1", "booster_throttle_level_2", "booster_throttle_level_3",
        "booster_throttle_switch_ratio_0", "booster_throttle_switch_ratio_1", "booster_throttle_switch_ratio_2",
        "cost", "fuel_used_kg", "orbit_error_m", "status"
    ])

global_iter_count = 0


def log_iteration(phase, iteration, params, results):
    """Helper to log a single optimizer iteration with de-scaled physics values."""
    meco_mach, pitch_time_0, pitch_angle_0, pitch_time_1, pitch_angle_1, pitch_time_2, pitch_angle_2, \
    pitch_time_3, pitch_angle_3, pitch_time_4, pitch_angle_4, \
    coast_s, upper_burn_s, \
    upper_throttle_level_0, upper_throttle_level_1, upper_throttle_level_2, upper_throttle_level_3, \
    upper_throttle_switch_ratio_0, upper_throttle_switch_ratio_1, upper_throttle_switch_ratio_2, \
    booster_throttle_level_0, booster_throttle_level_1, booster_throttle_level_2, booster_throttle_level_3, \
    booster_throttle_switch_ratio_0, booster_throttle_switch_ratio_1, booster_throttle_switch_ratio_2 = params
    with open(LOG_FILENAME, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            phase,
            iteration,
            f"{meco_mach:.4f}",
            f"{pitch_time_0:.1f}", f"{pitch_angle_0:.1f}",
            f"{pitch_time_1:.1f}", f"{pitch_angle_1:.1f}",
            f"{pitch_time_2:.1f}", f"{pitch_angle_2:.1f}",
            f"{pitch_time_3:.1f}", f"{pitch_angle_3:.1f}",
            f"{pitch_time_4:.1f}", f"{pitch_angle_4:.1f}",
            f"{coast_s:.1f}",
            f"{upper_burn_s:.1f}",
            f"{upper_throttle_level_0:.2f}", f"{upper_throttle_level_1:.2f}", f"{upper_throttle_level_2:.2f}", f"{upper_throttle_level_3:.2f}",
            f"{upper_throttle_switch_ratio_0:.2f}", f"{upper_throttle_switch_ratio_1:.2f}", f"{upper_throttle_switch_ratio_2:.2f}",
            f"{booster_throttle_level_0:.2f}", f"{booster_throttle_level_1:.2f}", f"{booster_throttle_level_2:.2f}", f"{booster_throttle_level_3:.2f}",
            f"{booster_throttle_switch_ratio_0:.2f}", f"{booster_throttle_switch_ratio_1:.2f}", f"{booster_throttle_switch_ratio_2:.2f}",
            f"{results['cost']:.2f}",
            f"{results['fuel']:.2f}",
            f"{results['orbit_error']:.2f}", # Changed "error" to "orbit_error" for clarity and consistency
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
    meco_mach, \
    pitch_time_0, pitch_angle_0, \
    pitch_time_1, pitch_angle_1, \
    pitch_time_2, pitch_angle_2, \
    pitch_time_3, pitch_angle_3, \
    pitch_time_4, pitch_angle_4, \
    coast_s, upper_burn_s, \
    upper_throttle_level_0, upper_throttle_level_1, upper_throttle_level_2, upper_throttle_level_3, \
    upper_throttle_switch_ratio_0, upper_throttle_switch_ratio_1, upper_throttle_switch_ratio_2, \
    booster_throttle_level_0, booster_throttle_level_1, booster_throttle_level_2, booster_throttle_level_3, \
    booster_throttle_switch_ratio_0, booster_throttle_switch_ratio_1, booster_throttle_switch_ratio_2 = scaled_params

    # Clamp pitch angles to be reasonable
    pitch_angle_0 = np.clip(pitch_angle_0, 0.0, 90.0)
    pitch_angle_1 = np.clip(pitch_angle_1, 0.0, 90.0)
    pitch_angle_2 = np.clip(pitch_angle_2, 0.0, 90.0)
    pitch_angle_3 = np.clip(pitch_angle_3, 0.0, 90.0)
    pitch_angle_4 = np.clip(pitch_angle_4, 0.0, 90.0)
    
    # Ensure pitch times are ordered and valid (s)
    pitch_times_raw = np.array([pitch_time_0, pitch_time_1, pitch_time_2, pitch_time_3, pitch_time_4])
    pitch_angles_deg_raw = np.array([pitch_angle_0, pitch_angle_1, pitch_angle_2, pitch_angle_3, pitch_angle_4])
    
    # Sort them to define the pitch program
    pitch_order = np.argsort(pitch_times_raw)
    pitch_times = pitch_times_raw[pitch_order]
    pitch_angles_deg = pitch_angles_deg_raw[pitch_order]

    # Ensure throttle levels and switch ratios are within [0, 1]
    upper_throttle_levels = np.clip([upper_throttle_level_0, upper_throttle_level_1, upper_throttle_level_2, upper_throttle_level_3], 0.0, 1.0)
    upper_throttle_switch_ratios = np.clip([upper_throttle_switch_ratio_0, upper_throttle_switch_ratio_1, upper_throttle_switch_ratio_2], 0.0, 1.0)
    upper_throttle_switch_ratios.sort() # Ensure ratios are increasing
    
    booster_throttle_levels = np.clip([booster_throttle_level_0, booster_throttle_level_1, booster_throttle_level_2, booster_throttle_level_3], 0.0, 1.0)
    booster_throttle_switch_ratios = np.clip([booster_throttle_switch_ratio_0, booster_throttle_switch_ratio_1, booster_throttle_switch_ratio_2], 0.0, 1.0)
    booster_throttle_switch_ratios.sort() # Ensure ratios are increasing

    # 2. UPDATE CONFIGURATION
    CFG.meco_mach = float(meco_mach)

    # Pitch guidance will be handled by a custom function
    CFG.pitch_guidance_mode = 'function' 
    
    # Second stage parameters
    CFG.separation_delay_s = float(coast_s)
    CFG.upper_ignition_delay_s = 2.0  # Re-inserted: Keep a small settling time

    # Construct upper stage throttle program
    upper_throttle_program = []
    current_time_ratio = 0.0
    upper_throttle_program.append([current_time_ratio * upper_burn_s, upper_throttle_levels[0]])
    
    for i in range(len(upper_throttle_switch_ratios)):
        switch_ratio = upper_throttle_switch_ratios[i]
        throttle_level = upper_throttle_levels[i+1]
        
        # Ensure switch ratios are distinct and increasing
        if switch_ratio > current_time_ratio:
            upper_throttle_program.append([switch_ratio * upper_burn_s, upper_throttle_levels[i]]) # Maintain previous throttle until switch time
            upper_throttle_program.append([switch_ratio * upper_burn_s + 1e-6, throttle_level]) # Small delta for switch
            current_time_ratio = switch_ratio
        else: # Handle cases where ratios might be too close or identical after sorting/clipping
            upper_throttle_program[-1][1] = throttle_level # Update previous throttle level if ratios are same
            
    # Ensure final segment lasts until upper_burn_s
    upper_throttle_program.append([upper_burn_s, upper_throttle_levels[-1]])
    # Add a cut-off
    upper_throttle_program.append([upper_burn_s + 1, 0.0])
    
    CFG.upper_stage_throttle_program = upper_throttle_program

    # Construct booster throttle program
    booster_throttle_program = []
    # Assuming booster starts at full throttle (or defined by first level) and we define relative switch times
    # Max duration for booster burn is roughly CFG.stage_1_burn_time_s (from rocket.py, often around 160s)
    # Let's use a proxy for booster burn duration for the ratios, e.g., 200 seconds.
    booster_burn_duration_proxy = 200.0 # A reasonable upper bound for booster burn time
    
    current_booster_time_ratio = 0.0
    booster_throttle_program.append([current_booster_time_ratio * booster_burn_duration_proxy, booster_throttle_levels[0]])
    
    for i in range(len(booster_throttle_switch_ratios)):
        switch_ratio = booster_throttle_switch_ratios[i]
        throttle_level = booster_throttle_levels[i+1]
        
        if switch_ratio > current_booster_time_ratio:
            booster_throttle_program.append([switch_ratio * booster_burn_duration_proxy, booster_throttle_levels[i]])
            booster_throttle_program.append([switch_ratio * booster_burn_duration_proxy + 1e-6, throttle_level])
            current_booster_time_ratio = switch_ratio
        else:
            booster_throttle_program[-1][1] = throttle_level
            
    # Ensure a final value is set, and it handles MECO internally
    booster_throttle_program.append([booster_burn_duration_proxy, booster_throttle_levels[-1]])
    booster_throttle_program.append([booster_burn_duration_proxy + 1, 0.0]) # Ensure it eventually cuts off
    
    CFG.booster_throttle_program = booster_throttle_program
    
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
        
        # Apply the parameterized throttle program for the booster
        booster_throttle_controller = ParameterizedThrottleProgram(
            schedule=CFG.booster_throttle_program)
        sim.rocket.booster_throttle_program = booster_throttle_controller
        
        # Create and apply pitch program
        pitch_program_points = [
            (pitch_times[0], pitch_angles_deg[0]),
            (pitch_times[1], pitch_angles_deg[1]),
            (pitch_times[2], pitch_angles_deg[2]),
            (pitch_times[3], pitch_angles_deg[3]),
            (pitch_times[4], pitch_angles_deg[4])
        ]
        sim.guidance.pitch_program = create_pitch_program_callable(pitch_program_points)
        
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


global_iter_count = 0

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

    last_cost_phase2 = results['cost']

    return results['cost']


def run_optimization():
    """Runs the two-phase optimization process."""
    global global_iter_count

    # --- Initial Guess & Bounds (SCALED) ---
    # Use km for altitudes to keep numbers in a similar magnitude
    # params: meco_mach, pitch_start_km, pitch_end_km, pitch_blend, coast_s, upper_burn_s
    global bounds # Make bounds global for soft_bounds_penalty
    bounds = [
        (2.0, 9.0),      # 0: MECO Mach
        # Pitch profile (5 points, time in seconds, angle in degrees relative to vertical)
        (0.0, 50.0),     # 1: pitch_time_0 (s)
        (70.0, 90.0),    # 2: pitch_angle_0 (deg, 90=vertical)
        (50.0, 150.0),   # 3: pitch_time_1 (s)
        (30.0, 80.0),    # 4: pitch_angle_1 (deg)
        (150.0, 300.0),  # 5: pitch_time_2 (s)
        (20.0, 60.0),    # 6: pitch_angle_2 (deg)
        (250.0, 450.0),  # 7: pitch_time_3 (s)
        (10.0, 40.0),    # 8: pitch_angle_3 (deg)
        (400.0, 600.0),  # 9: pitch_time_4 (s)
        (0.0, 20.0),     # 10: pitch_angle_4 (deg)
        # Staging and upper stage burn
        (10.0, 500.0),   # 11: Coast duration after MECO (s)
        (50.0, 400.0),   # 12: Upper stage burn duration (s)
        # Upper stage throttle profile (4 levels, 3 switch ratios)
        (0.5, 1.0),      # 13: upper_throttle_level_0 (0-1)
        (0.5, 1.0),      # 14: upper_throttle_level_1 (0-1)
        (0.5, 1.0),      # 15: upper_throttle_level_2 (0-1)
        (0.5, 1.0),      # 16: upper_throttle_level_3 (0-1)
        (0.1, 0.3),      # 17: upper_throttle_switch_ratio_0 (0-1, fraction of burn duration)
        (0.3, 0.7),      # 18: upper_throttle_switch_ratio_1 (0-1, fraction of burn duration)
        (0.7, 0.9),      # 19: upper_throttle_switch_ratio_2 (0-1, fraction of burn duration)
        # Booster throttle profile (4 levels, 3 switch ratios)
        (0.5, 1.0),      # 20: booster_throttle_level_0 (0-1)
        (0.5, 1.0),      # 21: booster_throttle_level_1 (0-1)
        (0.5, 1.0),      # 22: booster_throttle_level_2 (0-1)
        (0.5, 1.0),      # 23: booster_throttle_level_3 (0-1)
        (0.1, 0.3),      # 24: booster_throttle_switch_ratio_0 (0-1, fraction of booster burn duration)
        (0.3, 0.7),      # 25: booster_throttle_switch_ratio_1 (0-1, fraction of booster burn duration)
        (0.7, 0.9),      # 26: booster_throttle_switch_ratio_2 (0-1, fraction of booster burn duration)
    ]

    print(f"=== PHASE 1: TARGETING ORBIT (Logging to {LOG_FILENAME}) ===", flush=True)
    global_iter_count = 0
    print("DEBUG: About to call minimize for Phase 1", flush=True)
    res1 = differential_evolution(
        objective_phase1,
        bounds, # differential_evolution takes bounds as its second argument
        maxiter=500, # Max generations, increased for larger parameter space
        disp=True # Display convergence messages
    )

    print(f"\n--- Phase 1 Complete (Differential Evolution) ---", flush=True)
    print(f"Best Error: {res1.fun/1000:.1f} km", flush=True)
    final_params1 = res1.x
    print(f"Phase 1 Optimal Parameters (summary): Mach={final_params1[0]:.2f}, Coast={final_params1[11]:.1f}s, Upper Burn={final_params1[12]:.1f}s. Full details in {LOG_FILENAME}", flush=True)

    if res1.fun > 50000:  # If we're still more than 50km off, it's probably not a good solution
        print("\nFailed to find a stable orbit in Phase 1. Stopping.", flush=True)
        return

    print("\n=== PHASE 2: MINIMIZING FUEL (Differential Evolution) ===", flush=True)
    global_iter_count = 0  # Reset for phase 2 logging
    # Start phase 2 from the best point of phase 1 is not directly applicable for DE,
    # as DE does not take an initial guess. It will start its own population within the bounds.
    # We will use the 'bounds' as defined for the entire search space.

    res2 = differential_evolution(
        objective_phase2,
        bounds, # differential_evolution takes bounds as its second argument
        maxiter=500, # Max generations, increased for larger parameter space
        disp=True # Display convergence messages
    )

    print("\n=== OPTIMIZATION COMPLETE ===", flush=True)
    final_params2 = res2.x
    # Run one last time for final numbers
    final_results = run_simulation_wrapper(final_params2)

    print(f"Final Fuel Used: {final_results['fuel']:.1f} kg", flush=True)
    print(f"Final Orbit Error: {final_results['error']/1000:.1f} km", flush=True)
    print(f"Optimal Parameters (summary): Mach={final_params2[0]:.2f}, Coast={final_params2[11]:.1f}s, Upper Burn={final_params2[12]:.1f}s. Full details in {LOG_FILENAME}", flush=True)


if __name__ == "__main__":
    try:
        run_optimization()
    except Exception as e:
        print(f"ERROR: An unhandled exception occurred during optimization: {e}", flush=True)
        import traceback
        traceback.print_exc()
