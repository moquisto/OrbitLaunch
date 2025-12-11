"""
optimize_twostage.py

Solves the launch problem in two distinct phases for maximum efficiency:
Phase 1: "Targeting" - Find ANY parameters that hit the target orbit (Ignore fuel).
Phase 2: "Optimizing" - From that valid orbit, minimize fuel usage while staying in orbit.
"""

import numpy as np
import time
import os # Import os for cpu_count
from scipy.optimize import differential_evolution
from main import build_simulation, orbital_elements_from_state # Removed MU_EARTH, R_EARTH
from config import CFG

# --- Configuration (for optimizer) ---
TARGET_TOLERANCE = 2000.0 # We need to be within 2km to consider it "Orbit"

def run_simulation_wrapper(params, cfg_instance):
    """Helper to run sim and return (fuel_used, orbit_error_m)"""
    # Access CFG attributes directly here, they are guaranteed to be initialized when this function is called.
    current_target_alt_m = cfg_instance.target_orbit.target_orbit_alt_m
    current_mu_earth = cfg_instance.central_body.earth_mu
    current_r_earth = cfg_instance.central_body.earth_radius_m

    # Unpack Scaled Params
    meco_mach = params[0]
    p_start_m = params[1] * 1000.0
    p_end_m   = params[2] * 1000.0
    p_blend   = params[3]

    # New upper stage throttle parameters
    ap_factor = params[4]
    coast_s   = params[5]
    circ_thr  = params[6]
    first_thr = params[7]
    initial_pitch_deg = params[8] # New parameter for initial pitch
    separation_alt_m = params[9] * 1000.0 # New parameter for separation altitude

    # Update Config (use cfg_instance)
    cfg_instance.staging.meco_mach = float(meco_mach)
    cfg_instance.pitch_guidance.pitch_turn_start_m = float(p_start_m)
    cfg_instance.pitch_guidance.pitch_turn_end_m = float(p_end_m)
    cfg_instance.pitch_guidance.pitch_blend_exp = float(p_blend)
    cfg_instance.throttle_guidance.upper_stage_first_burn_target_ap_factor = float(ap_factor)
    cfg_instance.throttle_guidance.upper_stage_coast_duration_target_s = float(coast_s)
    cfg_instance.throttle_guidance.upper_stage_circ_burn_throttle_setpoint = float(circ_thr)
    cfg_instance.throttle_guidance.upper_stage_first_burn_throttle_setpoint = float(first_thr)
    cfg_instance.pitch_guidance.initial_pitch_deg = float(initial_pitch_deg) # Set the initial pitch angle
    cfg_instance.staging.separation_altitude_m = float(separation_alt_m) # Set separation altitude
    cfg_instance.orbit_tolerances.orbit_alt_tol = 100_000.0 # Tolerance for exit_on_orbit (hardcoded for sim exit)
    cfg_instance.orbit_tolerances.exit_on_orbit = True

    try:
        sim, state0, t0 = build_simulation(cfg_instance) # Pass cfg_instance
        log = sim.run(t0, duration=4000.0, dt=cfg_instance.simulation_timing.main_dt_s, state0=state0,
                      orbit_target_radius=current_r_earth + current_target_alt_m, # Use local vars
                      exit_on_orbit=cfg_instance.orbit_tolerances.exit_on_orbit)

        fuel_used = state0.m - log.m[-1]

        r, v = log.r[-1], log.v[-1]
        a, rp, ra = orbital_elements_from_state(r, v, current_mu_earth) # Use local vars

        if rp is None or ra is None:
            return fuel_used, 1e9 # Crash

        target_r = current_r_earth + current_target_alt_m # Use local vars
        error = abs(rp - target_r) + abs(ra - target_r)
        return fuel_used, error

    except Exception: # Catching all exceptions is broad, but for optimization robustness, it's safer.
        return 0.0, 1e9

# --- PHASE 1: TARGETING ---
def objective_phase1(params, _cfg_instance): # Accept _cfg_instance
    _, error = run_simulation_wrapper(params, _cfg_instance) # Pass _cfg_instance
    print(f"[Phase 1] Error: {error/1000:.1f} km")
    return error

# --- PHASE 2: OPTIMIZING ---
def objective_phase2(params, _cfg_instance): # Accept _cfg_instance
    fuel, error = run_simulation_wrapper(params, _cfg_instance) # Pass _cfg_instance
    
    # We add a "Wall" penalty: If error > TARGET_TOLERANCE, cost shoots up.
    # Otherwise, cost is just fuel.
    if error > TARGET_TOLERANCE:
        penalty = (error - TARGET_TOLERANCE) * 1000.0
        cost = fuel + penalty
    else:
        cost = fuel
        
    print(f"[Phase 2] Fuel: {fuel:.0f} kg | Error: {error/1000:.1f} km | Cost: {cost:.0f}")
    return cost

def run_optimization():
    # Initial Guess (No longer used by differential_evolution, but bounds are)
    # x0 = [3.5, 1.5, 85.0, 0.85, 1.05, 2500.0, 1.0, 1.0] 
    bounds = [(2.0, 9.0), (0.5, 10.0), (40.0, 150.0), (0.4, 1.5), (1.01, 1.2), (1000.0, 4000.0), (0.5, 1.0), (0.5, 1.0), (80.0, 90.0), (50.0, 120.0)]

    num_workers = os.cpu_count() - 1 if os.cpu_count() is not None and os.cpu_count() > 1 else 1

    print("=== PHASE 1: TARGETING ORBIT (using Differential Evolution) ===")
    res1 = differential_evolution(objective_phase1, bounds, args=(CFG,), maxiter=50, popsize=15, tol=0.01, workers=num_workers) # Using differential_evolution
    
    print(f"\nPhase 1 Complete. Best Error: {res1.fun/1000:.1f} km")
    print(f"Params: {res1.x}")
    
    # differential_evolution returns res.fun for the minimum value
    if res1.fun > 20000: # 20km error threshold for Phase 1 success 
        print("Failed to find stable orbit in Phase 1. Stopping.")
        return

    print("\n=== PHASE 2: MINIMIZING FUEL (using Differential Evolution) ===")
    # Start from the valid orbit we just found (differential_evolution doesn't use x0 directly, but we can pass it if we were using a local optimizer)
    # For differential_evolution, we rely on its population generation, but the result from res1.x is a good indication of a promising region.
    # However, differential_evolution requires bounds to be passed directly.
    res2 = differential_evolution(objective_phase2, bounds, args=(CFG,), maxiter=100, popsize=15, tol=0.001, workers=num_workers) # Using differential_evolution
    
    print("\n=== OPTIMIZATION COMPLETE ===")
    print(f"Final Fuel Used: {res2.fun:.1f} kg")
    print(f"Optimal Params: Mach={res2.x[0]:.2f}, Start={res2.x[1]*1000:.0f}m, End={res2.x[2]*1000:.0f}m, Blend={res2.x[3]:.2f}, ApFactor={res2.x[4]:.3f}, Coast={res2.x[5]:.0f}s, CircThr={res2.x[6]:.2f}, FirstThr={res2.x[7]:.2f}, Pitch0={res2.x[8]:.1f}deg, SepAlt={res2.x[9]*1000:.0f}m")

if __name__ == "__main__":
    run_optimization()

if __name__ == "__main__":
    run_optimization()
