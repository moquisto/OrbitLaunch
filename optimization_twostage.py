"""
optimize_twostage.py

Solves the launch problem in two distinct phases for maximum efficiency:
Phase 1: "Targeting" - Find ANY parameters that hit the target orbit (Ignore fuel).
Phase 2: "Optimizing" - From that valid orbit, minimize fuel usage while staying in orbit.
"""

import numpy as np
import time
from scipy.optimize import minimize
from main import build_simulation, orbital_elements_from_state, MU_EARTH, R_EARTH
from config import CFG

# --- Configuration ---
TARGET_ALT_M = 200_000.0   # 200 km
TARGET_TOLERANCE = 10000.0 # We need to be within 10km to consider it "Orbit"

def run_simulation_wrapper(params):
    """Helper to run sim and return (fuel_used, orbit_error_m)"""
    # Unpack Scaled Params
    meco_mach = params[0]
    p_start_m = params[1] * 1000.0
    p_end_m   = params[2] * 1000.0
    p_blend   = params[3]

    # Update Config
    CFG.meco_mach = float(meco_mach)
    CFG.pitch_turn_start_m = float(p_start_m)
    CFG.pitch_turn_end_m = float(p_end_m)
    CFG.pitch_blend_exp = float(p_blend)
    CFG.orbit_alt_tol = 1e6    
    CFG.exit_on_orbit = True 

    try:
        sim, state0, t0 = build_simulation()
        log = sim.run(t0, duration=4000.0, dt=1.0, state0=state0, 
                      orbit_target_radius=R_EARTH + TARGET_ALT_M,
                      exit_on_orbit=True)
        
        fuel_used = state0.m - log.m[-1]
        
        r, v = log.r[-1], log.v[-1]
        a, rp, ra = orbital_elements_from_state(r, v, MU_EARTH)
        
        if rp is None or ra is None:
            return fuel_used, 1e9 # Crash
            
        target_r = R_EARTH + TARGET_ALT_M
        error = abs(rp - target_r) + abs(ra - target_r)
        return fuel_used, error

    except Exception:
        return 0.0, 1e9

# --- PHASE 1: TARGETING ---
def objective_phase1(params):
    _, error = run_simulation_wrapper(params)
    print(f"[Phase 1] Error: {error/1000:.1f} km")
    return error

# --- PHASE 2: OPTIMIZING ---
def objective_phase2(params):
    fuel, error = run_simulation_wrapper(params)
    
    # We add a "Wall" penalty: If error > 10km, cost shoots up.
    # Otherwise, cost is just fuel.
    if error > TARGET_TOLERANCE:
        penalty = (error - TARGET_TOLERANCE) * 1000.0
        cost = fuel + penalty
    else:
        cost = fuel
        
    print(f"[Phase 2] Fuel: {fuel:.0f} kg | Error: {error/1000:.1f} km | Cost: {cost:.0f}")
    return cost

def run_optimization():
    # Initial Guess (Active Booster settings)
    x0 = [3.5, 1.5, 85.0, 0.85] 
    bounds = [(2.0, 9.0), (0.5, 10.0), (40.0, 150.0), (0.4, 1.5)]

    print("=== PHASE 1: TARGETING ORBIT ===")
    res1 = minimize(objective_phase1, x0, method='Nelder-Mead', bounds=bounds, options={'maxiter': 50, 'xatol': 0.1})
    
    print(f"\nPhase 1 Complete. Best Error: {res1.fun/1000:.1f} km")
    print(f"Params: {res1.x}")
    
    if res1.fun > 50000:
        print("Failed to find stable orbit in Phase 1. Stopping.")
        return

    print("\n=== PHASE 2: MINIMIZING FUEL ===")
    # Start from the valid orbit we just found
    x0_phase2 = res1.x
    
    res2 = minimize(objective_phase2, x0_phase2, method='Nelder-Mead', bounds=bounds, options={'maxiter': 100, 'xatol': 0.01})
    
    print("\n=== OPTIMIZATION COMPLETE ===")
    print(f"Final Fuel Used: {res2.fun:.1f} kg")
    print(f"Optimal Params: Mach={res2.x[0]:.2f}, Start={res2.x[1]*1000:.0f}m, End={res2.x[2]*1000:.0f}m")

if __name__ == "__main__":
    run_optimization()