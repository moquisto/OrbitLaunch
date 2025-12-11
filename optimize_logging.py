"""
optimize_logging.py (Fixed & Optimized)

Implements Direct Shooting optimization with:
1. Input Scaling (Normalizes parameters so the optimizer works efficiently).
2. "Active Control" Initialization (Starts with a feasible MECO Mach).
3. Detailed CSV Logging of real-world physics values.
"""

import csv
import numpy as np
import time
from scipy.optimize import minimize
from main import build_simulation, orbital_elements_from_state, MU_EARTH, R_EARTH
from config import CFG

# --- Configuration ---
TARGET_ALT_M = 420_000.0   # 420 km
LOG_FILENAME = "optimization_log.csv"

# Cost Weights
WEIGHT_FUEL = 1.0          # 1 kg fuel = 1 cost
WEIGHT_ERROR = 100.0       # 1 m error = 100 cost (High priority on accuracy)
PENALTY_CRASH = 1e9        # "Soft Wall" for failed orbits

# --- Initialize CSV ---
# We overwrite the file on start to ensure a clean log
with open(LOG_FILENAME, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "iteration", "meco_mach", "pitch_start_m", "pitch_end_m", 
        "pitch_blend", "cost", "fuel_used_kg", "orbit_error_m", "status"
    ])

global_iter_count = 0

def objective_function(scaled_params):
    """
    The optimizer sees 'scaled_params' (small numbers).
    We convert them to 'physics_params' (real units) for the simulation.
    """
    global global_iter_count
    global_iter_count += 1
    
    # 1. DE-SCALE PARAMETERS
    # Optimizer sees:  [3.5,  1.5,    85.0,     0.85]
    # Physics uses:    [3.5,  1500.0, 85000.0,  0.85]
    meco_mach = scaled_params[0]
    p_start_m = scaled_params[1] * 1000.0  # Scale km -> m
    p_end_m   = scaled_params[2] * 1000.0  # Scale km -> m
    p_blend   = scaled_params[3]

    # 2. UPDATE CONFIGURATION
    CFG.meco_mach = float(meco_mach)
    CFG.pitch_turn_start_m = float(p_start_m)
    CFG.pitch_turn_end_m = float(p_end_m)
    CFG.pitch_blend_exp = float(p_blend)
    
    # Reset tolerances for the "Shot"
    CFG.orbit_alt_tol = 1e6    
    CFG.exit_on_orbit = True 

    # 3. RUN SIMULATION
    cost = 0.0
    fuel_used = 0.0
    orbit_error = 0.0
    status = "INIT"

    try:
        sim, state0, t0 = build_simulation()
        initial_mass = state0.m
        
        # Run sim
        log = sim.run(t0, duration=4000.0, dt=1.0, state0=state0, 
                      orbit_target_radius=R_EARTH + TARGET_ALT_M,
                      exit_on_orbit=True)
        
        # 4. CALCULATE RESULT
        fuel_used = initial_mass - log.m[-1]
        
        r_final = log.r[-1]
        v_final = log.v[-1]
        a, rp, ra = orbital_elements_from_state(r_final, v_final, MU_EARTH)
        target_r = R_EARTH + TARGET_ALT_M

        if rp is None or ra is None:
            # Crash or Hyperbolic
            cost = PENALTY_CRASH
            status = "CRASH"
            orbit_error = np.nan
        else:
            # Error = deviation from circular target altitude
            orbit_error = abs(rp - target_r) + abs(ra - target_r)
            
            cost = (fuel_used * WEIGHT_FUEL) + (orbit_error * WEIGHT_ERROR)
            
            # Simple status check
            if orbit_error < 5000: status = "PERFECT"
            elif orbit_error < 50000: status = "GOOD"
            else: status = "OK"

    except Exception as e:
        print(f"Simulation Error: {e}")
        cost = PENALTY_CRASH
        status = "ERROR"

    # 5. LOGGING (Log REAL values, not scaled ones)
    with open(LOG_FILENAME, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            global_iter_count, 
            f"{meco_mach:.4f}", 
            f"{p_start_m:.1f}", 
            f"{p_end_m:.1f}", 
            f"{p_blend:.3f}", 
            f"{cost:.2f}", 
            f"{fuel_used:.2f}", 
            f"{orbit_error:.2f}", 
            status
        ])

    # Console Output
    print(f"Iter {global_iter_count:3d} | Mach {meco_mach:.2f} | Start {p_start_m:5.0f}m | End {p_end_m/1000:3.0f}km | Cost {cost:.2e} | {status}")

    return cost

def run_optimization():
    print(f"Starting optimization with SCALED inputs...")
    print(f"Logging to: {LOG_FILENAME}")
    
    # --- Initial Guess (SCALED) ---
    # Mach 3.5: Ensures booster is NOT empty, giving the optimizer control
    # Start 1.5: Represents 1500m
    # End 85.0: Represents 85000m
    x0 = [3.5, 1.5, 85.0, 0.85] 

    # --- Bounds (SCALED) ---
    bounds = [
        (2.0, 9.0),      # Mach
        (0.5, 10.0),     # Start (500m - 10km)
        (40.0, 150.0),   # End (40km - 150km)
        (0.4, 1.5)       # Blend
    ]

    # --- Run Optimizer ---
    # Nelder-Mead is used because the simulation result is slightly "bumpy" 
    # (due to discrete time steps), which can confuse gradient-based solvers.
    res = minimize(
        objective_function, 
        x0, 
        method='Nelder-Mead', 
        bounds=bounds, 
        options={'maxiter': 150, 'xatol': 1e-2, 'fatol': 1.0}
    )

    print("\n" + "="*60)
    print(f"Optimization Finished. Success: {res.success}")
    print(f"Best SCALED Params: {res.x}")
    
    # Print Real-World Best
    real_best = [res.x[0], res.x[1]*1000, res.x[2]*1000, res.x[3]]
    print(f"Best REAL Params: Mach={real_best[0]:.2f}, Start={real_best[1]:.0f}m, End={real_best[2]:.0f}m, Blend={real_best[3]:.2f}")

if __name__ == "__main__":
    run_optimization()