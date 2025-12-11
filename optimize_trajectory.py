"""
optimize_trajectory.py

Implements a Direct Shooting Method to find the optimal launch parameters
for a specific target orbit, minimizing propellant usage.

Uses scipy.optimize.minimize (Nelder-Mead) to search the 4-dimensional
parameter space:
1. MECO Mach
2. Pitch Start Altitude
3. Pitch End Altitude
4. Pitch Blend Exponent
"""

import importlib
import numpy as np
import time
from scipy.optimize import minimize
from dataclasses import dataclass

# Import your existing environment
from gravity import MU_EARTH, R_EARTH
from main import ParameterizedThrottleProgram, build_simulation, orbital_elements_from_state
from config import CFG

# --- Optimization Configuration ---
TARGET_ALT_M = 200_000.0   # 200 km
LAUNCH_LAT = 25.997        # Boca Chica
LAUNCH_LON = -97.155

# Weights for the Cost Function
WEIGHT_FUEL = 1.0          # Cost per kg of fuel
WEIGHT_ERROR = 100.0       # Cost per meter of orbit error (High accuracy priority)
PENALTY_CRASH = 1e9        # Massive penalty for non-orbit trajectories

# Global counter for progress tracking
ITERATION_COUNT = 0

def objective_function(params):
    """
    The 'Black Box' function.
    Input: List of parameters [meco_mach, pitch_start, pitch_end, pitch_blend]
    Output: Single cost value (float)
    """
    global ITERATION_COUNT
    ITERATION_COUNT += 1

    # 1. Unpack Parameters
    meco_mach, p_start, p_end, p_blend = params

    # 2. Update Configuration (The "Knobs")
    CFG.launch_lat_deg = LAUNCH_LAT
    CFG.launch_lon_deg = LAUNCH_LON
    CFG.target_orbit_alt_m = TARGET_ALT_M
    
    CFG.meco_mach = float(meco_mach)
    CFG.pitch_turn_start_m = float(p_start)
    CFG.pitch_turn_end_m = float(p_end)
    CFG.pitch_blend_exp = float(p_blend)

    # Relax tolerances so the sim doesn't "exit early" on a bad orbit, 
    # allowing us to calculate the error magnitude for the optimizer.
    CFG.orbit_alt_tol = 1e6    
    CFG.exit_on_orbit = True 

    # 3. Run Simulation (The "Shot")
    try:
        sim, state0, t0 = build_simulation(CFG)
        orbit_radius = CFG.earth_radius_m + TARGET_ALT_M
        if CFG.throttle_guidance_mode == 'parameterized':
            controller = ParameterizedThrottleProgram(schedule=CFG.upper_stage_throttle_program)
        elif CFG.throttle_guidance_mode == 'function':
            module_name, class_name = CFG.throttle_guidance_function_class.rsplit('.', 1)
            module = importlib.import_module(module_name)
            ControllerClass = getattr(module, class_name)
            controller = ControllerClass(target_radius=orbit_radius, mu=CFG.earth_mu)
        else:
            raise ValueError(f"Unknown throttle_guidance_mode: '{CFG.throttle_guidance_mode}'")
        sim.guidance.throttle_schedule = controller
        initial_prop = sum(stage.prop_mass for stage in sim.rocket.stages)
        
        # Run with a generous timeout
        log = sim.run(t0, duration=4000.0, dt=1.0, state0=state0, 
                      orbit_target_radius=R_EARTH + TARGET_ALT_M,
                      exit_on_orbit=True) # It will exit if it hits the loose tolerance
    except Exception as e:
        print(f"Sim Failed: {e}")
        return PENALTY_CRASH

    # 4. Calculate Costs
    
    # Fuel Cost
    fuel_used = initial_prop - sum(sim.rocket.stage_prop_remaining)
    
    # Orbit Error Cost
    # We want a circular orbit at TARGET_ALT_M
    r_final = log.r[-1]
    v_final = log.v[-1]
    a, rp, ra = orbital_elements_from_state(r_final, v_final, MU_EARTH)
    
    target_r = R_EARTH + TARGET_ALT_M

    if rp is None or ra is None:
        # Crash or Hyperbolic - High Penalty
        cost = PENALTY_CRASH
        status = "CRASH"
    else:
        # Calculate deviation from circular target
        # Error = |Perigee - Target| + |Apogee - Target|
        rp_error = abs(rp - target_r)
        ra_error = abs(ra - target_r)
        total_error_m = rp_error + ra_error
        
        cost = (fuel_used * WEIGHT_FUEL) + (total_error_m * WEIGHT_ERROR)
        status = f"Orbit {rp_error/1000:.1f}km/{ra_error/1000:.1f}km off"

    # Optional: Print progress every 10 iterations
    if ITERATION_COUNT % 5 == 0:
        print(f"Iter {ITERATION_COUNT:3d} | Mach {meco_mach:.2f} | Start {p_start:.0f} | End {p_end:.0f} | Cost {cost:.2e} | {status}")

    return cost


def run_optimization():
    print(f"=== Trajectory Optimization (Direct Shooting) ===")
    print(f"Target: {TARGET_ALT_M/1000:.1f} km Circular Orbit")
    print(f"Optimizer: Nelder-Mead")
    print("-" * 60)

    # --- Initial Guess ---
    # Based on your previous manual tests, start with something "sane"
    x0 = [
        5.5,      # MECO Mach
        1500.0,   # Pitch Start (m)
        85000.0,  # Pitch End (m)
        0.85      # Blend Exponent
    ]

    # --- Bounds (Constraints) ---
    # format: (min, max) for each parameter
    bounds = [
        (3.0, 9.0),         # MECO Mach
        (500.0, 10000.0),   # Pitch Start
        (40000.0, 150000.0),# Pitch End
        (0.4, 2.0)          # Blend Exp
    ]

    # --- Run Optimizer ---
    start_time = time.time()
    
    # We use Nelder-Mead because the landscape is slightly "bumpy" due to time steps
    res = minimize(
        objective_function, 
        x0, 
        method='Nelder-Mead', 
        bounds=bounds, 
        options={'maxiter': 100, 'disp': True, 'xatol': 1e-2}
    )

    end_time = time.time()

    # --- Report Results ---
    print("\n" + "="*60)
    if res.success:
        print("OPTIMIZATION SUCCESSFUL")
    else:
        print("OPTIMIZATION TERMINATED (Check if converged)")
    
    print(f"Time Elapsed: {end_time - start_time:.1f} seconds")
    print(f"Iterations: {res.nit}")
    print("-" * 60)
    print(f"Best Parameters:")
    print(f"  MECO Mach       : {res.x[0]:.4f}")
    print(f"  Pitch Start     : {res.x[1]:.1f} m")
    print(f"  Pitch End       : {res.x[2]:.1f} m")
    print(f"  Pitch Blend     : {res.x[3]:.3f}")
    print("-" * 60)
    
    # --- Final Verification Run ---
    print("\nRunning Verification Simulation with Optimal Parameters...")
    verify_solution(res.x)

def verify_solution(best_params):
    """Runs one final simulation with detailed logging using the best found parameters."""
    CFG.meco_mach = best_params[0]
    CFG.pitch_turn_start_m = best_params[1]
    CFG.pitch_turn_end_m = best_params[2]
    CFG.pitch_blend_exp = best_params[3]
    
    # Use stricter tolerance for verification to see if we ACTUALLY hit it
    CFG.orbit_alt_tol = 5000.0 # 5km tolerance
    CFG.exit_on_orbit = True

    sim, state0, t0 = build_simulation(CFG)
    orbit_radius = CFG.earth_radius_m + TARGET_ALT_M
    if CFG.throttle_guidance_mode == 'parameterized':
        controller = ParameterizedThrottleProgram(schedule=CFG.upper_stage_throttle_program)
    elif CFG.throttle_guidance_mode == 'function':
        module_name, class_name = CFG.throttle_guidance_function_class.rsplit('.', 1)
        module = importlib.import_module(module_name)
        ControllerClass = getattr(module, class_name)
        controller = ControllerClass(target_radius=orbit_radius, mu=CFG.earth_mu)
    else:
        raise ValueError(f"Unknown throttle_guidance_mode: '{CFG.throttle_guidance_mode}'")
    sim.guidance.throttle_schedule = controller
    initial_prop = sum(stage.prop_mass for stage in sim.rocket.stages)
    log = sim.run(t0, duration=5000, dt=0.5, state0=state0, 
                  orbit_target_radius=R_EARTH + TARGET_ALT_M)

    fuel_used = initial_prop - sum(sim.rocket.stage_prop_remaining)
    r = log.r[-1]
    v = log.v[-1]
    a, rp, ra = orbital_elements_from_state(r, v, MU_EARTH)
    
    print(f"Fuel Used       : {fuel_used:.1f} kg")
    if rp:
        print(f"Final Perigee   : {(rp - R_EARTH)/1000:.2f} km")
        print(f"Final Apogee    : {(ra - R_EARTH)/1000:.2f} km")
        print(f"Orbit Error     : {(abs(rp-(R_EARTH+TARGET_ALT_M)) + abs(ra-(R_EARTH+TARGET_ALT_M)))/1000:.2f} km")
    else:
        print("Final Result    : No Orbit")

if __name__ == "__main__":
    run_optimization()
