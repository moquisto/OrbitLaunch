"""
optimize_twostage.py

Solves the launch problem in two distinct phases for maximum efficiency:
Phase 1: "Targeting" - Find ANY parameters that hit the target orbit (Ignore fuel).
Phase 2: "Optimizing" - From that valid orbit, minimize fuel usage while staying in orbit.
"""

import importlib
import numpy as np
import time
import os  # Import os for cpu_count
import copy  # Import copy for deepcopy
from scipy.optimize import differential_evolution
from main import ParameterizedThrottleProgram, build_simulation
from custom_guidance import orbital_elements_from_state
from config import CFG

# --- Configuration (for optimizer) ---
TARGET_TOLERANCE = 2000.0 # We need to be within 2km to consider it "Orbit"
# Faster evaluation settings to keep the optimizer responsive
EVAL_DT = 5.0
EVAL_DURATION = 300.0


def _install_throttle(sim, cfg_instance):
    """Match the throttle controller used in main() so optimization is consistent."""
    orbit_radius = cfg_instance.central_body.earth_radius_m + cfg_instance.target_orbit.target_orbit_alt_m
    if cfg_instance.throttle_guidance.throttle_guidance_mode == 'parameterized':
        controller = ParameterizedThrottleProgram(schedule=cfg_instance.throttle_guidance.upper_stage_throttle_program)
    elif cfg_instance.throttle_guidance.throttle_guidance_mode == 'function':
        module_name, class_name = cfg_instance.throttle_guidance.throttle_guidance_function_class.rsplit('.', 1)
        module = importlib.import_module(module_name)
        ControllerClass = getattr(module, class_name)
        try:
            controller = ControllerClass(target_radius=orbit_radius, mu=cfg_instance.central_body.earth_mu, cfg=cfg_instance)
        except TypeError:
            controller = ControllerClass(target_radius=orbit_radius, mu=cfg_instance.central_body.earth_mu)
    else:
        raise ValueError(f"Unknown throttle_guidance_mode: '{cfg_instance.throttle_guidance.throttle_guidance_mode}'")
    sim.guidance.throttle_schedule = controller
    return orbit_radius

def run_simulation_wrapper(params, cfg_instance):
    """Helper to run sim and return (fuel_used, orbit_error_m)

    Uses a fresh config copy per evaluation to avoid cross-contamination between candidates.
    """
    cfg_local = copy.deepcopy(cfg_instance)
    current_target_alt_m = cfg_local.target_orbit.target_orbit_alt_m
    current_mu_earth = cfg_local.central_body.earth_mu
    current_r_earth = cfg_local.central_body.earth_radius_m

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
    cfg_local.staging.meco_mach = float(meco_mach)
    cfg_local.pitch_guidance.pitch_turn_start_m = float(p_start_m)
    cfg_local.pitch_guidance.pitch_turn_end_m = float(p_end_m)
    cfg_local.pitch_guidance.pitch_blend_exp = float(p_blend)
    cfg_local.throttle_guidance.upper_stage_first_burn_target_ap_factor = float(ap_factor)
    cfg_local.throttle_guidance.upper_stage_coast_duration_target_s = float(coast_s)
    cfg_local.throttle_guidance.upper_stage_circ_burn_throttle_setpoint = float(circ_thr)
    cfg_local.throttle_guidance.upper_stage_first_burn_throttle_setpoint = float(first_thr)
    cfg_local.pitch_guidance.initial_pitch_deg = float(initial_pitch_deg) # Set the initial pitch angle
    cfg_local.staging.separation_altitude_m = float(separation_alt_m) # Set separation altitude
    cfg_local.orbit_tolerances.orbit_alt_tol = 100_000.0 # Tolerance for exit_on_orbit (hardcoded for sim exit)
    cfg_local.orbit_tolerances.exit_on_orbit = True

    try:
        sim, state0, t0 = build_simulation(cfg_local)
        orbit_radius = _install_throttle(sim, cfg_local)
        initial_prop = sum(stage.prop_mass for stage in sim.rocket.stages)
        # Use a coarser/faster sim for the optimizer to avoid stalls
        local_dt = max(cfg_local.simulation_timing.main_dt_s, EVAL_DT)
        local_duration = min(EVAL_DURATION, cfg_local.simulation_timing.main_duration_s)
        log = sim.run(
            t0,
            duration=local_duration,
            dt=local_dt,
            state0=state0,
            orbit_target_radius=current_r_earth + current_target_alt_m,
            exit_on_orbit=cfg_local.orbit_tolerances.exit_on_orbit,
        )

        fuel_used = initial_prop - sum(sim.rocket.stage_prop_remaining)

        r, v = log.r[-1], log.v[-1]
        a, rp, ra = orbital_elements_from_state(r, v, current_mu_earth) # Use local vars

        if rp is None or ra is None:
            return fuel_used, 1e9 # Crash

        target_r = current_r_earth + current_target_alt_m # Use local vars
        error = abs(rp - target_r) + abs(ra - target_r)
        return fuel_used, error

    except Exception: # Catching all exceptions is broad, but for optimization robustness, it's safer.
        return 0.0, 1e9

class OptimizationProgress:
    def __init__(self, phase_name, objective_func, cfg_instance, log_file):
        self.phase_name = phase_name
        self.iteration = 0
        self.objective_func = objective_func
        self.cfg_instance = cfg_instance
        self.log_file = log_file

    def report(self, xk, convergence=None):
        self.iteration += 1
        # Re-evaluate objective for the best vector (xk) for reporting
        current_objective_value = self.objective_func(xk, self.cfg_instance)
        self.log_file.write(f"[{self.phase_name}] Iteration {self.iteration}: Best Objective = {current_objective_value:.3f}\n")
        self.log_file.flush() # Ensure immediate write to file

# --- PHASE 1: TARGETING ---
def objective_phase1(params, _cfg_instance): # Accept _cfg_instance
    _, error = run_simulation_wrapper(params, _cfg_instance) # Pass _cfg_instance
    return error

# --- PHASE 2: OPTIMIZING ---
def objective_phase2(params, _cfg_instance): # Accept _cfg_instance
    fuel, error = run_simulation_wrapper(params, _cfg_instance)
    
    # We add a "Wall" penalty: If error > TARGET_TOLERANCE, cost shoots up.
    # Otherwise, cost is just fuel.
    if error > TARGET_TOLERANCE:
        penalty = (error - TARGET_TOLERANCE) * 1000.0
        cost = fuel + penalty
    else:
        cost = fuel
        
    return cost

def run_optimization():
    log_file_path = os.path.join(os.environ.get("GEMINI_TEMP_DIR", "."), "optimization_progress.log")
    with open(log_file_path, "w") as log_file:
        log_file.write("Starting optimization...\n")
        log_file.flush()

        # Check if guidance modes are set correctly for optimization
        if CFG.pitch_guidance.pitch_guidance_mode != "function":
            log_file.write(f"WARNING: Pitch guidance mode is '{CFG.pitch_guidance.pitch_guidance_mode}'. "
                          "Optimization parameters for pitch guidance will NOT be used unless set to 'function'.\n")
        if CFG.throttle_guidance.throttle_guidance_mode != "function":
            log_file.write(f"WARNING: Throttle guidance mode is '{CFG.throttle_guidance.throttle_guidance_mode}'. "
                          "Optimization parameters for upper stage throttle will NOT be used unless set to 'function'.\n")
        log_file.flush()
        
        # Bounds for differential evolution
        bounds = [(2.0, 9.0), (0.5, 10.0), (40.0, 150.0), (0.4, 1.5), (1.01, 1.2), (1000.0, 4000.0), (0.5, 1.0), (0.5, 1.0), (80.0, 90.0), (50.0, 120.0)]

        # Choose worker count conservatively to avoid multiprocessing hangs; single-threaded is safest here.
        num_workers = 1

        # Create a deep copy of CFG to pass to worker processes to avoid multiprocessing global state issues
        _deep_copied_cfg = copy.deepcopy(CFG)
        
        phase1_progress = OptimizationProgress('Phase 1', objective_phase1, _deep_copied_cfg, log_file)
        log_file.write("=== PHASE 1: TARGETING ORBIT (using Differential Evolution) ===\n")
        log_file.flush()
        res1 = differential_evolution(
            objective_phase1,
            bounds,
            args=(_deep_copied_cfg,),
            maxiter=5,
            popsize=6,
            tol=0.1,
            workers=num_workers,
            updating="deferred",
            polish=False,
            callback=phase1_progress.report,
        )
        
        log_file.write(f"\nPhase 1 Complete. Best Error: {res1.fun/1000:.1f} km\n")
        log_file.write(f"Params: {res1.x}\n")
        log_file.flush()
        
        # differential_evolution returns res.fun for the minimum value
        if res1.fun > 20000: # 20km error threshold for Phase 1 success 
            log_file.write("Failed to find stable orbit in Phase 1. Stopping.\n")
            log_file.flush()
            return

        phase2_progress = OptimizationProgress('Phase 2', objective_phase2, _deep_copied_cfg, log_file)
        log_file.write("\n=== PHASE 2: MINIMIZING FUEL (using Differential Evolution) ===\n")
        log_file.flush()
        # Start from the valid orbit we just found (differential_evolution doesn't use x0 directly, but we can pass it if we were using a local optimizer)
        # For differential_evolution, we rely on its population generation, but the result from res1.x is a good indication of a promising region.
        # However, differential_evolution requires bounds to be passed directly.
        res2 = differential_evolution(
            objective_phase2,
            bounds,
            args=(_deep_copied_cfg,),
            maxiter=6,
            popsize=6,
            tol=0.05,
            workers=num_workers,
            updating="deferred",
            polish=False,
            callback=phase2_progress.report,
        )
        
        log_file.write("\n=== OPTIMIZATION COMPLETE ===\n")
        log_file.write(f"Final Fuel Used: {res2.fun:.1f} kg\n")
        log_file.write(f"Optimal Params: Mach={res2.x[0]:.2f}, Start={res2.x[1]*1000:.0f}m, End={res2.x[2]*1000:.0f}m, Blend={res2.x[3]:.2f}, ApFactor={res2.x[4]:.3f}, Coast={res2.x[5]:.0f}s, CircThr={res2.x[6]:.2f}, FirstThr={res2.x[7]:.2f}, Pitch0={res2.x[8]:.1f}deg, SepAlt={res2.x[9]*1000:.0f}m\n")
        log_file.flush()
    
    print(f"\nOptimization complete. Progress and results saved to: {log_file_path}")

if __name__ == "__main__":
    run_optimization()
