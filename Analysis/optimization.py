try:
    import cma
    CMA_AVAILABLE = True
except Exception:
    CMA_AVAILABLE = False

"""
optimization_twostage.py

Solves the launch problem in two distinct phases for maximum efficiency:
Phase 1: "Targeting" - Find ANY parameters that hit the target orbit (Ignore fuel).
Phase 2: "Optimizing" - From that valid orbit, minimize fuel usage while staying in orbit.

This version incorporates input scaling, coarse-to-fine simulation, and detailed logging.
"""
import multiprocessing
import copy
import traceback
import numpy as np
from scipy.optimize import differential_evolution

# New imports
from main import main_orchestrator
from Environment.gravity import orbital_elements_from_state
from Environment.config import EnvironmentConfig
from Hardware.config import HardwareConfig
from Software.config import SoftwareConfig
from Main.config import SimulationConfig
from Logging.config import LoggingConfig
from Analysis.config import AnalysisConfig, OptimizationParams, OptimizationBounds # Import OptimizationBounds
from Software.guidance import create_pitch_program_callable, ParameterizedThrottleProgram, configure_software_for_optimization
from Logging.generate_logs import log_iteration, ensure_log_header, LOG_FILENAME # Import from logging module
from Analysis.cost_functions import evaluate_simulation_results, PENALTY_CRASH, TARGET_TOLERANCE_M # Import new function, PENALTY_CRASH, and TARGET_TOLERANCE_M

print("optimization_twostage.py script started.", flush=True)

# Shared counter for iteration tracking across processes
global_iter_count = None

def init_worker(shared_counter):
    """Initializer for worker processes to share the iteration counter."""
    global global_iter_count
    global_iter_count = shared_counter

class ObjectiveFunctionWrapper:
    def __init__(self, phase, env_config, hw_config, sw_config, sim_config, log_config, analysis_config, bounds):
        self.phase = phase
        self.env_config = env_config
        self.hw_config = hw_config
        self.sw_config = sw_config
        self.sim_config = sim_config
        self.log_config = log_config
        self.analysis_config = analysis_config
        self.bounds = bounds

    def __call__(self, scaled_params: np.ndarray):
        global global_iter_count
        
        # Increment shared counter safely
        current_iter = 0
        if global_iter_count is not None:
            with global_iter_count.get_lock():
                global_iter_count.value += 1
                current_iter = global_iter_count.value
        
        params_obj = OptimizationParams(*scaled_params)
        results = run_simulation_wrapper(
            params_obj,
            self.env_config,
            self.hw_config,
            self.sw_config,
            self.sim_config,
            self.log_config
        )
        
        if self.phase == 1:
            results['cost'] = results['error']
            bound_penalty = soft_bounds_penalty(scaled_params, self.bounds)
            results['cost'] += bound_penalty

            log_iteration("Phase 1", current_iter, params_obj, results)
            print(
                f"[Phase 1] Iter {current_iter:3d} | Error: {results['error']/1000:.1f} km | Status: {results['status']}", flush=True)

            return results['cost']
        else:
            fuel, error = results["fuel"], results["error"]
            if error > TARGET_TOLERANCE_M: # Uses the TARGET_TOLERANCE_M constant
                penalty = (error - TARGET_TOLERANCE_M) * 10.0
                cost = fuel + penalty
            else:
                cost = fuel
            
            bound_penalty = soft_bounds_penalty(scaled_params, self.bounds)
            cost += bound_penalty
            results['cost'] = cost

            log_iteration("Phase 2", current_iter, params_obj, results)
            print(f"[Phase 2] Iter {current_iter:3d} | Fuel: {fuel:.0f} kg | Error: {error/1000:.1f} km | Cost: {cost:.0f} | Status: {results['status']}", flush=True)

            return cost



def _evaluate_candidate(args):
    """Helper for multiprocessing pool; keeps objective picklable and guarded."""
    objective_fn_wrapper, cand = args
    try:
        return float(objective_fn_wrapper(np.array(cand, dtype=float)))
    except Exception as exc:  # pragma: no cover - defensive logging for worker issues
        print(f"[CMA worker] candidate failed: {exc}", flush=True)
        return float(PENALTY_CRASH) # Ensure return type is float

def run_simulation_wrapper(params: OptimizationParams, env_config: EnvironmentConfig, hw_config: HardwareConfig, sw_config: SoftwareConfig, sim_config: SimulationConfig, log_config: LoggingConfig):
    """
    Runs the simulation with a structured parameter object.
    This function de-scales parameters into real physics units for the simulation
    and returns a dictionary with detailed results.
    """
    if not isinstance(params, OptimizationParams):
        params = OptimizationParams(*params)
    
    # Create a deep copy of configs to avoid modifying the originals
    # These configs will be passed to main_orchestrator as the base
    # and then modified by the optimization parameters
    cfg_env = copy.deepcopy(env_config)
    cfg_hw = copy.deepcopy(hw_config)
    cfg_sw = copy.deepcopy(sw_config)
    cfg_sim = copy.deepcopy(sim_config)
    cfg_log = copy.deepcopy(log_config)

    cfg_sw, cfg_sim = configure_software_for_optimization(params, cfg_sw, cfg_sim, cfg_env)

    # Initialize results with a default "CRASH" status in case the simulation fails early
    results = {"fuel": 0.0, "error": PENALTY_CRASH, "status": "INIT", "cost": PENALTY_CRASH}
    
    try:
        sim, state0, t0, _log_config, _analysis_config = main_orchestrator(
            env_config=cfg_env,
            hw_config=cfg_hw,
            sw_config=cfg_sw,
            sim_config=cfg_sim,
            log_config=cfg_log,
        )
        initial_mass = state0.m # Capture initial mass after orchestration

        # Run full simulation
        log = sim.run(t0, duration=10000.0, dt=1.0, state0=state0)
        max_altitude = max(log.altitude) if log.altitude else 0.0 # Get max altitude for evaluation

        results = evaluate_simulation_results(log, initial_mass, cfg_env, cfg_sim, max_altitude)

    except IndexError:
        results["status"] = "SIM_FAIL_INDEX"
        results["error"] = PENALTY_CRASH
    except Exception:
        results["status"] = "SIM_FAIL_UNKNOWN"
        results["error"] = PENALTY_CRASH
        print(f"Simulation wrapper encountered an unexpected error: {traceback.format_exc()}", flush=True) # Added more detailed error logging
    finally:
        results["orbit_error"] = results.get("error", PENALTY_CRASH)

    return results

def soft_bounds_penalty(scaled_params, bounds):
    penalty = 0.0
    for i, param in enumerate(scaled_params):
        lower, upper = bounds[i]
        if param < lower:
            penalty += (lower - param) * 1e5  # Large penalty for going below lower bound
        elif param > upper:
            penalty += (param - upper) * 1e5  # Large penalty for going above upper bound
    return penalty

def run_cma_phase(objective_fn_wrapper, bounds, shared_counter, start=None, sigma_scale=0.2, maxiter=200, popsize=None):
    """
    Run a CMA-ES loop for a given objective and bounds. Returns the cma result object.
    """
    if not CMA_AVAILABLE:
        raise RuntimeError("CMA-ES requested but `cma` package is not available.")

    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    if start is None:
        start = (lb + ub) / 2.0
    else:
        start = np.clip(np.array(start, dtype=float), lb, ub)
    sigma0 = sigma_scale * float(np.mean(ub - lb))

    # Keep popsize modest for speed; users can override via argument.
    default_popsize = 8 + int(1.5 * np.log(len(lb)))
    opts = {
        "bounds": [lb.tolist(), ub.tolist()],
        "maxiter": maxiter,
        "popsize": popsize or default_popsize,
        "verb_disp": 1,
        "CMA_diagonal": True,  # faster early iterations
    }
    es = cma.CMAEvolutionStrategy(start.tolist(), sigma0, opts)

    with multiprocessing.Pool(initializer=init_worker, initargs=(shared_counter,)) as pool:
        while not es.stop():
            candidates = es.ask()
            work = [(objective_fn_wrapper, cand) for cand in candidates]
            costs = pool.map(_evaluate_candidate, work)
            gen_best = float(np.min(costs)) if costs else np.inf
            print(f"[CMA] gen {es.countiter:3d} | best this gen: {gen_best:.2f}", flush=True)
            es.tell(candidates, costs)
    return es.result


def run_optimization():
    """Runs the two-phase optimization process."""
    env_config = EnvironmentConfig()
    hw_config = HardwareConfig()
    sw_config = SoftwareConfig()
    sim_config = SimulationConfig()
    log_config = LoggingConfig()
    analysis_config = AnalysisConfig()

    # Initialize shared counter
    global global_iter_count
    global_iter_count = multiprocessing.Value('i', 0)

    # --- Initial Guess & Bounds (SCALED) ---
    # The bounds are now managed centrally in Analysis/config.py
    bounds = OptimizationBounds.get_bounds()

    def tighten_bounds(bounds_in, seed, margin=0.2):
        tightened = []
        for (lo, hi), s in zip(bounds_in, seed):
            width = hi - lo
            new_lo = max(lo, s - margin * width)
            new_hi = min(hi, s + margin * width)
            if new_lo > new_hi:
                new_lo, new_hi = lo, hi
            tightened.append((new_lo, new_hi))
        return tightened

    # Build start params either from manual seed or config-derived defaults.
    if analysis_config.optimizer_manual_seed and len(analysis_config.optimizer_manual_seed) == 35:
        start_params = np.array(analysis_config.optimizer_manual_seed, dtype=float)
        bounds = tighten_bounds(bounds, start_params, margin=0.2)
    else:
        start_params = (
            np.array([b[0] for b in bounds], dtype=float) + np.array([b[1] for b in bounds], dtype=float)
        ) / 2.0

    ensure_log_header()
    print(f"=== PHASE 1: TARGETING ORBIT (Logging to {LOG_FILENAME}) ===", flush=True)
    with global_iter_count.get_lock():
        global_iter_count.value = 0

    # Ensure seed respects the active bounds to avoid CMA boundary errors.
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    start_params = np.clip(start_params, lb, ub)
    
    objective_phase1 = ObjectiveFunctionWrapper(
        phase=1,
        env_config=env_config,
        hw_config=hw_config,
        sw_config=sw_config,
        sim_config=sim_config,
        log_config=log_config,
        analysis_config=analysis_config,
        bounds=bounds
    )

    if CMA_AVAILABLE:
        print("Using CMA-ES for Phase 1", flush=True)
        res1 = run_cma_phase(objective_phase1, bounds, global_iter_count, start=start_params, sigma_scale=0.15, maxiter=120, popsize=14)
        best1 = res1.xbest
        best1_cost = res1.fbest
    else:
        print("CMA-ES not available, falling back to Differential Evolution for Phase 1", flush=True)
        res = differential_evolution(
            objective_phase1,
            bounds,
            maxiter=300,
            disp=True
        )
        best1 = res.x
        best1_cost = res.fun

    print(f"\n--- Phase 1 Complete ---", flush=True)
    print(f"Best Error/Cost: {best1_cost/1000:.1f} km", flush=True)
    print(f"Phase 1 Optimal Parameters (summary): Mach={best1[0]:.2f}, Coast={best1[11]:.1f}s, Upper Burn={best1[12]:.1f}s. Full details in {LOG_FILENAME}", flush=True)

    if best1_cost > 50000:  # If we're still more than 50km off, it's probably not a good solution
        print("\nFailed to find a stable orbit in Phase 1. Stopping.", flush=True)
        return

    print("\n=== PHASE 2: MINIMIZING FUEL (CMA-ES if available) ===", flush=True)
    with global_iter_count.get_lock():
        global_iter_count.value = 0  # Reset for phase 2 logging

    objective_phase2 = ObjectiveFunctionWrapper(
        phase=2,
        env_config=env_config,
        hw_config=hw_config,
        sw_config=sw_config,
        sim_config=sim_config,
        log_config=log_config,
        analysis_config=analysis_config,
        bounds=bounds
    )

    if CMA_AVAILABLE:
        # Seed phase 2 from phase 1 best with a smaller sigma to focus search.
        res2 = run_cma_phase(
            objective_phase2,
            bounds,
            global_iter_count,
            start=best1,
            sigma_scale=0.1,
            maxiter=120,
            popsize=14
        )
        best2 = res2.xbest
        best2_cost = res2.fbest
    else:
        res = differential_evolution(
            objective_phase2,
            bounds,
            maxiter=300,
            disp=True
        )
        best2 = res.x
        best2_cost = res.fun

    print("\n=== OPTIMIZATION COMPLETE ===", flush=True)
    final_params2 = best2
    # Run one last time for final numbers
    final_results = run_simulation_wrapper(
        final_params2,
        env_config,
        hw_config,
        sw_config,
        sim_config,
        log_config
    )

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
