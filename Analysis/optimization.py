# Last modified: 2025-12-13 11:54:18.000000
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
import sys
from pathlib import Path

# Allow running this file directly (e.g. `python3 Analysis/optimization.py`) by
# ensuring the repo root is on sys.path so `import main` works.
if __package__ in (None, ""):
    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

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
from Analysis.cost_functions import evaluate_simulation_results, PENALTY_CRASH # Import new function and PENALTY_CRASH

# Shared counter for iteration tracking across processes
global_iter_count = None
global_log_lock = None

def init_worker(shared_counter, log_lock):
    """Initializer for worker processes to share global state safely."""
    global global_iter_count, global_log_lock
    global_iter_count = shared_counter
    global_log_lock = log_lock

class ObjectiveFunctionWrapper:
    def __init__(
        self,
        phase,
        env_config,
        hw_config,
        sw_config,
        sim_config,
        log_config,
        analysis_config,
        bounds,
        *,
        param_space: str = "physical",
    ):
        self.phase = phase
        self.env_config = env_config
        self.hw_config = hw_config
        self.sw_config = sw_config
        self.sim_config = sim_config
        self.log_config = log_config
        self.analysis_config = analysis_config
        self.bounds = bounds
        self.param_space = param_space
        self._lb = np.array([b[0] for b in bounds], dtype=float)
        self._ub = np.array([b[1] for b in bounds], dtype=float)
        self._span = self._ub - self._lb

    def __call__(self, scaled_params: np.ndarray):
        global global_iter_count, global_log_lock
        
        # Increment shared counter safely
        current_iter = 0
        if global_iter_count is not None:
            with global_iter_count.get_lock():
                global_iter_count.value += 1
                current_iter = global_iter_count.value
        
        raw_params = np.asarray(scaled_params, dtype=float)
        if self.param_space == "unit":
            unit_params = np.clip(raw_params, 0.0, 1.0)
            phys_params = self._lb + unit_params * self._span
        else:
            phys_params = raw_params

        params_obj = OptimizationParams(*phys_params.tolist())
        results = run_simulation_wrapper(
            params_obj,
            self.env_config,
            self.hw_config,
            self.sw_config,
            self.sim_config,
            self.log_config,
            self.phase  # Pass phase to the wrapper
        )
        
        cost = results.get('cost', PENALTY_CRASH) # Use the cost calculated in the results
        if global_log_lock is None:
            log_iteration(f"Phase {self.phase}", current_iter, params_obj, results)
        else:
            with global_log_lock:
                log_iteration(f"Phase {self.phase}", current_iter, params_obj, results)

        if self.phase == 1:
            print(
                f"[Phase 1] Iter {current_iter:3d} | Cost: {cost:.1f} | Status: {results['status']}", 
                flush=True
            )
        else: # Phase 2
            fuel = results.get('fuel', 0)
            error = results.get('orbital_error', 0)
            print(
                f"[Phase 2] Iter {current_iter:3d} | Fuel: {fuel:.0f} kg | Error: {error/1000:.1f} km | Cost: {cost:.0f} | Status: {results['status']}", 
                flush=True
            )

        return cost



def _evaluate_candidate(args):
    """Helper for multiprocessing pool; keeps objective picklable and guarded."""
    objective_fn_wrapper, cand = args
    try:
        return float(objective_fn_wrapper(np.array(cand, dtype=float)))
    except Exception as exc:  # pragma: no cover - defensive logging for worker issues
        print(f"[CMA worker] candidate failed: {exc}", flush=True)
        return float(PENALTY_CRASH) # Ensure return type is float

def run_simulation_wrapper(
    params: OptimizationParams,
    env_config: EnvironmentConfig,
    hw_config: HardwareConfig,
    sw_config: SoftwareConfig,
    sim_config: SimulationConfig,
    log_config: LoggingConfig,
    phase: int,
    *,
    dt_s: float | None = None,
    duration_s: float | None = None,
):
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

    # Use a phase-dependent simulation fidelity: phase 1 can be coarser, phase 2
    # should be finer to make fuel comparisons meaningful.
    if phase == 1:
        cfg_sim.main_dt_s = 0.5
    else:
        cfg_sim.main_dt_s = 0.25

    # Simulate long enough to include the full burn program and a short coast so
    # orbital elements are evaluated after cutoff, without relying on "exit on orbit"
    # (which can otherwise stop mid-burn and bias fuel usage downward).
    burn_and_coast = (
        float(params.upper_burn_s)
        + float(params.coast_s)
        + float(params.upper_ignition_delay_s)
        + 700.0
    )
    cfg_sim.main_duration_s = float(np.clip(burn_and_coast, 800.0, 6000.0))

    # Allow callers (e.g., final evaluation) to override fidelity.
    if dt_s is not None:
        cfg_sim.main_dt_s = float(dt_s)
    if duration_s is not None:
        cfg_sim.main_duration_s = float(duration_s)

    # Initialize results with a default "CRASH" status in case the simulation fails early
    results = {"fuel": 0.0, "status": "INIT", "cost": PENALTY_CRASH}
    
    try:
        sim, state0, t0, _log_config, _analysis_config = main_orchestrator(
            env_config=cfg_env,
            hw_config=cfg_hw,
            sw_config=cfg_sw,
            sim_config=cfg_sim,
            log_config=cfg_log,
        )
        initial_mass = state0.m # Capture initial mass after orchestration

        # Run simulation (dt/duration tuned above)
        log = sim.run(t0, duration=float(cfg_sim.main_duration_s), dt=float(cfg_sim.main_dt_s), state0=state0)
        max_altitude = max(log.altitude) if log.altitude else 0.0 # Get max altitude for evaluation

        results = evaluate_simulation_results(log, initial_mass, cfg_env, cfg_sim, max_altitude, phase)

    except IndexError:
        results["status"] = "SIM_FAIL_INDEX"
    except Exception:
        results["status"] = "SIM_FAIL_UNKNOWN"
        print(f"Simulation wrapper encountered an unexpected error: {traceback.format_exc()}", flush=True) # Added more detailed error logging
    
    # Ensure cost is present in results, even in failure cases not caught by evaluate_simulation_results
    if 'cost' not in results:
        # This will use the new calculate_cost function via evaluate_simulation_results
        # but we need to ensure it's called even if an early exception happens.
        # For simplicity, let's just assign a penalty. A more robust way would be to
        # call `calculate_cost` here, but that requires more inputs.
        # The refactored `evaluate_simulation_results` should handle this.
        # Let's check if the status provides enough info.
        if results['status'] in ["SIM_FAIL_INDEX", "SIM_FAIL_UNKNOWN"]:
            from Analysis.cost_functions import calculate_cost
            # We pass a minimal results dictionary to the cost function
            results['cost'] = calculate_cost(
                results, phase, sim_config.target_orbit_alt_m, env_config.earth_radius_m
            )
        else:
            results['cost'] = PENALTY_CRASH

    return results



def run_cma_phase(
    objective_fn_wrapper,
    bounds,
    shared_counter,
    log_lock,
    start=None,
    sigma_scale=0.2,
    maxiter=200,
    popsize=None,
):
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
        # Start diagonal for speed, then allow correlations to be learned.
        "CMA_diagonal": 30,
        "CMA_active": True,
    }
    es = cma.CMAEvolutionStrategy(start.tolist(), sigma0, opts)

    with multiprocessing.Pool(initializer=init_worker, initargs=(shared_counter, log_lock)) as pool:
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

    # Initialize shared counter and a lock for logging from worker processes.
    global global_iter_count
    global_iter_count = multiprocessing.Value('i', 0)
    global global_log_lock
    global_log_lock = multiprocessing.Lock()

    # --- Initial Guess & Bounds (SCALED) ---
    # The bounds are now managed centrally in Analysis/config.py
    bounds_phys = OptimizationBounds.get_bounds()
    bounds_unit = [(0.0, 1.0)] * len(bounds_phys)
    lb_phys = np.array([b[0] for b in bounds_phys], dtype=float)
    ub_phys = np.array([b[1] for b in bounds_phys], dtype=float)
    span_phys = ub_phys - lb_phys
    span_phys = np.where(span_phys == 0.0, 1.0, span_phys)

    def to_unit(x_phys: np.ndarray) -> np.ndarray:
        return (np.asarray(x_phys, dtype=float) - lb_phys) / span_phys

    def from_unit(x_unit: np.ndarray) -> np.ndarray:
        u = np.asarray(x_unit, dtype=float)
        return lb_phys + np.clip(u, 0.0, 1.0) * span_phys

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

    if analysis_config.optimizer_manual_seed and len(analysis_config.optimizer_manual_seed) == 35:
        start_params_phys = np.array(analysis_config.optimizer_manual_seed, dtype=float)
    else:
        start_params_phys = (lb_phys + ub_phys) / 2.0

    ensure_log_header()
    print(f"=== PHASE 1: TARGETING ORBIT (Logging to {LOG_FILENAME}) ===", flush=True)
    with global_iter_count.get_lock():
        global_iter_count.value = 0
    
    
    # Ensure seed respects the active bounds to avoid CMA boundary errors.
    # Clip seed to physical bounds, then map into unit space for better conditioning.
    start_params_phys = np.clip(start_params_phys, lb_phys, ub_phys)
    start_params_unit = np.clip(to_unit(start_params_phys), 0.0, 1.0)
    
    objective_phase1 = ObjectiveFunctionWrapper(
        phase=1,
        env_config=env_config,
        hw_config=hw_config,
        sw_config=sw_config,
        sim_config=sim_config,
        log_config=log_config,
        analysis_config=analysis_config,
        bounds=bounds_phys,
        param_space="unit",
    )

    if CMA_AVAILABLE:
        print("Using CMA-ES for Phase 1", flush=True)
        res1 = run_cma_phase(
            objective_phase1,
            bounds_unit,
            global_iter_count,
            global_log_lock,
            start=start_params_unit,
            sigma_scale=0.35,
            maxiter=200,
            popsize=24,
        )
        best1_unit = res1.xbest
        best1_cost = res1.fbest
    else:
        print("CMA-ES not available, falling back to Differential Evolution for Phase 1", flush=True)
        res = differential_evolution(
            objective_phase1,
            bounds_unit,
            maxiter=200,
            disp=True
        )
        best1_unit = res.x
        best1_cost = res.fun

    best1 = from_unit(best1_unit)

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
        bounds=bounds_phys,
        param_space="unit",
    )

    if CMA_AVAILABLE:
        # Seed phase 2 from phase 1 best with a smaller sigma to focus search.
        res2 = run_cma_phase(
            objective_phase2,
            tighten_bounds(bounds_unit, best1_unit, margin=0.15),
            global_iter_count,
            global_log_lock,
            start=best1_unit,
            sigma_scale=0.15,
            maxiter=250,
            popsize=24,
        )
        best2_unit = res2.xbest
        best2_cost = res2.fbest
    else:
        res = differential_evolution(
            objective_phase2,
            bounds_unit,
            maxiter=250,
            disp=True
        )
        best2_unit = res.x
        best2_cost = res.fun

    print("\n=== OPTIMIZATION COMPLETE ===", flush=True)
    final_params2 = from_unit(best2_unit)
    # Run one last time for final numbers on a finer timestep for a more
    # trustworthy fuel figure.
    final_results = run_simulation_wrapper(
        final_params2,
        env_config,
        hw_config,
        sw_config,
        sim_config,
        log_config,
        phase=2,  # Explicitly use phase 2 for final evaluation
        dt_s=sim_config.main_dt_s,
    )

    print(f"Final Fuel Used: {final_results['fuel']:.1f} kg", flush=True)
    print(f"Final Orbit Error: {final_results['orbital_error']/1000:.1f} km", flush=True)
    print(f"Optimal Parameters (summary): Mach={final_params2[0]:.2f}, Coast={final_params2[11]:.1f}s, Upper Burn={final_params2[12]:.1f}s. Full details in {LOG_FILENAME}", flush=True)


if __name__ == "__main__":
    try:
        run_optimization()
    except Exception as e:
        print(f"ERROR: An unhandled exception occurred during optimization: {e}", flush=True)
        import traceback
        traceback.print_exc()
