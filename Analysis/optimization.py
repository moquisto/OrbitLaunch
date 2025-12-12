"""
optimization_twostage.py

Solves the launch problem in two distinct phases for maximum efficiency:
Phase 1: "Targeting" - Find ANY parameters that hit the target orbit (Ignore fuel).
Phase 2: "Optimizing" - From that valid orbit, minimize fuel usage while staying in orbit.

This version incorporates input scaling, coarse-to-fine simulation, and detailed logging.
"""
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
from Analysis.config import AnalysisConfig, OptimizationParams # Import OptimizationParams from Analysis.config
from Software.guidance import create_pitch_program_callable, ParameterizedThrottleProgram
from Logging.generate_logs import log_iteration, ensure_log_header # Import from logging module

import traceback

try:
    import cma
    CMA_AVAILABLE = True
except Exception:
    CMA_AVAILABLE = False

print("optimization_twostage.py script started.", flush=True)

# --- Configuration ---
TARGET_ALT_M = 420_000.0   # 420 km
# We need to be within 10km to consider it "Orbit" for Phase 2
TARGET_TOLERANCE_M = 10000.0
# LOG_FILENAME is now in Logging/generate_logs.py
PENALTY_CRASH = 1e9        # "Soft Wall" for failed orbits
PERIGEE_FLOOR_M = 120_000.0  # Minimum acceptable perigee altitude
ECC_TOLERANCE = 0.01         # Maximum eccentricity tolerated before penalty

class Counter:
    def __init__(self, initial_value=0):
        self.value = initial_value

global_iter_count = Counter(0)
# Bounds used by objectives (set in run_optimization); initialized for tests.
bounds = []


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
        global_iter_count.value += 1
        
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

            log_iteration("Phase 1", global_iter_count.value, params_obj, results)
            print(
                f"[Phase 1] Iter {global_iter_count.value:3d} | Error: {results['error']/1000:.1f} km | Status: {results['status']}", flush=True)

            return results['cost']
        else:
            fuel, error = results["fuel"], results["error"]
            if error > TARGET_TOLERANCE_M:
                penalty = (error - TARGET_TOLERANCE_M) * 10.0
                cost = fuel + penalty
            else:
                cost = fuel
            
            bound_penalty = soft_bounds_penalty(scaled_params, self.bounds)
            cost += bound_penalty
            results['cost'] = cost

            log_iteration("Phase 2", global_iter_count.value, params_obj, results)
            print(f"[Phase 2] Iter {global_iter_count.value:3d} | Fuel: {fuel:.0f} kg | Error: {error/1000:.1f} km | Cost: {cost:.0f} | Status: {results['status']}", flush=True)

            return cost

def _evaluate_candidate(args):
    """Helper for multiprocessing pool; keeps objective picklable and guarded."""
    objective_fn_wrapper, cand, bnds = args
    # Ensure spawned workers see the current bounds snapshot
    global bounds
    bounds = bnds
    try:
        return float(objective_fn_wrapper(np.array(cand, dtype=float)))
    except Exception as exc:  # pragma: no cover - defensive logging for worker issues
        print(f"[CMA worker] candidate failed: {exc}", flush=True)
        return PENALTY_CRASH








import copy

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

    # 1. DE-SCALE AND PREPARE PARAMETERS for physics simulation
    # Pitch angles are treated consistently with the config: 0 = horizontal, 90 = vertical.
    booster_pitch_points_list = [
        (params.booster_pitch_time_0, np.clip(params.booster_pitch_angle_0, 0.0, 90.0)),
        (params.booster_pitch_time_1, np.clip(params.booster_pitch_angle_1, 0.0, 90.0)),
        (params.booster_pitch_time_2, np.clip(params.booster_pitch_angle_2, 0.0, 90.0)),
        (params.booster_pitch_time_3, np.clip(params.booster_pitch_angle_3, 0.0, 90.0)),
        (params.booster_pitch_time_4, np.clip(params.booster_pitch_angle_4, 0.0, 90.0))
    ]
    # Sort by time
    booster_pitch_points_list.sort(key=lambda x: x[0])

    upper_pitch_points_list = [
        (params.upper_pitch_time_0, np.clip(params.upper_pitch_angle_0, 0.0, 90.0)),
        (params.upper_pitch_time_1, np.clip(params.upper_pitch_angle_1, 0.0, 90.0)),
        (params.upper_pitch_time_2, np.clip(params.upper_pitch_angle_2, 0.0, 90.0))
    ]
    # Sort by time
    upper_pitch_points_list.sort(key=lambda x: x[0])

    # Ensure throttle levels and switch ratios are within [0, 1]
    upper_throttle_levels = np.clip([params.upper_throttle_level_0, params.upper_throttle_level_1, params.upper_throttle_level_2, params.upper_throttle_level_3], 0.0, 1.0)
    upper_throttle_switch_ratios = np.clip([params.upper_throttle_switch_ratio_0, params.upper_throttle_switch_ratio_1, params.upper_throttle_switch_ratio_2], 0.0, 1.0)
    upper_throttle_switch_ratios.sort()  # Ensure ratios are increasing
    
    booster_throttle_levels = np.clip([params.booster_throttle_level_0, params.booster_throttle_level_1, params.booster_throttle_level_2, params.booster_throttle_level_3], 0.0, 1.0)
    booster_throttle_switch_ratios = np.clip([params.booster_throttle_switch_ratio_0, params.booster_throttle_switch_ratio_1, params.booster_throttle_switch_ratio_2], 0.0, 1.0)
    booster_throttle_switch_ratios.sort()  # Ensure ratios are increasing

    # 2. UPDATE CONFIGURATION (on the copy)
    cfg_sw.meco_mach = float(params.meco_mach)
    cfg_sw.pitch_guidance_mode = 'parameterized' 
    cfg_sw.separation_delay_s = float(params.coast_s)
    cfg_sw.upper_ignition_delay_s = float(params.upper_ignition_delay_s)

    # Construct upper stage throttle program schedule
    upper_throttle_program_schedule = []
    current_time_ratio = 0.0
    upper_throttle_program_schedule.append([current_time_ratio * params.upper_burn_s, upper_throttle_levels[0]])
    
    for i in range(len(upper_throttle_switch_ratios)):
        switch_ratio = upper_throttle_switch_ratios[i]
        throttle_level = upper_throttle_levels[i+1]
        
        if switch_ratio > current_time_ratio:
            upper_throttle_program_schedule.append([switch_ratio * params.upper_burn_s, upper_throttle_levels[i]])
            upper_throttle_program_schedule.append([switch_ratio * params.upper_burn_s + 1e-6, throttle_level])
            current_time_ratio = switch_ratio
        else:
            upper_throttle_program_schedule[-1][1] = throttle_level
            
    upper_throttle_program_schedule.append([params.upper_burn_s, upper_throttle_levels[-1]])
    upper_throttle_program_schedule.append([params.upper_burn_s + 1, 0.0])
    cfg_sw.upper_stage_throttle_program = upper_throttle_program_schedule # Update cfg_sw as well

    # Construct booster throttle program schedule
    booster_throttle_program_schedule = []
    # Note: sim_config.G0 has been moved to env_config.G0
    mdot_approx = cfg_hw.booster_thrust_sl / (cfg_hw.booster_isp_sl * cfg_env.G0) 
    booster_burn_duration_proxy = cfg_hw.booster_prop_mass / mdot_approx if mdot_approx > 0 else 160.0
    
    current_booster_time_ratio = 0.0
    booster_throttle_program_schedule.append([current_booster_time_ratio * booster_burn_duration_proxy, booster_throttle_levels[0]])
    
    for i in range(len(booster_throttle_switch_ratios)):
        switch_ratio = booster_throttle_switch_ratios[i]
        throttle_level = booster_throttle_levels[i+1]
        
        if switch_ratio > current_booster_time_ratio:
            booster_throttle_program_schedule.append([switch_ratio * booster_burn_duration_proxy, booster_throttle_levels[i]])
            booster_throttle_program_schedule.append([switch_ratio * booster_burn_duration_proxy + 1e-6, throttle_level])
            current_booster_time_ratio = switch_ratio
        else:
            booster_throttle_program_schedule[-1][1] = throttle_level
            
    booster_throttle_program_schedule.append([booster_burn_duration_proxy, booster_throttle_levels[-1]])
    booster_throttle_program_schedule.append([booster_burn_duration_proxy + 1, 0.0])
    cfg_sw.booster_throttle_program = booster_throttle_program_schedule # Update cfg_sw as well
    
    cfg_sim.orbit_alt_tol = 1e6
    cfg_sim.exit_on_orbit = False
    
    results = {"fuel": 0.0, "error": PENALTY_CRASH, "status": "INIT", "cost": PENALTY_CRASH}
    
    try:
        # Create Pitch Program instance
        cfg_sw.pitch_program = booster_pitch_points_list # Temporarily set to generate correct pitch program
        cfg_sw.upper_pitch_program = upper_pitch_points_list # Temporarily set to generate correct pitch program
        pitch_program_instance = cfg_sw.create_pitch_program(cfg_env)
        
        # Create Upper Throttle Program instance
        # Note: create_throttle_program in sw_config creates an upper stage one by default
        target_orbit_radius = cfg_env.earth_radius_m + cfg_sim.target_orbit_alt_m
        upper_throttle_program_instance = cfg_sw.create_throttle_program(target_orbit_radius, cfg_env.earth_mu)
        upper_throttle_program_instance.schedule = upper_throttle_program_schedule # Override with optimized schedule
        
        sim, state0, t0, _log_config, _analysis_config = main_orchestrator(
            env_config=cfg_env,
            hw_config=cfg_hw,
            sw_config=cfg_sw,
            sim_config=cfg_sim,
            log_config=cfg_log,
            # Pass custom guidance program instances to main_orchestrator
            pitch_program_instance=pitch_program_instance,
            upper_throttle_program_instance=upper_throttle_program_instance,
            booster_throttle_schedule_list=booster_throttle_program_schedule,
        )
        
        initial_mass = state0.m

        coarse_log = sim.run(t0, duration=2000.0, dt=2.0, state0=state0)
        max_altitude_coarse = max(coarse_log.altitude) if coarse_log.altitude else 0.0
        if max_altitude_coarse < 50_000.0:
            results["status"] = "CRASH_COARSE"
            results["error"] = 1e7 + (TARGET_ALT_M - max_altitude_coarse) * 10 # Penalize heavily for low altitude crashes
            return results

        log = sim.run(t0, duration=3000.0, dt=1.0, state0=state0)
        results["fuel"] = initial_mass - log.m[-1]
        max_altitude = max(log.altitude) if log.altitude else 0.0

        r, v = log.r[-1], log.v[-1]
        a, rp, ra = orbital_elements_from_state(r, v, cfg_env.earth_mu)

        if rp is None or ra is None or rp < (cfg_env.earth_radius_m - 5000):
            results["status"] = "CRASH"
            results["error"] = 1e7 + (TARGET_ALT_M - max_altitude)
            return results

        target_r = cfg_env.earth_radius_m + TARGET_ALT_M
        results["error"] = abs(rp - target_r) + abs(ra - target_r)

        perigee_alt = rp - cfg_env.earth_radius_m
        ecc = abs((ra - rp) / (ra + rp)) if (ra + rp) != 0 else 0
        results["perigee_alt_m"] = perigee_alt
        results["eccentricity"] = ecc

        perigee_penalty = max(0.0, PERIGEE_FLOOR_M - perigee_alt) * 100.0
        ecc_penalty = max(0.0, (ecc or 0.0) - ECC_TOLERANCE) * target_r
        results["error"] += perigee_penalty + ecc_penalty

        if perigee_alt < PERIGEE_FLOOR_M:
            results["status"] = "SUBORBIT"
        elif results["error"] < 5000:
            results["status"] = "PERFECT"
        elif results["error"] < 50000:
            results["status"] = "GOOD"
        else:
            results["status"] = "OK"

    except IndexError:
        results["status"] = "SIM_FAIL_INDEX"
    except Exception:
        results["status"] = "SIM_FAIL_UNKNOWN"
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

# ...





def run_cma_phase(objective_fn_wrapper, bounds, start=None, sigma_scale=0.2, maxiter=200, popsize=None):
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

    with multiprocessing.Pool() as pool:
        while not es.stop():
            candidates = es.ask()
            work = [(objective_fn_wrapper, cand, bounds) for cand in candidates]
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

    global global_iter_count
    
    # Use config values for optimization constants
    global TARGET_ALT_M
    TARGET_ALT_M = sim_config.target_orbit_alt_m

    # --- Initial Guess & Bounds (SCALED) ---
    # Use km for altitudes to keep numbers in a similar magnitude
    # params: meco_mach, pitch_start_km, pitch_end_km, pitch_blend, coast_s, upper_burn_s
    global bounds # Make bounds global for soft_bounds_penalty
    bounds = [
        (4.5, 6.5),      # 0: MECO Mach
        # Pitch profile (5 points, time in seconds, angle in degrees relative to horizontal)
        (0.0, 20.0),     # 1: pitch_time_0 (s) - Initial liftoff phase
        (80.0, 90.0),     # 2: pitch_angle_0 (deg, 90=vertical)
        (20.0, 60.0),     # 3: pitch_time_1 (s) - Start of gravity turn
        (60.0, 85.0),     # 4: pitch_angle_1 (deg)
        (50.0, 90.0),     # 5: pitch_time_2 (s) - Approaching max Q
        (40.0, 70.0),     # 6: pitch_angle_2 (deg)
        (80.0, 120.0),    # 7: pitch_time_3 (s) - Mid-atmosphere flight
        (20.0, 50.0),     # 8: pitch_angle_3 (deg)
        (110.0, 150.0),   # 9: pitch_time_4 (s) - Final moments before MECO
        (0.0, 20.0),      # 10: pitch_angle_4 (deg)
        # Staging and upper stage burn
        (5.0, 200.0),    # 11: Coast duration after MECO (s)
        (100.0, 300.0),  # 12: Upper stage burn duration (s)
        (0.0, 60.0),     # 13: Upper stage ignition delay after separation (s)
        (-15.0, 15.0),   # 14: Azimuth heading (deg from east toward north)
        # Upper-stage pitch profile (time from upper ignition, deg from horizontal)
        (0.0, 60.0),     # 15: upper_pitch_time_0 (s)
        (5.0, 45.0),     # 16: upper_pitch_angle_0 (deg)
        (40.0, 180.0),   # 17: upper_pitch_time_1 (s)
        (0.0, 30.0),     # 18: upper_pitch_angle_1 (deg)
        (150.0, 300.0),  # 19: upper_pitch_time_2 (s)
        (0.0, 15.0),     # 20: upper_pitch_angle_2 (deg)
        # Upper stage throttle profile (4 levels, 3 switch ratios)
        (0.3, 1.0),      # 21: upper_throttle_level_0 (0-1)
        (0.3, 1.0),      # 22: upper_throttle_level_1 (0-1)
        (0.3, 1.0),      # 23: upper_throttle_level_2 (0-1)
        (0.3, 1.0),      # 24: upper_throttle_level_3 (0-1)
        (0.05, 0.4),     # 25: upper_throttle_switch_ratio_0 (0-1, fraction of burn duration)
        (0.25, 0.8),     # 26: upper_throttle_switch_ratio_1 (0-1, fraction of burn duration)
        (0.6, 0.95),     # 27: upper_throttle_switch_ratio_2 (0-1, fraction of burn duration)
        # Booster throttle profile (4 levels, 3 switch ratios)
        (0.3, 1.0),      # 28: booster_throttle_level_0 (0-1)
        (0.3, 1.0),      # 29: booster_throttle_level_1 (0-1)
        (0.3, 1.0),      # 30: booster_throttle_level_2 (0-1)
        (0.3, 1.0),      # 31: booster_throttle_level_3 (0-1)
        (0.05, 0.4),     # 32: booster_throttle_switch_ratio_0 (0-1, fraction of booster burn duration)
        (0.25, 0.8),     # 33: booster_throttle_switch_ratio_1 (0-1, fraction of booster burn duration)
        (0.6, 0.95),     # 34: booster_throttle_switch_ratio_2 (0-1, fraction of booster burn duration)
    ]

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
        res1 = run_cma_phase(objective_phase1, bounds, start=start_params, sigma_scale=0.15, maxiter=120, popsize=14)
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
