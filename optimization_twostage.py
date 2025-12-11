"""
optimization_twostage.py

Solves the launch problem in two distinct phases for maximum efficiency:
Phase 1: "Targeting" - Find ANY parameters that hit the target orbit (Ignore fuel).
Phase 2: "Optimizing" - From that valid orbit, minimize fuel usage while staying in orbit.

This version incorporates input scaling, coarse-to-fine simulation, and detailed logging.
"""
import csv
import multiprocessing
import os
import numpy as np
from scipy.optimize import differential_evolution
from main import build_simulation, ParameterizedThrottleProgram
from gravity import MU_EARTH, R_EARTH, orbital_elements_from_state
from config import CFG
from custom_guidance import create_pitch_program_callable
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
LOG_FILENAME = "optimization_twostage_log.csv"
PENALTY_CRASH = 1e9        # "Soft Wall" for failed orbits

class Counter:
    def __init__(self, initial_value=0):
        self.value = initial_value

global_iter_count = Counter(0)
# Bounds used by objectives (set in run_optimization); initialized for tests.
bounds = []


def _evaluate_candidate(args):
    """Helper for multiprocessing pool; keeps objective picklable and guarded."""
    objective_fn, cand, bnds = args
    # Ensure spawned workers see the current bounds snapshot
    global bounds
    bounds = bnds
    try:
        return float(objective_fn(np.array(cand, dtype=float)))
    except Exception as exc:  # pragma: no cover - defensive logging for worker issues
        print(f"[CMA worker] candidate failed: {exc}", flush=True)
        return PENALTY_CRASH


def build_default_params_from_config():
    """Construct a starting parameter vector derived from the active CFG programs."""

    def pick_pitch_points(schedule, count, default_angle=0.0):
        times = []
        angles = []
        for t, ang in schedule[:count]:
            times.append(float(t))
            angles.append(float(ang))
        # Pad if not enough points
        while len(times) < count:
            times.append(times[-1] + 20.0 if times else 0.0)
            angles.append(default_angle)
        return times[:count], angles[:count]

    booster_pitch_times, booster_pitch_angles = pick_pitch_points(CFG.pitch_program, 5, default_angle=0.0)
    upper_pitch_times, upper_pitch_angles = pick_pitch_points(CFG.upper_pitch_program, 3, default_angle=0.0)

    def throttle_to_levels_and_ratios(schedule, desired_levels=4):
        """Convert an absolute time schedule to evenly clipped levels/ratios."""
        levels = []
        times = []
        for t, lvl in schedule:
            times.append(float(t))
            levels.append(float(lvl))
        if not times:
            return [1.0] * desired_levels, [0.2, 0.5, 0.8]
        burn_duration = max(times)
        # Normalize switch times to ratios, ensure we have desired_levels entries
        ratios = []
        prev_level = levels[0]
        prev_time = times[0]
        norm_switches = []
        for t, lvl in zip(times[1:], levels[1:]):
            if lvl != prev_level:
                norm_switches.append(max(0.0, min(0.99, t / burn_duration if burn_duration > 0 else 0.0)))
                prev_level = lvl
                prev_time = t
        # Build levels list by sampling the schedule at switch points (clipped to [0,1])
        sampled_levels = []
        last_idx = 0
        unique_switches = []
        for sw in norm_switches:
            if unique_switches and abs(unique_switches[-1] - sw) < 1e-6:
                continue
            unique_switches.append(sw)
        # Start level
        sampled_levels.append(levels[0])
        # Next levels from switches
        for _ in unique_switches:
            idx = min(last_idx + 1, len(levels) - 1)
            sampled_levels.append(levels[idx])
            last_idx = idx
        # Pad/trim to desired_levels
        while len(sampled_levels) < desired_levels:
            sampled_levels.append(sampled_levels[-1])
        sampled_levels = sampled_levels[:desired_levels]
        # Ratios must be length desired_levels-1
        while len(unique_switches) < desired_levels - 1:
            unique_switches.append(unique_switches[-1] + 0.1 if unique_switches else 0.2)
        unique_switches = unique_switches[:desired_levels - 1]
        return [np.clip(l, 0.0, 1.0) for l in sampled_levels], sorted(unique_switches)

    upper_throttle_levels, upper_throttle_ratios = throttle_to_levels_and_ratios(CFG.upper_stage_throttle_program)
    booster_throttle_levels, booster_throttle_ratios = throttle_to_levels_and_ratios(CFG.booster_throttle_program)

    return np.array([
        CFG.meco_mach,
        booster_pitch_times[0], booster_pitch_angles[0],
        booster_pitch_times[1], booster_pitch_angles[1],
        booster_pitch_times[2], booster_pitch_angles[2],
        booster_pitch_times[3], booster_pitch_angles[3],
        booster_pitch_times[4], booster_pitch_angles[4],
        30.0,   # coast_s
        200.0,  # upper_burn_s
        10.0,   # upper_ignition_delay_s
        0.0,    # azimuth_deg (east)
        upper_pitch_times[0], upper_pitch_angles[0],
        upper_pitch_times[1], upper_pitch_angles[1],
        upper_pitch_times[2], upper_pitch_angles[2],
        *upper_throttle_levels,
        *upper_throttle_ratios,
        *booster_throttle_levels,
        *booster_throttle_ratios,
    ], dtype=float)


def log_iteration(phase, iteration, params, results):
    """Helper to log a single optimizer iteration with de-scaled physics values."""
    meco_mach, pitch_time_0, pitch_angle_0, pitch_time_1, pitch_angle_1, pitch_time_2, pitch_angle_2, \
    pitch_time_3, pitch_angle_3, pitch_time_4, pitch_angle_4, \
    coast_s, upper_burn_s, upper_ignition_delay_s, azimuth_deg, \
    upper_pitch_time_0, upper_pitch_angle_0, upper_pitch_time_1, upper_pitch_angle_1, upper_pitch_time_2, upper_pitch_angle_2, \
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
            f"{upper_ignition_delay_s:.1f}",
            f"{azimuth_deg:.1f}",
            f"{upper_pitch_time_0:.1f}", f"{upper_pitch_angle_0:.1f}",
            f"{upper_pitch_time_1:.1f}", f"{upper_pitch_angle_1:.1f}",
            f"{upper_pitch_time_2:.1f}", f"{upper_pitch_angle_2:.1f}",
            f"{upper_throttle_level_0:.2f}", f"{upper_throttle_level_1:.2f}", f"{upper_throttle_level_2:.2f}", f"{upper_throttle_level_3:.2f}",
            f"{upper_throttle_switch_ratio_0:.2f}", f"{upper_throttle_switch_ratio_1:.2f}", f"{upper_throttle_switch_ratio_2:.2f}",
            f"{booster_throttle_level_0:.2f}", f"{booster_throttle_level_1:.2f}", f"{booster_throttle_level_2:.2f}", f"{booster_throttle_level_3:.2f}",
            f"{booster_throttle_switch_ratio_0:.2f}", f"{booster_throttle_switch_ratio_1:.2f}", f"{booster_throttle_switch_ratio_2:.2f}",
            f"{results.get('cost', 0.0):.2f}",
            f"{results.get('fuel', 0.0):.2f}",
            f"{results.get('orbit_error', results.get('error', 0.0)):.2f}",
            results.get('status', 'UNKNOWN')
        ])


def ensure_log_header():
    """Create the CSV log file with a header if it's missing or empty."""
    if not os.path.exists(LOG_FILENAME) or os.path.getsize(LOG_FILENAME) == 0:
        with open(LOG_FILENAME, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "phase", "iteration", "meco_mach",
                "pitch_time_0", "pitch_angle_0",
                "pitch_time_1", "pitch_angle_1",
                "pitch_time_2", "pitch_angle_2",
                "pitch_time_3", "pitch_angle_3",
                "pitch_time_4", "pitch_angle_4",
                "coast_s", "upper_burn_s", "upper_ignition_delay_s",
                "azimuth_deg",
                "upper_pitch_time_0", "upper_pitch_angle_0",
                "upper_pitch_time_1", "upper_pitch_angle_1",
                "upper_pitch_time_2", "upper_pitch_angle_2",
                "upper_throttle_level_0", "upper_throttle_level_1", "upper_throttle_level_2", "upper_throttle_level_3",
                "upper_throttle_switch_ratio_0", "upper_throttle_switch_ratio_1", "upper_throttle_switch_ratio_2",
                "booster_throttle_level_0", "booster_throttle_level_1", "booster_throttle_level_2", "booster_throttle_level_3",
                "booster_throttle_switch_ratio_0", "booster_throttle_switch_ratio_1", "booster_throttle_switch_ratio_2",
                "cost", "fuel", "orbit_error", "status",
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
    coast_s, upper_burn_s, upper_ignition_delay_s, azimuth_deg, \
    upper_pitch_time_0, upper_pitch_angle_0, upper_pitch_time_1, upper_pitch_angle_1, upper_pitch_time_2, upper_pitch_angle_2, \
    upper_throttle_level_0, upper_throttle_level_1, upper_throttle_level_2, upper_throttle_level_3, \
    upper_throttle_switch_ratio_0, upper_throttle_switch_ratio_1, upper_throttle_switch_ratio_2, \
    booster_throttle_level_0, booster_throttle_level_1, booster_throttle_level_2, booster_throttle_level_3, \
    booster_throttle_switch_ratio_0, booster_throttle_switch_ratio_1, booster_throttle_switch_ratio_2 = scaled_params

    # Pitch angles are treated consistently with the config: 0 = horizontal, 90 = vertical.
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

    # Upper-stage pitch schedule (relative to upper ignition)
    upper_pitch_times_raw = np.array([upper_pitch_time_0, upper_pitch_time_1, upper_pitch_time_2])
    upper_pitch_angles_deg_raw = np.clip(
        np.array([upper_pitch_angle_0, upper_pitch_angle_1, upper_pitch_angle_2]),
        0.0, 90.0,
    )
    upper_pitch_order = np.argsort(upper_pitch_times_raw)
    upper_pitch_times = upper_pitch_times_raw[upper_pitch_order]
    upper_pitch_angles_deg = upper_pitch_angles_deg_raw[upper_pitch_order]

    # Ensure throttle levels and switch ratios are within [0, 1]
    upper_throttle_levels = np.clip([upper_throttle_level_0, upper_throttle_level_1, upper_throttle_level_2, upper_throttle_level_3], 0.0, 1.0)
    upper_throttle_switch_ratios = np.clip([upper_throttle_switch_ratio_0, upper_throttle_switch_ratio_1, upper_throttle_switch_ratio_2], 0.0, 1.0)
    upper_throttle_switch_ratios.sort() # Ensure ratios are increasing
    
    booster_throttle_levels = np.clip([booster_throttle_level_0, booster_throttle_level_1, booster_throttle_level_2, booster_throttle_level_3], 0.0, 1.0)
    booster_throttle_switch_ratios = np.clip([booster_throttle_switch_ratio_0, booster_throttle_switch_ratio_1, booster_throttle_switch_ratio_2], 0.0, 1.0)
    booster_throttle_switch_ratios.sort() # Ensure ratios are increasing

    # 2. UPDATE CONFIGURATION
    CFG.meco_mach = float(meco_mach)

    # Keep the factory setup happy, then override with our custom pitch program below.
    CFG.pitch_guidance_mode = 'parameterized' 
    
    # Second stage parameters
    CFG.separation_delay_s = float(coast_s)
    CFG.upper_ignition_delay_s = float(upper_ignition_delay_s)

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
    # Let's use a proxy for booster burn duration for the ratios, calculated from config.
    mdot_approx = CFG.booster_thrust_sl / (CFG.booster_isp_sl * CFG.G0)
    booster_burn_duration_proxy = CFG.booster_prop_mass / mdot_approx if mdot_approx > 0 else 160.0
    
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
            schedule=CFG.booster_throttle_program, apply_to_stage0=True)
        sim.rocket.booster_throttle_program = booster_throttle_controller
        
        # Create and apply stage-aware pitch program
        booster_pitch_points = [
            (pitch_times[0], pitch_angles_deg[0]),
            (pitch_times[1], pitch_angles_deg[1]),
            (pitch_times[2], pitch_angles_deg[2]),
            (pitch_times[3], pitch_angles_deg[3]),
            (pitch_times[4], pitch_angles_deg[4])
        ]
        upper_pitch_points = [
            (upper_pitch_times[0], upper_pitch_angles_deg[0]),
            (upper_pitch_times[1], upper_pitch_angles_deg[1]),
            (upper_pitch_times[2], upper_pitch_angles_deg[2]),
        ]

        booster_pitch_fn = create_pitch_program_callable(booster_pitch_points, azimuth_deg=azimuth_deg)
        upper_pitch_fn = create_pitch_program_callable(upper_pitch_points, azimuth_deg=azimuth_deg)

        def stage_pitch_program(t, state, t_stage=None, stage_index=None):
            idx = stage_index if stage_index is not None else getattr(state, "stage_index", 0)
            if idx <= 0:
                return booster_pitch_fn(t, state)
            t_rel = t_stage if t_stage is not None else t
            return upper_pitch_fn(t_rel, state)

        sim.guidance.pitch_program = stage_pitch_program
        
        initial_mass = state0.m

        # Coarse pass for speed; refine only if promising
        coarse_log = sim.run(t0, duration=2000.0, dt=2.0, state0=state0,
                             orbit_target_radius=R_EARTH + TARGET_ALT_M,
                             exit_on_orbit=False)
        max_altitude_coarse = max(coarse_log.altitude) if coarse_log.altitude else 0.0
        if max_altitude_coarse < 50_000.0:
            results["status"] = "CRASH"
            results["error"] = 1e7 + (TARGET_ALT_M - max_altitude_coarse)
            return results

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
    finally:
        # Ensure both keys exist for downstream logging
        results["orbit_error"] = results.get("orbit_error", results.get("error", PENALTY_CRASH))

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


def objective_phase1(scaled_params):
    global global_iter_count, last_params_phase1, last_cost_phase1, stuck_counter_phase1
    print(f"DEBUG: Entering objective_phase1, iteration {global_iter_count.value}", flush=True) # Added flush=True here too

    global_iter_count.value += 1

    results = run_simulation_wrapper(scaled_params)
    results['cost'] = results['error']  # Cost for phase 1 is just the error
    # Apply soft bounds penalty
    bound_penalty = soft_bounds_penalty(scaled_params, bounds)
    results['cost'] += bound_penalty

    log_iteration("Phase 1", global_iter_count.value, scaled_params, results)
    print(
        f"[Phase 1] Iter {global_iter_count.value:3d} | Error: {results['error']/1000:.1f} km | Status: {results['status']}", flush=True)

    last_cost_phase1 = results['cost']

    return results['cost']


def objective_phase2(scaled_params):
    global global_iter_count, last_params_phase2, last_cost_phase2, stuck_counter_phase2

    global_iter_count.value += 1

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

    log_iteration("Phase 2", global_iter_count.value, scaled_params, results)
    print(f"[Phase 2] Iter {global_iter_count.value:3d} | Fuel: {fuel:.0f} kg | Error: {error/1000:.1f} km | Cost: {cost:.0f} | Status: {results['status']}", flush=True)

    last_cost_phase2 = results['cost']

    return results['cost']


def run_cma_phase(objective_fn, bounds, start=None, sigma_scale=0.2, maxiter=200, popsize=None):
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
            work = [(objective_fn, cand, bounds) for cand in candidates]
            costs = pool.map(_evaluate_candidate, work)
            gen_best = float(np.min(costs)) if costs else np.inf
            print(f"[CMA] gen {es.countiter:3d} | best this gen: {gen_best:.2f}", flush=True)
            es.tell(candidates, costs)
    return es.result


def run_optimization():
    """Runs the two-phase optimization process."""
    global global_iter_count

    # --- Initial Guess & Bounds (SCALED) ---
    # Use km for altitudes to keep numbers in a similar magnitude
    # params: meco_mach, pitch_start_km, pitch_end_km, pitch_blend, coast_s, upper_burn_s
    global bounds # Make bounds global for soft_bounds_penalty
    bounds = [
        (4.5, 6.5),      # 0: MECO Mach
        # Pitch profile (5 points, time in seconds, angle in degrees relative to horizontal)
        (0.0, 20.0),      # 1: pitch_time_0 (s) - Initial liftoff phase
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

    ensure_log_header()
    print(f"=== PHASE 1: TARGETING ORBIT (Logging to {LOG_FILENAME}) ===", flush=True)
    global_iter_count.value = 0
    start_params = build_default_params_from_config()
    # Ensure seed respects the active bounds to avoid CMA boundary errors.
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    start_params = np.clip(start_params, lb, ub)
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
