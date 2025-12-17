# Last modified: 2025-12-13 11:54:18.000000

try:
    import cma
    CMA_AVAILABLE = True
except Exception:  # pragma: no cover
    cma = None
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
import os

# Allow running this file directly (e.g. `python3 Analysis/optimization.py`) by
# ensuring the repo root is on sys.path so `import main` works.
if __package__ in (None, ""):
    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

import multiprocessing
import copy
import traceback
import dataclasses
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
        label: str | None = None,
        param_space: str = "physical",
        active_indices: list[int] | None = None,
        base_params_phys: np.ndarray | list[float] | OptimizationParams | None = None,
        dt_s: float | None = None,
        duration_s: float | None = None,
        enable_logging: bool = False,
    ):
        self.phase = phase
        self.label = str(label) if label is not None else f"Phase {phase}"
        self.env_config = env_config
        self.hw_config = hw_config
        self.sw_config = sw_config
        self.sim_config = sim_config
        self.log_config = log_config
        self.analysis_config = analysis_config
        self.bounds = bounds
        self.param_space = param_space
        self.active_indices = list(active_indices) if active_indices is not None else None
        self.dt_s = dt_s
        self.duration_s = duration_s
        self.enable_logging = bool(enable_logging)
        self._lb = np.array([b[0] for b in bounds], dtype=float)
        self._ub = np.array([b[1] for b in bounds], dtype=float)
        self._span = self._ub - self._lb

        if self.active_indices is not None:
            if base_params_phys is None:
                raise ValueError("base_params_phys is required when active_indices is provided.")
            if isinstance(base_params_phys, OptimizationParams):
                base_vec = np.array(dataclasses.astuple(base_params_phys), dtype=float)
            else:
                base_vec = np.asarray(base_params_phys, dtype=float)
            if base_vec.shape[0] != 35:
                raise ValueError(f"base_params_phys must have length 35, got {base_vec.shape[0]}")
            self.base_params_phys = base_vec
        else:
            self.base_params_phys = None

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
            phys_params_active = self._lb + unit_params * self._span
        else:
            phys_params_active = raw_params

        if self.active_indices is None:
            phys_params_full = phys_params_active
        else:
            phys_params_full = np.array(self.base_params_phys, dtype=float)
            phys_params_full[self.active_indices] = phys_params_active

        params_obj = OptimizationParams(*np.asarray(phys_params_full, dtype=float).tolist())
        results = run_simulation_wrapper(
            params_obj,
            self.env_config,
            self.hw_config,
            self.sw_config,
            self.sim_config,
            self.log_config,
            self.phase,  # Pass phase to the wrapper
            dt_s=self.dt_s,
            duration_s=self.duration_s,
        )
        
        cost = results.get('cost', PENALTY_CRASH) # Use the cost calculated in the results
        if self.enable_logging:
            if global_log_lock is None:
                log_iteration(self.label, current_iter, params_obj, results)
            else:
                with global_log_lock:
                    log_iteration(self.label, current_iter, params_obj, results)

        if self.phase == 1:
            print(
                f"[{self.label}] Iter {current_iter:3d} | Cost: {cost:.1f} | Status: {results['status']}", 
                flush=True
            )
        else: # Phase 2
            fuel = results.get('fuel', 0)
            error = results.get('orbital_error', 0)
            print(
                f"[{self.label}] Iter {current_iter:3d} | Fuel: {fuel:.0f} kg | Error: {error/1000:.1f} km | Cost: {cost:.0f} | Status: {results['status']}", 
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
    return_log: bool = False,
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

    # Optimization runs call the atmosphere model thousands of times; enable the
    # fast US76 lookup table for performance. (This is still deterministic and
    # numerically close to direct US76 queries.)
    cfg_env.use_fast_atmosphere_lookup = True

    cfg_sw, cfg_sim = configure_software_for_optimization(params, cfg_sw, cfg_sim, cfg_env)

    # Use a phase-dependent simulation fidelity: phase 1 can be coarser, phase 2
    # should be finer to make fuel comparisons meaningful.
    if phase == 1:
        cfg_sim.main_dt_s = 1.0
        cfg_sim.integrator = "velocity_verlet"
    else:
        cfg_sim.main_dt_s = 0.5
        # RK4 is substantially more accurate for the non-conservative forces in
        # ascent (thrust + drag), especially at dt=0.5s used in the coarse stage.
        cfg_sim.integrator = "rk4"

    # Simulate only until shortly after SECO, then stop. Orbital elements can be
    # computed from the state at cutoff (no need to coast to apoapsis), and
    # shortening the horizon massively speeds up optimization.
    booster_time_est = max(float(params.booster_pitch_time_4), 120.0) + 80.0
    upper_start_est = booster_time_est + float(params.coast_s) + float(params.upper_ignition_delay_s)
    upper_cutoff_est = upper_start_est + float(params.upper_burn_s) + 2.0  # +1s shutdown point + margin
    post_cutoff_coast_s = 60.0 if phase == 1 else 120.0
    cfg_sim.main_duration_s = float(np.clip(upper_cutoff_est + post_cutoff_coast_s, 300.0, 3000.0))

    # Allow callers (e.g., final evaluation) to override fidelity.
    if dt_s is not None:
        cfg_sim.main_dt_s = float(dt_s)
        # For phase 1 we keep velocity_verlet for speed unless the caller
        # requests a fine timestep; phase 2 always uses RK4 (set above).
        if phase == 1 and cfg_sim.main_dt_s <= 0.25:
            cfg_sim.integrator = "rk4"
    if duration_s is not None:
        cfg_sim.main_duration_s = float(duration_s)

    # Initialize results with a default "CRASH" status in case the simulation fails early
    results = {"fuel": 0.0, "status": "INIT", "cost": PENALTY_CRASH}
    sim_log = None
    
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
        sim_log = sim.run(t0, duration=float(cfg_sim.main_duration_s), dt=float(cfg_sim.main_dt_s), state0=state0)
        max_altitude = max(sim_log.altitude) if sim_log.altitude else 0.0 # Get max altitude for evaluation

        results = evaluate_simulation_results(sim_log, initial_mass, cfg_env, cfg_sim, max_altitude, phase)

    except IndexError:
        results["status"] = "SIM_FAIL_INDEX"
    except Exception:
        results["status"] = "SIM_FAIL_UNKNOWN"
        print(f"Simulation wrapper encountered an unexpected error: {traceback.format_exc()}", flush=True) # Added more detailed error logging
    
    # Optional: simulate a two-burn insertion by circularizing at apoapsis.
    #
    # This turns the "post-circ" approximation into an actual (impulsive) burn
    # inside the trajectory returned by the program: burn -> coast -> circularize.
    if sim_log is not None:
        # Optional apoapsis circularization burn model. This is disabled by
        # default because the intended baseline problem is single-burn direct
        # insertion. Enable explicitly via env var if desired.
        enable_circ = str(os.getenv("ORBITLAUNCH_SIMULATE_CIRCULARIZATION_BURN", "0")).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        if enable_circ:
            try:
                from Analysis.cost_functions import TARGET_TOLERANCE_M, calculate_cost

                cutoff_reason_pre = str(results.get("cutoff_reason", "") or "")
                pre_status = str(results.get("status", "UNKNOWN"))
                perigee_pre = float(results.get("perigee_alt_m", float("nan")))
                apo_pre = float(results.get("apoapsis_alt_m", float("nan")))
                ecc_pre = float(results.get("eccentricity", float("nan")))
                orbit_error_pre = float(results.get("orbital_error", float("nan")))

                rp_m = float(results.get("rp_m", float("nan")))
                ra_m = float(results.get("ra_m", float("nan")))
                if np.isfinite(rp_m) and np.isfinite(ra_m) and rp_m > 0.0 and ra_m > 0.0:
                    a_m = 0.5 * (rp_m + ra_m)
                else:
                    a_m = float("nan")

                # Only bound orbits have an apoapsis to circularize at.
                if np.isfinite(a_m) and np.isfinite(ra_m) and a_m > 0.0:
                    mu = float(cfg_env.earth_mu)
                    v_apo = float(np.sqrt(max(0.0, mu * (2.0 / ra_m - 1.0 / a_m))))
                    v_circ = float(np.sqrt(max(0.0, mu / ra_m)))
                    dv_circ_signed = float(v_circ - v_apo)
                    dv_circ = float(abs(dv_circ_signed))

                    g0 = 9.80665
                    isp_vac = float(getattr(cfg_hw, "upper_isp_vac", 0.0) or 0.0)
                    if isp_vac <= 0.0:
                        raise ValueError("upper_isp_vac must be > 0 to compute circularization fuel")

                    idx_eval = int(results.get("eval_index", len(getattr(sim_log, "m", [])) - 1))
                    m_hist = getattr(sim_log, "m", None)
                    stage_hist = getattr(sim_log, "stage", None)
                    r_hist = getattr(sim_log, "r", None)
                    v_hist = getattr(sim_log, "v", None)
                    if not m_hist or not r_hist or not v_hist:
                        raise ValueError("simulation log missing mass/state history")
                    idx_eval = max(0, min(idx_eval, len(m_hist) - 1))
                    mass_eval_kg = float(m_hist[idx_eval])
                    stage_eval = int(stage_hist[idx_eval]) if stage_hist and idx_eval < len(stage_hist) else None

                    # If stage info is present and we're not on the upper stage, skip.
                    if stage_eval is not None and stage_eval != 1:
                        raise ValueError("circularization burn requires upper stage (stage==1)")

                    fuel_main = max(0.0, float(results.get("fuel", 0.0) or 0.0))
                    fuel_circ = float(mass_eval_kg * (1.0 - np.exp(-dv_circ / (g0 * isp_vac))))
                    fuel_circ = max(0.0, fuel_circ)

                    # Feasibility check using the stage dry mass (payload is folded into dry mass in this model).
                    upper_dry = float(getattr(cfg_hw, "upper_dry_mass", 0.0) or 0.0)
                    prop_remaining_est = max(0.0, mass_eval_kg - upper_dry)
                    circ_feasible = fuel_circ <= prop_remaining_est + 1e-6

                    # Preserve pre-circularization values for debugging/logging.
                    results["status_pre_circ"] = pre_status
                    results["perigee_alt_pre_circ_m"] = float(perigee_pre) if np.isfinite(perigee_pre) else float("nan")
                    results["apoapsis_alt_pre_circ_m"] = float(apo_pre) if np.isfinite(apo_pre) else float("nan")
                    results["eccentricity_pre_circ"] = float(ecc_pre) if np.isfinite(ecc_pre) else float("nan")
                    results["orbit_error_pre_circ_m"] = float(orbit_error_pre) if np.isfinite(orbit_error_pre) else float("nan")

                    results["circ_applied"] = bool(circ_feasible)
                    results["circ_dv_mps"] = float(dv_circ)
                    results["fuel_main_kg"] = float(fuel_main)
                    results["fuel_circ_kg"] = float(fuel_circ)
                    results["fuel_total_kg"] = float(fuel_main + fuel_circ)
                    results["mass_eval_kg"] = float(mass_eval_kg)
                    if stage_eval is not None:
                        results["stage_eval"] = int(stage_eval)

                    if circ_feasible:
                        # After circularization at apoapsis, the orbit is circular at ra.
                        alt_post = float(ra_m - float(cfg_env.earth_radius_m))
                        results["perigee_alt_m"] = alt_post
                        results["apoapsis_alt_m"] = alt_post
                        results["eccentricity"] = 0.0
                        results["rp_m"] = float(ra_m)
                        results["ra_m"] = float(ra_m)
                        results["cutoff_reason_pre_circ"] = cutoff_reason_pre
                        if not cutoff_reason_pre or cutoff_reason_pre == "impact":
                            results["cutoff_reason"] = "circ_applied"

                        target_r = float(cfg_env.earth_radius_m) + float(cfg_sim.target_orbit_alt_m)
                        ra_error = abs(float(ra_m) - target_r)
                        results["perigee_error_m"] = float(ra_error)
                        results["apoapsis_error_m"] = float(ra_error)
                        results["orbital_error"] = float(ra_error)

                        if results["orbital_error"] < TARGET_TOLERANCE_M * 0.5:
                            results["status"] = "PERFECT"
                        elif results["orbital_error"] < TARGET_TOLERANCE_M * 2:
                            results["status"] = "GOOD"
                        else:
                            results["status"] = "OK"

                        results["fuel"] = float(results["fuel_total_kg"])
                        results["cost"] = calculate_cost(
                            results, phase, cfg_sim.target_orbit_alt_m, cfg_env.earth_radius_m
                        )

                        # If requested, stitch the returned log so the final plot
                        # shows the coast to apoapsis and the circular orbit after
                        # the burn (two-body propagation).
                        if return_log:
                            try:
                                from Main.telemetry import Logger

                                def _copy_prefix(src: Logger, end_idx: int) -> Logger:
                                    out = Logger()
                                    n = max(0, int(end_idx) + 1)
                                    for attr in out.__dict__.keys():
                                        dst_val = getattr(out, attr, None)
                                        src_val = getattr(src, attr, None)
                                        if isinstance(dst_val, list) and isinstance(src_val, list):
                                            setattr(out, attr, src_val[:n].copy())
                                    out.orbit_achieved = bool(getattr(src, "orbit_achieved", False))
                                    out.cutoff_reason = str(getattr(src, "cutoff_reason", "") or "")
                                    return out

                                def _rk4_two_body_step(r_m: np.ndarray, v_mps: np.ndarray, mu_val: float, dt_val: float):
                                    def accel(rr: np.ndarray) -> np.ndarray:
                                        r_norm = float(np.linalg.norm(rr))
                                        return (-float(mu_val) * rr) / max(r_norm**3, 1e-9)

                                    k1_r = v_mps
                                    k1_v = accel(r_m)

                                    r2 = r_m + 0.5 * dt_val * k1_r
                                    v2 = v_mps + 0.5 * dt_val * k1_v
                                    k2_r = v2
                                    k2_v = accel(r2)

                                    r3 = r_m + 0.5 * dt_val * k2_r
                                    v3 = v_mps + 0.5 * dt_val * k2_v
                                    k3_r = v3
                                    k3_v = accel(r3)

                                    r4 = r_m + dt_val * k3_r
                                    v4 = v_mps + dt_val * k3_v
                                    k4_r = v4
                                    k4_v = accel(r4)

                                    r_next = r_m + (dt_val / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
                                    v_next = v_mps + (dt_val / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
                                    return r_next, v_next

                                # Find apoapsis after cutoff in the simulated log.
                                idx_cutoff = int(results.get("eval_index", len(sim_log.t_sim) - 1))
                                idx_cutoff = max(0, min(idx_cutoff, len(sim_log.t_sim) - 1))
                                if getattr(sim_log, "altitude", None):
                                    alt = np.asarray(sim_log.altitude, dtype=float)
                                    n_alt = min(len(alt), len(sim_log.t_sim))
                                    start = min(idx_cutoff, n_alt - 1)
                                    idx_apo = start + int(np.argmax(alt[start:n_alt]))
                                else:
                                    r_norms = np.array([float(np.linalg.norm(r)) for r in sim_log.r], dtype=float)
                                    start = min(idx_cutoff, len(r_norms) - 1)
                                    idx_apo = start + int(np.argmax(r_norms[start:]))
                                idx_apo = max(0, min(idx_apo, len(sim_log.t_sim) - 1))

                                stitched = _copy_prefix(sim_log, idx_apo)
                                stitched.cutoff_reason = "circ_applied"
                                stitched.orbit_achieved = True

                                r_apo_vec = np.asarray(stitched.r[-1], dtype=float)
                                v_apo_vec = np.asarray(stitched.v[-1], dtype=float)
                                r_norm = float(np.linalg.norm(r_apo_vec))
                                if r_norm <= 0.0:
                                    raise ValueError("invalid apoapsis state")
                                r_hat = r_apo_vec / r_norm
                                v_tan = v_apo_vec - float(np.dot(v_apo_vec, r_hat)) * r_hat
                                v_tan_norm = float(np.linalg.norm(v_tan))
                                if v_tan_norm <= 1e-9:
                                    raise ValueError("apoapsis tangential velocity too small")
                                tan_hat = v_tan / v_tan_norm
                                v_circ_local = float(np.sqrt(mu / r_norm))
                                v_post = tan_hat * v_circ_local

                                mass_apo = float(stitched.m[-1])
                                mass_post = max(0.0, mass_apo - fuel_circ)

                                # Overwrite the apoapsis sample with the post-burn state.
                                stitched.r[-1] = r_apo_vec.copy()
                                stitched.v[-1] = v_post.copy()
                                stitched.m[-1] = float(mass_post)
                                stitched.stage[-1] = int(stitched.stage[-1]) if stitched.stage else 1
                                if stitched.altitude:
                                    stitched.altitude[-1] = float(r_norm - float(cfg_env.earth_radius_m))
                                if stitched.speed:
                                    stitched.speed[-1] = float(np.linalg.norm(v_post))
                                if stitched.thrust_mag:
                                    stitched.thrust_mag[-1] = 0.0
                                if stitched.drag_mag:
                                    stitched.drag_mag[-1] = 0.0
                                if stitched.mdot:
                                    stitched.mdot[-1] = 0.0
                                if stitched.dynamic_pressure:
                                    stitched.dynamic_pressure[-1] = 0.0
                                if stitched.rho:
                                    stitched.rho[-1] = 0.0
                                if stitched.mach:
                                    stitched.mach[-1] = 0.0

                                # Propagate under two-body for the remainder of the horizon.
                                t = float(stitched.t_sim[-1])
                                t_end = float(cfg_sim.main_duration_s)
                                dt_val = float(cfg_sim.main_dt_s)
                                r_curr = r_apo_vec.copy()
                                v_curr = v_post.copy()
                                stage_val = int(stitched.stage[-1]) if stitched.stage else 1
                                while t + dt_val <= t_end + 1e-9:
                                    r_curr, v_curr = _rk4_two_body_step(r_curr, v_curr, mu, dt_val)
                                    t = t + dt_val

                                    stitched.t_sim.append(float(t))
                                    stitched.t_env.append(float(t))
                                    stitched.r.append(np.asarray(r_curr, dtype=float).copy())
                                    stitched.v.append(np.asarray(v_curr, dtype=float).copy())
                                    stitched.m.append(float(mass_post))
                                    stitched.stage.append(stage_val)

                                    r_norm_step = float(np.linalg.norm(r_curr))
                                    altitude = r_norm_step - float(cfg_env.earth_radius_m)
                                    speed = float(np.linalg.norm(v_curr))
                                    r_hat_step = (
                                        r_curr / r_norm_step
                                        if r_norm_step > 0.0
                                        else np.array([0.0, 0.0, 1.0], dtype=float)
                                    )
                                    v_vertical = float(np.dot(v_curr, r_hat_step)) if r_norm_step > 0.0 else 0.0
                                    v_horizontal = float(np.sqrt(max(0.0, speed * speed - v_vertical * v_vertical)))
                                    fpa_deg = (
                                        float(np.degrees(np.arctan2(v_vertical, v_horizontal)))
                                        if (v_horizontal > 0.0 or v_vertical != 0.0)
                                        else 0.0
                                    )
                                    specific_energy = 0.5 * speed * speed - mu / max(r_norm_step, 1e-6)

                                    stitched.altitude.append(float(altitude))
                                    stitched.speed.append(float(speed))
                                    stitched.thrust_mag.append(0.0)
                                    stitched.drag_mag.append(0.0)
                                    stitched.mdot.append(0.0)
                                    stitched.dynamic_pressure.append(0.0)
                                    stitched.rho.append(0.0)
                                    stitched.mach.append(0.0)
                                    stitched.flight_path_angle_deg.append(float(fpa_deg))
                                    stitched.v_vertical.append(float(v_vertical))
                                    stitched.v_horizontal.append(float(v_horizontal))
                                    stitched.specific_energy.append(float(specific_energy))

                                sim_log = stitched
                            except Exception:
                                pass
                    else:
                        # Cannot circularize with remaining propellant -> keep the raw simulation.
                        results["circ_applied"] = False
            except Exception:
                # Keep the optimizer robust: if circularization logic fails, fall back to raw simulation metrics.
                pass

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

    if return_log:
        results["log"] = sim_log
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
    def _env_int(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            return int(default)
        try:
            return max(1, int(float(raw)))
        except Exception:
            return int(default)

    def _env_float(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            return float(default)
        try:
            return float(raw)
        except Exception:
            return float(default)

    env_config = EnvironmentConfig()
    hw_config = HardwareConfig()
    sw_config = SoftwareConfig()
    sim_config = SimulationConfig()
    log_config = LoggingConfig()
    analysis_config = AnalysisConfig()

    # Initialize shared counter and a lock for logging from worker processes.
    global global_iter_count, global_log_lock
    global_iter_count = multiprocessing.Value("i", 0)
    global_log_lock = multiprocessing.Lock()

    # --- Initial Guess & Bounds (SCALED) ---
    # The bounds are now managed centrally in Analysis/config.py
    bounds_phys_full = OptimizationBounds.get_bounds()
    lb_phys_full = np.array([b[0] for b in bounds_phys_full], dtype=float)
    ub_phys_full = np.array([b[1] for b in bounds_phys_full], dtype=float)
    span_phys_full = ub_phys_full - lb_phys_full
    fixed_mask_phys_full = span_phys_full == 0.0
    span_phys_full_safe = np.where(fixed_mask_phys_full, 1.0, span_phys_full)

    def to_unit(x_phys: np.ndarray) -> np.ndarray:
        x = np.asarray(x_phys, dtype=float)
        u = (x - lb_phys_full) / span_phys_full_safe
        # Fixed dimensions map to 0 in unit space (any value is equivalent).
        if np.any(fixed_mask_phys_full):
            u = np.where(fixed_mask_phys_full, 0.0, u)
        return u

    def from_unit(x_unit: np.ndarray) -> np.ndarray:
        u = np.asarray(x_unit, dtype=float)
        # IMPORTANT: use the true span (can be 0 for fixed dims) so those
        # parameters remain fixed at lb_phys.
        return lb_phys_full + np.clip(u, 0.0, 1.0) * span_phys_full

    def tighten_bounds_unit(bounds_in: list[tuple[float, float]], seed: np.ndarray, margin: float = 0.2):
        tightened = []
        seed = np.asarray(seed, dtype=float)
        for (lo, hi), s in zip(bounds_in, seed):
            width = hi - lo
            new_lo = max(lo, s - margin * width)
            new_hi = min(hi, s + margin * width)
            if new_lo > new_hi:
                new_lo, new_hi = lo, hi
            tightened.append((new_lo, new_hi))
        return tightened

    def make_unit_mapper(bounds_phys: list[tuple[float, float]]):
        lb = np.array([b[0] for b in bounds_phys], dtype=float)
        ub = np.array([b[1] for b in bounds_phys], dtype=float)
        span = ub - lb
        fixed = span == 0.0
        span_safe = np.where(fixed, 1.0, span)

        def to_unit_local(x_phys: np.ndarray) -> np.ndarray:
            x = np.asarray(x_phys, dtype=float)
            u = (x - lb) / span_safe
            if np.any(fixed):
                u = np.where(fixed, 0.0, u)
            return u

        def from_unit_local(x_unit: np.ndarray) -> np.ndarray:
            u = np.asarray(x_unit, dtype=float)
            return lb + np.clip(u, 0.0, 1.0) * span

        return lb, ub, to_unit_local, from_unit_local

    def apply_fixed_defaults(params_phys_full: np.ndarray) -> np.ndarray:
        """Remove nuisance degrees of freedom for faster convergence."""
        x = np.array(params_phys_full, dtype=float, copy=True)
        # Azimuth is not constrained by the cost function (no inclination target),
        # so fixing it avoids wasting search effort.
        x[14] = 0.0

        # By default we allow the optimizer to shape the *upper-stage* throttle
        # profile. This extra DOF is often required to achieve true direct
        # insertion (both perigee and apoapsis at the target altitude) without a
        # separate circularization burn.
        optimize_upper_throttle = str(os.getenv("ORBITLAUNCH_OPTIMIZE_UPPER_THROTTLE", "1")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        optimize_booster_throttle = str(os.getenv("ORBITLAUNCH_OPTIMIZE_BOOSTER_THROTTLE", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        if not optimize_upper_throttle:
            x[21:25] = 1.0
            x[25:28] = np.array([0.1, 0.5, 0.9], dtype=float)

        if not optimize_booster_throttle:
            x[28:32] = 1.0
            x[32:35] = np.array([0.1, 0.5, 0.9], dtype=float)
        return x

    optimize_upper_throttle = str(os.getenv("ORBITLAUNCH_OPTIMIZE_UPPER_THROTTLE", "1")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    optimize_booster_throttle = str(os.getenv("ORBITLAUNCH_OPTIMIZE_BOOSTER_THROTTLE", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    fixed_indices = {14}
    if not optimize_upper_throttle:
        fixed_indices.update(range(21, 28))
    if not optimize_booster_throttle:
        fixed_indices.update(range(28, 35))
    active_indices = [i for i in range(35) if i not in fixed_indices]

    if analysis_config.optimizer_manual_seed and len(analysis_config.optimizer_manual_seed) == 35:
        start_params_phys_full = np.array(analysis_config.optimizer_manual_seed, dtype=float)
    else:
        start_params_phys_full = (lb_phys_full + ub_phys_full) / 2.0

    # Ensure seed respects bounds and fixed defaults.
    start_params_phys_full = np.clip(start_params_phys_full, lb_phys_full, ub_phys_full)
    start_params_phys_full = apply_fixed_defaults(start_params_phys_full)
    start_params_phys_full = np.clip(start_params_phys_full, lb_phys_full, ub_phys_full)

    # Pre-build bounds/mappers for the active subset.
    bounds_active_phys = [bounds_phys_full[i] for i in active_indices]
    bounds_active_unit = [(0.0, 1.0)] * len(active_indices)
    _lb_active, _ub_active, to_unit_active, from_unit_active = make_unit_mapper(bounds_active_phys)

    ensure_log_header()
    print(f"=== PHASE 1: TARGETING ORBIT (Logging to {LOG_FILENAME}) ===", flush=True)
    with global_iter_count.get_lock():
        global_iter_count.value = 0

    # Evaluate the seed once to choose reasonable optimizer hyperparameters.
    seed_results = run_simulation_wrapper(
        start_params_phys_full,
        env_config,
        hw_config,
        sw_config,
        sim_config,
        log_config,
        phase=1,
    )
    seed_status = seed_results.get("status", "UNKNOWN")
    seed_orbit_error = float(seed_results.get("orbital_error", PENALTY_CRASH) or PENALTY_CRASH)

    if seed_status in {"OK", "GOOD", "PERFECT"}:
        phase1_maxiter = 80
        phase1_popsize = 16
        sigma1 = 0.20
    else:
        phase1_maxiter = 200
        phase1_popsize = 24
        sigma1 = 0.35

    # Optional overrides for experimentation / quick smoke runs.
    phase1_maxiter = _env_int("ORBITLAUNCH_PHASE1_MAXITER", phase1_maxiter)
    phase1_popsize = _env_int("ORBITLAUNCH_PHASE1_POPSIZE", phase1_popsize)

    start_active_unit = np.clip(to_unit_active(start_params_phys_full[active_indices]), 0.0, 1.0)

    objective_phase1 = ObjectiveFunctionWrapper(
        phase=1,
        label="Phase 1",
        env_config=env_config,
        hw_config=hw_config,
        sw_config=sw_config,
        sim_config=sim_config,
        log_config=log_config,
        analysis_config=analysis_config,
        bounds=bounds_active_phys,
        param_space="unit",
        active_indices=active_indices,
        base_params_phys=start_params_phys_full,
        enable_logging=True,
    )

    if CMA_AVAILABLE:
        print("Using CMA-ES for Phase 1", flush=True)
        res1 = run_cma_phase(
            objective_phase1,
            bounds_active_unit,
            global_iter_count,
            global_log_lock,
            start=start_active_unit,
            sigma_scale=sigma1,
            maxiter=phase1_maxiter,
            popsize=phase1_popsize,
        )
        best1_unit_active = np.asarray(res1.xbest, dtype=float)
        best1_cost = res1.fbest
    else:
        print("CMA-ES not available, falling back to Differential Evolution for Phase 1", flush=True)
        res = differential_evolution(
            objective_phase1,
            bounds_active_unit,
            maxiter=phase1_maxiter,
            disp=True
        )
        best1_unit_active = np.asarray(res.x, dtype=float)
        best1_cost = res.fun

    best1_active_phys = from_unit_active(best1_unit_active)
    best1_phys_full = np.array(start_params_phys_full, dtype=float, copy=True)
    best1_phys_full[active_indices] = best1_active_phys
    best1_phys_full = apply_fixed_defaults(best1_phys_full)
    best1_phys_full = np.clip(best1_phys_full, lb_phys_full, ub_phys_full)

    # Re-evaluate the best phase-1 candidate to confirm it is a bound, survivable
    # orbit (some optimizers can report a best point that is only marginally
    # better due to numerical noise).
    best1_results = run_simulation_wrapper(
        best1_phys_full,
        env_config,
        hw_config,
        sw_config,
        sim_config,
        log_config,
        phase=1,
    )

    # Never allow Phase 1 to get worse than the (already orbiting) manual seed.
    # CMA-ES does not automatically evaluate the starting point as a candidate.
    seed_cost1 = float(seed_results.get("cost", PENALTY_CRASH) or PENALTY_CRASH)
    best1_cost1 = float(best1_results.get("cost", best1_cost) or best1_cost)
    seed_is_orbit = seed_status in {"OK", "GOOD", "PERFECT"}
    best1_is_orbit = best1_results.get("status") in {"OK", "GOOD", "PERFECT"}
    if seed_is_orbit and (not best1_is_orbit or seed_cost1 <= best1_cost1):
        best1_phys_full = np.array(start_params_phys_full, dtype=float, copy=True)
        best1_results = seed_results
        best1_cost1 = seed_cost1
        best1_is_orbit = True

    print(f"\n--- Phase 1 Complete ---", flush=True)
    print(f"Seed status: {seed_status} | Seed orbit error: {seed_orbit_error/1000:.1f} km", flush=True)
    print(f"Best Error/Cost: {best1_cost1/1000:.1f} km", flush=True)
    print(
        f"Phase 1 Best (summary): Mach={best1_phys_full[0]:.2f}, Coast={best1_phys_full[11]:.1f}s, "
        f"Upper Burn={best1_phys_full[12]:.1f}s. Full details in {LOG_FILENAME}",
        flush=True,
    )

    if not best1_is_orbit:
        print(
            f"\nFailed to find a stable orbit in Phase 1 (best status: {best1_results.get('status')}). Stopping.",
            flush=True,
        )
        return

    best1_orbit_error = float(best1_results.get("orbital_error", best1_cost1) or best1_cost1)

    print("\n=== PHASE 2: MINIMIZING FUEL (CMA-ES if available) ===", flush=True)
    with global_iter_count.get_lock():
        global_iter_count.value = 0  # Reset for phase 2 logging

    base_params_phase2_phys_full = np.array(best1_phys_full, dtype=float, copy=True)
    base_params_phase2_phys_full = apply_fixed_defaults(base_params_phase2_phys_full)
    base_params_phase2_phys_full = np.clip(base_params_phase2_phys_full, lb_phys_full, ub_phys_full)
    start2_unit_active = np.clip(to_unit_active(base_params_phase2_phys_full[active_indices]), 0.0, 1.0)

    # Baseline phase-2 cost at the phase-1 best orbit; we'll never accept a worse result.
    baseline2_results = run_simulation_wrapper(
        base_params_phase2_phys_full,
        env_config,
        hw_config,
        sw_config,
        sim_config,
        log_config,
        phase=2,
    )
    baseline2_cost = float(baseline2_results.get("cost", PENALTY_CRASH) or PENALTY_CRASH)

    objective_phase2_coarse = ObjectiveFunctionWrapper(
        phase=2,
        label="Phase 2 (coarse)",
        env_config=env_config,
        hw_config=hw_config,
        sw_config=sw_config,
        sim_config=sim_config,
        log_config=log_config,
        analysis_config=analysis_config,
        bounds=bounds_active_phys,
        param_space="unit",
        active_indices=active_indices,
        base_params_phys=base_params_phase2_phys_full,
        enable_logging=True,
    )

    # NOTE: For direct insertion we often need to move substantially from the
    # phase-1 solution to lift the perigee to the target altitude. Tightening
    # too aggressively can trap the search in a local minimum where apoapsis is
    # near-target but perigee remains low. Use a looser default tightening
    # schedule unless phase 1 already got very close.
    if best1_orbit_error <= 30_000.0:
        bounds2_unit_active = tighten_bounds_unit(bounds_active_unit, start2_unit_active, margin=0.15)
        sigma2 = 0.15
    elif best1_orbit_error <= 200_000.0:
        bounds2_unit_active = tighten_bounds_unit(bounds_active_unit, start2_unit_active, margin=0.30)
        sigma2 = 0.25
    elif best1_orbit_error <= 1_000_000.0:
        bounds2_unit_active = tighten_bounds_unit(bounds_active_unit, start2_unit_active, margin=0.40)
        sigma2 = 0.35
    else:
        bounds2_unit_active = bounds_active_unit
        sigma2 = 0.45

    phase2_coarse_maxiter = _env_int("ORBITLAUNCH_PHASE2_COARSE_MAXITER", 200)
    phase2_coarse_popsize = _env_int("ORBITLAUNCH_PHASE2_COARSE_POPSIZE", 16)

    if CMA_AVAILABLE:
        res2 = run_cma_phase(
            objective_phase2_coarse,
            bounds2_unit_active,
            global_iter_count,
            global_log_lock,
            start=start2_unit_active,
            sigma_scale=sigma2,
            maxiter=phase2_coarse_maxiter,
            popsize=phase2_coarse_popsize,
        )
        best2_unit_active = np.asarray(res2.xbest, dtype=float)
        best2_cost = float(res2.fbest)
    else:
        res = differential_evolution(
            objective_phase2_coarse,
            bounds2_unit_active,
            maxiter=phase2_coarse_maxiter,
            disp=True,
        )
        best2_unit_active = np.asarray(res.x, dtype=float)
        best2_cost = float(res.fun)

    best2_active_phys = from_unit_active(best2_unit_active)
    best2_phys_full = np.array(base_params_phase2_phys_full, dtype=float, copy=True)
    best2_phys_full[active_indices] = best2_active_phys
    best2_phys_full = apply_fixed_defaults(best2_phys_full)
    best2_phys_full = np.clip(best2_phys_full, lb_phys_full, ub_phys_full)

    best2_results = run_simulation_wrapper(
        best2_phys_full,
        env_config,
        hw_config,
        sw_config,
        sim_config,
        log_config,
        phase=2,
    )
    best2_cost_eval = float(best2_results.get("cost", best2_cost) or best2_cost)
    best2_is_orbit = best2_results.get("status") in {"OK", "GOOD", "PERFECT"}
    baseline2_is_orbit = baseline2_results.get("status") in {"OK", "GOOD", "PERFECT"}
    if baseline2_is_orbit and (not best2_is_orbit or baseline2_cost <= best2_cost_eval):
        best2_unit_active = start2_unit_active
        best2_phys_full = np.array(base_params_phase2_phys_full, dtype=float, copy=True)
        best2_cost_eval = baseline2_cost
        best2_results = baseline2_results

    # --- Phase 2 Polish (higher-fidelity dt) ---
    print("\n=== PHASE 2: POLISH (higher fidelity) ===", flush=True)
    with global_iter_count.get_lock():
        global_iter_count.value = 0

    # Tighten around the best coarse solution, relative to the already-tight bounds.
    bounds2_polish_unit_active = tighten_bounds_unit(bounds2_unit_active, best2_unit_active, margin=0.20)

    objective_phase2_polish = ObjectiveFunctionWrapper(
        phase=2,
        label="Phase 2 (polish)",
        env_config=env_config,
        hw_config=hw_config,
        sw_config=sw_config,
        sim_config=sim_config,
        log_config=log_config,
        analysis_config=analysis_config,
        bounds=bounds_active_phys,
        param_space="unit",
        active_indices=active_indices,
        base_params_phys=base_params_phase2_phys_full,
        dt_s=_env_float("ORBITLAUNCH_PHASE2_POLISH_DT_S", 0.25),
        enable_logging=True,
    )

    phase2_polish_maxiter = _env_int("ORBITLAUNCH_PHASE2_POLISH_MAXITER", 80)
    phase2_polish_popsize = _env_int("ORBITLAUNCH_PHASE2_POLISH_POPSIZE", 16)

    if CMA_AVAILABLE:
        res2p = run_cma_phase(
            objective_phase2_polish,
            bounds2_polish_unit_active,
            global_iter_count,
            global_log_lock,
            start=best2_unit_active,
            sigma_scale=0.10,
            maxiter=phase2_polish_maxiter,
            popsize=phase2_polish_popsize,
        )
        best2p_unit_active = np.asarray(res2p.xbest, dtype=float)
        best2p_cost = float(res2p.fbest)
    else:
        res = differential_evolution(
            objective_phase2_polish,
            bounds2_polish_unit_active,
            maxiter=phase2_polish_maxiter,
            disp=True,
        )
        best2p_unit_active = np.asarray(res.x, dtype=float)
        best2p_cost = float(res.fun)

    best2p_active_phys = from_unit_active(best2p_unit_active)
    final_params2 = np.array(base_params_phase2_phys_full, dtype=float, copy=True)
    final_params2[active_indices] = best2p_active_phys
    final_params2 = apply_fixed_defaults(final_params2)
    final_params2 = np.clip(final_params2, lb_phys_full, ub_phys_full)

    print("\n=== OPTIMIZATION COMPLETE ===", flush=True)

    # Compare the polished solution to the best coarse/baseline solution at the
    # same (fine) fidelity before reporting final numbers.
    eval_dt = _env_float("ORBITLAUNCH_FINAL_EVAL_DT_S", _env_float("ORBITLAUNCH_PHASE2_POLISH_DT_S", 0.25))
    eval_dt = max(1e-4, float(eval_dt))
    candidate_results_polish = run_simulation_wrapper(
        final_params2,
        env_config,
        hw_config,
        sw_config,
        sim_config,
        log_config,
        phase=2,
        dt_s=eval_dt,
    )
    candidate_results_coarse = run_simulation_wrapper(
        best2_phys_full,
        env_config,
        hw_config,
        sw_config,
        sim_config,
        log_config,
        phase=2,
        dt_s=eval_dt,
    )
    candidate_cost_polish = float(candidate_results_polish.get("cost", PENALTY_CRASH) or PENALTY_CRASH)
    candidate_cost_coarse = float(candidate_results_coarse.get("cost", PENALTY_CRASH) or PENALTY_CRASH)

    if candidate_cost_coarse <= candidate_cost_polish:
        final_params2 = np.array(best2_phys_full, dtype=float, copy=True)
        final_results = candidate_results_coarse
    else:
        final_results = candidate_results_polish

    print(f"Final Fuel Used: {final_results['fuel']:.1f} kg", flush=True)
    print(f"Final Orbit Error: {final_results['orbital_error']/1000:.1f} km", flush=True)
    print(
        f"Optimal Parameters (summary): Mach={final_params2[0]:.2f}, Coast={final_params2[11]:.1f}s, "
        f"Upper Burn={final_params2[12]:.1f}s. Full details in {LOG_FILENAME}",
        flush=True,
    )

    # --- Final trajectory plot + summary ---
    plot_final = str(os.getenv("ORBITLAUNCH_PLOT_FINAL", "1")).strip().lower() not in {"0", "false", "no", "off"}
    animate_final = str(os.getenv("ORBITLAUNCH_ANIMATE_FINAL", "0")).strip().lower() in {"1", "true", "yes", "on"}
    if plot_final or animate_final:
        from Analysis.plotting import plot_trajectory_3d, animate_trajectory

        final_dt_s = _env_float("ORBITLAUNCH_FINAL_DT_S", 0.25)
        final_duration_s = _env_float("ORBITLAUNCH_FINAL_DURATION_S", 6000.0)
        final_dt_s = max(1e-4, float(final_dt_s))
        final_duration_s = max(1.0, float(final_duration_s))

        print(
            f"\n=== FINAL TRAJECTORY (dt={final_dt_s:.3f}s, duration={final_duration_s:.0f}s) ===",
            flush=True,
        )
        final_traj_results = run_simulation_wrapper(
            final_params2,
            env_config,
            hw_config,
            sw_config,
            sim_config,
            log_config,
            phase=2,
            dt_s=final_dt_s,
            duration_s=final_duration_s,
            return_log=True,
        )
        final_log = final_traj_results.get("log")
        if final_log is None:
            print(
                f"Final plot skipped: simulation log unavailable (status={final_traj_results.get('status', 'UNKNOWN')})",
                flush=True,
            )
            return

        # Print a concise summary at the end of the program.
        try:
            final_time_s = float(final_log.t_sim[-1]) if final_log.t_sim else 0.0
            final_alt_km = (
                (float(np.linalg.norm(final_log.r[-1])) - float(env_config.earth_radius_m)) / 1000.0
                if getattr(final_log, "r", None)
                else 0.0
            )
            final_speed_mps = float(np.linalg.norm(final_log.v[-1])) if getattr(final_log, "v", None) else 0.0
            final_mass_kg = float(final_log.m[-1]) if getattr(final_log, "m", None) else 0.0
            max_alt_km = (float(max(final_log.altitude)) / 1000.0) if getattr(final_log, "altitude", None) else 0.0
            max_q_kpa = (float(max(final_log.dynamic_pressure)) / 1000.0) if getattr(final_log, "dynamic_pressure", None) else 0.0

            stage_switch_times = []
            if getattr(final_log, "stage", None) and getattr(final_log, "t_sim", None):
                stage_switch_times = [
                    float(final_log.t_sim[i])
                    for i in range(1, min(len(final_log.stage), len(final_log.t_sim)))
                    if final_log.stage[i] != final_log.stage[i - 1]
                ]

            per_km = float(final_traj_results.get("perigee_alt_m", float("nan"))) / 1000.0
            apo_km = float(final_traj_results.get("apoapsis_alt_m", float("nan"))) / 1000.0
            ecc = float(final_traj_results.get("eccentricity", float("nan")))
            err_km = float(final_traj_results.get("orbital_error", float("nan"))) / 1000.0

            circ_applied = bool(final_traj_results.get("circ_applied", False))
            per_pre_km = float(final_traj_results.get("perigee_alt_pre_circ_m", float("nan"))) / 1000.0
            apo_pre_km = float(final_traj_results.get("apoapsis_alt_pre_circ_m", float("nan"))) / 1000.0
            ecc_pre = float(final_traj_results.get("eccentricity_pre_circ", float("nan")))
            err_pre_km = float(final_traj_results.get("orbit_error_pre_circ_m", float("nan"))) / 1000.0
            dv_circ = float(final_traj_results.get("circ_dv_mps", float("nan")))
            fuel_main_kg = float(final_traj_results.get("fuel_main_kg", float("nan")))
            fuel_circ_kg = float(final_traj_results.get("fuel_circ_kg", float("nan")))
            fuel_kg = float(final_traj_results.get("fuel", float("nan")))
            cutoff_reason = str(final_traj_results.get("cutoff_reason", "") or "")
            status = str(final_traj_results.get("status", "UNKNOWN"))

            # Optional orbital period estimate (elliptical orbits only).
            a_m = None
            try:
                rp_m = float(final_traj_results.get("rp_m", float("nan")))
                ra_m = float(final_traj_results.get("ra_m", float("nan")))
                if np.isfinite(rp_m) and np.isfinite(ra_m) and rp_m > 0.0 and ra_m > 0.0:
                    a_m = 0.5 * (rp_m + ra_m)
            except Exception:
                a_m = None
            period_s = None
            if a_m is not None and a_m > 0.0:
                period_s = float(2.0 * np.pi * np.sqrt(a_m**3 / float(env_config.earth_mu)))

            print("\n--- Final Trajectory Summary ---", flush=True)
            print(f"Status: {status} | Cutoff: {cutoff_reason}", flush=True)
            print(f"t_end: {final_time_s:.1f} s | Max alt: {max_alt_km:.1f} km | Max Q: {max_q_kpa:.1f} kPa", flush=True)
            print(
                f"Final alt: {final_alt_km:.1f} km | Final speed: {final_speed_mps:.1f} m/s | Final mass: {final_mass_kg:.0f} kg",
                flush=True,
            )
            print(
                f"Perigee: {per_km:.1f} km | Apoapsis: {apo_km:.1f} km | e: {ecc:.4f} | Orbit error: {err_km:.2f} km",
                flush=True,
            )

            print(f"Propellant burned: {fuel_kg:.1f} kg", flush=True)

            print_fuel_breakdown = str(os.getenv("ORBITLAUNCH_PRINT_FUEL_BREAKDOWN", "0")).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if print_fuel_breakdown and circ_applied and np.isfinite(fuel_main_kg) and np.isfinite(fuel_circ_kg):
                print(f"  - main burn: {fuel_main_kg:.1f} kg", flush=True)
                print(f"  - circularization: {fuel_circ_kg:.1f} kg", flush=True)

            print_circ_details = str(os.getenv("ORBITLAUNCH_PRINT_CIRC_DETAILS", "0")).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if print_circ_details and circ_applied:
                print(f"Circularization: Δv {dv_circ:.2f} m/s | Fuel {fuel_circ_kg:.1f} kg", flush=True)
                if np.isfinite(per_pre_km) and np.isfinite(apo_pre_km) and np.isfinite(ecc_pre) and np.isfinite(err_pre_km):
                    print(
                        f"Pre-circ orbit: Perigee {per_pre_km:.1f} km | Apoapsis {apo_pre_km:.1f} km | e {ecc_pre:.4f} | Error {err_pre_km:.2f} km",
                        flush=True,
                    )
            if stage_switch_times:
                print(f"Stage switches at t={', '.join(f'{t:.1f}s' for t in stage_switch_times)}", flush=True)
            if period_s is not None and np.isfinite(period_s):
                print(f"Estimated orbital period: {period_s/60.0:.1f} min", flush=True)
        except Exception:
            print("WARNING: failed to print final trajectory summary.", flush=True)

        if plot_final:
            plot_trajectory_3d(final_log, env_config.earth_radius_m)
        if animate_final:
            animate_trajectory(final_log, env_config.earth_radius_m)


if __name__ == "__main__":
    try:
        run_optimization()
    except Exception as e:
        print(f"ERROR: An unhandled exception occurred during optimization: {e}", flush=True)
        import traceback
        traceback.print_exc()
