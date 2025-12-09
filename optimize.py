"""
Simple fuel-minimization search for a 2-stage rocket to reach a 420 km circular orbit.

This script samples propellant masses, throttle levels, and a simple altitude-based
pitch profile, runs the simulation, and records results to CSV.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
try:
    import cma
    CMA_AVAILABLE = True
except ImportError:
    CMA_AVAILABLE = False

from aerodynamics import Aerodynamics, CdModel
from atmosphere import AtmosphereModel
from gravity import EarthModel, MU_EARTH, OMEGA_EARTH, R_EARTH
from integrators import RK4, State
from rocket import Engine, Rocket, Stage
from simulation import Guidance, Simulation, ControlCommand
from config import CFG

TARGET_ORBIT_RADIUS = R_EARTH + CFG.target_orbit_alt_m
G0 = 9.80665
ORBIT_ALT_TOL = CFG.orbit_alt_tol
ORBIT_ECC_TOL = CFG.orbit_ecc_tol
ORBIT_SPEED_TOL = CFG.orbit_speed_tol
ORBIT_RADIAL_TOL = CFG.orbit_radial_tol


@dataclass
class SampleParams:
    prop1: float
    prop2: float
    throttle1_a: float
    throttle1_b: float
    t1_split: float
    throttle2_a: float
    throttle2_b: float
    t2_split: float
    pitch_alt1: float
    pitch_alt2: float
    pitch_alt3: float
    pitch_alt4: float
    pitch_ang1_deg: float
    pitch_ang2_deg: float
    pitch_ang3_deg: float
    pitch_ang4_deg: float


def make_pitch_program(alts: list[float], angs_deg: list[float]) -> Callable:
    def pitch_prog(t: float, state: State) -> np.ndarray:
        r = np.asarray(state.r_eci, dtype=float)
        v = np.asarray(state.v_eci, dtype=float)
        r_norm = np.linalg.norm(r)
        if r_norm == 0.0:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        r_hat = r / r_norm
        east = np.cross([0.0, 0.0, 1.0], r_hat)
        east_norm = np.linalg.norm(east)
        east = east / east_norm if east_norm > 0.0 else np.array([1.0, 0.0, 0.0], dtype=float)

        alt = r_norm - R_EARTH
        # Interpolate pitch angle vs altitude
        ang = np.interp(alt, alts, angs_deg)
        gamma = np.deg2rad(ang)
        direction = np.cos(gamma) * east + np.sin(gamma) * r_hat
        n = np.linalg.norm(direction)
        return direction / n if n > 0.0 else r_hat

    return pitch_prog


def make_throttle_schedule(t_split: float, throttle_a: float, throttle_b: float) -> Callable:
    def throttle_fn(t: float, state: State) -> float:
        return float(throttle_a if t < t_split else throttle_b)

    return throttle_fn


def build_simulation(params: SampleParams) -> tuple[Simulation, State, float]:
    earth = EarthModel(mu=MU_EARTH, radius=R_EARTH, omega_vec=OMEGA_EARTH)
    atmosphere = AtmosphereModel(lat_deg=CFG.launch_lat_deg, lon_deg=CFG.launch_lon_deg)
    cd_model = CdModel(2.0)

    booster_engine = Engine(
        thrust_vac=CFG.booster_thrust_vac,
        thrust_sl=CFG.booster_thrust_sl,
        isp_vac=CFG.booster_isp_vac,
        isp_sl=CFG.booster_isp_sl,
    )
    upper_engine = Engine(
        thrust_vac=CFG.upper_thrust_vac,
        thrust_sl=CFG.upper_thrust_sl,
        isp_vac=CFG.upper_isp_vac,
        isp_sl=CFG.upper_isp_sl,
    )
    booster_stage = Stage(
        dry_mass=CFG.booster_dry_mass,
        prop_mass=params.prop1,
        engine=booster_engine,
        ref_area=CFG.ref_area_m2,
    )
    upper_stage = Stage(
        dry_mass=CFG.upper_dry_mass,
        prop_mass=params.prop2,
        engine=upper_engine,
        ref_area=CFG.ref_area_m2,
    )
    rocket = Rocket(
        stages=[booster_stage, upper_stage],
        separation_delay=CFG.separation_delay_s,
        upper_ignition_delay=CFG.upper_ignition_delay_s,
        separation_altitude_m=None,
        earth_radius=R_EARTH,
    )

    # Ensure altitudes are sorted with padding at ends
    alts = sorted([params.pitch_alt1, params.pitch_alt2, params.pitch_alt3, params.pitch_alt4])
    angs = [params.pitch_ang1_deg, params.pitch_ang2_deg, params.pitch_ang3_deg, params.pitch_ang4_deg]
    pitch_prog = make_pitch_program(alts, angs)

    throttle_fn = make_throttle_schedule(params.t1_split, params.throttle1_a, params.throttle1_b)
    guidance = Guidance(pitch_program=pitch_prog, throttle_schedule=throttle_fn)

    aero = Aerodynamics(atmosphere=atmosphere, cd_model=cd_model, reference_area=None)
    integrator = RK4()
    sim = Simulation(
        earth=earth,
        atmosphere=atmosphere,
        aerodynamics=aero,
        rocket=rocket,
        integrator=integrator,
        guidance=guidance,
    )

    # Initial state at launch site
    lat = math.radians(CFG.launch_lat_deg)
    lon = math.radians(CFG.launch_lon_deg)
    r0 = R_EARTH * np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)], dtype=float)
    v0 = earth.atmosphere_velocity(r0)
    m0 = booster_stage.total_mass() + upper_stage.total_mass()
    state0 = State(r_eci=r0, v_eci=v0, m=m0, stage_index=0)
    t_env0 = 0.0
    return sim, state0, t_env0


def evaluate(
    params: SampleParams,
    duration: float = 1200.0,
    dt: float = 0.2,
    return_log: bool = False,
    speed_tol: float = ORBIT_SPEED_TOL,
    radial_tol: float = ORBIT_RADIAL_TOL,
    alt_tol: float = ORBIT_ALT_TOL,
    simulate_two_burn: bool = True,
):
    sim, state0, t_env0 = build_simulation(params)

    # Replace throttle schedule for stage 2 (simple: constant throttle2 after ignition)
    def throttle_sched(t: float, state: State) -> float:
        if getattr(state, "stage_index", 0) == 0:
            return params.throttle1_a if t < params.t1_split else params.throttle1_b
        return params.throttle2_a if t < params.t2_split else params.throttle2_b

    sim.guidance.throttle_schedule = throttle_sched

    log = sim.run(
        t_env0,
        duration,
        dt,
        state0,
        orbit_target_radius=TARGET_ORBIT_RADIUS,
        orbit_speed_tolerance=speed_tol,
        orbit_radial_tolerance=radial_tol,
        orbit_alt_tolerance=alt_tol,
    )

    initial_prop = sim.rocket.stages[0].prop_mass + sim.rocket.stages[1].prop_mass
    remaining_prop = sum(sim.rocket.stage_prop_remaining)
    prop_used = max(initial_prop - remaining_prop, 0.0)
    total_prop = initial_prop

    # Orbit evaluation using orbital elements
    r_vec = np.array(log.r[-1], dtype=float)
    v_vec = np.array(log.v[-1], dtype=float)
    r_norm = np.linalg.norm(r_vec)
    v_norm = np.linalg.norm(v_vec)

    h_vec = np.cross(r_vec, v_vec)
    h_norm = np.linalg.norm(h_vec)
    energy = 0.5 * v_norm**2 - MU_EARTH / r_norm
    if energy != 0:
        a = -MU_EARTH / (2 * energy)
    else:
        a = np.inf
    ecc_vec = np.cross(v_vec, h_vec) / MU_EARTH - r_vec / r_norm
    ecc = np.linalg.norm(ecc_vec)

    if ecc < 1.0 and a > 0:
        rp = a * (1 - ecc)
        ra = a * (1 + ecc)
    else:
        rp = np.nan
    ra = np.nan if ecc >= 1.0 or a <= 0 else a * (1 + ecc)

    r_hat = r_vec / r_norm
    vr = float(np.dot(v_vec, r_hat))
    fpa = math.degrees(math.asin(vr / v_norm)) if v_norm > 0 else 0.0

    orbit_ok = False

    penalty = 0.0

    # Liftoff T/W check (using sea-level thrust)
    thrust_sl = sim.rocket.stages[0].engine.thrust_sl * params.throttle1_a
    tw0 = thrust_sl / (state0.m * G0)
    tw_violation = tw0 < 1.2
    if tw0 < 1.2:
        penalty += 1e6 * (1.2 - tw0)

    # Max-Q constraint (optional)
    max_q = max(log.dynamic_pressure) if len(log.dynamic_pressure) else 0.0
    max_q_violation = False
    if CFG.max_q_limit is not None and max_q > CFG.max_q_limit:
        penalty += 1e6 * (max_q - CFG.max_q_limit) / CFG.max_q_limit
        max_q_violation = True

    # Stage coast enforcement: ensure no thrust during the 60 s after separation
    t_arr = np.array(log.t_sim)
    thrust_arr = np.array(log.thrust_mag)
    stage_arr = np.array(log.stage)
    idx_switch = np.where(np.diff(stage_arr) != 0)[0]
    coast_violation = False
    if len(idx_switch) > 0:
        t_switch = t_arr[idx_switch[0]]
        mask = (t_arr >= t_switch) & (t_arr <= t_switch + 60.0)
        if thrust_arr[mask].size > 0 and (thrust_arr[mask] > 1e-3).any():
            penalty += 1e6
            coast_violation = True

    # Max acceleration constraint (approximate)
    accel_arr = []
    for r_vec, m_val, th_mag, dr_mag in zip(log.r, log.m, log.thrust_mag, log.drag_mag):
        r_norm_local = np.linalg.norm(r_vec)
        g_mag = MU_EARTH / (r_norm_local**2) if r_norm_local > 0 else 0.0
        a_mag = (th_mag + dr_mag) / max(m_val, 1e-6) + g_mag
        accel_arr.append(a_mag)
    max_accel = max(accel_arr) if accel_arr else 0.0
    max_accel_violation = False
    if CFG.max_accel_limit is not None and max_accel > CFG.max_accel_limit:
        penalty += 1e6 * (max_accel - CFG.max_accel_limit) / CFG.max_accel_limit
        max_accel_violation = True

    # Circularization check at apogee (approximate two-burn plan)
    prop_left = remaining_prop if remaining_prop > 0 else 0.0
    circularization_feasible = False
    delta_v_needed = float("nan")
    prop_needed_circ = float("nan")
    # Direct orbit check (if guidance hit target already)
    if not orbit_ok:
        orbit_ok = (
            log.orbit_achieved
            and ecc < ORBIT_ECC_TOL
            and not math.isnan(rp)
            and rp >= TARGET_ORBIT_RADIUS - alt_tol
            and ra <= TARGET_ORBIT_RADIUS + alt_tol
        )
    # Simulate a coarse coast+burn if not in orbit and feasible
    if simulate_two_burn and not orbit_ok and ecc < 1.0 and a > 0 and not math.isnan(ra) and prop_left > 0:
        # Prepare state at end of ascent
        state_last = State(r_eci=log.r[-1], v_eci=log.v[-1], m=log.m[-1], stage_index=log.stage[-1])
        # If still on stage 0 but prop depleted, drop booster dry mass and move to stage 1
        if state_last.stage_index == 0:
            state_last.m = max(state_last.m - sim.rocket.stages[0].dry_mass, 1e-6)
            state_last.stage_index = 1
        # Reset rocket time bookkeeping
        sim.rocket._last_time = 0.0
        # Guidance: coast until near apogee (vr <= 0 and r close to ra), then burn prograde
        class CircularizeGuidance:
            def __init__(self, ra_target: float, alt_tol_local: float):
                self.ra_target = ra_target
                self.alt_tol = alt_tol_local
            def compute_command(self, t: float, state: State):
                r_vec = np.asarray(state.r_eci, dtype=float)
                v_vec = np.asarray(state.v_eci, dtype=float)
                r_norm_local = np.linalg.norm(r_vec)
                v_norm_local = np.linalg.norm(v_vec)
                r_hat_local = r_vec / r_norm_local if r_norm_local > 0 else np.array([0.0, 0.0, 1.0])
                vr_local = float(np.dot(v_vec, r_hat_local))
                burn = (abs(r_norm_local - self.ra_target) <= self.alt_tol) and vr_local <= 0.0
                if v_norm_local > 0:
                    dir_vec = v_vec / v_norm_local
                else:
                    dir_vec = np.array([0.0, 0.0, 1.0])
                throttle = 1.0 if burn else 0.0
                return ControlCommand(throttle=throttle, thrust_direction=dir_vec)
        # Build a new simulation with prograde guidance
        circ_guidance = CircularizeGuidance(ra_target=ra, alt_tol_local=alt_tol)
        sim.guidance = circ_guidance
        log2 = sim.run(
            t_env_start=t_env0 + log.t_sim[-1],
            duration=duration,
            dt=dt,
            state0=state_last,
            orbit_target_radius=TARGET_ORBIT_RADIUS,
            orbit_speed_tolerance=speed_tol,
            orbit_radial_tolerance=radial_tol,
            orbit_alt_tolerance=alt_tol,
        )
        log = log2  # overwrite log with coast+burn segment
        # Recompute orbit metrics after circularization attempt
        r_vec = np.array(log.r[-1], dtype=float)
        v_vec = np.array(log.v[-1], dtype=float)
        r_norm = np.linalg.norm(r_vec)
        v_norm = np.linalg.norm(v_vec)
        h_vec = np.cross(r_vec, v_vec)
        h_norm = np.linalg.norm(h_vec)
        energy = 0.5 * v_norm**2 - MU_EARTH / r_norm
        a = -MU_EARTH / (2 * energy) if energy != 0 else np.inf
        ecc_vec = np.cross(v_vec, h_vec) / MU_EARTH - r_vec / r_norm
        ecc = np.linalg.norm(ecc_vec)
        if ecc < 1.0 and a > 0:
            rp = a * (1 - ecc)
            ra = a * (1 + ecc)
        else:
            rp = np.nan
            ra = np.nan if ecc >= 1.0 or a <= 0 else a * (1 + ecc)
        r_hat = r_vec / r_norm
        vr = float(np.dot(v_vec, r_hat))
        fpa = math.degrees(math.asin(vr / v_norm)) if v_norm > 0 else 0.0
        orbit_ok = (
            log.orbit_achieved
            and ecc < ORBIT_ECC_TOL
            and not math.isnan(rp)
            and rp >= TARGET_ORBIT_RADIUS - alt_tol
            and ra <= TARGET_ORBIT_RADIUS + alt_tol
        )
        # Update remaining prop after burn
        remaining_prop = sum(sim.rocket.stage_prop_remaining)
        prop_left = remaining_prop if remaining_prop > 0 else 0.0

    if not orbit_ok:
        penalty += 1e9

    cost = prop_used + penalty

    result = {
        "orbit_ok": orbit_ok,
        "cost": cost,
        "prop_used": prop_used,
        "rp_m": rp,
        "ra_m": ra,
        "ecc": ecc,
        "a_m": a,
        "spec_energy": energy,
        "fpa_deg": fpa,
        "delta_v_needed": delta_v_needed,
        "prop_needed_circ": prop_needed_circ,
        "prop_left": prop_left,
        "circularization_feasible": circularization_feasible,
        "prop1": params.prop1,
        "prop2": params.prop2,
        "throttle1_a": params.throttle1_a,
        "throttle1_b": params.throttle1_b,
        "t1_split": params.t1_split,
        "throttle2_a": params.throttle2_a,
        "throttle2_b": params.throttle2_b,
        "t2_split": params.t2_split,
        "pitch_alt1": params.pitch_alt1,
        "pitch_alt2": params.pitch_alt2,
        "pitch_alt3": params.pitch_alt3,
        "pitch_alt4": params.pitch_alt4,
        "pitch_ang1_deg": params.pitch_ang1_deg,
        "pitch_ang2_deg": params.pitch_ang2_deg,
        "pitch_ang3_deg": params.pitch_ang3_deg,
        "pitch_ang4_deg": params.pitch_ang4_deg,
        "max_alt": max(log.altitude),
        "max_speed": max(log.speed),
        "max_q": max(log.dynamic_pressure),
        "final_alt": log.altitude[-1],
        "final_speed": log.speed[-1],
        "tw0": tw0,
        "tw_violation": tw_violation,
        "coast_violation": coast_violation,
        "total_prop": total_prop,
        "max_q_violation": max_q_violation,
        "max_q": max_q,
        "max_accel": max_accel,
        "max_accel_violation": max_accel_violation,
    }
    return (result, log) if return_log else result


# --- Parameter packing for optimizer ---
def params_to_vector(p: SampleParams) -> np.ndarray:
    return np.array(
        [
            p.prop1,
            p.prop2,
            p.throttle1_a,
            p.throttle1_b,
            p.t1_split,
            p.throttle2_a,
            p.throttle2_b,
            p.t2_split,
            p.pitch_alt1,
            p.pitch_alt2,
            p.pitch_alt3,
            p.pitch_alt4,
            p.pitch_ang1_deg,
            p.pitch_ang2_deg,
            p.pitch_ang3_deg,
            p.pitch_ang4_deg,
        ],
        dtype=float,
    )


def vector_to_params(x: np.ndarray, bounds: dict) -> SampleParams:
    def clamp(val, lo, hi):
        return max(lo, min(hi, val))

    return SampleParams(
        prop1=clamp(x[0], *bounds["prop1"]),
        prop2=clamp(x[1], *bounds["prop2"]),
        throttle1_a=clamp(x[2], *bounds["throttle1"]),
        throttle1_b=clamp(x[3], *bounds["throttle1"]),
        t1_split=clamp(x[4], *bounds["t_split"]),
        throttle2_a=clamp(x[5], *bounds["throttle2"]),
        throttle2_b=clamp(x[6], *bounds["throttle2"]),
        t2_split=clamp(x[7], *bounds["t_split"]),
        pitch_alt1=clamp(x[8], *bounds["pitch_alt"]),
        pitch_alt2=clamp(x[9], *bounds["pitch_alt"]),
        pitch_alt3=clamp(x[10], *bounds["pitch_alt"]),
        pitch_alt4=clamp(x[11], *bounds["pitch_alt"]),
        pitch_ang1_deg=clamp(x[12], *bounds["pitch_ang"]),
        pitch_ang2_deg=clamp(x[13], *bounds["pitch_ang"]),
        pitch_ang3_deg=clamp(x[14], *bounds["pitch_ang"]),
        pitch_ang4_deg=clamp(x[15], *bounds["pitch_ang"]),
    )


def random_sample(n: int) -> list[SampleParams]:
    samples = []
    for _ in range(n):
        prop1 = random.uniform(*CFG.prop1_bounds)
        prop2 = random.uniform(*CFG.prop2_bounds)
        throttle1 = random.uniform(*CFG.throttle1_bounds)
        throttle1_b = random.uniform(*CFG.throttle1_bounds)
        t1_split = random.uniform(*CFG.throttle_split_time_bounds)
        throttle2 = random.uniform(*CFG.throttle2_bounds)
        throttle2_b = random.uniform(*CFG.throttle2_bounds)
        t2_split = random.uniform(*CFG.throttle_split_time_bounds)
        # Generate four altitude breakpoints and corresponding angles
        alts = sorted(
            [
                random.uniform(*CFG.pitch_alt_bounds),
                random.uniform(*CFG.pitch_alt_bounds),
                random.uniform(*CFG.pitch_alt_bounds),
                random.uniform(*CFG.pitch_alt_bounds),
            ]
        )
        angs = [
            random.uniform(*CFG.pitch_angle_bounds_deg),
            random.uniform(*CFG.pitch_angle_bounds_deg),
            random.uniform(*CFG.pitch_angle_bounds_deg),
            random.uniform(*CFG.pitch_angle_bounds_deg),
        ]
        samples.append(
            SampleParams(
                prop1=prop1,
                prop2=prop2,
                throttle1_a=throttle1,
                throttle1_b=throttle1_b,
                t1_split=t1_split,
                throttle2_a=throttle2,
                throttle2_b=throttle2_b,
                t2_split=t2_split,
                pitch_alt1=alts[0],
                pitch_alt2=alts[1],
                pitch_alt3=alts[2],
                pitch_alt4=alts[3],
                pitch_ang1_deg=angs[0],
                pitch_ang2_deg=angs[1],
                pitch_ang3_deg=angs[2],
                pitch_ang4_deg=angs[3],
            )
        )
    return samples


def plot_trajectory(log, filename: str | None = None, target_radius: float | None = None):
    """Static 3D plot of trajectory; saves to file if filename provided."""
    positions = np.array(log.r)
    if positions.shape[0] < 2:
        return

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Earth sphere
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = R_EARTH * np.outer(np.cos(u), np.sin(v))
    y = R_EARTH * np.outer(np.sin(u), np.sin(v))
    z = R_EARTH * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, cmap="Blues", alpha=0.08, linewidth=0, antialiased=False)
    ax.plot_wireframe(x, y, z, color="lightblue", alpha=0.2, linewidth=0.3)

    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color="tab:red", lw=1.5, label="Trajectory")
    if target_radius is not None:
        theta = np.linspace(0, 2 * np.pi, 200)
        circ_x = target_radius * np.cos(theta)
        circ_y = target_radius * np.sin(theta)
        circ_z = np.zeros_like(theta)
        ax.plot(circ_x, circ_y, circ_z, "--", color="gray", alpha=0.6, label="Target orbit (equatorial)")

    r_max = max(np.linalg.norm(p) for p in positions)
    lim = 1.05 * max(R_EARTH, r_max)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=35)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Trajectory")
    ax.legend(loc="upper right")
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150)
        plt.close(fig)
        print(f"Saved trajectory plot to {filename}")
    else:
        plt.show()


def main():
    # Settings: coarse sweep then local refinement with Nelder-Mead
    n_random = CFG.opt_n_random
    n_heuristic = CFG.opt_n_heuristic
    top_k = CFG.opt_top_k
    coarse_duration = CFG.opt_coarse_duration_s
    coarse_dt = CFG.opt_coarse_dt_s
    refine_duration = CFG.opt_refine_duration_s
    refine_dt = CFG.opt_refine_dt_s
    plot_each = CFG.opt_plot_each  # set True to save every sample trajectory

    bounds = {
        "prop1": CFG.prop1_bounds,
        "prop2": CFG.prop2_bounds,
        "throttle1": CFG.throttle1_bounds,
        "throttle2": CFG.throttle2_bounds,
        "pitch_alt": CFG.pitch_alt_bounds,
        "pitch_ang": CFG.pitch_angle_bounds_deg,
        "t_split": CFG.throttle_split_time_bounds,
    }

    results = []
    best_res = None
    best_params = None
    best_log = None

    # Initial seeds
    seeds = random_sample(n_random)
    heuristic = []
    for _ in range(n_heuristic):
        heuristic.append(
            SampleParams(
                prop1=3.1e6,
                prop2=1.1e6,
                throttle1_a=0.95,
                throttle1_b=0.9,
                t1_split=150.0,
                throttle2_a=0.95,
                throttle2_b=0.9,
                t2_split=400.0,
                pitch_alt1=5000.0,
                pitch_alt2=20000.0,
                pitch_alt3=60000.0,
                pitch_alt4=100000.0,
                pitch_ang1_deg=85.0,
                pitch_ang2_deg=60.0,
                pitch_ang3_deg=30.0,
                pitch_ang4_deg=5.0,
            )
        )
    seeds.extend(heuristic)

    seed_results = []
    for i, params in enumerate(seeds, start=1):
        print(f"Seed {i}/{len(seeds)} (dt={coarse_dt})...")
        res, log = evaluate(
            params,
            duration=coarse_duration,
            dt=coarse_dt,
            return_log=True,
            speed_tol=CFG.orbit_speed_tol_coarse,
            radial_tol=CFG.orbit_radial_tol_coarse,
            alt_tol=CFG.orbit_alt_tol_coarse,
            simulate_two_burn=True,
        )
        seed_results.append((res, params, log))
        results.append(res)
        if plot_each:
            plot_trajectory(log, f"trajectory_seed_{i}.png", TARGET_ORBIT_RADIUS)
        if best_res is None or res["cost"] < best_res["cost"]:
            best_res = res
            best_params = params
            best_log = log

    # Cost wrappers
    def cost_wrapper(x, duration, dt, speed_tol, radial_tol, alt_tol):
        p = vector_to_params(x, bounds)
        res = evaluate(
            p,
            duration=duration,
            dt=dt,
            return_log=False,
            speed_tol=speed_tol,
            radial_tol=radial_tol,
            alt_tol=alt_tol,
        )
        return res["cost"]

    # Global search with CMA-ES if available, otherwise DE on best seed
    best_x = None
    lower_bounds = [
        bounds["prop1"][0],
        bounds["prop2"][0],
        bounds["throttle1"][0],
        bounds["throttle1"][0],
        bounds["t_split"][0],
        bounds["throttle2"][0],
        bounds["throttle2"][0],
        bounds["t_split"][0],
        bounds["pitch_alt"][0],
        bounds["pitch_alt"][0],
        bounds["pitch_alt"][0],
        bounds["pitch_alt"][0],
        bounds["pitch_ang"][0],
        bounds["pitch_ang"][0],
        bounds["pitch_ang"][0],
        bounds["pitch_ang"][0],
    ]
    upper_bounds = [
        bounds["prop1"][1],
        bounds["prop2"][1],
        bounds["throttle1"][1],
        bounds["throttle1"][1],
        bounds["t_split"][1],
        bounds["throttle2"][1],
        bounds["throttle2"][1],
        bounds["t_split"][1],
        bounds["pitch_alt"][1],
        bounds["pitch_alt"][1],
        bounds["pitch_alt"][1],
        bounds["pitch_alt"][1],
        bounds["pitch_ang"][1],
        bounds["pitch_ang"][1],
        bounds["pitch_ang"][1],
        bounds["pitch_ang"][1],
    ]
    if CMA_AVAILABLE and CFG.opt_use_cma:
        x0 = [(lo + hi) / 2.0 for lo, hi in zip(lower_bounds, upper_bounds)]
        sigma = [
            CFG.opt_cma_sigma_scale * max(1e-6, hi - lo) for lo, hi in zip(lower_bounds, upper_bounds)
        ]
        print("Running CMA-ES global search...")
        es = cma.CMAEvolutionStrategy(
            x0,
            sigma,
            {
                "bounds": [lower_bounds, upper_bounds],
                "maxiter": CFG.opt_cma_maxiter,
                "verb_disp": 1,
            },
        )
        while not es.stop():
            xs = es.ask()
            costs = [
                cost_wrapper(
                    x,
                    duration=coarse_duration,
                    dt=coarse_dt,
                    speed_tol=CFG.orbit_speed_tol_coarse,
                    radial_tol=CFG.orbit_radial_tol_coarse,
                    alt_tol=CFG.orbit_alt_tol_coarse,
                )
                for x in xs
            ]
            es.tell(xs, costs)
        best_x = es.result.xbest
    else:
        # Fallback: take best seed params as starting point and run DE
        seed_results.sort(key=lambda t: t[0]["cost"])
        top_seeds = seed_results[: min(top_k, len(seed_results))]
        if top_seeds:
            _, p0, _ = top_seeds[0]
            x0 = params_to_vector(p0)
        else:
            x0 = [(lo + hi) / 2.0 for lo, hi in zip(lower_bounds, upper_bounds)]
        de_bounds = list(zip(lower_bounds, upper_bounds))
        print("Running DE global search (fallback)...")
        de_res = differential_evolution(
            lambda x: cost_wrapper(
                x,
                duration=coarse_duration,
                dt=coarse_dt,
                speed_tol=CFG.orbit_speed_tol_coarse,
                radial_tol=CFG.orbit_radial_tol_coarse,
                alt_tol=CFG.orbit_alt_tol_coarse,
            ),
            de_bounds,
            maxiter=20,
            popsize=10,
            polish=False,
            disp=False,
        )
        best_x = de_res.x

    # Nelder-Mead polish at finer dt
    print("Nelder-Mead polish on global best...")
    nm_res = minimize(
        lambda x: cost_wrapper(
            x,
            duration=refine_duration,
            dt=refine_dt,
            speed_tol=ORBIT_SPEED_TOL,
            radial_tol=ORBIT_RADIAL_TOL,
            alt_tol=ORBIT_ALT_TOL,
        ),
        best_x,
        method="Nelder-Mead",
        options={"maxiter": CFG.opt_nm_maxiter, "disp": False},
    )
    p_nm = vector_to_params(nm_res.x, bounds)
    res_nm, log_nm = evaluate(p_nm, duration=refine_duration, dt=refine_dt, return_log=True)
    results.append(res_nm)
    if plot_each:
        plot_trajectory(log_nm, f"trajectory_nm_best.png", TARGET_ORBIT_RADIUS)
    if best_res is None or res_nm["cost"] < best_res["cost"]:
        best_res = res_nm
        best_params = p_nm
        best_log = log_nm

    # Sort by cost
    results.sort(key=lambda r: r["cost"])

    # Write CSV
    fieldnames = list(results[0].keys()) if results else []
    with open("optimization_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    if results:
        best = results[0]
        print("\nBest sample:")
        for k, v in best.items():
            print(f"  {k}: {v}")
    else:
        print("No results.")

    # Plot final best trajectory
    if best_log is not None:
        plot_trajectory(best_log, "trajectory_best.png", TARGET_ORBIT_RADIUS)
        # dt sensitivity check
        res_dt, _ = evaluate(best_params, duration=refine_duration, dt=max(refine_dt * 0.5, 0.05), return_log=False)
        print("\nDT sensitivity check (half dt):")
        for k in ["cost", "orbit_ok", "rp_m", "ra_m", "ecc", "prop_used", "max_q", "max_accel"]:
            if k in res_dt and k in best_res:
                print(f"  {k}: coarse={best_res.get(k)} fine={res_dt.get(k)}")


if __name__ == "__main__":
    main()
