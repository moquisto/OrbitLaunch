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
from scipy.optimize import minimize

from aerodynamics import Aerodynamics, CdModel
from atmosphere import AtmosphereModel
from gravity import EarthModel, MU_EARTH, OMEGA_EARTH, R_EARTH
from integrators import RK4, State
from rocket import Engine, Rocket, Stage
from simulation import Guidance, Simulation
from config import CFG

TARGET_ORBIT_RADIUS = R_EARTH + CFG.target_orbit_alt_m
G0 = 9.80665
ORBIT_ALT_TOL = CFG.orbit_alt_tol
ORBIT_ECC_TOL = 0.01
ORBIT_SPEED_TOL = CFG.orbit_speed_tol
ORBIT_RADIAL_TOL = CFG.orbit_radial_tol


@dataclass
class SampleParams:
    prop1: float
    prop2: float
    throttle1: float
    throttle2: float
    pitch_start_alt: float
    pitch_end_alt: float


def make_pitch_program(pitch_start_alt: float, pitch_end_alt: float) -> Callable:
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
        if alt < pitch_start_alt:
            return r_hat
        elif alt < pitch_end_alt:
            w = (alt - pitch_start_alt) / max(1.0, (pitch_end_alt - pitch_start_alt))
            direction = (1.0 - w) * r_hat + w * east
        else:
            speed = np.linalg.norm(v)
            direction = v / speed if speed > 1.0 else east
        n = np.linalg.norm(direction)
        return direction / n if n > 0.0 else r_hat

    return pitch_prog


def make_throttle_schedule(throttle_val: float) -> Callable:
    def throttle_fn(t: float, state: State) -> float:
        return float(throttle_val)

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

    pitch_prog = make_pitch_program(params.pitch_start_alt, params.pitch_end_alt)
    throttle_fn = make_throttle_schedule(params.throttle1)
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


def evaluate(params: SampleParams, duration: float = 1200.0, dt: float = 0.2, return_log: bool = False):
    sim, state0, t_env0 = build_simulation(params)

    # Replace throttle schedule for stage 2 (simple: constant throttle2 after ignition)
    def throttle_sched(t: float, state: State) -> float:
        if getattr(state, "stage_index", 0) == 0:
            return params.throttle1
        return params.throttle2

    sim.guidance.throttle_schedule = throttle_sched

    log = sim.run(
        t_env0,
        duration,
        dt,
        state0,
        orbit_target_radius=TARGET_ORBIT_RADIUS,
        orbit_speed_tolerance=ORBIT_SPEED_TOL,
        orbit_radial_tolerance=ORBIT_RADIAL_TOL,
        orbit_alt_tolerance=ORBIT_ALT_TOL,
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
    thrust_sl = sim.rocket.stages[0].engine.thrust_sl * params.throttle1
    tw0 = thrust_sl / (state0.m * G0)
    tw_violation = tw0 < 1.2
    if tw0 < 1.2:
        penalty += 1e6 * (1.2 - tw0)

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

    # Circularization check at apogee (approximate two-burn plan)
    prop_left = remaining_prop if remaining_prop > 0 else 0.0
    circularization_feasible = False
    delta_v_needed = float("nan")
    prop_needed_circ = float("nan")
    if ecc < 1.0 and a > 0 and not math.isnan(ra):
        # If apogee at or above target, estimate burn to circularize at ra
        if ra >= TARGET_ORBIT_RADIUS - ORBIT_ALT_TOL:
            v_apo = math.sqrt(MU_EARTH * (2.0 / ra - 1.0 / a))
            v_circ = math.sqrt(MU_EARTH / ra)
            delta_v_needed = max(0.0, v_circ - v_apo)
            isp_stage2 = sim.rocket.stages[1].engine.isp_vac
            if delta_v_needed > 0 and isp_stage2 > 0 and prop_left > 0:
                m0 = log.m[-1]
                m1 = m0 * math.exp(-delta_v_needed / (isp_stage2 * G0))
                prop_needed_circ = max(0.0, m0 - m1)
                if prop_left >= prop_needed_circ:
                    circularization_feasible = True
                    prop_used += prop_needed_circ
                    prop_left -= prop_needed_circ
                    # After hypothetical circularization, treat orbit as achieved
                    ecc = 0.0
                    rp = ra
                    a = ra
                    orbit_ok = True
    # Direct orbit check (if guidance hit target already)
    if not orbit_ok:
        orbit_ok = (
            log.orbit_achieved
            and ecc < ORBIT_ECC_TOL
            and not math.isnan(rp)
            and rp >= TARGET_ORBIT_RADIUS - ORBIT_ALT_TOL
            and ra <= TARGET_ORBIT_RADIUS + ORBIT_ALT_TOL
        )

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
        "throttle1": params.throttle1,
        "throttle2": params.throttle2,
        "pitch_start_alt": params.pitch_start_alt,
        "pitch_end_alt": params.pitch_end_alt,
        "max_alt": max(log.altitude),
        "max_speed": max(log.speed),
        "max_q": max(log.dynamic_pressure),
        "final_alt": log.altitude[-1],
        "final_speed": log.speed[-1],
        "tw0": tw0,
        "tw_violation": tw_violation,
        "coast_violation": coast_violation,
        "total_prop": total_prop,
    }
    return (result, log) if return_log else result


# --- Parameter packing for optimizer ---
def params_to_vector(p: SampleParams) -> np.ndarray:
    return np.array(
        [
            p.prop1,
            p.prop2,
            p.throttle1,
            p.throttle2,
            p.pitch_start_alt,
            p.pitch_end_alt,
        ],
        dtype=float,
    )


def vector_to_params(x: np.ndarray, bounds: dict) -> SampleParams:
    def clamp(val, lo, hi):
        return max(lo, min(hi, val))

    return SampleParams(
        prop1=clamp(x[0], *bounds["prop1"]),
        prop2=clamp(x[1], *bounds["prop2"]),
        throttle1=clamp(x[2], *bounds["throttle1"]),
        throttle2=clamp(x[3], *bounds["throttle2"]),
        pitch_start_alt=clamp(x[4], *bounds["pitch_start_alt"]),
        pitch_end_alt=clamp(x[5], *bounds["pitch_end_alt"]),
    )


def random_sample(n: int) -> list[SampleParams]:
    samples = []
    for _ in range(n):
        prop1 = random.uniform(2.5e6, 3.6e6)
        prop2 = random.uniform(0.8e6, 1.4e6)
        throttle1 = random.uniform(0.8, 1.0)
        throttle2 = random.uniform(0.8, 1.0)
        pitch_start_alt = random.uniform(3_000.0, 8_000.0)
        pitch_end_alt = random.uniform(40_000.0, 120_000.0)
        samples.append(SampleParams(prop1, prop2, throttle1, throttle2, pitch_start_alt, pitch_end_alt))
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
        "prop1": (2.5e6, 3.6e6),
        "prop2": (0.8e6, 1.4e6),
        "throttle1": (0.8, 1.0),
        "throttle2": (0.8, 1.0),
        "pitch_start_alt": (3_000.0, 8_000.0),
        "pitch_end_alt": (40_000.0, 120_000.0),
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
                throttle1=0.95,
                throttle2=0.95,
                pitch_start_alt=5000.0,
                pitch_end_alt=70000.0,
            )
        )
    seeds.extend(heuristic)

    seed_results = []
    for i, params in enumerate(seeds, start=1):
        print(f"Seed {i}/{len(seeds)} (dt={coarse_dt})...")
        res, log = evaluate(params, duration=coarse_duration, dt=coarse_dt, return_log=True)
        seed_results.append((res, params, log))
        results.append(res)
        if plot_each:
            plot_trajectory(log, f"trajectory_seed_{i}.png", TARGET_ORBIT_RADIUS)
        if best_res is None or res["cost"] < best_res["cost"]:
            best_res = res
            best_params = params
            best_log = log

    # Select top_k seeds
    seed_results.sort(key=lambda t: t[0]["cost"])
    top_seeds = seed_results[: min(top_k, len(seed_results))]

    # Local Nelder-Mead refinement on each top seed (coarse dt for speed)
    def cost_wrapper(x):
        p = vector_to_params(x, bounds)
        res = evaluate(p, duration=coarse_duration, dt=coarse_dt, return_log=False)
        return res["cost"]

    for idx, (res0, p0, _) in enumerate(top_seeds, start=1):
        x0 = params_to_vector(p0)
        print(f"Nelder-Mead refine seed {idx}/{len(top_seeds)}...")
        nm_res = minimize(
            cost_wrapper,
            x0,
            method="Nelder-Mead",
            options={"maxiter": CFG.opt_nm_maxiter, "disp": False},
        )
        p_nm = vector_to_params(nm_res.x, bounds)
        res_nm, log_nm = evaluate(p_nm, duration=refine_duration, dt=refine_dt, return_log=True)
        results.append(res_nm)
        if plot_each:
            plot_trajectory(log_nm, f"trajectory_nm_{idx}.png", TARGET_ORBIT_RADIUS)
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


if __name__ == "__main__":
    main()
