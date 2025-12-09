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

from aerodynamics import Aerodynamics, CdModel
from atmosphere import AtmosphereModel
from gravity import EarthModel, MU_EARTH, OMEGA_EARTH, R_EARTH
from integrators import RK4, State
from rocket import Engine, Rocket, Stage
from simulation import Guidance, Simulation

# Launch site
LAUNCH_LAT_DEG = 28.60839
LAUNCH_LON_DEG = -80.60433

TARGET_ORBIT_RADIUS = R_EARTH + 420_000.0
G0 = 9.80665
ORBIT_ALT_TOL = 5_000.0  # meters
ORBIT_ECC_TOL = 0.01
ORBIT_SPEED_TOL = 50.0
ORBIT_RADIAL_TOL = 50.0


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
    atmosphere = AtmosphereModel(lat_deg=LAUNCH_LAT_DEG, lon_deg=LAUNCH_LON_DEG)
    cd_model = CdModel(2.0)

    booster_engine = Engine(
        thrust_vac=7.35e7,
        thrust_sl=7.0e7,
        isp_vac=347.0,
        isp_sl=327.0,
    )
    upper_engine = Engine(
        thrust_vac=1.5e7,
        thrust_sl=1.2e7,
        isp_vac=380.0,
        isp_sl=330.0,
    )
    booster_stage = Stage(
        dry_mass=2.7e5,
        prop_mass=params.prop1,
        engine=booster_engine,
        ref_area=np.pi * (4.5**2),
    )
    upper_stage = Stage(
        dry_mass=1.3e5,
        prop_mass=params.prop2,
        engine=upper_engine,
        ref_area=np.pi * (4.5**2),
    )
    rocket = Rocket(
        stages=[booster_stage, upper_stage],
        separation_delay=120.0,
        upper_ignition_delay=120.0,
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
    lat = math.radians(LAUNCH_LAT_DEG)
    lon = math.radians(LAUNCH_LON_DEG)
    r0 = R_EARTH * np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)], dtype=float)
    v0 = earth.atmosphere_velocity(r0)
    m0 = booster_stage.total_mass() + upper_stage.total_mass()
    state0 = State(r_eci=r0, v_eci=v0, m=m0, stage_index=0)
    t_env0 = 0.0
    return sim, state0, t_env0


def evaluate(params: SampleParams, duration: float = 1200.0, dt: float = 1.0):
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

    prop_used = (sim.rocket.stages[0].prop_mass + sim.rocket.stages[1].prop_mass) - (log.m[-1] - sim.rocket.stages[0].dry_mass - sim.rocket.stages[1].dry_mass)
    prop_used = max(prop_used, 0.0)
    total_prop = sim.rocket.stages[0].prop_mass + sim.rocket.stages[1].prop_mass

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

    # Stage coast enforcement: ensure no thrust during the 120 s after separation
    t_arr = np.array(log.t_sim)
    thrust_arr = np.array(log.thrust_mag)
    stage_arr = np.array(log.stage)
    idx_switch = np.where(np.diff(stage_arr) != 0)[0]
    coast_violation = False
    if len(idx_switch) > 0:
        t_switch = t_arr[idx_switch[0]]
        mask = (t_arr >= t_switch) & (t_arr <= t_switch + 120.0)
        if thrust_arr[mask].size > 0 and (thrust_arr[mask] > 1e-3).any():
            penalty += 1e6
            coast_violation = True

    # Circularization check at apogee (approximate two-burn plan)
    prop_left = max(log.m[-1] - (sim.rocket.stages[0].dry_mass + sim.rocket.stages[1].dry_mass), 0.0)
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
    return result


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


def main():
    n_samples = 20
    duration = 1200.0
    dt = 1.0
    results = []

    for i, params in enumerate(random_sample(n_samples), start=1):
        print(f"Running sample {i}/{n_samples}...")
        res = evaluate(params, duration=duration, dt=dt)
        results.append(res)

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


if __name__ == "__main__":
    main()
