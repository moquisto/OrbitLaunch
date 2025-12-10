"""
Two-phase optimizer to minimize propellant (maximize final mass) for a 420 km circular orbit.
Uses closed-loop guidance with:
  - Pitch turn (start/end altitudes, blend exponent).
  - Two-phase upper-stage throttle (boost to target apoapsis, coast, circularize) with throttle caps.

Phase 1: random search for a feasible orbit.
Phase 2: local refinement around the best feasible solution.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import matplotlib

# Headless for optimization; main simulation still plots.
matplotlib.use("Agg")

from config import CFG
from gravity import R_EARTH
from main import TwoPhaseUpperThrottle, build_simulation, orbital_elements_from_state


@dataclass
class OptResult:
    params: dict
    final_mass: float
    orbit_achieved: bool
    perigee_alt_m: float | None
    apoapsis_alt_m: float | None


def evaluate(p_start: float, p_end: float, ap_target: float, blend_exp: float, boost_cap: float, circ_cap: float) -> OptResult:
    # Apply overrides
    CFG.pitch_turn_start_m = p_start
    CFG.pitch_turn_end_m = p_end
    CFG.pitch_blend_exp = blend_exp
    CFG.target_orbit_alt_m = ap_target
    CFG.upper_boost_throttle_cap = boost_cap
    CFG.upper_circ_throttle_cap = circ_cap

    sim, state0, t_env0 = build_simulation()
    orbit_radius = R_EARTH + CFG.target_orbit_alt_m
    controller = TwoPhaseUpperThrottle(target_radius=orbit_radius)
    sim.guidance.throttle_schedule = controller

    duration = min(CFG.main_duration_s, 1600.0)
    dt = max(CFG.main_dt_s, 1.2)

    log = sim.run(
        t_env0,
        duration,
        dt,
        state0,
        orbit_target_radius=orbit_radius,
        orbit_speed_tolerance=CFG.orbit_speed_tol,
        orbit_radial_tolerance=CFG.orbit_radial_tol,
        orbit_alt_tolerance=CFG.orbit_alt_tol,
        exit_on_orbit=True,
        post_orbit_coast_s=0.0,
    )

    a, rp, ra = orbital_elements_from_state(log.r[-1], log.v[-1], sim.earth.mu)
    rp_alt = rp - R_EARTH if rp is not None else None
    ra_alt = ra - R_EARTH if ra is not None else None

    return OptResult(
        params={
            "pitch_start_m": p_start,
            "pitch_end_m": p_end,
            "target_ap_alt_m": ap_target,
            "pitch_blend_exp": blend_exp,
            "boost_cap": boost_cap,
            "circ_cap": circ_cap,
        },
        final_mass=log.m[-1],
        orbit_achieved=log.orbit_achieved,
        perigee_alt_m=rp_alt,
        apoapsis_alt_m=ra_alt,
    )


def score(result: OptResult, target_alt: float = 420_000.0) -> float:
    if result.orbit_achieved and result.perigee_alt_m is not None and result.apoapsis_alt_m is not None and result.perigee_alt_m > 0:
        per_err = abs(result.perigee_alt_m - target_alt)
        ap_err = abs(result.apoapsis_alt_m - target_alt)
        return result.final_mass - 0.05 * (per_err + ap_err)
    per_pen = abs(result.perigee_alt_m) if result.perigee_alt_m is not None else 1e6
    return -1e9 - per_pen


def random_search(n_samples: int = 12) -> OptResult | None:
    best: OptResult | None = None
    best_score = float("-inf")
    for i in range(n_samples):
        p_start = random.uniform(3000.0, 9000.0)
        p_end = random.uniform(30_000.0, 80_000.0)
        if p_end <= p_start:
            p_end = p_start + 10_000.0
        ap_target = random.uniform(400_000.0, 440_000.0)
        blend = random.uniform(0.8, 1.5)
        boost_cap = random.uniform(0.7, 1.0)
        circ_cap = random.uniform(0.4, 1.0)

        res = evaluate(p_start, p_end, ap_target, blend, boost_cap, circ_cap)
        sc = score(res)
        if sc > best_score:
            best = res
            best_score = sc
            print(f"[coarse {i+1}/{n_samples}] New best: mass={res.final_mass:.1f} kg, "
                  f"orbit={res.orbit_achieved}, perigee={res.perigee_alt_m}, apoapsis={res.apoapsis_alt_m}, "
                  f"params={res.params}")
            if res.orbit_achieved:
                # Early exit if good orbit
                break
    return best


def refine_search(anchor: OptResult, n_samples: int = 20) -> OptResult:
    best = anchor
    best_score = score(anchor)

    for i in range(n_samples):
        p_start = max(1000.0, random.gauss(anchor.params["pitch_start_m"], 800.0))
        p_end = max(p_start + 5000.0, random.gauss(anchor.params["pitch_end_m"], 6000.0))
        ap_target = max(390_000.0, random.gauss(anchor.params["target_ap_alt_m"], 8000.0))
        blend = max(0.5, random.gauss(anchor.params["pitch_blend_exp"], 0.15))
        boost_cap = min(1.0, max(0.4, random.gauss(anchor.params["boost_cap"], 0.1)))
        circ_cap = min(1.0, max(0.4, random.gauss(anchor.params["circ_cap"], 0.1)))

        res = evaluate(p_start, p_end, ap_target, blend, boost_cap, circ_cap)
        sc = score(res)
        if sc > best_score:
            best = res
            best_score = sc
            print(f"[refine {i+1}/{n_samples}] New best: mass={res.final_mass:.1f} kg, "
                  f"orbit={res.orbit_achieved}, perigee={res.perigee_alt_m}, apoapsis={res.apoapsis_alt_m}, "
                  f"params={res.params}")
    return best


if __name__ == "__main__":
    random.seed(0)
    coarse = random_search()
    if coarse:
        print("\n=== Coarse result ===")
        print(f"Final mass: {coarse.final_mass:.1f} kg, orbit: {coarse.orbit_achieved}")
        print(f"Perigee: {coarse.perigee_alt_m/1000:.2f} km" if coarse.perigee_alt_m else "Perigee: n/a")
        print(f"Apoapsis: {coarse.apoapsis_alt_m/1000:.2f} km" if coarse.apoapsis_alt_m else "Apoapsis: n/a")
        print(f"Params: {coarse.params}")
        refined = refine_search(coarse)
        print("\n=== Refined result ===")
        print(f"Final mass: {refined.final_mass:.1f} kg, orbit: {refined.orbit_achieved}")
        print(f"Perigee: {refined.perigee_alt_m/1000:.2f} km" if refined.perigee_alt_m else "Perigee: n/a")
        print(f"Apoapsis: {refined.apoapsis_alt_m/1000:.2f} km" if refined.apoapsis_alt_m else "Apoapsis: n/a")
        print(f"Params: {refined.params}")
    else:
        print("No result from coarse search.")
