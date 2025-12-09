"""
Simple random-search optimizer to maximize final mass (minimize prop use)
for reaching the target circular orbit with the current BFR-like model.

Tunable parameters:
  - pitch_turn_start_m
  - pitch_turn_end_m
  - upper boost target apoapsis (via TwoPhaseUpperThrottle default target)

This runs headless (no plotting) and uses the existing Simulation pipeline.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from main import build_simulation, orbital_elements_from_state
from config import CFG
from gravity import R_EARTH


@dataclass
class OptResult:
    params: dict
    final_mass: float
    orbit_achieved: bool
    perigee_alt_m: float | None
    apoapsis_alt_m: float | None


def evaluate_once(pitch_start_m: float, pitch_end_m: float, target_ap_alt_m: float) -> OptResult:
    # Apply parameter overrides
    CFG.pitch_turn_start_m = pitch_start_m
    CFG.pitch_turn_end_m = pitch_end_m
    CFG.target_orbit_alt_m = target_ap_alt_m
    # Ensure orbit exit so we don't overrun
    CFG.exit_on_orbit = True
    CFG.post_orbit_coast_s = 0.0

    sim, state0, t_env0 = build_simulation()
    duration = min(CFG.main_duration_s, 2000.0)
    dt = max(CFG.main_dt_s, 1.2)
    orbit_radius = R_EARTH + CFG.target_orbit_alt_m

    log = sim.run(
        t_env0,
        duration,
        dt,
        state0,
        orbit_target_radius=orbit_radius,
        orbit_speed_tolerance=CFG.orbit_speed_tol,
        orbit_radial_tolerance=CFG.orbit_radial_tol,
        orbit_alt_tolerance=CFG.orbit_alt_tol,
        exit_on_orbit=CFG.exit_on_orbit,
        post_orbit_coast_s=CFG.post_orbit_coast_s,
    )

    a, rp, ra = orbital_elements_from_state(log.r[-1], log.v[-1], sim.earth.mu)
    rp_alt = rp - R_EARTH if rp is not None else None
    ra_alt = ra - R_EARTH if ra is not None else None

    return OptResult(
        params={
            "pitch_start_m": pitch_start_m,
            "pitch_end_m": pitch_end_m,
            "target_ap_alt_m": target_ap_alt_m,
        },
        final_mass=log.m[-1],
        orbit_achieved=log.orbit_achieved,
        perigee_alt_m=rp_alt,
        apoapsis_alt_m=ra_alt,
    )


def random_search(n_samples: int = 15) -> OptResult:
    best: OptResult | None = None
    best_score: float = -math.inf
    for i in range(n_samples):
        # Sample pitch turn start/end and apoapsis target in reasonable bounds
        p_start = random.uniform(3000.0, 7000.0)
        p_end = random.uniform(30000.0, 70000.0)
        if p_end <= p_start:
            p_end = p_start + 10000.0
        ap_target = random.uniform(350_000.0, 500_000.0)

        res = evaluate_once(p_start, p_end, ap_target)
        achieved = res.orbit_achieved and res.perigee_alt_m is not None and res.perigee_alt_m > 0
        # Penalize non-achievement and negative perigee
        if achieved:
            score = res.final_mass
        else:
            perigee_pen = abs(res.perigee_alt_m) if res.perigee_alt_m is not None else 1e6
            score = -1e9 - perigee_pen

        if score > best_score:
            best = res
            best_score = score
            print(f"[{i+1}/{n_samples}] New best: mass={res.final_mass:.1f} kg, "
                  f"orbit={res.orbit_achieved}, p_start={p_start:.0f}, p_end={p_end:.0f}, "
                  f"ap_target={ap_target/1000:.0f} km")
    return best


if __name__ == "__main__":
    random.seed(0)
    result = random_search()
    if result:
        print("\n=== Best result ===")
        print(f"Final mass: {result.final_mass:.1f} kg")
        print(f"Orbit achieved: {result.orbit_achieved}")
        print(f"Perigee alt: {result.perigee_alt_m/1000:.2f} km" if result.perigee_alt_m else "Perigee: n/a")
        print(f"Apoapsis alt: {result.apoapsis_alt_m/1000:.2f} km" if result.apoapsis_alt_m else "Apoapsis: n/a")
        print(f"Params: {result.params}")
