"""
Structured optimizer to minimize propellant (maximize final mass) for reaching
the target circular orbit using the existing closed-loop guidance:
  - Pitch program defined by start/end altitudes.
  - Two-phase upper-stage throttle (boost to target apoapsis, coast, circularize).

We search over a small set of setpoints:
  * pitch_turn_start_m
  * pitch_turn_end_m
  * target_ap_alt_m (upper boost target)
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import matplotlib

# Keep optimizer headless; main simulation still plots interactively.
matplotlib.use("Agg")

from config import CFG
from gravity import R_EARTH
from main import (
    TwoPhaseUpperThrottle,
    build_simulation,
    orbital_elements_from_state,
)


@dataclass
class OptResult:
    params: dict
    final_mass: float
    orbit_achieved: bool
    perigee_alt_m: float | None
    apoapsis_alt_m: float | None


def evaluate(p_start: float, p_end: float, ap_target: float) -> OptResult:
    # Apply parameter overrides
    CFG.pitch_turn_start_m = p_start
    CFG.pitch_turn_end_m = p_end
    # Pitch blend exponent fixed in this optimizer; could be varied too
    CFG.target_orbit_alt_m = ap_target
    # Optional: adjust throttle caps and limits if desired (kept as config defaults here)

    sim, state0, t_env0 = build_simulation()
    orbit_radius = R_EARTH + CFG.target_orbit_alt_m
    controller = TwoPhaseUpperThrottle(target_radius=orbit_radius)
    sim.guidance.throttle_schedule = controller

    duration = min(CFG.main_duration_s, 1400.0)
    dt = max(CFG.main_dt_s, 1.5)

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
        params={"pitch_start_m": p_start, "pitch_end_m": p_end, "target_ap_alt_m": ap_target},
        final_mass=log.m[-1],
        orbit_achieved=log.orbit_achieved,
        perigee_alt_m=rp_alt,
        apoapsis_alt_m=ra_alt,
    )


def random_search(n_samples: int = 12) -> OptResult:
    best: OptResult | None = None
    best_score = float("-inf")
    target_alt = 420_000.0

    for i in range(n_samples):
        p_start = random.uniform(3000.0, 9000.0)
        p_end = random.uniform(30_000.0, 80_000.0)
        if p_end <= p_start:
            p_end = p_start + 10_000.0
        ap_target = random.uniform(400_000.0, 440_000.0)

        res = evaluate(p_start, p_end, ap_target)
        achieved = res.orbit_achieved and res.perigee_alt_m is not None and res.perigee_alt_m > 0

        if achieved and res.perigee_alt_m is not None and res.apoapsis_alt_m is not None:
            perigee_err = abs(res.perigee_alt_m - target_alt)
            apo_err = abs(res.apoapsis_alt_m - target_alt)
            score = res.final_mass - 0.05 * (perigee_err + apo_err)
        else:
            perigee_pen = abs(res.perigee_alt_m) if res.perigee_alt_m is not None else 1e6
            score = -1e9 - perigee_pen

        if score > best_score:
            best = res
            best_score = score
            print(
                f"[{i+1}/{n_samples}] New best: mass={res.final_mass:.1f} kg, "
                f"orbit={res.orbit_achieved}, perigee={res.perigee_alt_m}, "
                f"apoapsis={res.apoapsis_alt_m}, params={res.params}"
            )

    return best


if __name__ == "__main__":
    random.seed(0)
    result = random_search()
    if result:
        print("\n=== Best result ===")
        print(f"Final mass: {result.final_mass:.1f} kg")
        print(f"Orbit achieved: {result.orbit_achieved}")
        if result.perigee_alt_m is not None:
            print(f"Perigee alt: {result.perigee_alt_m/1000:.2f} km")
        else:
            print("Perigee: n/a")
        if result.apoapsis_alt_m is not None:
            print(f"Apoapsis alt: {result.apoapsis_alt_m/1000:.2f} km")
        else:
            print("Apoapsis: n/a")
        print(f"Params: {result.params}")
