"""
orbit_guidance_sweep.py

Sweeps different pitch-turn guidance profiles and selects the one that achieves
a perfectly circular equatorial orbit at the target altitude using the least
propellant.

Parameter swept:
    - pitch_turn_end_m  (end of gravity turn transition altitude)

For each guidance profile:
    - Build simulation
    - Override pitch program
    - Run ascent
    - Check orbit quality
    - Record propellant use

Outputs:
    - Best profile
    - Propellant vs guidance parameter
    - Orbit error vs guidance parameter
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from main import (
    build_simulation,
    R_EARTH,
    MU_EARTH,
    orbital_elements_from_state,
)
from config import CFG


# ------------------------------------------------------------
# Orbit checking utilities
# ------------------------------------------------------------

def compute_orbit_errors(r, v, target_r):
    """
    Compute robust orbit error metrics for a circular equatorial orbit.
    Returns dictionary of errors in SI units (m, m/s, radians).
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)

    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)

    if r_norm <= 0.0:
        # Invalid state — return large errors
        return {
            "altitude_error": np.inf,
            "speed_error": np.inf,
            "radial_velocity_error": np.inf,
            "inclination_error": np.inf,
        }

    # radial velocity
    vr = float(np.dot(v, r / r_norm))

    # circular speed target
    v_circ = np.sqrt(MU_EARTH / target_r)

    # inclination: use angular momentum vector h = r x v
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    if h_norm <= 0.0:
        inc = np.pi / 2.0  # degenerate; set large inclination error
    else:
        cos_inc = float(h[2] / h_norm)
        # numerical safety
        cos_inc = np.clip(cos_inc, -1.0, 1.0)
        inc = float(np.arccos(cos_inc))

    return {
        "altitude_error": abs(r_norm - target_r),
        "speed_error": abs(v_norm - v_circ),
        "radial_velocity_error": abs(vr),
        "inclination_error": abs(inc),
    }


def orbit_success(err):
    """
    Strict success criteria. Tweak tolerances here if necessary.
    """
    return (
        err["altitude_error"] < 30.0
        and err["speed_error"] < 5.0
        and err["radial_velocity_error"] < 0.5
        and err["inclination_error"] < np.deg2rad(0.05)
    )


# ------------------------------------------------------------
# Customizable pitch program generator
# ------------------------------------------------------------

def make_pitch_program(pitch_turn_start_m, pitch_turn_end_m):
    """
    Returns a pitch program with variable pitch_turn_end_m.
    """

    def pitch_program(t, state):
        r = np.asarray(state.r_eci, float)
        v = np.asarray(state.v_eci, float)
        r_norm = np.linalg.norm(r)
        if r_norm == 0.0:
            return np.array([0.0, 0.0, 1.0], float)
        r_hat = r / r_norm

        # Local horizontal east direction (simple approximation)
        east = np.cross([0.0, 0.0, 1.0], r_hat)
        east_norm = np.linalg.norm(east)
        if east_norm > 0.0:
            east = east / east_norm
        else:
            east = np.array([1.0, 0.0, 0.0], float)

        alt = r_norm - R_EARTH

        if alt < pitch_turn_start_m:
            return r_hat
        elif alt < pitch_turn_end_m and (pitch_turn_end_m - pitch_turn_start_m) > 0.0:
            w = (alt - pitch_turn_start_m) / \
                (pitch_turn_end_m - pitch_turn_start_m)
            direction = (1.0 - w) * r_hat + w * east
            n = np.linalg.norm(direction)
            if n > 0.0:
                return direction / n
            return r_hat
        else:
            speed = np.linalg.norm(v)
            if speed > 0.0:
                return v / speed
            return east

    return pitch_program


# ------------------------------------------------------------
# Run a single simulation under a given guidance profile
# ------------------------------------------------------------

def run_orbit_sim(pitch_turn_end_m):
    target_r = R_EARTH + CFG.target_orbit_alt_m

    # Build sim and initial state
    sim, state0, t_env0 = build_simulation()

    # Save initial mass
    initial_mass = float(state0.m)

    # Patch guidance pitch program
    sim.guidance.pitch_program = make_pitch_program(
        CFG.pitch_turn_start_m,
        pitch_turn_end_m,
    )

    # Run simulation — allow it to exit when orbit is detected so we capture prop usage at injection
    log = sim.run(
        t_env_start=t_env0,
        duration=CFG.main_duration_s,
        dt=CFG.main_dt_s,
        state0=state0,
        orbit_target_radius=target_r,
        orbit_speed_tolerance=CFG.orbit_speed_tol,
        orbit_radial_tolerance=CFG.orbit_radial_tol,
        orbit_alt_tolerance=CFG.orbit_alt_tol,
        exit_on_orbit=True,          # <--- important: stop when orbit achieved
        post_orbit_coast_s=0.0,
    )

    # If the sim terminated early due to impact/escape etc, log.r may still have last state.
    r_final = log.r[-1]
    v_final = log.v[-1]
    m_final = float(log.m[-1])

    # Orbit analysis
    err = compute_orbit_errors(r_final, v_final, target_r)
    ok = orbit_success(err) or bool(log.orbit_achieved)

    # Classical orbital elements (may be None)
    a, rp, ra = orbital_elements_from_state(r_final, v_final, MU_EARTH)

    prop_used = initial_mass - m_final

    return {
        "success": ok,
        "errors": err,
        "prop_used": prop_used,
        "pitch_end": pitch_turn_end_m,
        "a": a,
        "rp": rp,
        "ra": ra,
        "cutoff_reason": log.cutoff_reason,
    }


# ------------------------------------------------------------
# Main sweep
# ------------------------------------------------------------

def main():
    # Parameter sweep range (meters)
    pitch_end_values = np.linspace(20000, 120000, 5)  # 20–120 km

    results = []

    print("\n=== Starting guidance sweep ===")
    t_start = time.time()
    for pe in pitch_end_values:
        print(f"Testing pitch_turn_end_m = {pe/1000:.1f} km ...")
        result = run_orbit_sim(pe)
        results.append(result)
    t_end = time.time()
    print(f"\nSweep completed in {t_end - t_start:.1f} s (wall time).")

    # Extract arrays for plotting
    pitch_vals = np.array([r["pitch_end"] for r in results])
    prop_used = np.array([r["prop_used"] for r in results])

    # Compose a simple scalar orbit error metric for plotting (weighted sum)
    orbit_errors = np.array([
        r["errors"]["altitude_error"]
        + r["errors"]["speed_error"]
        + 10.0 * r["errors"]["radial_velocity_error"]
        + 10000.0 * r["errors"]["inclination_error"]
        for r in results
    ])

    success_mask = np.array([r["success"] for r in results])

    # Find best valid result
    valid_results = [r for r in results if r["success"]]
    if valid_results:
        best = min(valid_results, key=lambda r: r["prop_used"])
        print("\n=== BEST GUIDANCE PROFILE FOUND ===")
        print(f"Pitch turn end altitude: {best['pitch_end']/1000:.2f} km")
        print(f"Propellant used:        {best['prop_used']:.2f} kg")
        print(f"Orbit errors:           {best['errors']}")
        print(f"Cutoff reason:          {best.get('cutoff_reason')}")
    else:
        print("\nNo successful orbits achieved in the sweep.")
        best = None

    # --------------------------------------------------------
    # Plotting
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))
    plt.plot(pitch_vals / 1000.0, prop_used, '-o')
    plt.xlabel("Pitch turn end altitude [km]")
    plt.ylabel("Propellant used [kg]")
    plt.title("Propellant Consumption vs Guidance Profile")
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(10, 5))
    plt.plot(pitch_vals / 1000.0, orbit_errors, '-o')
    plt.xlabel("Pitch turn end altitude [km]")
    plt.ylabel("Orbit error metric (weighted)")
    plt.title("Orbit Error vs Guidance Profile")
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
