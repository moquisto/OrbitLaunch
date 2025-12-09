import numpy as np
import matplotlib.pyplot as plt

from main import (
    build_simulation,
    orbital_elements_from_state,
    R_EARTH,
    MU_EARTH,
    CFG,
)


def run_with_forced_meco(meco_time_s, target_radius):
    """Run a simulation but force booster cutoff at a given MECO time."""
    sim, state0, t_env0 = build_simulation()

    # --- Patch throttle schedule ---
    def forced_meco_throttle(t, state):
        if t < meco_time_s:
            return 1.0    # full power until cutoff
        else:
            return 0.0    # engine off (force MECO)

    sim.guidance.throttle_schedule = forced_meco_throttle

    log = sim.run(
        t_env0,
        duration=2000,         # short run, since we are not circularizing
        dt=1.0,
        state0=state0,
        orbit_target_radius=target_radius,
        orbit_speed_tolerance=1e9,   # disable orbit exit
        orbit_radial_tolerance=1e9,
        orbit_alt_tolerance=1e9,
        exit_on_orbit=False,
    )

    # Orbit elements from final state
    a, rp, ra = orbital_elements_from_state(log.r[-1], log.v[-1], MU_EARTH)

    # Propellant used = initial - final
    m_used = state0.m - log.m[-1]

    # Orbit error for circular orbit:
    # deviation = |rp - r_target| + |ra - r_target|
    if rp is not None and ra is not None:
        orbit_error = abs(rp - target_radius) + abs(ra - target_radius)
    else:
        orbit_error = np.inf

    return m_used, orbit_error


def main():
    # --- Sweep MECO times ---
    meco_times = np.linspace(50, 300, 60)  # seconds

    target_radius = R_EARTH + 20000e3  # e.g., 20,000 km MEO

    prop_used_list = []
    orbit_error_list = []

    for meco in meco_times:
        print(f"Running MECO = {meco:.1f}s ...")
        m_used, err = run_with_forced_meco(meco, target_radius)
        prop_used_list.append(m_used)
        orbit_error_list.append(err)

    meco_times = np.array(meco_times)
    prop_used_list = np.array(prop_used_list)
    orbit_error_list = np.array(orbit_error_list)

    # --- Plot 1: Fuel used vs MECO time ---
    plt.figure(figsize=(10, 5))
    plt.plot(meco_times, prop_used_list, lw=2)
    plt.xlabel("MECO time [s]")
    plt.ylabel("Propellant used [kg]")
    plt.title("Propellant used vs MECO time")
    plt.grid(True)
    plt.tight_layout()

    # --- Plot 2: Orbit error vs MECO time ---
    plt.figure(figsize=(10, 5))
    plt.plot(meco_times, orbit_error_list, lw=2)
    plt.xlabel("MECO time [s]")
    plt.ylabel("Orbit error [m]")
    plt.title("Orbit deviation from target MEO vs MECO time")
    plt.yscale("log")   # helpful because errors vary wildly
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()

    # J=α(propellant)+β(orbit error) - maybe add this later as a measurement of the best main engine cutoff time.
