"""
Entry point to run a simple end-to-end launch simulation.

This builds the Earth/atmosphere/aero/rocket stack, runs a fixed-step
integration, and prints a short summary. Parameters are loosely inspired by
SpaceX's BFR/Starship system but reduced to two stages with simplified thrust
and mass numbers for a lightweight demo.
"""

from __future__ import annotations

import datetime as dt
import numpy as np

from aerodynamics import Aerodynamics, CdModel
from atmosphere import AtmosphereModel
from gravity import EarthModel, MU_EARTH, OMEGA_EARTH, R_EARTH
from integrators import RK4, State
from rocket import Engine, Rocket, Stage
from simulation import Guidance, Simulation
from config import CFG
import matplotlib.pyplot as plt
from matplotlib import animation

def simple_pitch_program(t: float, state: State) -> np.ndarray:
    """
    Gravity-turn inspired pitch:
    - <5 km: vertical
    - 5–60 km: blend from vertical toward downrange (east)
    - >60 km: downrange / prograde if velocity is established
    """
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
    if alt < 5_000.0:
        return r_hat
    elif alt < 60_000.0:
        w = (alt - 5_000.0) / 55_000.0
        direction = (1.0 - w) * r_hat + w * east
    else:
        speed = np.linalg.norm(v)
        if speed > 1.0:
            direction = v / speed  # prograde
        else:
            direction = east
    n = np.linalg.norm(direction)
    return direction / n if n > 0.0 else r_hat


def throttle_schedule(t: float, state: State) -> float:
    """Hold full throttle; adapt here if you want to throttle for max-Q, etc."""
    return 1.0


def build_rocket() -> Rocket:
    # Approximate BFR-like stages (order-of-magnitude values)
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
        prop_mass=CFG.booster_prop_mass,
        engine=booster_engine,
        ref_area=CFG.ref_area_m2,
    )
    upper_stage = Stage(
        dry_mass=CFG.upper_dry_mass,
        prop_mass=CFG.upper_prop_mass,
        engine=upper_engine,
        ref_area=CFG.ref_area_m2,
    )

    return Rocket(
        stages=[booster_stage, upper_stage],
        main_engine_ramp_time=CFG.main_engine_ramp_time,
        upper_engine_ramp_time=CFG.upper_engine_ramp_time,
        meco_mach=6.0,
        separation_delay=CFG.separation_delay_s,
        upper_ignition_delay=CFG.upper_ignition_delay_s,
        separation_altitude_m=None,  # stage on depletion trigger, but separation is time-based
        earth_radius=R_EARTH,
    )


def build_simulation() -> tuple[Simulation, State, float]:
    earth = EarthModel(mu=MU_EARTH, radius=R_EARTH, omega_vec=OMEGA_EARTH)
    atmosphere = AtmosphereModel(lat_deg=CFG.launch_lat_deg, lon_deg=CFG.launch_lon_deg)
    cd_model = CdModel(2.0)
    rocket = build_rocket()
    aero = Aerodynamics(atmosphere=atmosphere, cd_model=cd_model, reference_area=None)
    guidance = Guidance(pitch_program=simple_pitch_program, throttle_schedule=throttle_schedule)
    integrator = RK4()
    sim = Simulation(earth=earth, atmosphere=atmosphere, aerodynamics=aero, rocket=rocket, integrator=integrator, guidance=guidance)

    # Initial state: surface at launch site, co-rotating atmosphere
    lat = np.deg2rad(CFG.launch_lat_deg)
    lon = np.deg2rad(CFG.launch_lon_deg)
    r0 = R_EARTH * np.array(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        dtype=float,
    )
    v0 = earth.atmosphere_velocity(r0)  # start with Earth's rotation speed
    m0 = sum(stage.total_mass() for stage in rocket.stages)
    state0 = State(r_eci=r0, v_eci=v0, m=m0, stage_index=0)

    # Use current UTC time as simulation start (seconds since 2000-01-01)
    now = dt.datetime.now(dt.timezone.utc)
    base = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
    t0 = (now - base).total_seconds()

    return sim, state0, t0


def main():
    sim, state0, t_env0 = build_simulation()
    duration = CFG.main_duration_s
    dt = CFG.main_dt_s

    orbit_radius = R_EARTH + CFG.target_orbit_alt_m  # ISS-like LEO target
    log = sim.run(
        t_env0,
        duration,
        dt,
        state0,
        orbit_target_radius=orbit_radius,
        orbit_speed_tolerance=CFG.orbit_speed_tol,
        orbit_radial_tolerance=CFG.orbit_radial_tol,
        orbit_alt_tolerance=CFG.orbit_alt_tol,
    )

    # Summary
    final_alt_km = (np.linalg.norm(log.r[-1]) - R_EARTH) / 1000.0
    final_speed = np.linalg.norm(log.v[-1])
    final_mass = log.m[-1]
    final_stage = log.stage[-1]
    max_alt_km = max(log.altitude) / 1000.0
    max_speed = max(log.speed)
    max_q = max(log.dynamic_pressure)
    stage_switch_times = [log.t_sim[i] for i in range(1, len(log.stage)) if log.stage[i] != log.stage[i - 1]]

    print("\n=== Simulation summary ===")
    print(f"Steps: {len(log.t_sim)}")
    print(f"Final sim time  : {log.t_sim[-1]:.1f} s")
    print(f"Final altitude  : {final_alt_km:.2f} km")
    print(f"Final speed     : {final_speed:.1f} m/s")
    print(f"Final mass      : {final_mass:.1f} kg")
    print(f"Final stage idx : {final_stage}")
    print(f"Max altitude    : {max_alt_km:.2f} km")
    print(f"Max speed       : {max_speed:.1f} m/s")
    print(f"Max q           : {max_q:.1f} Pa")
    print(f"Stage switches  : {stage_switch_times}")
    if log.orbit_achieved:
        print("Orbit target met within tolerances.")
    else:
        print("Orbit target NOT met.")

    save_log_to_txt(log, "simulation_log.txt")
    plot_trajectory_3d(log, R_EARTH)
    animate_trajectory(log, R_EARTH)


def plot_trajectory_3d(log, r_earth: float):
    """Static 3D plot of trajectory around a spherical Earth."""
    positions = np.array(log.r)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Earth sphere (light, semi-transparent) and wireframe for context
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x = r_earth * np.outer(np.cos(u), np.sin(v))
    y = r_earth * np.outer(np.sin(u), np.sin(v))
    z = r_earth * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, cmap="Blues", alpha=0.1, linewidth=0, antialiased=False)
    ax.plot_wireframe(x, y, z, color="lightblue", alpha=0.2, linewidth=0.3)

    # Trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color="tab:red", label="Trajectory", lw=2)
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color="green", s=30, label="Launch")
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color="black", s=30, label="Final")

    # Symmetric limits based on max radial distance
    r_max = max(np.linalg.norm(p) for p in positions)
    lim = 1.05 * max(r_earth, r_max)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=35)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("3D Trajectory")
    ax.legend()
    plt.tight_layout()
    plt.show()


def animate_trajectory(log, r_earth: float):
    """Simple 3D animation of the trajectory."""
    positions = np.array(log.r)
    if positions.shape[0] < 2:
        return

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Earth sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x = r_earth * np.outer(np.cos(u), np.sin(v))
    y = r_earth * np.outer(np.sin(u), np.sin(v))
    z = r_earth * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, cmap="Blues", alpha=0.05, linewidth=0, antialiased=False)
    ax.plot_wireframe(x, y, z, color="lightblue", alpha=0.2, linewidth=0.3)

    traj_line, = ax.plot([], [], [], color="tab:red", lw=2)
    point, = ax.plot([], [], [], "o", color="black", markersize=5)

    r_max = max(np.linalg.norm(p) for p in positions)
    lim = 1.05 * max(r_earth, r_max)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=35)

    def init():
        traj_line.set_data([], [])
        traj_line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        return traj_line, point

    def update(frame):
        traj_line.set_data(positions[: frame + 1, 0], positions[: frame + 1, 1])
        traj_line.set_3d_properties(positions[: frame + 1, 2])
        point.set_data([positions[frame, 0]], [positions[frame, 1]])
        point.set_3d_properties([positions[frame, 2]])
        return traj_line, point

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(positions),
        init_func=init,
        interval=50,
        blit=True,
        repeat=False,
    )
    ax.set_title("Trajectory Animation")
    plt.show()


def save_log_to_txt(log, filename: str):
    """Write simulation states to a text (CSV-style) file for analysis."""
    with open(filename, "w") as f:
        f.write(
            "# t_sim_s,t_env_s,alt_m,speed_mps,mass_kg,stage,"
            "thrust_N,drag_N,mdot_kgps,q_Pa,rho_kgpm3,mach,"
            "pos_x_m,pos_y_m,pos_z_m,vel_x_mps,vel_y_mps,vel_z_mps\n"
        )
        n = len(log.t_sim)
        for i in range(n):
            r = log.r[i]
            v = log.v[i]
            f.write(
                f"{log.t_sim[i]:.3f},{log.t_env[i]:.3f},"
                f"{log.altitude[i]:.3f},{log.speed[i]:.3f},"
                f"{log.m[i]:.3f},{log.stage[i]},"
                f"{log.thrust_mag[i]:.3f},{log.drag_mag[i]:.3f},"
                f"{log.mdot[i]:.6f},{log.dynamic_pressure[i]:.3f},"
                f"{log.rho[i]:.6e},{log.mach[i]:.3f},"
                f"{r[0]:.3f},{r[1]:.3f},{r[2]:.3f},"
                f"{v[0]:.3f},{v[1]:.3f},{v[2]:.3f}\n"
            )
    print(f"Saved simulation log to {filename}")


if __name__ == "__main__":
    main()
