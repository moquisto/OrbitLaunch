"""
Entry point to run a simple end-to-end launch simulation.

This builds the Earth/atmosphere/aero/rocket stack, runs a fixed-step
integration, and prints a short summary. Parameters are loosely inspired by
SpaceX's BFR/Starship system but reduced to two stages with simplified thrust
and mass numbers for a lightweight demo.
"""

from __future__ import annotations

import datetime as dt
import importlib
import numpy as np

from aerodynamics import Aerodynamics, CdModel, mach_dependent_cd
from atmosphere import AtmosphereModel
from custom_guidance import orbital_elements_from_state
from gravity import EarthModel
from integrators import RK4, VelocityVerlet, State
from rocket import Engine, Rocket, Stage
from simulation import Guidance, Simulation
from config import CFG
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from typing import Any, Tuple

class ParameterizedPitchProgram:
    """
    Interpolates a pitch angle schedule based on altitude.
    The schedule is defined as a list of [altitude_m, angle_deg] pairs.
    - Angle is degrees from the local horizontal (0=horizontal, 90=vertical).
    - Below the first altitude point, the first angle is held.
    - Above the last altitude point, transitions to prograde guidance.
    """

    def __init__(self, schedule: list[list[float]], prograde_threshold: float, earth_radius: float):
        # Sort schedule by altitude
        self.schedule = sorted(schedule, key=lambda p: p[0])
        self.prograde_threshold = prograde_threshold
        self.earth_radius = earth_radius
        self.alt_points = np.array([p[0] for p in self.schedule])
        self.angle_points_rad = np.deg2rad([p[1] for p in self.schedule])

    def __call__(self, t: float, state: State) -> np.ndarray:
        r = np.asarray(state.r_eci, dtype=float)
        v = np.asarray(state.v_eci, dtype=float)
        r_norm = np.linalg.norm(r)
        if r_norm == 0.0:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        r_hat = r / r_norm

        # Define local orientation vectors (up, east)
        east = np.cross([0.0, 0.0, 1.0], r_hat)
        east_norm = np.linalg.norm(east)
        east = east / east_norm if east_norm > 0.0 else np.array([1.0, 0.0, 0.0], dtype=float)

        alt = r_norm - self.earth_radius
        final_alt = self.alt_points[-1]

        if alt > final_alt:
            # Above pitch program, go prograde if fast enough
            speed = np.linalg.norm(v)
            if speed > self.prograde_threshold:
                direction = v / speed
            else:
                # If speed is low, point horizontally
                direction = east
        else:
            # Interpolate angle from schedule
            pitch_rad = np.interp(alt, self.alt_points, self.angle_points_rad)
            direction = np.cos(pitch_rad) * east + np.sin(pitch_rad) * r_hat

        n = np.linalg.norm(direction)
        return direction / n if n > 0.0 else r_hat


def throttle_schedule(t: float, state: State, cfg_instance: Any) -> float: # Added cfg_instance
    """Hold full throttle; adapt here if you want to throttle for max-Q, etc."""
    return cfg_instance.throttle_guidance.base_throttle_cmd


class ParameterizedThrottleProgram:
    """
    Interpolates a throttle schedule for the upper stage based on time since
    ignition. The schedule is a list of [time_sec, throttle_level] pairs.
    """

    def __init__(self, schedule: list[list[float]]):
        self.schedule = sorted(schedule, key=lambda p: p[0])
        self.time_points = np.array([p[0] for p in self.schedule])
        self.throttle_points = np.array([p[1] for p in self.schedule])

    def __call__(self, t: float, state: State) -> float:
        stage_idx = getattr(state, "stage_index", 0)
        if stage_idx == 0:
            return 1.0  # Booster always full throttle

        ignition_time = getattr(state, "upper_ignition_start_time", None)
        if ignition_time is None:
            return 0.0  # Not yet ignited

        time_since_ignition = t - ignition_time
        if time_since_ignition < 0:
            return 0.0

        # Interpolate throttle level from the schedule
        throttle = np.interp(
            time_since_ignition, self.time_points, self.throttle_points, left=0.0, right=0.0
        )
        return float(throttle)


def build_rocket(cfg_instance) -> Rocket: # Added cfg_instance parameter
    # Approximate BFR-like stages (order-of-magnitude values)
    booster_engine = Engine(
        thrust_vac=cfg_instance.vehicle.booster_thrust_vac,
        thrust_sl=cfg_instance.vehicle.booster_thrust_sl,
        isp_vac=cfg_instance.vehicle.booster_isp_vac,
        isp_sl=cfg_instance.vehicle.booster_isp_sl,
        cfg=cfg_instance,
    )
    upper_engine = Engine(
        thrust_vac=cfg_instance.vehicle.upper_thrust_vac,
        thrust_sl=cfg_instance.vehicle.upper_thrust_sl,
        isp_vac=cfg_instance.vehicle.upper_isp_vac,
        isp_sl=cfg_instance.vehicle.upper_isp_sl,
        cfg=cfg_instance,
    )

    booster_stage = Stage(
        dry_mass=cfg_instance.vehicle.booster_dry_mass,
        prop_mass=cfg_instance.vehicle.booster_prop_mass,
        engine=booster_engine,
        ref_area=cfg_instance.vehicle.ref_area_m2,
    )
    upper_stage = Stage(
        dry_mass=cfg_instance.vehicle.upper_dry_mass,
        prop_mass=cfg_instance.vehicle.upper_prop_mass,
        engine=upper_engine,
        ref_area=cfg_instance.vehicle.ref_area_m2,
    )

    return Rocket(
        stages=[booster_stage, upper_stage],
        cfg_instance=cfg_instance,
        main_engine_ramp_time=cfg_instance.staging.main_engine_ramp_time,
        upper_engine_ramp_time=cfg_instance.staging.upper_engine_ramp_time,
        meco_mach=cfg_instance.staging.meco_mach,
        separation_delay=cfg_instance.staging.separation_delay_s,
        upper_ignition_delay=cfg_instance.staging.upper_ignition_delay_s,
        separation_altitude_m=cfg_instance.staging.separation_altitude_m,  # stage on depletion trigger, but separation is time-based
        earth_radius=cfg_instance.central_body.earth_radius_m,
        min_throttle=cfg_instance.vehicle.engine_min_throttle,
        shutdown_ramp_time=cfg_instance.staging.engine_shutdown_ramp_s,
        throttle_shape_full_threshold=cfg_instance.vehicle.throttle_full_shape_threshold,
        mach_ref_speed=cfg_instance.vehicle.mach_reference_speed,
        booster_throttle_program=cfg_instance.throttle_guidance.booster_throttle_program,
    )


def build_simulation(cfg_instance) -> tuple[Simulation, State, float]: # Added cfg_instance parameter
    earth = EarthModel(
        mu=cfg_instance.central_body.earth_mu,
        radius=cfg_instance.central_body.earth_radius_m,
        omega_vec=np.array(cfg_instance.central_body.earth_omega_vec, dtype=float),
        j2=cfg_instance.central_body.j2_coeff if cfg_instance.central_body.use_j2 else None,
    )
    atmosphere = AtmosphereModel(
        h_switch=cfg_instance.atmosphere.atmosphere_switch_alt_m,
        lat_deg=cfg_instance.launch_site.launch_lat_deg,
        lon_deg=cfg_instance.launch_site.launch_lon_deg,
        f107=cfg_instance.atmosphere.atmosphere_f107,
        f107a=cfg_instance.atmosphere.atmosphere_f107a,
        ap=cfg_instance.atmosphere.atmosphere_ap,
    )
    cd_model = CdModel(mach_dependent_cd, cfg_instance) 
    rocket = build_rocket(cfg_instance) # Pass cfg_instance
    aero = Aerodynamics(atmosphere=atmosphere, cd_model=cd_model, reference_area=None, cfg=cfg_instance)
    # --- Pitch Program Selection ---
    if cfg_instance.pitch_guidance.pitch_guidance_mode == 'parameterized':
        pitch_program = ParameterizedPitchProgram(
            schedule=cfg_instance.pitch_guidance.pitch_program,
            prograde_threshold=cfg_instance.pitch_guidance.pitch_prograde_speed_threshold,
            earth_radius=cfg_instance.central_body.earth_radius_m,
        )
    elif cfg_instance.pitch_guidance.pitch_guidance_mode == 'function':
        try:
            module_name, func_name = cfg_instance.pitch_guidance.pitch_guidance_function.rsplit('.', 1)
            module = importlib.import_module(module_name)
            pitch_program = getattr(module, func_name)
            # When using function, pass cfg_instance as a partial application
            pitch_program = lambda t, state: getattr(module, func_name)(t, state, cfg_instance)
        except (ImportError, AttributeError, ValueError) as e:
            raise ImportError(f"Could not load pitch guidance function '{cfg_instance.pitch_guidance.pitch_guidance_function}': {e}")
    else:
        raise ValueError(f"Unknown pitch_guidance_mode: '{cfg_instance.pitch_guidance.pitch_guidance_mode}'")

    guidance = Guidance(pitch_program=pitch_program, throttle_schedule=lambda t, state: throttle_schedule(t, state, cfg_instance)) # Pass cfg_instance
    integrator_name = str(getattr(cfg_instance.simulation_timing, "integrator", "rk4")).lower()
    if integrator_name in ("rk4", "runge-kutta", "rk"):
        integrator = RK4()
    elif integrator_name in ("velocity_verlet", "verlet", "vv"):
        integrator = VelocityVerlet()
    else:
        raise ValueError(f"Unknown integrator '{cfg_instance.simulation_timing.integrator}'. Expected 'rk4' or 'velocity_verlet'.")
    sim = Simulation(
        earth=earth,
        atmosphere=atmosphere,
        aerodynamics=aero,
        rocket=rocket,
        cfg_instance=cfg_instance,
        integrator=integrator,
        guidance=guidance,
        max_q_limit=cfg_instance.path_constraints.max_q_limit,
        max_accel_limit=cfg_instance.path_constraints.max_accel_limit,
        impact_altitude_buffer_m=cfg_instance.termination_logic.impact_altitude_buffer_m,
        escape_radius_factor=cfg_instance.termination_logic.escape_radius_factor,
    )

    # Initial state: surface at launch site, co-rotating atmosphere
    lat = np.deg2rad(cfg_instance.launch_site.launch_lat_deg)
    lon = np.deg2rad(cfg_instance.launch_site.launch_lon_deg)
    r0 = cfg_instance.central_body.earth_radius_m * np.array(
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


def main(cfg_instance): # Accept cfg_instance
    sim, state0, t_env0 = build_simulation(cfg_instance) # Pass cfg_instance to build_simulation
    duration = cfg_instance.simulation_timing.main_duration_s
    dt = cfg_instance.simulation_timing.main_dt_s

    # --- Throttle Controller Selection ---
    orbit_radius = cfg_instance.central_body.earth_radius_m + cfg_instance.target_orbit.target_orbit_alt_m
    if cfg_instance.throttle_guidance.throttle_guidance_mode == 'parameterized':
        controller = ParameterizedThrottleProgram(schedule=cfg_instance.throttle_guidance.upper_stage_throttle_program)
    elif cfg_instance.throttle_guidance.throttle_guidance_mode == 'function':
        try:
            module_name, class_name = cfg_instance.throttle_guidance.throttle_guidance_function_class.rsplit('.', 1)
            module = importlib.import_module(module_name)
            ControllerClass = getattr(module, class_name)
            # The original controller class requires target_radius and mu
            try:
                controller = ControllerClass(target_radius=orbit_radius, mu=cfg_instance.central_body.earth_mu, cfg=cfg_instance)
            except TypeError:
                controller = ControllerClass(target_radius=orbit_radius, mu=cfg_instance.central_body.earth_mu)
        except (ImportError, AttributeError, ValueError) as e:
            raise ImportError(f"Could not load throttle guidance class '{cfg_instance.throttle_guidance.throttle_guidance_function_class}': {e}")
    else:
        raise ValueError(f"Unknown throttle_guidance_mode: '{cfg_instance.throttle_guidance.throttle_guidance_mode}'")

    sim.guidance.throttle_schedule = controller
    log = sim.run(
        t_env0,
        duration,
        dt,
        state0,
        orbit_target_radius=orbit_radius,
        orbit_speed_tolerance=cfg_instance.orbit_tolerances.orbit_speed_tol,
        orbit_radial_tolerance=cfg_instance.orbit_tolerances.orbit_radial_tol,
        orbit_alt_tolerance=cfg_instance.orbit_tolerances.orbit_alt_tol,
        exit_on_orbit=cfg_instance.orbit_tolerances.exit_on_orbit,
        post_orbit_coast_s=cfg_instance.orbit_tolerances.post_orbit_coast_s,
    )

    # Summary
    earth_radius = cfg_instance.central_body.earth_radius_m
    final_alt_km = (np.linalg.norm(log.r[-1]) - earth_radius) / 1000.0
    final_speed = np.linalg.norm(log.v[-1])
    final_mass = log.m[-1]
    final_stage = log.stage[-1]
    max_alt_km = max(log.altitude) / 1000.0
    max_speed = max(log.speed)
    max_q = max(log.dynamic_pressure)
    stage_switch_times = [log.t_sim[i] for i in range(1, len(log.stage)) if log.stage[i] != log.stage[i - 1]]
    # Basic orbital diagnostics from final state
    a, rp, ra = orbital_elements_from_state(log.r[-1], log.v[-1], cfg_instance.central_body.earth_mu)
    rp_alt_km = (rp - earth_radius) / 1000.0 if rp is not None else None
    ra_alt_km = (ra - earth_radius) / 1000.0 if ra is not None else None
    def print_state(label: str, idx: int):
        a_i, rp_i, ra_i = orbital_elements_from_state(log.r[idx], log.v[idx], cfg_instance.central_body.earth_mu)
        rp_alt_i = (rp_i - earth_radius) / 1000.0 if rp_i is not None else None
        ra_alt_i = (ra_i - earth_radius) / 1000.0 if ra_i is not None else None
        rp_str = f"{rp_alt_i:.2f}" if rp_alt_i is not None else "n/a"
        ra_str = f"{ra_alt_i:.2f}" if ra_alt_i is not None else "n/a"
        print(
            f"{label} @ t={log.t_sim[idx]:.1f}s: "
            f"alt={log.altitude[idx]/1000:.2f} km, "
            f"speed={log.speed[idx]:.1f} m/s, "
            f"fpa={log.flight_path_angle_deg[idx]:.2f} deg, "
            f"q={log.dynamic_pressure[idx]:.0f} Pa, "
            f"rp={rp_str} km, ra={ra_str} km"
        )

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
    if a is not None:
        print(f"Semi-major axis : {a/1000:.2f} km")
    if rp_alt_km is not None and ra_alt_km is not None:
        print(f"Perigee altitude: {rp_alt_km:.2f} km")
        print(f"Apoapsis altitude: {ra_alt_km:.2f} km")
    if log.orbit_achieved:
        print("Orbit target met within tolerances.")
    else:
        print("Orbit target NOT met.")
    # Stage fuel/engine timing diagnostics
    booster_empty = sim.rocket.stage_fuel_empty_time[0]
    upper_empty = sim.rocket.stage_fuel_empty_time[1]
    booster_off = sim.rocket.stage_engine_off_complete_time[0]
    upper_off = sim.rocket.stage_engine_off_complete_time[1]
    if booster_empty is not None:
        print(f"Booster fuel empty at t = {booster_empty:.1f} s")
    if booster_off is not None:
        print(f"Booster engine off at t = {booster_off:.1f} s")
    if upper_empty is not None:
        print(f"Upper fuel empty at t = {upper_empty:.1f} s")
    if upper_off is not None:
        print(f"Upper engine off at t = {upper_off:.1f} s")
    print(f"Remaining prop (booster, upper): {sim.rocket.stage_prop_remaining}")

    # Calculate indices for print_state calls
    idx_max_alt = np.argmax(log.altitude)
    idx_max_speed = np.argmax(log.speed)
    
    print_state("Max altitude", idx_max_alt)
    print_state("Max speed", idx_max_speed)
    if upper_off is not None:
        idx_upper_off = np.argmin(np.abs(np.array(log.t_sim) - upper_off))
        print_state("Upper engine off", idx_upper_off)

    save_log_to_txt(log, cfg_instance.output.log_filename)
    # Enable static trajectory plot; keep animation disabled for headless use.
    if cfg_instance.output.plot_trajectory:
        plot_trajectory_3d(log, cfg_instance.central_body.earth_radius_m)
    if cfg_instance.output.animate_trajectory:
        animate_trajectory(log, cfg_instance.central_body.earth_radius_m)


def plot_trajectory_3d(log, r_earth: float):
    """Static 3D plot of trajectory around a spherical Earth."""
    positions = np.array(log.r)
    times = np.array(log.t_sim)

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

    # Trajectory with time-based color gradient
    if positions.shape[0] > 1:
        segments = np.stack([positions[:-1], positions[1:]], axis=1)
        t_norm = (times - times.min()) / max(times.ptp(), 1e-9)
        colors = plt.cm.plasma(t_norm[:-1])
        lc = Line3DCollection(segments, colors=colors, linewidths=2, label="Trajectory")
        ax.add_collection3d(lc)
    else:
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
    plt.legend()
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
            "fpa_deg,v_vertical_mps,v_horizontal_mps,specific_energy_Jpkg,"
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
                f"{log.flight_path_angle_deg[i]:.3f},{log.v_vertical[i]:.3f},{log.v_horizontal[i]:.3f},{log.specific_energy[i]:.3f},"
                f"{r[0]:.3f},{r[1]:.3f},{r[2]:.3f},"
                f"{v[0]:.3f},{v[1]:.3f},{v[2]:.3f}\n"
            )
    print(f"Saved simulation log to {filename}")


if __name__ == "__main__":
    main(CFG)
