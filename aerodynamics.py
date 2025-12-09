"""
Aerodynamics module structure: drag-only interface.
Implementations are placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union, Any

import numpy as np
from atmosphere import AtmosphereModel


class CdModel:
    """
    Drag coefficient model. Accepts either a constant value or a callable
    returning Cd as a function of Mach number.
    """

    def __init__(self, value_or_callable: Union[float, Callable[[float], float]] = 2.0):
        self.value_or_callable = value_or_callable

    def cd(self, mach: float) -> float:
        """Return drag coefficient for a given Mach number.

        If value_or_callable is a constant, that value is returned.
        If it is a callable, it is evaluated as Cd(Mach).
        """
        if callable(self.value_or_callable):
            return float(self.value_or_callable(mach))
        return float(self.value_or_callable)


@dataclass
class Aerodynamics:
    atmosphere: AtmosphereModel
    cd_model: CdModel
    reference_area: float  # [m^2], reference/frontal area of the rocket

    def drag_force(self, state: Any, earth: Any, t: float) -> np.ndarray:
        """Compute aerodynamic drag force in the ECI frame.

        Assumptions
        -----------
        - The rocket's longitudinal axis is aligned with the air-relative
          velocity (zero angle of attack).
        - Drag acts purely opposite to the air-relative velocity vector.
        - The reference area is constant and represents the effective
          frontal area normal to the flow.
        - `state` provides `r_eci` and `v_eci` attributes (position and
          velocity in ECI, both as 3-vectors in meters / m/s).
        - `earth` provides `radius` and `atmosphere_velocity(r_eci)`.
        """
        # Extract position and velocity in ECI.
        r = np.asarray(state.r_eci, dtype=float)
        v = np.asarray(state.v_eci, dtype=float)

        r_norm = np.linalg.norm(r)
        if r_norm == 0.0:
            return np.zeros(3)

        # Altitude above mean surface.
        altitude = r_norm - float(earth.radius)
        if altitude < 0.0:
            altitude = 0.0

        # Atmospheric properties at this altitude and time.
        props = self.atmosphere.properties(altitude, t)
        rho = float(props.rho)
        T = float(props.T)

        if rho <= 0.0:
            return np.zeros(3)

        # Air-relative velocity: rocket velocity minus co-rotating atmosphere.
        v_atm = np.asarray(earth.atmosphere_velocity(r), dtype=float)
        v_rel = v - v_atm
        v_rel_mag = np.linalg.norm(v_rel)
        if v_rel_mag == 0.0:
            return np.zeros(3)

        # Speed of sound (ideal gas, dry air) and Mach number.
        gamma = 1.4
        R_air = 287.05  # J/(kg*K)
        a = np.sqrt(max(gamma * R_air * T, 0.0))
        mach = v_rel_mag / a if a > 0.0 else 0.0

        # Drag coefficient from model.
        cd = self.cd_model.cd(mach)

        # Drag magnitude: 0.5 * rho * |v_rel|^2 * Cd * A.
        A = float(self.reference_area)
        q = 0.5 * rho * v_rel_mag ** 2
        F_mag = q * cd * A

        # Direction opposite to air-relative velocity.
        F_vec = -F_mag * v_rel / v_rel_mag
        return F_vec


if __name__ == "__main__":
    """
    Simple test plots for drag forces.

    We run two 1D vertical-ascent style tests including Earth's rotation,
    with a co-rotating atmosphere (v_atm = omega × r):

    1) A purely ballistic projectile launched straight up from the surface
       with a given initial speed, under gravity + drag only (no thrust).

    2) A rocket with constant thrust in the radial (upward) direction,
       plus gravity + drag.

    The goal is to visualize how drag behaves as a function of time and
    altitude for these two cases, using the current AtmosphereModel and
    Aerodynamics implementation.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from gravity import EarthModel, MU_EARTH, R_EARTH, OMEGA_EARTH

    # Reuse the AtmosphereModel that was already imported at the top.
    # Here we include Earth's rotation: the atmosphere co-rotates with the
    # planet, so v_rel = v_eci - v_atm.
    earth = EarthModel(mu=MU_EARTH, radius=R_EARTH, omega_vec=OMEGA_EARTH)
    atm = AtmosphereModel()

    # Simple Cd model; use different reference areas for projectile and rocket.
    cd_model = CdModel(2.0)

    # Projectile: smaller, more streamlined body.
    diameter_proj = 0.5  # m
    radius_proj = diameter_proj / 2.0
    A_proj = float(np.pi * radius_proj**2)

    # Rocket: larger body.
    diameter_roc = 3.0  # m
    radius_roc = diameter_roc / 2.0
    A_roc = float(np.pi * radius_roc**2)

    aero_projectile = Aerodynamics(atmosphere=atm, cd_model=cd_model, reference_area=A_proj)
    aero_rocket = Aerodynamics(atmosphere=atm, cd_model=cd_model, reference_area=A_roc)

    class State:
        """Minimal state container with r_eci and v_eci attributes."""
        def __init__(self, r_eci: np.ndarray, v_eci: np.ndarray):
            self.r_eci = r_eci
            self.v_eci = v_eci

    def simulate_vertical(mass: float,
                          thrust_mag: float,
                          v0_up: float,
                          t_final: float,
                          dt: float,
                          aero: Aerodynamics):
        """
        Integrate a simple vertical (radial) motion with gravity + drag
        and optional constant thrust in the radial direction.

        Parameters
        ----------
        mass : float
            Vehicle mass [kg].
        thrust_mag : float
            Constant thrust magnitude [N]. Use 0.0 for a pure projectile.
        v0_up : float
            Initial upward speed [m/s] at the surface.
        t_final : float
            Final simulation time [s].
        dt : float
            Time step [s].
        aero : Aerodynamics
            Aerodynamics model (atmosphere + Cd + reference area).

        Returns
        -------
        times, altitudes_km, speeds, drag_mags : np.ndarray
        """
        # Initial position at surface, along +x.
        r = np.array([R_EARTH, 0.0, 0.0], dtype=float)
        r_norm = np.linalg.norm(r)
        r_hat = r / r_norm

        # Initial velocity: co-rotating with the atmosphere plus a radial
        # upward component. This makes the initial relative wind purely
        # vertical with magnitude v0_up.
        v_atm0 = earth.atmosphere_velocity(r)
        v = v_atm0 + v0_up * r_hat

        state = State(r_eci=r.copy(), v_eci=v.copy())

        times = []
        altitudes_km = []
        speeds = []
        drag_mags = []

        t = 0.0
        while t <= t_final:
            r = state.r_eci
            v = state.v_eci
            r_norm = np.linalg.norm(r)
            if r_norm == 0.0:
                break

            altitude = r_norm - earth.radius
            # Stop if we have fallen back below the surface (except at t=0).
            if altitude < 0.0 and t > 0.0:
                break

            altitude_km = max(altitude, 0.0) / 1000.0
            r_hat = r / r_norm

            # Gravity acceleration and force
            a_g = earth.gravity_accel(r)
            F_g = mass * a_g

            # Thrust in radial direction (upward)
            F_thrust = thrust_mag * r_hat if altitude >= 0.0 else np.zeros(3)

            # Drag force
            F_drag = aero.drag_force(state, earth, t)

            # Net force and acceleration
            F_net = F_g + F_thrust + F_drag
            a_net = F_net / mass

            # Store diagnostics
            times.append(t)
            altitudes_km.append(altitude_km)
            speeds.append(np.linalg.norm(v))
            drag_mags.append(np.linalg.norm(F_drag))

            # Advance state (explicit Euler)
            state.v_eci = v + a_net * dt
            state.r_eci = r + state.v_eci * dt

            t += dt

        return (np.array(times),
                np.array(altitudes_km),
                np.array(speeds),
                np.array(drag_mags))

    # ------------------------------------------------------------------
    # Run two test cases
    # ------------------------------------------------------------------
    dt = 0.1  # s

    # 1) Pure ballistic projectile (no thrust)
    # Choose parameters so that drag is significant but the projectile still
    # reaches several kilometers altitude, making the curve visible on the
    # same axes as the rocket.
    mass_projectile = 500.0      # kg
    v0_projectile = 600.0        # m/s initial upward speed
    t_final_projectile = 200.0   # s
    (t_proj,
     h_proj_km,
     v_proj,
     Fd_proj) = simulate_vertical(mass_projectile,
                                  thrust_mag=0.0,
                                  v0_up=v0_projectile,
                                  t_final=t_final_projectile,
                                  dt=dt,
                                  aero=aero_projectile)

    # 2) Simple rocket with constant thrust
    # Thrust and mass chosen to give T/W ~ 2 at sea level (reasonable for
    # a large launch vehicle), with a shorter total burn/plot time to keep
    # altitudes in a visually useful range.
    mass_rocket = 1.0e5          # kg
    thrust_rocket = 2.0e6        # N (T/W ~2 at sea level)
    v0_rocket = 0.0              # start from rest
    t_final_rocket = 150.0       # s
    (t_roc,
     h_roc_km,
     v_roc,
     Fd_roc) = simulate_vertical(mass_rocket,
                                 thrust_mag=thrust_rocket,
                                 v0_up=v0_rocket,
                                 t_final=t_final_rocket,
                                 dt=dt,
                                 aero=aero_rocket)

    # ------------------------------------------------------------------
    # Plot results
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Altitude vs time
    ax = axes[0, 0]
    ax.plot(t_proj, h_proj_km, label="Projectile (no thrust)")
    ax.plot(t_roc, h_roc_km, label="Rocket (with thrust)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Altitude [km]")
    ax.set_title("Altitude vs time")
    ax.grid(True)
    ax.legend()

    # Speed vs time
    ax = axes[0, 1]
    ax.plot(t_proj, v_proj, label="Projectile")
    ax.plot(t_roc, v_roc, label="Rocket")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [m/s]")
    ax.set_title("Speed vs time")
    ax.grid(True)
    ax.legend()

    # Drag magnitude vs time
    ax = axes[1, 0]
    ax.plot(t_proj, Fd_proj, label="Projectile")
    ax.plot(t_roc, Fd_roc, label="Rocket")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Drag force |F_drag| [N]")
    ax.set_title("Drag vs time")
    ax.grid(True)
    ax.legend()

    # Drag vs altitude (log scale)
    ax = axes[1, 1]
    ax.semilogy(h_proj_km, Fd_proj, label="Projectile")
    ax.semilogy(h_roc_km, Fd_roc, label="Rocket")
    ax.set_xlabel("Altitude [km]")
    ax.set_ylabel("Drag force |F_drag| [N]")
    ax.set_title("Drag vs altitude")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()