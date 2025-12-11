"""
Aerodynamics module structure: drag-only interface.
Implementations are placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union, Any, Optional

import numpy as np
from atmosphere import AtmosphereModel
from config import CFG


def get_wind_at_altitude(altitude: float, r_eci: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Returns a wind vector based on altitude to model the jet stream.
    The wind profile ramps up to a peak speed in the jet stream layer (8-13 km)
    and is zero outside this band. If a position vector is provided, the wind
    direction is projected into the local tangent plane so it remains “surface-
    following” rather than a fixed inertial vector.
    """
    alt_points = np.array(CFG.wind_alt_points)
    speed_points = np.array(CFG.wind_speed_points)

    wind_speed = np.interp(altitude, alt_points, speed_points)

    if r_eci is not None:
        r = np.asarray(r_eci, dtype=float)
        r_norm = np.linalg.norm(r)
        if r_norm > 0.0:
            up = r / r_norm
            east = np.cross([0.0, 0.0, 1.0], up)
            east_norm = np.linalg.norm(east)
            if east_norm < 1e-9:
                east = np.array([1.0, 0.0, 0.0], dtype=float)
                east_norm = 1.0
            east /= east_norm
            north = np.cross(up, east)
            north_norm = np.linalg.norm(north)
            if north_norm > 0.0:
                north /= north_norm
            else:
                north = np.array([0.0, 1.0, 0.0], dtype=float)

            dir_local = np.asarray(CFG.wind_direction_vec, dtype=float)
            dir_surface = dir_local[0] * east + dir_local[1] * north + dir_local[2] * up
            norm = np.linalg.norm(dir_surface)
            if norm > 0:
                direction = dir_surface / norm
            else:
                direction = east
        else:
            direction = np.zeros(3)
    else:
        direction = np.asarray(CFG.wind_direction_vec, dtype=float)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        else:
            direction = np.zeros(3)

    return direction * wind_speed


def mach_dependent_cd(mach: float) -> float:
    """
    A representative drag coefficient (Cd) curve for a generic launch vehicle,
    based on the Mach number. This captures the characteristic transonic drag
    rise and subsequent decrease in the supersonic regime.
    """
    mach_cd_map = np.array(CFG.mach_cd_map)
    mach_points = mach_cd_map[:, 0]
    cd_points = mach_cd_map[:, 1]
    return np.interp(mach, mach_points, cd_points)


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
    reference_area: Optional[float] = None  # fallback if rocket is not provided

    def drag_force(self, state: Any, earth: Any, t: float, rocket: Any = None) -> np.ndarray:
        """Compute aerodynamic drag force in the ECI frame.

        Assumptions
        -----------
        - The rocket's longitudinal axis is aligned with the air-relative
          velocity (zero angle of attack).
        - Drag acts purely opposite to the air-relative velocity vector.
        - Reference area is taken per stage from the rocket if provided;
          otherwise falls back to the constant reference_area on this class.
        - `state` provides `r_eci` and `v_eci` attributes (position and
          velocity in ECI, both as 3-vectors in meters / m/s).
        - `earth` provides `radius` and `atmosphere_velocity(r_eci)`.
        - `rocket` (optional) provides `reference_area(state)` for stage-aware drag.
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

        # Air-relative velocity: rocket velocity minus (co-rotating atmosphere + wind).
        v_atm_rotation = np.asarray(earth.atmosphere_velocity(r), dtype=float)
        
        if CFG.use_jet_stream_model:
            wind_vector = get_wind_at_altitude(altitude, r)
        else:
            wind_vector = np.array([0.0, 0.0, 0.0])

        v_air = v_atm_rotation + wind_vector
        v_rel = v - v_air
        v_rel_mag = np.linalg.norm(v_rel)
        if v_rel_mag == 0.0:
            return np.zeros(3)

        # Speed of sound (ideal gas, dry air) and Mach number.
        gamma = CFG.air_gamma
        R_air = CFG.air_gas_constant
        a = np.sqrt(max(gamma * R_air * T, 0.0))
        mach = v_rel_mag / a if a > 0.0 else 0.0

        # Drag coefficient from model.
        cd = self.cd_model.cd(mach)

        # Reference area: prefer stage-specific area from the rocket, else fallback.
        if rocket is not None and hasattr(rocket, "reference_area"):
            A = float(rocket.reference_area(state))
        elif self.reference_area is not None:
            A = float(self.reference_area)
        else:
            A = 0.0

        if A <= 0.0 or cd <= 0.0:
            return np.zeros(3)

        # Drag magnitude: 0.5 * rho * |v_rel|^2 * Cd * A.
        q = 0.5 * rho * v_rel_mag ** 2
        F_mag = q * cd * A

        # Direction opposite to air-relative velocity.
        F_vec = -F_mag * v_rel / v_rel_mag
        return F_vec


if __name__ == "__main__":
    """
    Simple test plots for drag forces and Cd curve.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from gravity import EarthModel, MU_EARTH, R_EARTH, OMEGA_EARTH

    # --- Plot 1: Representative Cd vs Mach curve ---
    mach_range = np.linspace(0, 10.0, 200)
    cd_values = [mach_dependent_cd(m) for m in mach_range]
    
    fig_cd, ax_cd = plt.subplots(figsize=(8, 5))
    ax_cd.plot(mach_range, cd_values)
    ax_cd.set_title("Representative Drag Coefficient (Cd) vs. Mach Number")
    ax_cd.set_xlabel("Mach Number")
    ax_cd.set_ylabel("Drag Coefficient (Cd)")
    ax_cd.grid(True)
    
    # Highlight key regions
    ax_cd.axvspan(0.8, 1.2, color='red', alpha=0.1, label='Transonic Region')
    ax_cd.legend()
    
    plt.tight_layout()
    plt.savefig("cd_curve.png")
    plt.close(fig_cd)
    print("Saved Cd vs. Mach curve plot to cd_curve.png")

    # --- Plot 2: Drag force simulation plots ---
    earth = EarthModel(mu=MU_EARTH, radius=R_EARTH, omega_vec=OMEGA_EARTH)
    atm = AtmosphereModel()

    # Simple Cd model for the old test cases
    cd_model_const = CdModel(2.0)

    # Projectile: smaller, more streamlined body.
    diameter_proj = 0.5  # m
    radius_proj = diameter_proj / 2.0
    A_proj = float(np.pi * radius_proj**2)

    # Rocket: larger body.
    diameter_roc = 3.0  # m
    radius_roc = diameter_roc / 2.0
    A_roc = float(np.pi * radius_roc**2)

    aero_projectile = Aerodynamics(atmosphere=atm, cd_model=cd_model_const, reference_area=A_proj)
    aero_rocket = Aerodynamics(atmosphere=atm, cd_model=cd_model_const, reference_area=A_roc)

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
        # ... (rest of the function is unchanged)
        # Initial position at surface, along +x.
        r = np.array([R_EARTH, 0.0, 0.0], dtype=float)
        r_norm = np.linalg.norm(r)
        r_hat = r / r_norm
        v_atm0 = earth.atmosphere_velocity(r)
        v = v_atm0 + v0_up * r_hat
        state = State(r_eci=r.copy(), v_eci=v.copy())
        times, altitudes_km, speeds, drag_mags = [], [], [], []
        t = 0.0
        while t <= t_final:
            r, v = state.r_eci, state.v_eci
            r_norm = np.linalg.norm(r)
            if r_norm == 0.0: break
            altitude = r_norm - earth.radius
            if altitude < 0.0 and t > 0.0: break
            r_hat = r / r_norm
            a_g = earth.gravity_accel(r)
            F_g = mass * a_g
            F_thrust = thrust_mag * r_hat if altitude >= 0.0 else np.zeros(3)
            F_drag = aero.drag_force(state, earth, t)
            F_net = F_g + F_thrust + F_drag
            a_net = F_net / mass
            times.append(t)
            altitudes_km.append(max(altitude, 0.0) / 1000.0)
            speeds.append(np.linalg.norm(v))
            drag_mags.append(np.linalg.norm(F_drag))
            state.v_eci = v + a_net * dt
            state.r_eci = r + state.v_eci * dt
            t += dt
        return (np.array(times), np.array(altitudes_km), np.array(speeds), np.array(drag_mags))

    dt = 0.1
    mass_projectile, v0_projectile, t_final_projectile = 500.0, 600.0, 200.0
    t_proj, h_proj_km, v_proj, Fd_proj = simulate_vertical(mass_projectile, 0.0, v0_projectile, t_final_projectile, dt, aero_projectile)
    mass_rocket, thrust_rocket, v0_rocket, t_final_rocket = 1.0e5, 2.0e6, 0.0, 150.0
    t_roc, h_roc_km, v_roc, Fd_roc = simulate_vertical(mass_rocket, thrust_rocket, v0_rocket, t_final_rocket, dt, aero_rocket)
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(t_proj, h_proj_km, label="Projectile (no thrust)")
    axes[0, 0].plot(t_roc, h_roc_km, label="Rocket (with thrust)")
    axes[0, 0].set_xlabel("Time [s]"); axes[0, 0].set_ylabel("Altitude [km]"); axes[0, 0].set_title("Altitude vs time"); axes[0, 0].grid(True); axes[0, 0].legend()
    axes[0, 1].plot(t_proj, v_proj, label="Projectile"); axes[0, 1].plot(t_roc, v_roc, label="Rocket")
    axes[0, 1].set_xlabel("Time [s]"); axes[0, 1].set_ylabel("Speed [m/s]"); axes[0, 1].set_title("Speed vs time"); axes[0, 1].grid(True); axes[0, 1].legend()
    axes[1, 0].plot(t_proj, Fd_proj, label="Projectile"); axes[1, 0].plot(t_roc, Fd_roc, label="Rocket")
    axes[1, 0].set_xlabel("Time [s]"); axes[1, 0].set_ylabel("Drag force |F_drag| [N]"); axes[1, 0].set_title("Drag vs time"); axes[1, 0].grid(True); axes[1, 0].legend()
    axes[1, 1].semilogy(h_proj_km, Fd_proj, label="Projectile"); axes[1, 1].semilogy(h_roc_km, Fd_roc, label="Rocket")
    axes[1, 1].set_xlabel("Altitude [km]"); axes[1, 1].set_ylabel("Drag force |F_drag| [N]"); axes[1, 1].set_title("Drag vs altitude"); axes[1, 1].grid(True); axes[1, 1].legend()
    plt.tight_layout()
    plt.savefig("drag_plots.png")
    plt.close(fig)
    print("Saved drag simulation plots to drag_plots.png")
