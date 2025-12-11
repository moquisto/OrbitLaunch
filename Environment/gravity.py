"""
Earth model structure: central gravity and co-rotation velocity interfaces.
Implementations are left to be filled in later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


@dataclass
class EarthModel:
    mu: float
    radius: float
    omega_vec: np.ndarray
    j2: float | None = None

    def gravity_accel(self, r_eci: np.ndarray) -> np.ndarray:
        """Return the central gravitational acceleration at a given ECI position.

        Parameters
        ----------
        r_eci : np.ndarray
            Position vector in the Earth-centered inertial (ECI) frame [m].

        Returns
        -------
        np.ndarray
            Gravitational acceleration vector in ECI [m/s^2].
        """
        r = np.asarray(r_eci, dtype=float)
        r_norm = np.linalg.norm(r)
        if r_norm == 0.0:
            raise ValueError("gravity_accel is undefined at r = 0")
        # Central (spherical) gravity: a = -mu * r / |r|^3
        a_central = -self.mu * r / r_norm**3

        if self.j2 is None or self.j2 == 0.0:
            return a_central

        # J2 oblateness perturbation (assuming z aligns with Earth's spin axis)
        mu = self.mu
        J2 = self.j2
        R = self.radius
        x, y, z = r
        r2 = r_norm * r_norm
        z2 = z * z
        factor = 1.5 * J2 * mu * (R**2) / (r_norm**5)
        k = 5.0 * z2 / r2
        a_j2_x = factor * x * (1.0 - k)
        a_j2_y = factor * y * (1.0 - k)
        a_j2_z = factor * z * (3.0 - k)
        a_j2 = np.array([a_j2_x, a_j2_y, a_j2_z], dtype=float)

        return a_central + a_j2

    def atmosphere_velocity(self, r_eci: np.ndarray) -> np.ndarray:
        """Return the local velocity of a rigidly co-rotating atmosphere.

        Parameters
        ----------
        r_eci : np.ndarray
            Position vector in the Earth-centered inertial (ECI) frame [m].

        Returns
        -------
        np.ndarray
            Atmosphere velocity at that point in the ECI frame [m/s].
        """
        r = np.asarray(r_eci, dtype=float)
        omega = np.asarray(self.omega_vec, dtype=float)
        # For a rigidly rotating planet, the atmosphere velocity in ECI is
        # given by v_atm = omega x r.
        return np.cross(omega, r)


def orbital_elements_from_state(r_vec: np.ndarray, v_vec: np.ndarray, mu: float) -> Tuple[float | None, float | None, float | None]:
    """
    Calculates orbital elements (semi-major axis, perigee radius, apoapsis radius)
    from position and velocity vectors.

    Parameters
    ----------
    r_vec : np.ndarray
        Position vector (m).
    v_vec : np.ndarray
        Velocity vector (m/s).
    mu : float
        Gravitational parameter of the central body (m^3/s^2).

    Returns
    -------
    Tuple[float | None, float | None, float | None]
        (semi_major_axis, perigee_radius, apoapsis_radius) in meters.
        Returns None for elements if the orbit is hyperbolic or parabolic,
        or if calculations yield non-physical results.
    """
    r_norm = np.linalg.norm(r_vec)
    v_norm = np.linalg.norm(v_vec)

    # Specific orbital energy
    epsilon = (v_norm**2 / 2.0) - (mu / r_norm)

    # Semi-major axis
    if epsilon == 0:  # Parabolic orbit
        a = np.inf
    elif epsilon > 0:  # Hyperbolic orbit
        a = -mu / (2 * epsilon)
    else:  # Elliptical orbit
        a = -mu / (2 * epsilon)

    # Specific angular momentum vector
    h_vec = np.cross(r_vec, v_vec)
    h_norm = np.linalg.norm(h_vec)

    if h_norm == 0:  # Radial trajectory
        return a, None, None

    # Eccentricity vector
    e_vec = (np.cross(v_vec, h_vec) / mu) - (r_vec / r_norm)
    e = np.linalg.norm(e_vec)

    rp, ra = None, None
    if e < 1:  # Elliptical orbit
        rp = a * (1 - e)
        ra = a * (1 + e)
    elif e == 1:  # Parabolic orbit
        rp = h_norm**2 / mu / 2.0
        ra = np.inf
    else:  # Hyperbolic orbit
        # For hyperbolic, apocenter is at infinity. Pericenter distance is meaningful.
        rp = h_norm**2 / (mu * (1 + e))
        ra = np.inf # Effectively infinite for hyperbolic

    # Handle cases where calculation might result in non-physical or undefined elements
    if np.isinf(a) or np.isnan(a) or (rp is not None and rp < 0) or (ra is not None and ra < 0):
        return None, None, None # Indicate non-orbital or invalid state

    return a, rp, ra





if __name__ == "__main__":
    """Simple test: sample gravity and atmosphere co-rotation and plot profiles.

    This block is only executed when running gravity.py directly. It constructs
    an EarthModel using the nominal Earth constants, evaluates the magnitude of
    gravitational acceleration and atmosphere velocity over a range of
    altitudes, prints a few sample values, and generates basic plots.
    """
    import matplotlib.pyplot as plt
    from .config import EnvironmentConfig

    config = EnvironmentConfig()

    # Create a nominal Earth model.
    earth = EarthModel(mu=config.earth_mu, radius=config.earth_radius_m, omega_vec=np.array(config.earth_omega_vec))

    # Radial grid from the surface to 5 Earth radii (in meters).
    r_vals = np.linspace(config.earth_radius_m, 5.0 * config.earth_radius_m, 200)
    altitudes_m = r_vals - config.earth_radius_m
    altitudes_km = altitudes_m / 1000.0

    g_magnitudes = []
    v_atm_magnitudes = []

    for r in r_vals:
        r_vec = np.array([r, 0.0, 0.0])

        # Gravitational acceleration magnitude.
        g_vec = earth.gravity_accel(r_vec)
        g_magnitudes.append(np.linalg.norm(g_vec))

        # Atmosphere co-rotation velocity magnitude (assuming equator).
        v_atm_vec = earth.atmosphere_velocity(r_vec)
        v_atm_magnitudes.append(np.linalg.norm(v_atm_vec))

    # Print a few sample points near the surface and at low Earth orbit altitudes.
    sample_altitudes_km = [0.0, 200.0, 400.0]
    print("Sample gravity and atmosphere velocity values:")
    for h_km in sample_altitudes_km:
        r = config.earth_radius_m + h_km * 1000.0
        r_vec = np.array([r, 0.0, 0.0])
        g = np.linalg.norm(earth.gravity_accel(r_vec))
        v_atm = np.linalg.norm(earth.atmosphere_velocity(r_vec))
        print(
            f"h = {h_km:6.1f} km: |g| = {g:6.3f} m/s^2, "
            f"|v_atm| (equator) = {v_atm:7.1f} m/s"
        )

    # Plot gravity and atmosphere velocity versus altitude.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Gravity profile.
    axes[0].plot(altitudes_km, g_magnitudes)
    axes[0].set_xlabel("Altitude [km]")
    axes[0].set_ylabel("Gravity magnitude [m/s^2]")
    axes[0].set_title("Central gravity vs altitude")
    axes[0].grid(True)

    # Atmosphere velocity profile at equator.
    axes[1].plot(altitudes_km, v_atm_magnitudes)
    axes[1].set_xlabel("Altitude [km]")
    axes[1].set_ylabel("Atmosphere speed [m/s]")
    axes[1].set_title("Co-rotating atmosphere speed vs altitude (equator)")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
