"""
Earth model structure: central gravity and co-rotation velocity interfaces.
Implementations are left to be filled in later.
"""

from __future__ import annotations

from dataclasses import dataclass

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


# Nominal constants for convenience (can be used by a real implementation)
MU_EARTH = 3.986_004_418e14  # m^3/s^2
R_EARTH = 6_371_000.0  # m
OMEGA_EARTH = np.array([0.0, 0.0, 7.292_115_9e-5])  # rad/s
J2_EARTH = 1.082_626_68e-3


if __name__ == "__main__":
    """Simple test: sample gravity and atmosphere co-rotation and plot profiles.

    This block is only executed when running gravity.py directly. It constructs
    an EarthModel using the nominal Earth constants, evaluates the magnitude of
    gravitational acceleration and atmosphere velocity over a range of
    altitudes, prints a few sample values, and generates basic plots.
    """
    import matplotlib.pyplot as plt

    # Create a nominal Earth model.
    earth = EarthModel(mu=MU_EARTH, radius=R_EARTH, omega_vec=OMEGA_EARTH)

    # Radial grid from the surface to 5 Earth radii (in meters).
    r_vals = np.linspace(R_EARTH, 5.0 * R_EARTH, 200)
    altitudes_m = r_vals - R_EARTH
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
        r = R_EARTH + h_km * 1000.0
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
