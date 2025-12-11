"""
Atmosphere module structure: outlines interfaces for US76, NRLMSIS2.1, and a
combined dispatcher. Implementations are intentionally left as placeholders.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import ussa1976
import pymsis


@dataclass
class AtmosphereProperties:
    """Basic thermodynamic properties of the atmosphere at a given altitude.

    Attributes
    ----------
    rho : float
        Mass density [kg/m^3].
    p : float
        Static pressure [Pa].
    T : float
        Temperature [K].
    """
    rho: float
    p: float
    T: float


class AtmosphereModel:
    """Unified atmosphere model wrapping US76 and NRLMSIS 2.1.

    This class provides a single interface for querying atmospheric
    properties. Below a switch altitude, it uses a US Standard Atmosphere
    1976-like model; at higher altitudes, it uses an NRLMSIS 2.1-like
    thermosphere model.

    The detailed implementations of the sub-models are intentionally left as
    placeholders; only the interface and dispatch logic are defined here.
    """

    def __init__(self, h_switch: float = 86000,
                 f107: float | None = None,
                 f107a: float | None = None,
                 ap: float | None = None,
                 lat_deg: float = 0.0,
                 lon_deg: float = 0.0):
        """Create an atmosphere model.

        Parameters
        ----------
        h_switch : float
            Altitude [m] at which to switch from the US76-like model to the
            NRLMSIS 2.1-like model.
        f107 : float | None
            10.7 cm solar radio flux index (optional, for NRLMSIS).
        f107a : float | None
            81-day average of f10.7 (optional, for NRLMSIS).
        ap : float | None
            Geomagnetic index (optional, for NRLMSIS).
        lat_deg : float
            Reference latitude in degrees for the high-altitude model.
        lon_deg : float
            Reference longitude in degrees for the high-altitude model.
        """
        self.h_switch = h_switch
        self.f107 = f107
        self.f107a = f107a
        self.ap = ap

        # Fixed reference location for the high-altitude model (NRLMSIS-like)
        self.lat_deg = lat_deg
        self.lon_deg = lon_deg

    def properties(self, altitude: float, t: float | None = None) -> AtmosphereProperties:
        """Return atmospheric properties at a given altitude.

        Parameters
        ----------
        altitude : float
            Geometric altitude above Earth's mean radius [m].
        t : float | None
            Simulation time [s]; may be used by the high-altitude model.
        """
        # Guard against small negative altitudes from numerical noise
        altitude = max(0.0, altitude)
        if altitude < self.h_switch:
            return self._us76_properties(altitude)
        return self._nrlmsis_properties(altitude, t)

    def _us76_properties(self, altitude: float) -> AtmosphereProperties:
        """Return properties from a US76-like model.

        Parameters
        ----------
        altitude : float
            Geometric altitude above mean sea level [m].
        """
        # Clamp altitude to the valid range of the USSA1976 model (0–1000 km)
        alt_clamped = float(np.clip(altitude, 0.0, 1_000_000.0))

        # ussa1976 expects an array of altitudes in meters
        ds = ussa1976.compute(
            z=np.array([alt_clamped], dtype=float),
            variables=["t", "p", "rho"],
        )

        # Extract scalars from the Dataset
        T = float(ds["t"][0])   # [K]
        p = float(ds["p"][0])   # [Pa]
        rho = float(ds["rho"][0])  # [kg/m^3]

        return AtmosphereProperties(rho=rho, p=p, T=T)

    def _nrlmsis_properties(self, altitude: float, t: float | None = None) -> AtmosphereProperties:
        """Return properties from an NRLMSIS 2.1-like model.

        Parameters
        ----------
        altitude : float
            Geometric altitude above mean sea level [m].
        t : float | None
            Simulation time [s] since an arbitrary reference epoch. If None,
            a fixed reference date is used.
        """
        # Convert altitude from meters to kilometers, and clamp to a reasonable
        # MSIS range (0--1000 km).
        alt_km = float(np.clip(altitude / 1000.0, 0.0, 1000.0))

        # Map simulation time t [s] to a calendar date. For this project we
        # simply interpret t as seconds since an arbitrary reference epoch.
        # If t is None, use the reference epoch directly.
        base_epoch = np.datetime64("2000-01-01T00:00")
        if t is None:
            date = base_epoch
        else:
            # Cast to integer seconds to avoid issues with non-integer timesteps.
            dt = np.timedelta64(int(t), "s")
            date = base_epoch + dt

        lon = float(self.lon_deg)
        lat = float(self.lat_deg)

        # Use provided solar/geomagnetic inputs if available; otherwise, fall
        # back to simple default values to avoid network downloads.
        f107 = 150.0 if self.f107 is None else float(self.f107)
        f107a = 150.0 if self.f107a is None else float(self.f107a)
        ap = 4.0 if self.ap is None else float(self.ap)

        # aps is expected to be an array-like of length 7. For a simple course
        # project we approximate all 7 Ap-related values with the same daily
        # Ap index.
        aps = np.array([[ap] * 7], dtype=float)

        output = pymsis.calculate(
            date,
            np.array([lon], dtype=float),
            np.array([lat], dtype=float),
            np.array([alt_km], dtype=float),
            np.array([f107], dtype=float),
            np.array([f107a], dtype=float),
            aps,
        )

        # output has shape (ndates, nlons, nlats, nalts, 11); here that is
        # (1, 1, 1, 1, 11). Squeeze to 1D for convenience.
        output = np.squeeze(output)

        # Total mass density [kg/m^3] and temperature [K] from pymsis.
        rho = float(output[pymsis.Variable.MASS_DENSITY])
        T = float(output[pymsis.Variable.TEMPERATURE])

        # Compute pressure using the ideal gas law with an effective mean
        # molecular mass. For a simple upper-atmosphere model, we approximate
        # the mean molar mass as a constant 28.96 g/mol, which is sufficient
        # for a first-order drag estimate in this project.
        R_universal = 8.314462618  # J/(mol*K)
        M_mean = 0.0289644         # kg/mol, approximate dry-air molar mass
        p = rho * (R_universal / M_mean) * T

        return AtmosphereProperties(rho=rho, p=p, T=T)


if __name__ == "__main__":
    """Simple test: sample the atmosphere from 0 to 1000 km and plot profiles.

    This block is only executed when running atmosphere.py directly. It
    constructs an AtmosphereModel with default parameters, evaluates the
    properties on a grid of altitudes, prints a few sample values, and
    generates basic plots of density, pressure and temperature versus
    altitude.
    """
    import matplotlib.pyplot as plt

    # Create an atmosphere model with default settings.
    atm = AtmosphereModel(f107=150.0, f107a=150.0, ap=4.0)

    # Altitude grid: 0--100 km in 5 km steps, then 150--1000 km in 50 km steps.
    altitudes_km = np.concatenate([
        np.arange(0.0, 100.0 + 1e-6, 1.0),
        np.arange(150.0, 1000.0 + 1e-6, 50.0),
    ])
    altitudes_m = altitudes_km * 1000.0

    rhos = []
    ps = []
    Ts = []

    for h in altitudes_m:
        props = atm.properties(h, t=0.0)
        rhos.append(props.rho)
        ps.append(props.p)
        Ts.append(props.T)

    # Print sample points at all grid altitudes (5 km up to 100 km, then 50 km).
    for h_km, rho, p, T in zip(altitudes_km, rhos, ps, Ts):
        print(f"h = {h_km:.1f} km: rho = {rho:.3e} kg/m^3, p = {p:.3e} Pa, T = {T:.1f} K")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Density vs altitude
    axes[0].plot(altitudes_km, rhos)
    axes[0].set_xlabel("Altitude [km]")
    axes[0].set_ylabel("Density [kg/m^3]")
    axes[0].set_title("Density profile")
    axes[0].grid(True)

    # Pressure vs altitude
    axes[1].plot(altitudes_km, ps)
    axes[1].set_xlabel("Altitude [km]")
    axes[1].set_ylabel("Pressure [Pa]")
    axes[1].set_title("Pressure profile")
    axes[1].grid(True)

    # Temperature vs altitude
    axes[2].plot(altitudes_km, Ts)
    axes[2].set_xlabel("Altitude [km]")
    axes[2].set_ylabel("Temperature [K]")
    axes[2].set_title("Temperature profile")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    # Additional diagnostic plots: zoomed density profiles.
    h_switch_km = atm.h_switch / 1000.0

    # 1) Density vs altitude (0–120 km) on a log scale.
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.semilogy(altitudes_km, rhos)
    ax2.set_xlim(0.0, 120.0)
    ax2.set_xlabel("Altitude [km]")
    ax2.set_ylabel("Density [kg/m^3]")
    ax2.set_title("Density profile (0–120 km, log scale)")
    ax2.grid(True, which="both", linestyle=":")

    # Mark the switch altitude on this plot.
    ax2.axvline(h_switch_km, color="k", linestyle="--", linewidth=1)
    ax2.text(
        h_switch_km + 2.0,
        rhos[0],
        f"h_switch = {h_switch_km:.1f} km",
        fontsize=8,
        va="top",
    )

    plt.tight_layout()
    plt.show()

    # 2) Zoomed-in density near the switch region on a log scale.
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.semilogy(altitudes_km, rhos)
    # Show +/- 20 km around the switch altitude.
    ax3.set_xlim(max(0.0, h_switch_km - 20.0), min(120.0, h_switch_km + 20.0))
    ax3.set_xlabel("Altitude [km]")
    ax3.set_ylabel("Density [kg/m^3]")
    ax3.set_title("Density near model switch (log scale)")
    ax3.grid(True, which="both", linestyle=":")

    ax3.axvline(h_switch_km, color="k", linestyle="--", linewidth=1)
    ax3.text(
        h_switch_km + 1.0,
        rhos[altitudes_km.tolist().index(h_switch_km)] if h_switch_km in altitudes_km.tolist() else rhos[-1],
        f"h_switch = {h_switch_km:.1f} km",
        fontsize=8,
        va="bottom",
    )

    plt.tight_layout()
    plt.show()
