"""
Integrator interfaces and State container for translational dynamics.
Implementations are placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


@dataclass
class State:
    r_eci: np.ndarray
    v_eci: np.ndarray
    m: float
    stage_index: int = 0

    def copy(self) -> "State":
        """
        Return a deep-ish copy of the state suitable for use in integrators.

        r_eci and v_eci are copied as new numpy arrays; m and stage_index are
        simple scalars.
        """
        return State(
            r_eci=self.r_eci.copy(),
            v_eci=self.v_eci.copy(),
            m=self.m,
            stage_index=self.stage_index,
        )


class Integrator:
    def step(self, deriv_fn: Callable[[float, State], Tuple[np.ndarray, np.ndarray, float]], state: State, t: float, dt: float) -> State:
        raise NotImplementedError("Implement integrator step")


class RK4(Integrator):
    def step(
        self,
        deriv_fn: Callable[[float, State], Tuple[np.ndarray, np.ndarray, float]],
        state: State,
        t: float,
        dt: float,
    ) -> State:
        """
        Classic 4th-order Runge–Kutta integrator for the translational
        state (r, v, m).

        deriv_fn(t, state) must return (dr_dt, dv_dt, dm_dt), where:
            dr_dt: np.ndarray (3,)   = velocity
            dv_dt: np.ndarray (3,)   = acceleration
            dm_dt: float             = mass rate [kg/s]
        """
        # k1 at (t, state)
        k1_r, k1_v, k1_m = deriv_fn(t, state)

        # k2 at (t + dt/2, state + dt/2 * k1)
        s2 = state.copy()
        s2.r_eci = state.r_eci + 0.5 * dt * k1_r
        s2.v_eci = state.v_eci + 0.5 * dt * k1_v
        s2.m = state.m + 0.5 * dt * k1_m
        k2_r, k2_v, k2_m = deriv_fn(t + 0.5 * dt, s2)

        # k3 at (t + dt/2, state + dt/2 * k2)
        s3 = state.copy()
        s3.r_eci = state.r_eci + 0.5 * dt * k2_r
        s3.v_eci = state.v_eci + 0.5 * dt * k2_v
        s3.m = state.m + 0.5 * dt * k2_m
        k3_r, k3_v, k3_m = deriv_fn(t + 0.5 * dt, s3)

        # k4 at (t + dt, state + dt * k3)
        s4 = state.copy()
        s4.r_eci = state.r_eci + dt * k3_r
        s4.v_eci = state.v_eci + dt * k3_v
        s4.m = state.m + dt * k3_m
        k4_r, k4_v, k4_m = deriv_fn(t + dt, s4)

        # Combine increments
        r_next = state.r_eci + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
        v_next = state.v_eci + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        m_next = state.m + (dt / 6.0) * (k1_m + 2.0 * k2_m + 2.0 * k3_m + k4_m)

        return State(r_eci=r_next, v_eci=v_next, m=m_next, stage_index=state.stage_index)


class VelocityVerlet(Integrator):
    def step(
        self,
        deriv_fn: Callable[[float, State], Tuple[np.ndarray, np.ndarray, float]],
        state: State,
        t: float,
        dt: float,
    ) -> State:
        """
        Velocity Verlet integrator for (r, v, m).

        This implementation:
          * Evaluates acceleration and mass rate at the beginning of the
            step: a_n, m_dot_n.
          * Predicts position r_{n+1} using a_n.
          * Predicts an intermediate velocity v* = v_n + a_n * dt.
          * Uses the predicted state at t + dt to compute a_{n+1}.
          * Corrects the velocity with the average acceleration:
                v_{n+1} = v_n + 0.5 * (a_n + a_{n+1}) * dt
          * Updates mass with a simple explicit step:
                m_{n+1} = m_n + m_dot_n * dt

        This assumes dm/dt varies slowly over a single time step, which is
        reasonable for standard rocket engines with small dt.
        """
        # Evaluate derivatives at the beginning of the step
        dr_dt_n, dv_dt_n, dm_dt_n = deriv_fn(t, state)
        a_n = dv_dt_n
        m_dot_n = dm_dt_n

        # Current values
        r_n = state.r_eci
        v_n = state.v_eci
        m_n = state.m

        # First update: position using a_n
        r_next = r_n + v_n * dt + 0.5 * a_n * dt * dt

        # Provisional mass update (explicit Euler for mass)
        m_next = m_n + m_dot_n * dt

        # Provisional velocity for computing a_{n+1}
        v_star = v_n + a_n * dt

        # Build a provisional state at t + dt
        s_prov = state.copy()
        s_prov.r_eci = r_next
        s_prov.v_eci = v_star
        s_prov.m = m_next

        # Acceleration at t + dt
        _, dv_dt_np1, _ = deriv_fn(t + dt, s_prov)
        a_np1 = dv_dt_np1

        # Correct velocity with average acceleration
        v_next = v_n + 0.5 * (a_n + a_np1) * dt

        return State(r_eci=r_next, v_eci=v_next, m=m_next, stage_index=state.stage_index)


if __name__ == "__main__":
    """
    Basic tests for RK4 and VelocityVerlet integrators.

    We integrate a 1D harmonic oscillator encoded into the State:
      - r_eci = [x, 0, 0]
      - v_eci = [vx, 0, 0]
      - m     = mass of the oscillator (constant)

    and compare:
      * Energy drift over many periods for a fixed dt.
      * Final energy error as a function of dt.

    This is useful for checking that the integrators behave as expected
    before wiring them into the full rocket simulation.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Harmonic oscillator parameters
    m_body = 1.0   # kg
    k_spring = 1.0 # N/m
    omega = np.sqrt(k_spring / m_body)
    period = 2.0 * np.pi / omega

    def harmonic_deriv(t: float, state: State):
        """
        Derivative for a 1D harmonic oscillator.

        x'' = -(k/m) x
        Encoded in 3D as motion along the x-axis.
        """
        x = state.r_eci[0]
        vx = state.v_eci[0]
        m = state.m

        # Acceleration in x
        ax = -(k_spring / m) * x

        dr_dt = np.array([vx, 0.0, 0.0])
        dv_dt = np.array([ax, 0.0, 0.0])
        dm_dt = 0.0  # mass constant

        return dr_dt, dv_dt, dm_dt

    def energy(state: State) -> float:
        """Total mechanical energy of the oscillator."""
        x = state.r_eci[0]
        vx = state.v_eci[0]
        m = state.m
        return 0.5 * m * vx**2 + 0.5 * k_spring * x**2

    def run_integrator(integrator: Integrator, dt: float, total_time: float):
        """Integrate the harmonic oscillator and return time, x, v, and energy."""
        n_steps = int(total_time / dt)
        # Initial state: x(0) = 1, v(0) = 0
        state = State(
            r_eci=np.array([1.0, 0.0, 0.0], dtype=float),
            v_eci=np.array([0.0, 0.0, 0.0], dtype=float),
            m=m_body,
        )
        t = 0.0

        ts = np.zeros(n_steps + 1)
        xs = np.zeros(n_steps + 1)
        vs = np.zeros(n_steps + 1)
        Es = np.zeros(n_steps + 1)

        ts[0] = t
        xs[0] = state.r_eci[0]
        vs[0] = state.v_eci[0]
        Es[0] = energy(state)

        for i in range(1, n_steps + 1):
            state = integrator.step(harmonic_deriv, state, t, dt)
            t += dt

            ts[i] = t
            xs[i] = state.r_eci[0]
            vs[i] = state.v_eci[0]
            Es[i] = energy(state)

        return ts, xs, vs, Es

    # ------------------------------------------------------------------
    # Test 1: Energy drift over many periods for a fixed dt
    # ------------------------------------------------------------------
    # Use a moderately small dt and integrate for many periods.
    # You can increase n_periods further (e.g. 500 or 1000) to see
    # very long-term behaviour; 200 is a good default.
    n_periods = 200
    dt = 0.05
    total_time = n_periods * period

    rk4 = RK4()
    vv = VelocityVerlet()

    t_rk4, x_rk4, v_rk4, E_rk4 = run_integrator(rk4, dt, total_time)
    t_vv, x_vv, v_vv, E_vv = run_integrator(vv, dt, total_time)

    E0 = E_rk4[0]  # same initial energy for both
    dE_rk4 = (E_rk4 - E0) / E0
    dE_vv = (E_vv - E0) / E0

    # Numeric summary for Test 1 (energy drift)
    max_abs_dE_rk4 = float(np.max(np.abs(dE_rk4)))
    max_abs_dE_vv = float(np.max(np.abs(dE_vv)))
    mean_abs_dE_rk4 = float(np.mean(np.abs(dE_rk4)))
    mean_abs_dE_vv = float(np.mean(np.abs(dE_vv)))

    print("\n=== Test 1: Harmonic oscillator energy drift ===")
    print(f"  n_periods = {n_periods}, dt = {dt}")
    print(f"  RK4:           max|ΔE/E0| = {max_abs_dE_rk4:.3e}, mean|ΔE/E0| = {mean_abs_dE_rk4:.3e}")
    print(f"  VelocityVerlet: max|ΔE/E0| = {max_abs_dE_vv:.3e}, mean|ΔE/E0| = {mean_abs_dE_vv:.3e}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t_rk4, dE_rk4, label="RK4")
    axes[0].plot(t_vv, dE_vv, label="VelocityVerlet")
    axes[0].set_ylabel("Relative energy error ΔE/E0")
    axes[0].set_title(f"Harmonic oscillator energy drift (dt = {dt})")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].semilogy(t_rk4, np.abs(dE_rk4) + 1e-16, label="RK4")
    axes[1].semilogy(t_vv, np.abs(dE_vv) + 1e-16, label="VelocityVerlet")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("|ΔE/E0| (log scale)")
    axes[1].grid(True)
    axes[1].legend()

    # Optional: mark every 10 periods to visualise long-term behaviour
    for k in range(0, n_periods + 1, 10):
        t_mark = k * period
        axes[0].axvline(t_mark, color="k", alpha=0.05)
        axes[1].axvline(t_mark, color="k", alpha=0.05)

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Test 2: Final energy error as a function of dt (convergence)
    # ------------------------------------------------------------------
    dt_values = [0.01, 0.02, 0.05, 0.1, 0.2]
    final_err_rk4 = []
    final_err_vv = []

    total_time_conv = 10.0 * period  # integrate for 10 periods

    for dt_test in dt_values:
        t_rk4_c, _, _, E_rk4_c = run_integrator(rk4, dt_test, total_time_conv)
        t_vv_c, _, _, E_vv_c = run_integrator(vv, dt_test, total_time_conv)

        # Exact energy at t=0
        E_exact = 0.5 * m_body * 0.0**2 + 0.5 * k_spring * 1.0**2  # 0.5
        final_err_rk4.append(abs(E_rk4_c[-1] - E_exact) / E_exact)
        final_err_vv.append(abs(E_vv_c[-1] - E_exact) / E_exact)

    print("\n=== Test 2: Final energy error vs dt (10 periods) ===")
    print("  dt        RK4_final_rel_error   VV_final_rel_error")
    for dt_val, err_rk4, err_vv in zip(dt_values, final_err_rk4, final_err_vv):
        print(f"  {dt_val:6.3f}   {err_rk4: .3e}          {err_vv: .3e}")

    plt.figure(figsize=(7, 5))
    plt.loglog(dt_values, final_err_rk4, "o-", label="RK4 final |ΔE|/E0")
    plt.loglog(dt_values, final_err_vv, "s-", label="VelocityVerlet final |ΔE|/E0")
    plt.xlabel("Time step dt")
    plt.ylabel("Final relative energy error")
    plt.title("Convergence of energy error with dt (10 periods)")
    plt.grid(True, which="both")
    plt.legend()


    # ------------------------------------------------------------------
    # Test 3: Time-reversibility (forward + backward integration)
    # ------------------------------------------------------------------
    def forward_backward_error(integrator: Integrator, dt_fb: float, total_time_fb: float) -> Tuple[float, float]:
        """
        Integrate forward for total_time_fb, then backwards with -dt_fb
        for the same duration, and return the norm of the difference
        between the final and initial state (position and velocity).
        """
        n_steps = int(total_time_fb / dt_fb)

        # Initial state: x(0) = 1, v(0) = 0
        state0 = State(
            r_eci=np.array([1.0, 0.0, 0.0], dtype=float),
            v_eci=np.array([0.0, 0.0, 0.0], dtype=float),
            m=m_body,
        )

        # Forward integration
        state_fwd = state0.copy()
        t_fb = 0.0
        for _ in range(n_steps):
            state_fwd = integrator.step(harmonic_deriv, state_fwd, t_fb, dt_fb)
            t_fb += dt_fb

        # Backward integration
        state_bwd = state_fwd.copy()
        for _ in range(n_steps):
            state_bwd = integrator.step(harmonic_deriv, state_bwd, t_fb, -dt_fb)
            t_fb -= dt_fb

        # Errors
        dr = state_bwd.r_eci - state0.r_eci
        dv = state_bwd.v_eci - state0.v_eci
        return np.linalg.norm(dr), np.linalg.norm(dv)

    dt_fb = 0.05
    total_time_fb = 200.0 * period  # many periods

    dr_rk4, dv_rk4 = forward_backward_error(rk4, dt_fb, total_time_fb)
    dr_vv, dv_vv = forward_backward_error(vv, dt_fb, total_time_fb)

    print("\n=== Test 3: Time-reversibility (harmonic oscillator) ===")
    print(f"  dt_fb = {dt_fb}, total_time_fb = {total_time_fb}")
    print(f"  RK4:           ||Δr|| = {dr_rk4:.3e}, ||Δv|| = {dv_rk4:.3e}")
    print(f"  VelocityVerlet:||Δr|| = {dr_vv:.3e}, ||Δv|| = {dv_vv:.3e}")


    # ------------------------------------------------------------------
    # Test 4: 2D Kepler orbit (central gravity) – energy and angular momentum
    # ------------------------------------------------------------------
    mu = 1.0  # gravitational parameter in dimensionless units

    def kepler_deriv(t: float, state: State):
        """
        Two-body problem with central gravity in 2D (x-y plane).
        r_eci = [x, y, 0], v_eci = [vx, vy, 0]
        """
        r_vec = state.r_eci
        v_vec = state.v_eci
        r_norm = np.linalg.norm(r_vec)
        # Avoid division by zero
        if r_norm == 0.0:
            a_vec = np.zeros(3)
        else:
            a_vec = -mu * r_vec / (r_norm**3)

        dr_dt = v_vec
        dv_dt = a_vec
        dm_dt = 0.0
        return dr_dt, dv_dt, dm_dt

    def kepler_energy_and_angmom(state: State):
        """Return energy and angular momentum magnitude for the Kepler problem."""
        r_vec = state.r_eci
        v_vec = state.v_eci
        r_norm = np.linalg.norm(r_vec)
        v_norm = np.linalg.norm(v_vec)
        E = 0.5 * v_norm**2 - mu / r_norm
        h_vec = np.cross(r_vec, v_vec)
        h_norm = np.linalg.norm(h_vec)
        return E, h_norm

    def run_kepler(integrator: Integrator, dt_k: float, n_orbits: int):
        """Integrate a circular orbit and track energy and angular momentum."""
        # Circular orbit: r0 = 1, v0 = sqrt(mu/r)
        r0 = np.array([1.0, 0.0, 0.0], dtype=float)
        v0 = np.array([0.0, np.sqrt(mu / 1.0), 0.0], dtype=float)
        state = State(r_eci=r0, v_eci=v0, m=1.0)

        # Orbital period for circular orbit
        T_orbit = 2.0 * np.pi * np.sqrt(1.0**3 / mu)
        total_time_k = n_orbits * T_orbit
        n_steps = int(total_time_k / dt_k)
        t_arr = np.zeros(n_steps + 1)
        E_arr = np.zeros(n_steps + 1)
        h_arr = np.zeros(n_steps + 1)

        E0, h0 = kepler_energy_and_angmom(state)
        t = 0.0
        t_arr[0] = t
        E_arr[0] = E0
        h_arr[0] = h0

        for i in range(1, n_steps + 1):
            state = integrator.step(kepler_deriv, state, t, dt_k)
            t += dt_k
            t_arr[i] = t
            E_arr[i], h_arr[i] = kepler_energy_and_angmom(state)

        return t_arr, E_arr, h_arr, E0, h0

    dt_k = 0.01
    n_orbits = 50

    t_k_rk4, E_k_rk4, h_k_rk4, E0_k, h0_k = run_kepler(rk4, dt_k, n_orbits)
    t_k_vv, E_k_vv, h_k_vv, _, _ = run_kepler(vv, dt_k, n_orbits)

    dE_k_rk4 = (E_k_rk4 - E0_k) / abs(E0_k)
    dE_k_vv = (E_k_vv - E0_k) / abs(E0_k)
    dh_k_rk4 = (h_k_rk4 - h0_k) / abs(h0_k)
    dh_k_vv = (h_k_vv - h0_k) / abs(h0_k)

    max_dE_k_rk4 = float(np.max(np.abs(dE_k_rk4)))
    max_dE_k_vv = float(np.max(np.abs(dE_k_vv)))
    max_dh_k_rk4 = float(np.max(np.abs(dh_k_rk4)))
    max_dh_k_vv = float(np.max(np.abs(dh_k_vv)))

    print("\n=== Test 4: 2D Kepler orbit (circular) ===")
    print(f"  dt_k = {dt_k}, n_orbits = {n_orbits}")
    print(f"  RK4:           max|ΔE/E0| = {max_dE_k_rk4:.3e}, max|Δh/h0| = {max_dh_k_rk4:.3e}")
    print(f"  VelocityVerlet: max|ΔE/E0| = {max_dE_k_vv:.3e}, max|Δh/h0| = {max_dh_k_vv:.3e}")

    fig_k, axes_k = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes_k[0].plot(t_k_rk4, dE_k_rk4, label="RK4")
    axes_k[0].plot(t_k_vv, dE_k_vv, label="VelocityVerlet")
    axes_k[0].set_ylabel("ΔE / |E0|")
    axes_k[0].set_title("Kepler orbit: energy drift")
    axes_k[0].grid(True)
    axes_k[0].legend()

    axes_k[1].plot(t_k_rk4, dh_k_rk4, label="RK4")
    axes_k[1].plot(t_k_vv, dh_k_vv, label="VelocityVerlet")
    axes_k[1].set_xlabel("Time [arb. units]")
    axes_k[1].set_ylabel("Δ|h| / |h0|")
    axes_k[1].set_title("Kepler orbit: angular momentum drift")
    axes_k[1].grid(True)
    axes_k[1].legend()

    plt.tight_layout()

    plt.show()
