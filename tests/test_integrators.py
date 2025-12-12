import numpy as np

from Main.integrators import RK4, VelocityVerlet
from Main.state import State


def harmonic_deriv(t, state):
    k_spring = 1.0
    x = state.r_eci[0]
    vx = state.v_eci[0]
    ax = -(k_spring / state.m) * x
    dr_dt = np.array([vx, 0.0, 0.0])
    dv_dt = np.array([ax, 0.0, 0.0])
    dm_dt = 0.0
    return dr_dt, dv_dt, dm_dt


def total_energy(state):
    k = 0.5 * state.m * np.dot(state.v_eci, state.v_eci)
    u = 0.5 * np.dot(state.r_eci, state.r_eci)
    return k + u


def run_integrator(integrator, steps=1000, dt=0.01):
    state = State(r_eci=np.array([1.0, 0.0, 0.0]), v_eci=np.array([0.0, 1.0, 0.0]), m=1.0)
    t = 0.0
    energies = []
    for _ in range(steps):
        energies.append(total_energy(state))
        state = integrator.step(harmonic_deriv, state, t, dt)
        t += dt
    energies.append(total_energy(state))
    return energies


def test_rk4_energy_conservation():
    energies = run_integrator(RK4(), steps=500, dt=0.005)
    drift = abs(energies[-1] - energies[0])
    assert drift < 1e-3


def test_velocity_verlet_energy_conservation():
    energies = run_integrator(VelocityVerlet(), steps=500, dt=0.0005)
    drift = abs(energies[-1] - energies[0])
    assert drift < 0.2
