import numpy as np
import types
import pytest

from simulation import Simulation, Guidance
from integrators import State, RK4


class DummyProps:
    def __init__(self):
        self.p = 0.0
        self.rho = 0.0
        self.T = 300.0


class DummyEarth:
    def __init__(self, mu=1e5, radius=1000.0):
        self.mu = mu
        self.radius = radius

    def gravity_accel(self, r):
        r = np.asarray(r, dtype=float)
        r_norm = np.linalg.norm(r)
        if r_norm == 0:
            return np.zeros(3)
        return -self.mu * r / (r_norm ** 3)

    def atmosphere_velocity(self, r):
        return np.zeros(3)


class DummyAtmosphere:
    def properties(self, alt, t_env):
        return DummyProps()


class DummyAero:
    def drag_force(self, state, earth, t_env, rocket):
        return np.zeros(3)


class DummyRocket:
    def __init__(self):
        self.stages = [types.SimpleNamespace(prop_mass=1.0)]
        self.stage_prop_remaining = [1.0]
        self.stage_fuel_empty_time = [None]
        self.stage_engine_off_complete_time = [None]
        self.separation_time_planned = None
        self.upper_ignition_start_time = None
        self._last_time = 0.0
        self.meco_time = None
        self.upper_ignition_delay = 0.0

    def thrust_and_mass_flow(self, control, state, p_amb):
        if self.separation_time_planned is None:
            self.separation_time_planned = 0.0
        return np.zeros(3), 0.0

    def reference_area(self, state):
        return 1.0

    def current_stage_index(self, state):
        return 0


def test_simulation_orbit_exit():
    earth = DummyEarth(mu=1e5, radius=1000.0)
    atmosphere = DummyAtmosphere()
    aero = DummyAero()
    rocket = DummyRocket()
    integrator = RK4()
    guidance = Guidance()
    sim = Simulation(earth, atmosphere, aero, rocket, integrator=integrator, guidance=guidance)

    r0 = np.array([earth.radius, 0.0, 0.0])
    v_circ = np.sqrt(earth.mu / earth.radius)
    state0 = State(r_eci=r0, v_eci=np.array([0.0, v_circ, 0.0]), m=1.0, stage_index=0)

    log = sim.run(
        t_env_start=0.0,
        duration=100.0,
        dt=1.0,
        state0=state0,
        orbit_target_radius=earth.radius,
        exit_on_orbit=True,
    )

    assert log.orbit_achieved is True
    assert log.cutoff_reason == "orbit_target_met"
    assert len(log.t_sim) < 5  # terminated early


def test_simulation_impact_terminates():
    earth = DummyEarth(mu=1e5, radius=1000.0)
    atmosphere = DummyAtmosphere()
    aero = DummyAero()
    rocket = DummyRocket()
    guidance = Guidance()
    sim = Simulation(earth, atmosphere, aero, rocket, integrator=RK4(), guidance=guidance)

    r0 = np.array([earth.radius - 200.0, 0.0, 0.0])  # below surface by 200 m
    state0 = State(r_eci=r0, v_eci=np.zeros(3), m=1.0, stage_index=0)

    log = sim.run(
        t_env_start=0.0,
        duration=10.0,
        dt=1.0,
        state0=state0,
        orbit_target_radius=None,
        exit_on_orbit=True,
    )

    assert log.orbit_achieved is False
    assert log.cutoff_reason == "impact"
    assert len(log.t_sim) == 1  # stopped immediately


def test_simulation_stage_separation_and_mass_drop():
    earth = DummyEarth(mu=1e5, radius=1000.0)
    atmosphere = DummyAtmosphere()
    aero = DummyAero()
    # Two-stage dummy rocket with known masses and an immediate separation time.
    rocket = DummyRocket()
    rocket.stages = [
        types.SimpleNamespace(dry_mass=10.0, prop_mass=5.0),
        types.SimpleNamespace(dry_mass=2.0, prop_mass=1.0),
    ]
    rocket.stage_prop_remaining = [5.0, 1.0]
    rocket.separation_time_planned = 0.0
    rocket.upper_ignition_delay = 0.0
    guidance = Guidance()
    sim = Simulation(earth, atmosphere, aero, rocket, integrator=RK4(), guidance=guidance)

    r0 = np.array([earth.radius + 10.0, 0.0, 0.0])
    state0 = State(r_eci=r0, v_eci=np.zeros(3), m=100.0, stage_index=0)

    log = sim.run(
        t_env_start=0.0,
        duration=2.0,
        dt=1.0,
        state0=state0,
        orbit_target_radius=None,
        exit_on_orbit=False,
    )

    # Stage should advance to 1 at or before the second sample, and mass should drop by booster dry+prop.
    assert max(log.stage) == 1
    assert min(log.m) <= 85.0 + 1e-6  # 100 - (10+5)
