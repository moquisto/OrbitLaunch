import numpy as np
import pytest

from simulation import Simulation, ControlCommand
from config import CFG


class DummyProps:
    def __init__(self, rho=1.0, T=300.0, p=0.0):
        self.rho = rho
        self.T = T
        self.p = p


class DummyAtmosphere:
    def __init__(self, props):
        self.props = props

    def properties(self, alt, t_env):
        return self.props


class DummyEarth:
    def __init__(self, radius=1000.0, mu=1.0):
        self.radius = radius
        self.mu = mu

    def gravity_accel(self, r):
        return np.zeros(3)

    def atmosphere_velocity(self, r):
        return np.zeros(3)


class DummyAero:
    def drag_force(self, state, earth, t_env, rocket):
        return np.zeros(3)


class DummyRocket:
    def __init__(self):
        self.last_throttle = None

    def thrust_and_mass_flow(self, control, state, p_amb):
        self.last_throttle = control["throttle"]
        # Force points along thrust direction; magnitude equals throttle for easy checking
        thrust_vec = control["throttle"] * np.asarray(control["thrust_dir_eci"], dtype=float)
        return thrust_vec, 0.0


def test_rhs_scales_thrust_by_max_accel(monkeypatch):
    monkeypatch.setattr(CFG, "use_jet_stream_model", False)
    props = DummyProps(rho=0.0, T=300.0, p=0.0)
    atmosphere = DummyAtmosphere(props)
    earth = DummyEarth(radius=1000.0, mu=0.0)
    aero = DummyAero()
    rocket = DummyRocket()

    sim = Simulation(earth, atmosphere, aero, rocket, max_accel_limit=1.0)
    state = type("S", (), {})()
    state.r_eci = np.array([earth.radius + 1.0, 0.0, 0.0])
    state.v_eci = np.array([0.0, 0.0, 0.0])
    state.m = 1.0
    state.stage_index = 0

    control = ControlCommand(throttle=1.0, thrust_direction=np.array([10.0, 0.0, 0.0]))

    _, dv_dt, _, _ = sim._rhs(t_env=0.0, t_sim=0.0, state=state, control=control)
    assert np.linalg.norm(dv_dt) <= 1.0 + 1e-6


def test_rhs_scales_throttle_by_max_q(monkeypatch):
    monkeypatch.setattr(CFG, "use_jet_stream_model", False)
    props = DummyProps(rho=1.0, T=300.0, p=0.0)
    atmosphere = DummyAtmosphere(props)
    earth = DummyEarth(radius=1000.0)
    aero = DummyAero()
    rocket = DummyRocket()

    sim = Simulation(earth, atmosphere, aero, rocket, max_q_limit=10.0)
    state = type("S", (), {})()
    state.r_eci = np.array([earth.radius + 1.0, 0.0, 0.0])
    state.v_eci = np.array([100.0, 0.0, 0.0])
    state.m = 1.0
    state.stage_index = 0

    control = ControlCommand(throttle=1.0, thrust_direction=np.array([1.0, 0.0, 0.0]))

    sim._rhs(t_env=0.0, t_sim=0.0, state=state, control=control)

    # q = 0.5 * rho * v^2 = 5000, so throttle scales by 10/5000
    assert rocket.last_throttle == pytest.approx(10.0 / 5000.0)
