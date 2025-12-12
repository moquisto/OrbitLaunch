import numpy as np

from Environment.aerodynamics import Aerodynamics, CdModel, get_wind_at_altitude
from Environment.config import EnvironmentConfig


class DummyAtmosphere:
    def __init__(self, rho=1.0, T=300.0, p=0.0):
        self._rho = rho
        self._T = T
        self._p = p

    def properties(self, alt, t_env):
        return type("Props", (), {"rho": self._rho, "T": self._T, "p": self._p})


class DummyEarth:
    def __init__(self, radius=1000.0):
        self.radius = radius

    def atmosphere_velocity(self, r):
        return np.zeros(3)


class DummyRocket:
    def reference_area(self, state):
        return 2.0


def test_drag_force_nonzero(monkeypatch):
    config = EnvironmentConfig()
    config.use_jet_stream_model = False
    atmo = DummyAtmosphere(rho=1.0, T=300.0, p=0.0)
    earth = DummyEarth(radius=1000.0)
    aero = Aerodynamics(atmosphere=atmo, cd_model=CdModel(2.0, config), env_config=config)
    rocket = DummyRocket()

    state = type("S", (), {})()
    state.r_eci = np.array([earth.radius + 100.0, 0.0, 0.0])
    state.v_eci = np.array([10.0, 0.0, 0.0])

    drag = aero.drag_force(state, earth, t=0.0, rocket=rocket)
    # q = 0.5*rho*v^2 = 50; F = q*cd*A = 50*2*2 = 200 directed opposite velocity
    np.testing.assert_allclose(drag, np.array([-200.0, 0.0, 0.0]), atol=1e-6)


def test_wind_profile_interpolation(monkeypatch):
    config = EnvironmentConfig()
    config.wind_alt_points = [0.0, 10_000.0, 20_000.0]
    config.wind_speed_points = [0.0, 50.0, 0.0]
    config.wind_direction_vec = [0.0, 1.0, 0.0]

    low = get_wind_at_altitude(0.0, config)
    mid = get_wind_at_altitude(10_000.0, config)
    high = get_wind_at_altitude(20_000.0, config)

    np.testing.assert_allclose(low, np.array([0.0, 0.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(mid, np.array([0.0, 50.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(high, np.array([0.0, 0.0, 0.0]), atol=1e-6)
