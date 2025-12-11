import numpy as np
import types
import pytest

import atmosphere
from atmosphere import AtmosphereModel, AtmosphereProperties
from gravity import EarthModel, orbital_elements_from_state, J2_EARTH


def test_atmosphere_dispatch(monkeypatch):
    calls = {"us76": 0, "msis": 0}

    def fake_us76(self, alt):
        calls["us76"] += 1
        return AtmosphereProperties(rho=1.0, p=2.0, T=3.0)

    def fake_msis(self, alt, t=None):
        calls["msis"] += 1
        return AtmosphereProperties(rho=4.0, p=5.0, T=6.0)

    monkeypatch.setattr(AtmosphereModel, "_us76_properties", fake_us76, raising=False)
    monkeypatch.setattr(AtmosphereModel, "_nrlmsis_properties", fake_msis, raising=False)

    model = AtmosphereModel(h_switch=100.0)

    low = model.properties(50.0, t=0.0)
    high = model.properties(150.0, t=0.0)

    assert (low.rho, low.p, low.T) == (1.0, 2.0, 3.0)
    assert (high.rho, high.p, high.T) == (4.0, 5.0, 6.0)
    assert calls["us76"] == 1 and calls["msis"] == 1


def test_earth_gravity_central_and_j2():
    mu = 1.0
    R = 1.0
    earth_central = EarthModel(mu=mu, radius=R, omega_vec=np.zeros(3), j2=None)
    r_vec = np.array([1.0, 0.0, 0.0])
    a = earth_central.gravity_accel(r_vec)
    np.testing.assert_allclose(a, np.array([-1.0, 0.0, 0.0]), atol=1e-9)

    earth_j2 = EarthModel(mu=mu, radius=R, omega_vec=np.zeros(3), j2=J2_EARTH)
    a_j2 = earth_j2.gravity_accel(np.array([1.0, 0.0, 0.0]))
    # x-component should be less negative due to J2 term
    assert a_j2[0] > a[0]


def test_atmosphere_velocity_cross_product():
    omega = np.array([0.0, 0.0, 2.0])
    earth = EarthModel(mu=1.0, radius=1.0, omega_vec=omega, j2=None)
    r_vec = np.array([1.0, 0.0, 0.0])
    v_atm = earth.atmosphere_velocity(r_vec)
    np.testing.assert_allclose(v_atm, np.array([0.0, 2.0, 0.0]), atol=1e-9)


def test_orbital_elements_from_state_hyperbolic_and_circular():
    mu = 1.0
    # Circular orbit at r=1, v = sqrt(mu/r)
    r = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])
    a, rp, ra = orbital_elements_from_state(r, v, mu)
    np.testing.assert_allclose(a, 1.0)
    np.testing.assert_allclose(rp, 1.0)
    np.testing.assert_allclose(ra, 1.0)

    # Hyperbolic: speed > escape
    v_hyp = np.array([0.0, 2.0, 0.0])
    a2, rp2, ra2 = orbital_elements_from_state(r, v_hyp, mu)
    assert a2 < 0  # hyperbolic
    assert rp2 is not None and ra2 == np.inf
