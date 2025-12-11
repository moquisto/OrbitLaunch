import numpy as np
import pytest
from unittest.mock import Mock, patch

from aerodynamics import Aerodynamics, CdModel, get_wind_at_altitude, mach_dependent_cd
from config import CFG


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
    monkeypatch.setattr(CFG, "use_jet_stream_model", False)
    atmo = DummyAtmosphere(rho=1.0, T=300.0, p=0.0)
    earth = DummyEarth(radius=1000.0)
    aero = Aerodynamics(atmosphere=atmo, cd_model=CdModel(2.0))
    rocket = DummyRocket()

    state = type("S", (), {})()
    state.r_eci = np.array([earth.radius + 100.0, 0.0, 0.0])
    state.v_eci = np.array([10.0, 0.0, 0.0])

    drag = aero.drag_force(state, earth, t=0.0, rocket=rocket)
    # q = 0.5*rho*v^2 = 50; F = q*cd*A = 50*2*2 = 200 directed opposite velocity
    np.testing.assert_allclose(drag, np.array([-200.0, 0.0, 0.0]), atol=1e-6)


def test_wind_profile_interpolation(monkeypatch):
    monkeypatch.setattr(CFG, "wind_alt_points", [0.0, 10_000.0, 20_000.0])
    monkeypatch.setattr(CFG, "wind_speed_points", [0.0, 50.0, 0.0])
    monkeypatch.setattr(CFG, "wind_direction_vec", [0.0, 1.0, 0.0])

    low = get_wind_at_altitude(0.0)
    mid = get_wind_at_altitude(10_000.0)
    high = get_wind_at_altitude(20_000.0)

    np.testing.assert_allclose(low, np.array([0.0, 0.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(mid, np.array([0.0, 50.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(high, np.array([0.0, 0.0, 0.0]), atol=1e-6)


def test_mach_dependent_cd(monkeypatch):
    """Test mach_dependent_cd with various Mach numbers."""
    # Define a simple, predictable mach_cd_map for testing
    test_mach_cd_map = np.array([
        [0.0, 0.5],
        [1.0, 0.8], # Transonic peak
        [2.0, 0.4],
        [5.0, 0.3],
    ])
    monkeypatch.setattr(CFG, "mach_cd_map", test_mach_cd_map)

    # Test cases
    assert mach_dependent_cd(0.0) == pytest.approx(0.5)
    assert mach_dependent_cd(0.5) == pytest.approx(0.65) # Interpolated
    assert mach_dependent_cd(1.0) == pytest.approx(0.8)
    assert mach_dependent_cd(1.5) == pytest.approx(0.6) # Interpolated
    assert mach_dependent_cd(2.0) == pytest.approx(0.4)
    assert mach_dependent_cd(6.0) == pytest.approx(0.3) # Extrapolated (clamped to last value)

def test_cdmodel_constant():
    """Test CdModel with a constant value."""
    model = CdModel(1.5)
    assert model.cd(0.1) == pytest.approx(1.5)
    assert model.cd(2.5) == pytest.approx(1.5)

def test_cdmodel_callable():
    """Test CdModel with a callable function."""
    def mock_cd_func(mach):
        return 0.1 * mach + 0.2

    model = CdModel(mock_cd_func)
    assert model.cd(1.0) == pytest.approx(0.3) # 0.1*1.0 + 0.2
    assert model.cd(5.0) == pytest.approx(0.7) # 0.1*5.0 + 0.2

def test_get_wind_at_altitude_with_r_eci_aligned_east(monkeypatch):
    """Test get_wind_at_altitude with r_eci for wind direction alignment (aligned east)."""
    monkeypatch.setattr(CFG, "wind_alt_points", [0.0, 10_000.0, 20_000.0])
    monkeypatch.setattr(CFG, "wind_speed_points", [0.0, 50.0, 0.0])
    # Wind direction primarily East [1, 0, 0] in local frame
    monkeypatch.setattr(CFG, "wind_direction_vec", [1.0, 0.0, 0.0]) 

    altitude = 10_000.0
    # Position vector in ECI, simulating a point on the equator along X-axis
    r_eci = np.array([6378e3, 0.0, 0.0]) 

    wind = get_wind_at_altitude(altitude, r_eci=r_eci)
    
    # Expected: wind speed of 50 m/s in the local East direction
    # At (R_EARTH, 0, 0), local East is (0, 1, 0) in ECI
    # This assertion is tricky because the internal 'east' calculation is relative to the cross product [0,0,1] and 'up'
    # Let's re-evaluate the expected output for r_eci = [R_EARTH, 0, 0]
    # Up vector (r_eci / |r_eci|) = [1, 0, 0]
    # East (cross([0,0,1], [1,0,0])) = [0, 1, 0]
    # North (cross([1,0,0], [0,1,0])) = [0, 0, 1]
    # dir_local = [1, 0, 0] (East in local frame)
    # dir_surface = 1 * [0, 1, 0] + 0 * [0, 0, 1] + 0 * [1, 0, 0] = [0, 1, 0]
    # So, wind should be [0.0, 50.0, 0.0]
    np.testing.assert_allclose(wind, np.array([0.0, 50.0, 0.0]), atol=1e-6)

def test_get_wind_at_altitude_with_r_eci_aligned_north(monkeypatch):
    """Test get_wind_at_altitude with r_eci for wind direction alignment (aligned north)."""
    monkeypatch.setattr(CFG, "wind_alt_points", [0.0, 10_000.0, 20_000.0])
    monkeypatch.setattr(CFG, "wind_speed_points", [0.0, 50.0, 0.0])
    # Wind direction primarily North [0, 1, 0] in local frame
    monkeypatch.setattr(CFG, "wind_direction_vec", [0.0, 1.0, 0.0]) 

    altitude = 10_000.0
    r_eci = np.array([6378e3, 0.0, 0.0]) 

    wind = get_wind_at_altitude(altitude, r_eci=r_eci)
    
    # Expected: wind speed of 50 m/s in the local North direction
    # At (R_EARTH, 0, 0), local North is (0, 0, 1) in ECI
    # Up vector = [1, 0, 0]
    # East = [0, 1, 0]
    # North = [0, 0, 1]
    # dir_local = [0, 1, 0] (North in local frame)
    # dir_surface = 0 * [0, 1, 0] + 1 * [0, 0, 1] + 0 * [1, 0, 0] = [0, 0, 1]
    # So, wind should be [0.0, 0.0, 50.0]
    np.testing.assert_allclose(wind, np.array([0.0, 0.0, 50.0]), atol=1e-6)


def test_drag_force_zero_rho(monkeypatch):
    """Test drag_force returns zero when atmosphere density is zero."""
    monkeypatch.setattr(CFG, "use_jet_stream_model", False)
    atmo = DummyAtmosphere(rho=0.0)
    earth = DummyEarth(radius=1000.0)
    aero = Aerodynamics(atmosphere=atmo, cd_model=CdModel(2.0), reference_area=1.0)

    state = type("S", (), {})()
    state.r_eci = np.array([earth.radius + 100.0, 0.0, 0.0])
    state.v_eci = np.array([10.0, 0.0, 0.0])

    drag = aero.drag_force(state, earth, t=0.0)
    np.testing.assert_allclose(drag, np.array([0.0, 0.0, 0.0]), atol=1e-9)

def test_drag_force_zero_v_rel(monkeypatch):
    """Test drag_force returns zero when relative velocity is zero."""
    monkeypatch.setattr(CFG, "use_jet_stream_model", False)
    atmo = DummyAtmosphere(rho=1.0)
    earth = DummyEarth(radius=1000.0)
    # Set atmosphere_velocity to match v_eci for zero relative velocity
    earth.atmosphere_velocity = Mock(return_value=np.array([10.0, 0.0, 0.0])) 
    aero = Aerodynamics(atmosphere=atmo, cd_model=CdModel(2.0), reference_area=1.0)

    state = type("S", (), {})()
    state.r_eci = np.array([earth.radius + 100.0, 0.0, 0.0])
    state.v_eci = np.array([10.0, 0.0, 0.0])

    drag = aero.drag_force(state, earth, t=0.0)
    np.testing.assert_allclose(drag, np.array([0.0, 0.0, 0.0]), atol=1e-9)

def test_drag_force_zero_reference_area(monkeypatch):
    """Test drag_force returns zero when reference area is zero."""
    monkeypatch.setattr(CFG, "use_jet_stream_model", False)
    atmo = DummyAtmosphere(rho=1.0)
    earth = DummyEarth(radius=1000.0)
    aero = Aerodynamics(atmosphere=atmo, cd_model=CdModel(2.0), reference_area=0.0) # Zero reference area

    state = type("S", (), {})()
    state.r_eci = np.array([earth.radius + 100.0, 0.0, 0.0])
    state.v_eci = np.array([10.0, 0.0, 0.0])

    drag = aero.drag_force(state, earth, t=0.0)
    np.testing.assert_allclose(drag, np.array([0.0, 0.0, 0.0]), atol=1e-9)

def test_drag_force_zero_cd(monkeypatch):
    """Test drag_force returns zero when drag coefficient is zero."""
    monkeypatch.setattr(CFG, "use_jet_stream_model", False)
    atmo = DummyAtmosphere(rho=1.0)
    earth = DummyEarth(radius=1000.0)
    aero = Aerodynamics(atmosphere=atmo, cd_model=CdModel(0.0), reference_area=1.0) # Zero Cd

    state = type("S", (), {})()
    state.r_eci = np.array([earth.radius + 100.0, 0.0, 0.0])
    state.v_eci = np.array([10.0, 0.0, 0.0])

    drag = aero.drag_force(state, earth, t=0.0)
    np.testing.assert_allclose(drag, np.array([0.0, 0.0, 0.0]), atol=1e-9)

def test_drag_force_r_norm_zero(monkeypatch):
    """Test drag_force returns zero when r_eci norm is zero."""
    monkeypatch.setattr(CFG, "use_jet_stream_model", False)
    atmo = DummyAtmosphere(rho=1.0)
    earth = DummyEarth(radius=1000.0)
    aero = Aerodynamics(atmosphere=atmo, cd_model=CdModel(1.0), reference_area=1.0)

    state = type("S", (), {})()
    state.r_eci = np.array([0.0, 0.0, 0.0]) # Zero position
    state.v_eci = np.array([10.0, 0.0, 0.0])

    drag = aero.drag_force(state, earth, t=0.0)
    np.testing.assert_allclose(drag, np.array([0.0, 0.0, 0.0]), atol=1e-9)


def test_drag_force_altitude_clipping(monkeypatch):
    """Test drag_force clips altitude to 0 if below surface."""
    monkeypatch.setattr(CFG, "use_jet_stream_model", False)
    atmo = DummyAtmosphere(rho=1.225) # Sea level density
    earth = DummyEarth(radius=6378e3)
    aero = Aerodynamics(atmosphere=atmo, cd_model=CdModel(0.5), reference_area=10.0)

    state = type("S", (), {})()
    # Position below surface
    state.r_eci = np.array([earth.radius - 100.0, 0.0, 0.0]) 
    state.v_eci = np.array([100.0, 0.0, 0.0])

    # Mock atmosphere.properties to confirm it's called with altitude=0
    with patch.object(atmo, 'properties', wraps=atmo.properties) as mock_properties:
        drag = aero.drag_force(state, earth, t=0.0)
        mock_properties.assert_called_with(0.0, 0.0) # Assert altitude is 0.0
    
    # Calculate expected drag for altitude=0, v_rel_mag=100, Cd=0.5, A=10.0, rho=1.225
    q = 0.5 * 1.225 * (100.0**2) # 6125.0
    expected_f_mag = q * 0.5 * 10.0 # 30625.0
    np.testing.assert_allclose(drag, np.array([-expected_f_mag, 0.0, 0.0]), atol=1e-6)

def test_drag_force_rocket_reference_area_priority(monkeypatch):
    """Test drag_force uses rocket's reference_area if available."""
    monkeypatch.setattr(CFG, "use_jet_stream_model", False)
    atmo = DummyAtmosphere(rho=1.0)
    earth = DummyEarth(radius=1000.0)
    aero = Aerodynamics(atmosphere=atmo, cd_model=CdModel(2.0), reference_area=5.0) # Aerodynamics has A=5

    state = type("S", (), {})()
    state.r_eci = np.array([earth.radius + 100.0, 0.0, 0.0])
    state.v_eci = np.array([10.0, 0.0, 0.0])

    # Rocket provides A=2.0
    rocket_with_area = DummyRocket() 
    
    drag = aero.drag_force(state, earth, t=0.0, rocket=rocket_with_area)
    # q = 0.5*rho*v^2 = 50; F = q*cd*A = 50*2*2.0 (from rocket) = 200
    np.testing.assert_allclose(drag, np.array([-200.0, 0.0, 0.0]), atol=1e-6)

def test_drag_force_class_reference_area_fallback(monkeypatch):
    """Test drag_force falls back to class's reference_area if rocket is None."""
    monkeypatch.setattr(CFG, "use_jet_stream_model", False)
    atmo = DummyAtmosphere(rho=1.0)
    earth = DummyEarth(radius=1000.0)
    aero = Aerodynamics(atmosphere=atmo, cd_model=CdModel(2.0), reference_area=5.0) # Aerodynamics has A=5

    state = type("S", (), {})()
    state.r_eci = np.array([earth.radius + 100.0, 0.0, 0.0])
    state.v_eci = np.array([10.0, 0.0, 0.0])

    drag = aero.drag_force(state, earth, t=0.0, rocket=None) # No rocket provided
    # q = 0.5*rho*v^2 = 50; F = q*cd*A = 50*2*5.0 (from class) = 500
    np.testing.assert_allclose(drag, np.array([-500.0, 0.0, 0.0]), atol=1e-6)

def test_drag_force_with_wind_model_enabled(monkeypatch):
    """Test drag_force correctly incorporates wind when CFG.use_jet_stream_model is True."""
    monkeypatch.setattr(CFG, "use_jet_stream_model", True)
    # Define a simple wind profile: 50 m/s wind in local East at 10km altitude
    monkeypatch.setattr(CFG, "wind_alt_points", [0.0, 10_000.0, 20_000.0])
    monkeypatch.setattr(CFG, "wind_speed_points", [0.0, 50.0, 0.0])
    monkeypatch.setattr(CFG, "wind_direction_vec", [1.0, 0.0, 0.0]) # Local East

    atmo = DummyAtmosphere(rho=1.0, T=300.0)
    earth = DummyEarth(radius=6378e3) # Use realistic radius for ECI transforms
    # atmosphere_velocity should return 0 for simplicity in testing wind directly
    earth.atmosphere_velocity = Mock(return_value=np.zeros(3)) 
    aero = Aerodynamics(atmosphere=atmo, cd_model=CdModel(2.0), reference_area=2.0)

    state = type("S", (), {})()
    # At 10km altitude, along X-axis
    state.r_eci = np.array([earth.radius + 10_000.0, 0.0, 0.0]) 
    state.v_eci = np.array([10.0, 0.0, 0.0]) # Rocket moving East relative to ECI

    drag = aero.drag_force(state, earth, t=0.0)

    # Expected wind at this altitude and position (from previous tests) is [0, 50, 0] ECI
    # v_atm_rotation = [0, 0, 0] (mocked)
    # wind_vector = [0, 50, 0]
    # v_air = [0, 50, 0]
    # v_rel = v_eci - v_air = [10, 0, 0] - [0, 50, 0] = [10, -50, 0]
    # v_rel_mag = sqrt(10^2 + (-50)^2) = sqrt(100 + 2500) = sqrt(2600) = 50.99
    # Mach calculation (assuming a=sqrt(gamma*R*T) for T=300, gamma=1.4, R=287 => a=347.16)
    # mach = 50.99 / 347.16 = 0.146
    # Cd(mach=0.146) = 2.0 (constant CdModel)
    # q = 0.5 * rho * v_rel_mag^2 = 0.5 * 1.0 * 2600 = 1300
    # F_mag = q * Cd * A = 1300 * 2.0 * 2.0 = 5200
    # F_vec = -F_mag * v_rel / v_rel_mag = -5200 * [10, -50, 0] / 50.99
    
    expected_v_rel = np.array([10.0, -50.0, 0.0])
    expected_v_rel_mag = np.linalg.norm(expected_v_rel)
    expected_q = 0.5 * 1.0 * expected_v_rel_mag**2
    expected_f_mag = expected_q * 2.0 * 2.0 # rho=1, Cd=2, A=2
    expected_drag_vec = -expected_f_mag * expected_v_rel / expected_v_rel_mag
    
    np.testing.assert_allclose(drag, expected_drag_vec, atol=1e-6)