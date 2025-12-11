import numpy as np
import pytest
from unittest.mock import Mock

from integrators import RK4, VelocityVerlet, State, Integrator


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


def test_state_initialization():
    """Test State dataclass initialization."""
    r = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0, 5.0, 6.0])
    s = State(r_eci=r, v_eci=v, m=100.0, stage_index=1, upper_ignition_start_time=10.5)
    
    np.testing.assert_allclose(s.r_eci, r)
    np.testing.assert_allclose(s.v_eci, v)
    assert s.m == 100.0
    assert s.stage_index == 1
    assert s.upper_ignition_start_time == 10.5

    # Test default values
    s_default = State(r_eci=r, v_eci=v, m=100.0)
    assert s_default.stage_index == 0
    assert s_default.upper_ignition_start_time is None

def test_state_copy_method():
    """Test the State.copy() method for deep-ish copy behavior."""
    r_orig = np.array([1.0, 2.0, 3.0])
    v_orig = np.array([4.0, 5.0, 6.0])
    s_orig = State(r_eci=r_orig, v_eci=v_orig, m=100.0, stage_index=1, upper_ignition_start_time=10.5)
    
    s_copy = s_orig.copy()

    # Assert new arrays are created for r_eci and v_eci
    assert s_copy.r_eci is not s_orig.r_eci
    assert s_copy.v_eci is not s_orig.v_eci
    np.testing.assert_allclose(s_copy.r_eci, s_orig.r_eci)
    np.testing.assert_allclose(s_copy.v_eci, s_orig.v_eci)

    # Assert scalar values are copied
    assert s_copy.m == s_orig.m
    assert s_copy.stage_index == s_orig.stage_index
    assert s_copy.upper_ignition_start_time == s_orig.upper_ignition_start_time

    # Modify copy and ensure original is unaffected
    s_copy.r_eci[0] = 99.0
    s_copy.v_eci[1] = 88.0
    s_copy.m = 50.0
    s_copy.stage_index = 2
    s_copy.upper_ignition_start_time = 20.0

    assert s_orig.r_eci[0] == 1.0
    assert s_orig.v_eci[1] == 5.0
    assert s_orig.m == 100.0
    assert s_orig.stage_index == 1
    assert s_orig.upper_ignition_start_time == 10.5

def test_integrator_base_class_raises_not_implemented_error():
    """Test that calling step on the Integrator base class raises NotImplementedError."""
    integrator = Integrator()
    dummy_deriv_fn = Mock()
    dummy_state = Mock(spec=State)
    t = 0.0
    dt = 1.0

    with pytest.raises(NotImplementedError, match="Implement integrator step"):
        integrator.step(dummy_deriv_fn, dummy_state, t, dt)

# Helper for single-step tests
def create_mock_deriv_fn(dr_dt_val, dv_dt_val, dm_dt_val):
    """Creates a mock deriv_fn that returns constant derivatives."""
    def mock_deriv_fn(t, state):
        return np.array(dr_dt_val), np.array(dv_dt_val), dm_dt_val
    return Mock(side_effect=mock_deriv_fn)

def test_rk4_single_step():
    """Test a single step of RK4 integrator with constant derivatives."""
    integrator = RK4()
    
    # Initial state
    r0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([0.0, 0.0, 0.0])
    m0 = 100.0
    initial_state = State(r_eci=r0, v_eci=v0, m=m0)

    # Constant derivatives
    dr_dt_const = np.array([1.0, 0.0, 0.0]) # constant velocity
    dv_dt_const = np.array([0.0, 1.0, 0.0]) # constant acceleration
    dm_dt_const = -1.0 # constant mass depletion

    mock_deriv_fn = create_mock_deriv_fn(dr_dt_const, dv_dt_const, dm_dt_const)
    
    dt = 1.0
    t = 0.0

    # For constant derivatives, RK4 should essentially behave like Euler,
    # since all k1, k2, k3, k4 values will be the same derivatives.
    # r_next = r0 + dt * dr_dt_const
    # v_next = v0 + dt * dv_dt_const
    # m_next = m0 + dt * dm_dt_const

    # The RK4 formula: (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    # If k1=k2=k3=k4=const_deriv, then (dt/6)*(6*const_deriv) = dt*const_deriv
    r_expected = r0 + dt * dr_dt_const
    v_expected = v0 + dt * dv_dt_const
    m_expected = m0 + dt * dm_dt_const

    next_state = integrator.step(mock_deriv_fn, initial_state, t, dt)

    np.testing.assert_allclose(next_state.r_eci, r_expected, atol=1e-9)
    np.testing.assert_allclose(next_state.v_eci, v_expected, atol=1e-9)
    assert next_state.m == pytest.approx(m_expected)
    assert next_state.stage_index == initial_state.stage_index
    assert next_state.upper_ignition_start_time == initial_state.upper_ignition_start_time

    # deriv_fn should be called 4 times for RK4
    assert mock_deriv_fn.call_count == 4


def test_velocity_verlet_single_step():
    """Test a single step of VelocityVerlet integrator with constant derivatives."""
    integrator = VelocityVerlet()

    # Initial state
    r0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([0.0, 0.0, 0.0])
    m0 = 100.0
    initial_state = State(r_eci=r0, v_eci=v0, m=m0)

    # Constant derivatives
    dr_dt_const = np.array([1.0, 0.0, 0.0]) # this is v0, but VV doesn't use dr_dt_n
    dv_dt_const = np.array([0.0, 1.0, 0.0]) # constant acceleration (a_n)
    dm_dt_const = -1.0 # constant mass depletion

    mock_deriv_fn = create_mock_deriv_fn(dr_dt_const, dv_dt_const, dm_dt_const)
    
    dt = 1.0
    t = 0.0

    # Expected values for Velocity Verlet with constant acceleration:
    # r_next = r_n + v_n * dt + 0.5 * a_n * dt^2
    # v_next = v_n + a_n * dt (since a_n = a_np1 for constant acceleration)
    # m_next = m_n + m_dot_n * dt
    
    a_n = dv_dt_const # Initial acceleration
    
    r_expected = r0 + v0 * dt + 0.5 * a_n * dt * dt
    v_expected = v0 + a_n * dt
    m_expected = m0 + dm_dt_const * dt

    next_state = integrator.step(mock_deriv_fn, initial_state, t, dt)

    np.testing.assert_allclose(next_state.r_eci, r_expected, atol=1e-9)
    np.testing.assert_allclose(next_state.v_eci, v_expected, atol=1e-9)
    assert next_state.m == pytest.approx(m_expected)
    assert next_state.stage_index == initial_state.stage_index
    assert next_state.upper_ignition_start_time == initial_state.upper_ignition_start_time

    # deriv_fn should be called 2 times for Velocity Verlet (once for a_n, once for a_np1)
    assert mock_deriv_fn.call_count == 2