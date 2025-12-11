import numpy as np
import pytest
from unittest.mock import Mock

from custom_guidance import create_pitch_program_callable, simple_pitch_program
from integrators import State

def test_pitch_program_interpolation():
    """
    Test if the pitch program interpolates angles correctly based on time.
    """
    pitch_points = [(0.0, 90.0), (100.0, 45.0), (200.0, 0.0)]
    pitch_program = create_pitch_program_callable(pitch_points)

    mock_state = Mock(spec=State, r_eci=np.array([0, 0, 6.371e6]), v_eci=np.array([0, 0, 1]))

    # Test exact points
    np.testing.assert_allclose(pitch_program(0.0, mock_state)[2], 1.0, atol=1e-6) # Vertical (z-axis)
    # Check approximately 45 degrees
    mid_vec = pitch_program(100.0, mock_state)
    angle_from_vertical = np.degrees(np.arccos(np.dot(mid_vec, np.array([0,0,1]))))
    assert angle_from_vertical == pytest.approx(45.0, abs=1.0) # Allow some tolerance due to horizontal component

    # Test interpolated points
    np.testing.assert_allclose(pitch_program(50.0, mock_state)[2], np.sin(np.radians(67.5)), atol=0.1) # Between 90 and 45
    np.testing.assert_allclose(pitch_program(150.0, mock_state)[2], np.sin(np.radians(22.5)), atol=0.1) # Between 45 and 0
    # Test outside bounds (should hold first/last value)
    np.testing.assert_allclose(pitch_program(-10.0, mock_state)[2], 1.0, atol=1e-6)
    np.testing.assert_allclose(pitch_program(210.0, mock_state)[2], np.sin(np.radians(0.0)), atol=1e-6)

def test_pitch_program_vertical_direction():
    """
    Test that thrust direction is vertical when pitch angle is 90 degrees.
    """
    pitch_points = [(0.0, 90.0)]
    pitch_program = create_pitch_program_callable(pitch_points)

    # State where rocket is at Earth's surface, aiming for vertical
    r_eci = np.array([0, 0, 6.371e6])
    v_eci = np.array([0, 0, 10]) # Small vertical velocity
    mock_state = Mock(spec=State, r_eci=r_eci, v_eci=v_eci)

    thrust_dir = pitch_program(0.0, mock_state)
    expected_dir = r_eci / np.linalg.norm(r_eci) # Should be purely radial (vertical)
    np.testing.assert_allclose(thrust_dir, expected_dir, atol=1e-6)

    # Another point, not on Z axis
    r_eci_diag = np.array([1, 1, 1]) * 6.371e6
    v_eci_diag = np.array([0.1, 0.1, 0.1])
    mock_state_diag = Mock(spec=State, r_eci=r_eci_diag, v_eci=v_eci_diag)
    thrust_dir_diag = pitch_program(0.0, mock_state_diag)
    expected_dir_diag = r_eci_diag / np.linalg.norm(r_eci_diag)
    np.testing.assert_allclose(thrust_dir_diag, expected_dir_diag, atol=1e-6)


def test_pitch_program_horizontal_direction():
    """
    Test that thrust direction is horizontal when pitch angle is 0 degrees.
    """
    pitch_points = [(0.0, 0.0)]
    pitch_program = create_pitch_program_callable(pitch_points)

    # State where rocket has some horizontal velocity
    r_eci = np.array([6.371e6, 0, 0])
    v_eci = np.array([0, 1000, 0]) # Pure tangential velocity
    mock_state = Mock(spec=State, r_eci=r_eci, v_eci=v_eci)

    thrust_dir = pitch_program(0.0, mock_state)
    
    # Expected direction should be tangential to Earth's surface, in the direction of velocity
    vertical_dir = r_eci / np.linalg.norm(r_eci)
    horizontal_dir_raw = v_eci - np.dot(v_eci, vertical_dir) * vertical_dir
    expected_dir = horizontal_dir_raw / np.linalg.norm(horizontal_dir_raw)
    
    np.testing.assert_allclose(thrust_dir, expected_dir, atol=1e-6)

    # State with mixed velocity
    r_eci_mixed = np.array([0, 0, 6.371e6])
    v_eci_mixed = np.array([1000, 0, 10]) # Mostly horizontal (X), small vertical (Z)
    mock_state_mixed = Mock(spec=State, r_eci=r_eci_mixed, v_eci=v_eci_mixed)
    thrust_dir_mixed = pitch_program(0.0, mock_state_mixed)
    
    vertical_dir_mixed = r_eci_mixed / np.linalg.norm(r_eci_mixed)
    horizontal_dir_raw_mixed = v_eci_mixed - np.dot(v_eci_mixed, vertical_dir_mixed) * vertical_dir_mixed
    expected_dir_mixed = horizontal_dir_raw_mixed / np.linalg.norm(horizontal_dir_raw_mixed)
    
    np.testing.assert_allclose(thrust_dir_mixed, expected_dir_mixed, atol=1e-6)


def test_pitch_program_arbitrary_angle():
    """
    Test thrust direction for an arbitrary pitch angle (e.g., 45 degrees).
    """
    pitch_points = [(0.0, 45.0)]
    pitch_program = create_pitch_program_callable(pitch_points)

    # Rocket on X-axis, velocity in Y-axis
    r_eci = np.array([6.371e6, 0, 0]) # Along X
    v_eci = np.array([0, 1000, 0])     # Along Y
    mock_state = Mock(spec=State, r_eci=r_eci, v_eci=v_eci)

    thrust_dir = pitch_program(0.0, mock_state)

    vertical_dir = r_eci / np.linalg.norm(r_eci) # (1, 0, 0)
    horizontal_dir = np.array([0, 1, 0]) # Should be (0, 1, 0) for this specific setup
    
    # Expected: cos(45) * vertical_dir + sin(45) * horizontal_dir
    expected_dir = np.cos(np.radians(45)) * vertical_dir + np.sin(np.radians(45)) * horizontal_dir
    np.testing.assert_allclose(thrust_dir, expected_dir, atol=1e-6)

def test_pitch_program_near_zero_velocity():
    """
    Test behavior when velocity is very small (near stationary).
    Should default to vertical.
    """
    pitch_points = [(0.0, 30.0)] # Pitch angle doesn't matter much if velocity is near zero
    pitch_program = create_pitch_program_callable(pitch_points)

    r_eci = np.array([0, 0, 6.371e6])
    v_eci_small = np.array([0.01, 0.001, 0.001]) # Very small velocity
    mock_state_small_v = Mock(spec=State, r_eci=r_eci, v_eci=v_eci_small)

    thrust_dir_small_v = pitch_program(0.0, mock_state_small_v)
    expected_dir_vertical = r_eci / np.linalg.norm(r_eci)
    np.testing.assert_allclose(thrust_dir_small_v, expected_dir_vertical, atol=1e-6)

    v_eci_zero = np.array([0.0, 0.0, 0.0]) # Zero velocity
    mock_state_zero_v = Mock(spec=State, r_eci=r_eci, v_eci=v_eci_zero)
    thrust_dir_zero_v = pitch_program(0.0, mock_state_zero_v)
    np.testing.assert_allclose(thrust_dir_zero_v, expected_dir_vertical, atol=1e-6)

def test_pitch_program_pure_vertical_motion_horizontal_fallback():
    """
    Test the fallback for purely vertical motion where horizontal_dir_raw is near zero.
    It should try to find an arbitrary horizontal direction.
    """
    pitch_points = [(0.0, 10.0)] # Some angle not 90
    pitch_program = create_pitch_program_callable(pitch_points)

    r_eci = np.array([0, 0, 6.371e6])
    v_eci = np.array([0, 0, 100]) # Pure vertical velocity
    mock_state = Mock(spec=State, r_eci=r_eci, v_eci=v_eci)

    thrust_dir = pitch_program(0.0, mock_state)
    
    vertical_dir = r_eci / np.linalg.norm(r_eci)
    
    # In custom_guidance, if horizontal_dir_raw is small for Z-axis vertical_dir:
    # tangent_dir = np.array([1.0, 0.0, 0.0]) - np.dot(np.array([1.0, 0.0, 0.0]), vertical_dir) * vertical_dir
    # Here, vertical_dir is (0,0,1). dot product is 0. So tangent_dir becomes (1,0,0)
    expected_tangent_dir = np.array([1.0, 0.0, 0.0])
    
    expected_dir = np.sin(np.radians(10)) * vertical_dir + np.cos(np.radians(10)) * expected_tangent_dir
    
    np.testing.assert_allclose(thrust_dir, expected_dir, atol=1e-6)

    # Test with a different r_eci where vertical_dir[2] < 0.9 (e.g., along X axis)
    r_eci_x = np.array([6.371e6, 0, 0])
    v_eci_x = np.array([100, 0, 0]) # Pure vertical velocity relative to Earth surface at launch site
    mock_state_x = Mock(spec=State, r_eci=r_eci_x, v_eci=v_eci_x)

    thrust_dir_x = pitch_program(0.0, mock_state_x)

    vertical_dir_x = r_eci_x / np.linalg.norm(r_eci_x) # (1,0,0)
    
    # In custom_guidance, if horizontal_dir_raw is small for X-axis vertical_dir:
    # tangent_dir = np.array([0.0, 0.0, 1.0]) - np.dot(np.array([0.0, 0.0, 1.0]), vertical_dir_x) * vertical_dir_x
    # Here, vertical_dir_x is (1,0,0). dot product is 0. So tangent_dir becomes (0,0,1)
    expected_tangent_dir_x = np.array([0.0, 0.0, 1.0])

    expected_dir_x = np.sin(np.radians(10)) * vertical_dir_x + np.cos(np.radians(10)) * expected_tangent_dir_x
    np.testing.assert_allclose(thrust_dir_x, expected_dir_x, atol=1e-6)


def test_simple_pitch_program_vertical_alignment():
    """simple_pitch_program should align thrust with the radial direction."""
    r_vec = np.array([1.0, 2.0, 3.0])
    mock_state = Mock(spec=State, r_eci=r_vec, v_eci=np.zeros(3))
    expected = r_vec / np.linalg.norm(r_vec)
    np.testing.assert_allclose(simple_pitch_program(0.0, mock_state), expected, atol=1e-6)

    zero_state = Mock(spec=State, r_eci=np.zeros(3), v_eci=np.zeros(3))
    np.testing.assert_allclose(simple_pitch_program(0.0, zero_state), np.array([0.0, 0.0, 1.0]), atol=1e-6)
