import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from custom_guidance import orbital_elements_from_state, simple_pitch_program, TwoPhaseUpperThrottle
from integrators import State # For type hinting/mocking
from config import CFG

class TestOrbitalElements(unittest.TestCase):

    def test_circular_orbit(self):
        # Earth-like scenario
        mu = 3.986004418e14  # Earth's standard gravitational parameter
        r_norm = 7_000_000.0  # ~6371km + ~629km altitude
        v_norm = np.sqrt(mu / r_norm) # Circular orbit speed
        
        r_vec = np.array([r_norm, 0.0, 0.0])
        v_vec = np.array([0.0, v_norm, 0.0])

        a, rp, ra = orbital_elements_from_state(r_vec, v_vec, mu)

        self.assertAlmostEqual(a, r_norm, places=0)
        self.assertAlmostEqual(rp, r_norm, places=0)
        self.assertAlmostEqual(ra, r_norm, places=0)
    
    def test_elliptical_orbit(self):
        # Example from a textbook (e.g., Vallado)
        mu = 3.986004418e14
        r_vec = np.array([7000e3, 0, 0])
        v_vec = np.array([0, 10e3, 0]) # High speed for an elliptical orbit

        a, rp, ra = orbital_elements_from_state(r_vec, v_vec, mu)

        self.assertIsNotNone(a)
        self.assertIsNotNone(rp)
        self.assertIsNotNone(ra)

        # Expected values can be calculated:
        # v_norm = 10e3, r_norm = 7000e3
        # E = 0.5 * v_norm^2 - mu / r_norm = 0.5 * (10e3)^2 - 3.986004418e14 / 7000e3
        #   = 5e7 - 5.6942920257e7 = -6.942920257e6
        # a = -mu / (2*E) = -3.986004418e14 / (2 * -6.942920257e6) = 2.8774e7
        # e = sqrt(1 + 2*E*h^2/mu^2) ... easier to check rp, ra directly

        # Perigee/Apoapsis calculations for these specific r, v:
        # e.g. using online calculator, these values should yield:
        # a ~ 28774 km
        # rp ~ 6010 km
        # ra ~ 51538 km
        self.assertAlmostEqual(a, 2.8774e7, delta=1e2)
        self.assertAlmostEqual(rp, 6.010e6, delta=1e2)
        self.assertAlmostEqual(ra, 5.1538e7, delta=1e2)


    def test_zero_r_norm(self):
        r_vec = np.array([0.0, 0.0, 0.0])
        v_vec = np.array([100.0, 0.0, 0.0])
        mu = 3.986004418e14
        a, rp, ra = orbital_elements_from_state(r_vec, v_vec, mu)
        self.assertIsNone(a)
        self.assertIsNone(rp)
        self.assertIsNone(ra)

    def test_zero_mu(self):
        r_vec = np.array([7e6, 0.0, 0.0])
        v_vec = np.array([100.0, 0.0, 0.0])
        mu = 0.0
        a, rp, ra = orbital_elements_from_state(r_vec, v_vec, mu)
        self.assertIsNone(a)
        self.assertIsNone(rp)
        self.assertIsNone(ra)
    
    def test_parabolic_orbit(self):
        # A parabolic orbit has E = 0, so a -> infinity
        mu = 3.986004418e14
        r_norm = 7_000_000.0
        v_norm_parabolic = np.sqrt(2 * mu / r_norm)
        r_vec = np.array([r_norm, 0.0, 0.0])
        v_vec = np.array([0.0, v_norm_parabolic, 0.0])

        a, rp, ra = orbital_elements_from_state(r_vec, v_vec, mu)
        self.assertIsNone(a)
        self.assertIsNone(rp)
        self.assertIsNone(ra)

    def test_radial_trajectory(self):
        # v and r are collinear, so h_vec is zero.
        mu = 3.986004418e14
        r_vec = np.array([7e6, 0.0, 0.0])
        v_vec = np.array([1e3, 0.0, 0.0]) # Velocity along radius

        a, rp, ra = orbital_elements_from_state(r_vec, v_vec, mu)
        self.assertIsNotNone(a) # a should be finite for sub-escape radial trajectory
        self.assertAlmostEqual(rp, 0.0, places=5) # Perigee is 0 for radial motion passing through center
        self.assertAlmostEqual(ra, 2 * a - 0.0, places=5) # Apoapsis is 2a for elliptical radial, if a is finite
        
        # Test for direct radial ascent from surface (v_vec = [1000,0,0], r_vec=[R,0,0])
        # This will hit h_norm_sq == 0 case.
        mu = 3.986004418e14
        r_vec = np.array([6.371e6, 0.0, 0.0])
        v_vec = np.array([1000.0, 0.0, 0.0])
        a, rp, ra = orbital_elements_from_state(r_vec, v_vec, mu)
        self.assertIsNotNone(a)
        self.assertAlmostEqual(rp, 0.0, places=5)
        self.assertAlmostEqual(ra, np.linalg.norm(r_vec), places=5) # radial in-bound, so ra is current r_norm


class MockState:
    def __init__(self, r_eci, v_eci, stage_index=0):
        self.r_eci = np.array(r_eci, dtype=float)
        self.v_eci = np.array(v_eci, dtype=float)
        self.stage_index = stage_index

class TestSimplePitchProgram(unittest.TestCase):

    def setUp(self):
        self.original_cfg_earth_radius_m = CFG.earth_radius_m
        CFG.earth_radius_m = 6_371_000.0

    def tearDown(self):
        CFG.earth_radius_m = self.original_cfg_earth_radius_m

    def test_below_start_altitude(self):
        # alt = 1000m, start = 5000m -> should return r_hat
        r_eci = np.array([CFG.earth_radius_m + 1000, 0.0, 0.0])
        v_eci = np.array([0.0, 10.0, 0.0])
        state = MockState(r_eci, v_eci)
        
        direction = simple_pitch_program(0.0, state)
        expected_direction = r_eci / np.linalg.norm(r_eci)
        np.testing.assert_allclose(direction, expected_direction)

    def test_between_start_and_end_altitude(self):
        # alt = 25000m, start = 5000m, end = 60000m
        # w = (25000 - 5000) / (60000 - 5000) = 20000 / 55000 = 4/11
        # direction = (1 - 4/11) * r_hat + (4/11) * east = (7/11)*r_hat + (4/11)*east
        r_eci = np.array([CFG.earth_radius_m + 25000, 0.0, 0.0])
        v_eci = np.array([0.0, 1000.0, 0.0])
        state = MockState(r_eci, v_eci)

        r_hat = r_eci / np.linalg.norm(r_eci)
        east = np.array([0.0, 1.0, 0.0]) # cross([0,0,1], r_hat=[1,0,0]) = [0,1,0]
        expected_direction = (7/11) * r_hat + (4/11) * east
        expected_direction = expected_direction / np.linalg.norm(expected_direction) # Normalize

        direction = simple_pitch_program(0.0, state)
        np.testing.assert_allclose(direction, expected_direction)

    def test_above_end_altitude_prograde(self):
        # alt = 70000m, end = 60000m. Speed > 100 -> prograde (v/speed)
        r_eci = np.array([CFG.earth_radius_m + 70000, 0.0, 0.0])
        v_eci = np.array([0.0, 500.0, 0.0]) # Speed > 100
        state = MockState(r_eci, v_eci)

        direction = simple_pitch_program(0.0, state)
        expected_direction = v_eci / np.linalg.norm(v_eci)
        np.testing.assert_allclose(direction, expected_direction)

    def test_above_end_altitude_east(self):
        # alt = 70000m, end = 60000m. Speed < 100 -> east
        r_eci = np.array([CFG.earth_radius_m + 70000, 0.0, 0.0])
        v_eci = np.array([0.0, 50.0, 0.0]) # Speed < 100
        state = MockState(r_eci, v_eci)

        direction = simple_pitch_program(0.0, state)
        # east is cross([0,0,1], r_hat)
        r_hat = r_eci / np.linalg.norm(r_eci)
        east_vec = np.cross(np.array([0.0, 0.0, 1.0]), r_hat)
        expected_direction = east_vec / np.linalg.norm(east_vec)
        np.testing.assert_allclose(direction, expected_direction)

    def test_zero_r_norm(self):
        r_eci = np.array([0.0, 0.0, 0.0])
        v_eci = np.array([0.0, 0.0, 0.0])
        state = MockState(r_eci, v_eci)
        direction = simple_pitch_program(0.0, state)
        np.testing.assert_allclose(direction, np.array([0.0, 0.0, 1.0])) # Default return if r_norm is 0


class TestTwoPhaseUpperThrottle(unittest.TestCase):

    def setUp(self):
        # Save original CFG values
        self.original_cfg_upper_throttle_vr_tolerance = CFG.upper_throttle_vr_tolerance
        self.original_cfg_upper_throttle_alt_tolerance = CFG.upper_throttle_alt_tolerance
        self.original_cfg_orbit_alt_tol = CFG.orbit_alt_tol

        # Set default values for testing
        CFG.upper_throttle_vr_tolerance = 2.0
        CFG.upper_throttle_alt_tolerance = 1000.0
        CFG.orbit_alt_tol = 50.0

    def tearDown(self):
        # Restore original CFG values
        CFG.upper_throttle_vr_tolerance = self.original_cfg_upper_throttle_vr_tolerance
        CFG.upper_throttle_alt_tolerance = self.original_cfg_upper_throttle_alt_tolerance
        CFG.orbit_alt_tol = self.original_cfg_orbit_alt_tol

    def test_init(self):
        throttle_guidance = TwoPhaseUpperThrottle(target_radius=7e6, mu=CFG.earth_mu)
        self.assertEqual(throttle_guidance.target_radius, 7e6)
        self.assertEqual(throttle_guidance.mu, CFG.earth_mu)
        self.assertEqual(throttle_guidance.phase, "boost")
        self.assertEqual(len(throttle_guidance.transitions), 1)
        self.assertEqual(throttle_guidance.transitions[0], ("boost", 0.0))
    
    def test_booster_stage_full_throttle(self):
        throttle_guidance = TwoPhaseUpperThrottle(target_radius=7e6, mu=CFG.earth_mu)
        state = MockState(r_eci=np.array([0,0,0]), v_eci=np.array([0,0,0]), stage_index=0)
        throttle_cmd = throttle_guidance(0.0, state)
        self.assertEqual(throttle_cmd, 1.0)
        self.assertEqual(throttle_guidance.phase, "boost") # Phase should not change for booster

    @patch('custom_guidance.orbital_elements_from_state')
    def test_boost_phase_to_coast_transition(self, mock_orbital_elements):
        throttle_guidance = TwoPhaseUpperThrottle(target_radius=7e6, mu=CFG.earth_mu)
        
        # Scenario 1: ra < target_radius, should remain in boost, throttle 1.0
        mock_orbital_elements.return_value = (None, None, 6.9e6) # a, rp, ra
        state = MockState(r_eci=np.array([7e6, 0, 0]), v_eci=np.array([0, 7e3, 0]), stage_index=1)
        throttle_cmd = throttle_guidance(10.0, state)
        self.assertEqual(throttle_cmd, 1.0)
        self.assertEqual(throttle_guidance.phase, "boost")
        self.assertEqual(len(throttle_guidance.transitions), 1)

        # Scenario 2: ra >= target_radius, should transition to coast, throttle 0.0
        mock_orbital_elements.return_value = (None, None, 7.1e6) # a, rp, ra
        throttle_cmd = throttle_guidance(20.0, state)
        self.assertEqual(throttle_cmd, 0.0)
        self.assertEqual(throttle_guidance.phase, "coast")
        self.assertEqual(len(throttle_guidance.transitions), 2)
        self.assertEqual(throttle_guidance.transitions[1], ("coast", 20.0))
        self.assertEqual(throttle_guidance.target_ap, 7.1e6)

    @patch('custom_guidance.orbital_elements_from_state')
    def test_coast_phase_to_circularize_transition(self, mock_orbital_elements):
        throttle_guidance = TwoPhaseUpperThrottle(target_radius=7e6, mu=CFG.earth_mu)
        throttle_guidance.phase = "coast"
        throttle_guidance.transitions.append(("boost", 0.0)) # Add a dummy transition
        throttle_guidance.target_ap = 7.5e6 # Assume it boosted to this apoapsis

        r_norm_at_ap = 7.5e6
        v_radial_low = CFG.upper_throttle_vr_tolerance - 0.1
        v_radial_high = CFG.upper_throttle_vr_tolerance + 0.1

        # Scenario 1: Still in coast, vr high, throttle 0.0
        mock_orbital_elements.return_value = (None, None, None) # Not relevant for this logic
        state_high_vr = MockState(r_eci=np.array([r_norm_at_ap, 0, 0]), v_eci=np.array([v_radial_high, 0, 0]), stage_index=1)
        throttle_cmd = throttle_guidance(100.0, state_high_vr)
        self.assertEqual(throttle_cmd, 0.0)
        self.assertEqual(throttle_guidance.phase, "coast")
        self.assertEqual(len(throttle_guidance.transitions), 2) # No new transition

        # Scenario 2: Still in coast, alt too far, throttle 0.0
        state_far_alt = MockState(r_eci=np.array([r_norm_at_ap + CFG.upper_throttle_alt_tolerance + 10, 0, 0]), 
                                  v_eci=np.array([v_radial_low, 0, 0]), stage_index=1)
        throttle_cmd = throttle_guidance(110.0, state_far_alt)
        self.assertEqual(throttle_cmd, 0.0)
        self.assertEqual(throttle_guidance.phase, "coast")
        self.assertEqual(len(throttle_guidance.transitions), 2)

        # Scenario 3: Transition to circularize, vr low and alt near ra, throttle 1.0
        state_near_ap = MockState(r_eci=np.array([r_norm_at_ap + CFG.upper_throttle_alt_tolerance - 10, 0, 0]), 
                                  v_eci=np.array([v_radial_low, 0, 0]), stage_index=1)
        throttle_cmd = throttle_guidance(120.0, state_near_ap)
        self.assertEqual(throttle_cmd, 1.0)
        self.assertEqual(throttle_guidance.phase, "circularize")
        self.assertEqual(len(throttle_guidance.transitions), 3)
        self.assertEqual(throttle_guidance.transitions[2], ("circularize", 120.0))

    @patch('custom_guidance.orbital_elements_from_state')
    def test_circularize_phase_to_done_transition(self, mock_orbital_elements):
        throttle_guidance = TwoPhaseUpperThrottle(target_radius=7e6, mu=CFG.earth_mu)
        throttle_guidance.phase = "circularize"
        throttle_guidance.transitions.append(("boost", 0.0))
        throttle_guidance.transitions.append(("coast", 10.0)) # Add dummy transitions

        # Scenario 1: Still circularizing, rp too low, throttle 1.0
        mock_orbital_elements.return_value = (None, throttle_guidance.target_radius - CFG.orbit_alt_tol - 100, None) # a, rp, ra
        state = MockState(r_eci=np.array([0,0,0]), v_eci=np.array([0,0,0]), stage_index=1)
        throttle_cmd = throttle_guidance(200.0, state)
        self.assertEqual(throttle_cmd, 1.0)
        self.assertEqual(throttle_guidance.phase, "circularize")
        self.assertEqual(len(throttle_guidance.transitions), 3)

        # Scenario 2: Transition to done, rp high enough, throttle 0.0
        mock_orbital_elements.return_value = (None, throttle_guidance.target_radius - CFG.orbit_alt_tol + 10, None) # a, rp, ra
        throttle_cmd = throttle_guidance(250.0, state)
        self.assertEqual(throttle_cmd, 0.0)
        self.assertEqual(throttle_guidance.phase, "done")
        self.assertEqual(len(throttle_guidance.transitions), 4)
        self.assertEqual(throttle_guidance.transitions[3], ("done", 250.0))

    @patch('custom_guidance.orbital_elements_from_state')
    def test_none_return_from_orbital_elements(self, mock_orbital_elements):
        throttle_guidance = TwoPhaseUpperThrottle(target_radius=7e6, mu=CFG.earth_mu)
        
        # Test boost phase when orbital_elements_from_state returns None
        mock_orbital_elements.return_value = (None, None, None)
        state = MockState(r_eci=np.array([7e6, 0, 0]), v_eci=np.array([0, 10e3, 0]), stage_index=1)
        throttle_cmd = throttle_guidance(10.0, state)
        self.assertEqual(throttle_cmd, 1.0) # Should remain in boost phase (return 1.0)

        throttle_guidance.phase = "coast"
        throttle_guidance.target_ap = 7.5e6
        # Test coast phase when orbital_elements_from_state returns None
        mock_orbital_elements.return_value = (None, None, None)
        state = MockState(r_eci=np.array([7e6, 0, 0]), v_eci=np.array([0, 10e3, 0]), stage_index=1)
        throttle_cmd = throttle_guidance(50.0, state)
        self.assertEqual(throttle_cmd, 0.0) # Should remain in coast phase (return 0.0)

        throttle_guidance.phase = "circularize"
        # Test circularize phase when orbital_elements_from_state returns None
        mock_orbital_elements.return_value = (None, None, None)
        state = MockState(r_eci=np.array([7e6, 0, 0]), v_eci=np.array([0, 10e3, 0]), stage_index=1)
        throttle_cmd = throttle_guidance(100.0, state)
        self.assertEqual(throttle_cmd, 1.0) # Should remain in circularize phase (return 1.0)


if __name__ == '__main__':
    unittest.main()
