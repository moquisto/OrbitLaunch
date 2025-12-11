"""
This file is for user-defined guidance functions.
You can create your own pitch and throttle programs here and then
reference them by name in the `config.py` file.

These functions are provided as examples and are the original, hardcoded
logic from the simulation.
"""

from __future__ import annotations
from typing import Tuple

import numpy as np

# These imports are necessary for the example functions.
# Your own functions may need more or fewer.
from integrators import State
from config import CFG

# This helper function is required by the example TwoPhaseUpperThrottle.
def orbital_elements_from_state(r_vec, v_vec, mu: float) -> Tuple[float | None, float | None, float | None]:
    """Return semi-major axis and perigee/apoapsis radii for a two-body orbit."""
    r = np.asarray(r_vec, dtype=float)
    v = np.asarray(v_vec, dtype=float)
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    if r_norm == 0.0 or mu <= 0.0:
        return None, None, None
    # Specific mechanical energy
    eps = 0.5 * v_norm**2 - mu / r_norm
    if abs(eps) < 1e-9: # Parabolic orbit, semi-major axis is infinite
        return None, None, None
    a = -mu / (2.0 * eps)
    # Eccentricity vector
    h_vec = np.cross(r, v)
    h_norm_sq = np.dot(h_vec, h_vec)
    if h_norm_sq == 0.0: # Radial trajectory
        return a, 0.0, abs(r_norm)
    e_vec = (np.cross(v, h_vec) / mu) - (r / r_norm)
    e = np.linalg.norm(e_vec)
    
    # Perigee and apoapsis
    rp = a * (1.0 - e)
    ra = a * (1.0 + e)
    return a, rp, ra


# --- Example Pitch Program ---

def simple_pitch_program(t: float, state: State, cfg_instance=CFG) -> np.ndarray:
    """
    Original gravity-turn inspired pitch program.
    Uses start/end altitudes from config.
    """
    r = np.asarray(state.r_eci, dtype=float)
    v = np.asarray(state.v_eci, dtype=float)
    r_norm = np.linalg.norm(r)
    if r_norm == 0.0:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    r_hat = r / r_norm

    east = np.cross([0.0, 0.0, 1.0], r_hat)
    east_norm = np.linalg.norm(east)
    east = east / east_norm if east_norm > 0.0 else np.array([1.0, 0.0, 0.0], dtype=float)

    alt = r_norm - cfg_instance.central_body.earth_radius_m
    
    start_alt = cfg_instance.pitch_guidance.pitch_turn_start_m
    end_alt = cfg_instance.pitch_guidance.pitch_turn_end_m
    
    if alt < start_alt:
        initial_pitch_rad = np.deg2rad(cfg_instance.pitch_guidance.initial_pitch_deg)
        # Blend r_hat (vertical) and east (horizontal) based on initial_pitch_deg
        direction = np.sin(initial_pitch_rad) * r_hat + np.cos(initial_pitch_rad) * east
        n = np.linalg.norm(direction)
        return direction / n if n > 0.0 else r_hat
    elif alt < end_alt:
        w = ((alt - start_alt) / max(end_alt - start_alt, 1.0)) ** cfg_instance.pitch_guidance.pitch_blend_exp
        direction = (1.0 - w) * r_hat + w * east
    else:
        speed = np.linalg.norm(v)
        if speed > 100.0: # Prograde threshold
            direction = v / speed  # prograde
        else:
            direction = east
            
    n = np.linalg.norm(direction)
    return direction / n if n > 0.0 else r_hat


# --- Example Throttle Program ---

class TwoPhaseUpperThrottle:
    """
    Original simple two-phase upper-stage throttle: boost to target apoapsis,
    coast to apoapsis, then circularize to raise perigee.
    """

    def __init__(self, target_radius: float, mu: float, cfg=None):
        self.target_radius = target_radius
        self.mu = mu
        self.cfg = cfg or CFG
        self.phase = "boost"
        self.target_ap = self.target_radius * self.cfg.throttle_guidance.upper_stage_first_burn_target_ap_factor
        self.transitions: list[tuple[str, float]] = [("boost", 0.0)]
        self.first_burn_throttle = self.cfg.throttle_guidance.upper_stage_first_burn_throttle_setpoint
        self.circ_burn_throttle = self.cfg.throttle_guidance.upper_stage_circ_burn_throttle_setpoint

    def __call__(self, t: float, state: State) -> float:
        stage_idx = getattr(state, "stage_index", 0)
        if stage_idx == 0:
            return 1.0  # booster always full throttle here

        r = np.asarray(state.r_eci, dtype=float)
        v = np.asarray(state.v_eci, dtype=float)
        a, rp, ra = orbital_elements_from_state(r, v, self.mu)
        r_norm = np.linalg.norm(r)
        vr = float(np.dot(v, r / r_norm)) if r_norm > 0 else 0.0

        if self.phase == "boost":
            if ra is not None and ra >= self.target_ap:
                self.phase = "coast"
                self.transitions.append(("coast", t))
                return 0.0
            return self.first_burn_throttle

        if self.phase == "coast":
            if ra is None:
                return 0.0
            # Coast until near apoapsis (radial velocity ~0 and radius near ra)
            if abs(vr) < self.cfg.throttle_guidance.upper_throttle_vr_tolerance and abs(r_norm - ra) < self.cfg.throttle_guidance.upper_throttle_alt_tolerance:
                self.phase = "circularize"
                self.transitions.append(("circularize", t))
                return self.circ_burn_throttle
            return 0.0

        if self.phase == "circularize":
            if rp is not None and rp >= self.target_radius - self.cfg.orbit_tolerances.orbit_alt_tol:
                self.phase = "done"
                self.transitions.append(("done", t))
                return 0.0
            return self.circ_burn_throttle

        return 0.0
