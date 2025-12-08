"""
Aerodynamics module structure: drag-only interface.
Implementations are placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np


from atmosphere import AtmosphereModel
from gravity import EarthModel, MU_EARTH, R_EARTH, OMEGA_EARTH


import matplotlib.pyplot as plt

from integrators import State, RK4


def speed_of_sound(T: float) -> float:
    """
    Speed of sound for dry air:
        a = sqrt(gamma * R * T)
    """
    gamma = 1.4
    R = 287.05
    return np.sqrt(gamma * R * T)


class CdModel:
    """
    Drag coefficient model. Accepts either a constant value or a callable
    returning Cd as a function of Mach number.
    """

    def __init__(self, value_or_callable: Union[float, Callable[[float], float]] = 2.0):
        self.value_or_callable = value_or_callable

    def cd(self, mach: float) -> float:
        if isinstance(self.value_or_callable, float):
            return self.value_or_callable
        return float(self.value_or_callable(mach))


@dataclass
class Aerodynamics:
    atmosphere: any
    cd_model: CdModel

    def drag_force(self, state, rocket, earth, t: float) -> np.ndarray:
        """
        Compute aerodynamic drag force (opposite to air-relative velocity).
        """
        """
        Standard drag formula: F_D = -1/2 rho(r) C_D(Mach) A |v_rel|v_rel
        check dimensions, rho is density m/volume, C_D is dimensionless, A is L^2, v_rel is L/T -> mass length /time^2. checks out
        drag coefficient is usually modeled through the mach number which is a quotient of the speed of sound. mach 1 is speed equal to speed of sound. mach 2 double and so on.
        so M = |v_rel|/a(r), where a is the speed of sound as a function of the distance from earth
        
        """
        r = state.r_eci
        v = state.v_eci

        # Altitude above Earth surface
        h = np.linalg.norm(r) - earth.radius
        if h < 0:
            h = 0

        props = self.atmosphere.properties(h, t)
        rho = props.rho

        if rho < 1e-12:  # negligible at high altitudes
            return np.zeros(3)

        # Air velocity from Earth rotation
        v_air = earth.atmosphere_velocity(r)

        # Relative velocity
        v_rel = v - v_air
        v_rel_mag = np.linalg.norm(v_rel)
        if v_rel_mag < 1e-9:
            return np.zeros(3)

        # Mach number
        a = speed_of_sound(props.T)
        mach = v_rel_mag / a

        # Cd
        Cd = self.cd_model.cd(mach)

        # Reference area
        A = rocket.reference_area(state)

        # Dynamic pressure
        q = 0.5 * rho * v_rel_mag**2

        # Drag magnitude
        drag_mag = Cd * A * q

        # Opposite direction of v_rel
        return -drag_mag * (v_rel / v_rel_mag)


"""
test, with minimal rocket class just to check with constant thrust and an object launched w.o thrust
"""


if __name__ == "__main__":

    # -------------------------------------------------------
    # Minimal Rocket class for testing
    # -------------------------------------------------------
    class Rocket:
        def __init__(self, area: float, mass: float, thrust: float = 0.0):
            self._area = area
            self._mass0 = mass
            self._thrust = thrust

        def reference_area(self, state):
            return self._area

        def thrust_force(self, state, t):
            # Always upward thrust along +z for simplicity
            return np.array([0.0, 0.0, self._thrust])

        def mass(self, state, t):
            # Constant mass for this test
            return self._mass0

    # -------------------------------------------------------
    # Derivative function used by integrator
    # -------------------------------------------------------

    def dynamics(t, state: State, rocket, earth: EarthModel, aero: Aerodynamics):
        r = state.r_eci
        v = state.v_eci

        # Gravity
        g = earth.gravity_accel(r)

        # Aerodynamic drag
        drag = aero.drag_force(state, rocket, earth, t)

        # Thrust
        thrust = rocket.thrust_force(state, t)

        # Total force
        F = thrust + drag + rocket.mass(state, t) * g
        a = F / rocket.mass(state, t)

        # No mass change in this test
        dm_dt = 0.0

        return v, a, dm_dt

    # -------------------------------------------------------
    # Helper: run simulation over time interval
    # -------------------------------------------------------

    def run_sim(initial_state: State, rocket, earth, aero, t_final=200, dt=0.1):
        integrator = RK4()

        def wrapped_deriv(t, s):
            drdt, dvdt, dmdt = dynamics(t, s, rocket, earth, aero)
            return drdt, dvdt, dmdt

        t = 0.0
        state = initial_state.copy()

        ts = []
        drags = []

        while t <= t_final:
            # Store drag magnitude
            drag_vec = aero.drag_force(state, rocket, earth, t)
            drag_mag = np.linalg.norm(drag_vec)

            ts.append(t)
            drags.append(drag_mag)

            # Integrate
            state = integrator.step(wrapped_deriv, state, t, dt)
            t += dt

        return np.array(ts), np.array(drags)

    # -------------------------------------------------------
    # RK4 IMPLEMENTATION (needed to run)
    # -------------------------------------------------------

    def rk4_step(state, t, dt, deriv_fn):
        r1, v1, m1 = deriv_fn(t, state)

        s2 = State(state.r_eci + 0.5 * dt * r1,
                   state.v_eci + 0.5 * dt * v1,
                   state.m)
        r2, v2, m2 = deriv_fn(t + 0.5 * dt, s2)

        s3 = State(state.r_eci + 0.5 * dt * r2,
                   state.v_eci + 0.5 * dt * v2,
                   state.m)
        r3, v3, m3 = deriv_fn(t + 0.5 * dt, s3)

        s4 = State(state.r_eci + dt * r3,
                   state.v_eci + dt * v3,
                   state.m)
        r4, v4, m4 = deriv_fn(t + dt, s4)

        r_next = state.r_eci + (dt / 6.0) * (r1 + 2*r2 + 2*r3 + r4)
        v_next = state.v_eci + (dt / 6.0) * (v1 + 2*v2 + 2*v3 + v4)

        return State(r_next, v_next, state.m)

    # Patch RK4 class
    RK4.step = lambda self, deriv, state, t, dt: rk4_step(state, t, dt, deriv)
    State.copy = lambda self: State(
        self.r_eci.copy(), self.v_eci.copy(), self.m, self.stage_index)

    # -------------------------------------------------------
    # MAIN TESTS
    # -------------------------------------------------------
    if __name__ == "__main__":
        # Earth + Atmosphere + Aerodynamics
        earth = EarthModel(mu=MU_EARTH, radius=R_EARTH, omega_vec=OMEGA_EARTH)
        atmosphere = AtmosphereModel()
        aero = Aerodynamics(atmosphere, CdModel(1.0))

        # ---------------------------------------------------
        # Test 1: constant upward velocity launch
        # ---------------------------------------------------
        rocket1 = Rocket(area=1.0, mass=100.0, thrust=0.0)

        initial_state1 = State(
            r_eci=np.array([0.0, 0.0, R_EARTH]),      # on the surface
            v_eci=np.array([0.0, 0.0, 300.0]),        # 300 m/s straight up
            m=100.0
        )

        t1, drag1 = run_sim(initial_state1, rocket1, earth, aero,
                            t_final=300, dt=0.5)

        # ---------------------------------------------------
        # Test 2: accelerating upward (constant thrust)
        # ---------------------------------------------------
        rocket2 = Rocket(area=1.0, mass=100.0, thrust=40000.0)  # ~40 kN

        initial_state2 = State(
            r_eci=np.array([0.0, 0.0, R_EARTH]),
            v_eci=np.array([0.0, 0.0, 0.0]),
            m=100.0
        )

        t2, drag2 = run_sim(initial_state2, rocket2, earth, aero,
                            t_final=200, dt=0.5)

        # ---------------------------------------------------
        # PLOTS (SEPARATE DRAG PLOTS)
        # ---------------------------------------------------

        # ----- Test 1: Constant upward velocity -----
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        ax1.plot(t1, drag1, color="blue")
        ax1.set_ylabel("Drag force [N]")
        ax1.set_xlabel("Time [s]")
        ax1.set_title("Test 1: Drag vs. Time (Constant Upward Velocity)")
        ax1.grid(True)
        plt.tight_layout()
        plt.show()

        # ----- Test 2: Constant thrust upward -----
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.plot(t2, drag2, color="red")
        ax2.set_ylabel("Drag force [N]")
        ax2.set_xlabel("Time [s]")
        ax2.set_title("Test 2: Drag vs. Time (Constant Thrust)")
        ax2.grid(True)
        plt.tight_layout()
        plt.show()
