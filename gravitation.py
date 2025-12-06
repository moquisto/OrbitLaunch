import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

#constants

G = 6.67430e-11  # gravitational constant
M_earth = 5.972e24 # earth mass

#info

#6378 kilometers = earth radius

#low earth orbit satellite 180 - 2000 kilometers above sea level
#MEO satellite 2000 - 35800 above sea level
#geostationary satellite 35780 above sea level or greater
#42164 km geosynchronous orbit since orbit speed matches earths rotation
#also called geostationary satellites. 
#after HEOs there are also the lagrange points where the pull from earth
#and the sun is cancelled out - 2 stable points, 3 unstable points.

#really good link https://earthobservatory.nasa.gov/features/OrbitsCatalog

#energy to orbit is dependent on height and inclination of the orbit
#a polar orbit requires more energy than one over the equator
#this is due to the earths momentum





def gravitational_acceleration(pos):
    """Return the gravitational acceleration vector at position pos."""
    r = np.linalg.norm(pos) # returns absolute distance from origin.
    return -G * M_earth * pos / r**3


def velocity_verlet_step(pos, vel, dt):
    """Perform one Velocity-Verlet integration step."""
    #returns position and velocity - hamiltonian ish
    a1 = gravitational_acceleration(pos)
    pos_new = pos + vel * dt + 0.5 * a1 * dt**2
    a2 = gravitational_acceleration(pos_new)
    vel_new = vel + 0.5 * (a1 + a2) * dt
    return pos_new, vel_new


# --------------------------------------------------------
# Simulation function
# --------------------------------------------------------

def simulate_orbit(pos0, vel0, m, dt=1, steps=20000):
    """Simulate 3D orbital motion."""

    pos = np.zeros((steps, 3))
    vel = np.zeros((steps, 3))

    pos[0] = pos0
    vel[0] = vel0

    for i in range(steps - 1):
        pos[i+1], vel[i+1] = velocity_verlet_step(pos[i], vel[i], dt)

    return pos, vel


# --------------------------------------------------------
# Animation
# --------------------------------------------------------

def animate_orbit(positions, interval=10):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Earth
    ax.scatter(0, 0, 0, color='blue', s=80)

    # Trajectory and moving point
    line, = ax.plot([], [], [], lw=1)
    point, = ax.plot([], [], [], marker='o', markersize=5, color='red')

    max_range = np.max(np.abs(positions))
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    def update(i):
        line.set_data(positions[:i, 0], positions[:i, 1])
        line.set_3d_properties(positions[:i, 2])

        point.set_data([positions[i, 0]], [positions[i, 1]])
        point.set_3d_properties([positions[i, 2]])

        return line, point

    ani = FuncAnimation(fig, update, frames=len(positions),
                        interval=interval, blit=False)

    plt.show()


if __name__ == "__main__":

    # Satellite parameters (adjust freely)
    #around 99 mins for nasas low earth orbit satellite to cycle
    #99*60 = 6000-60 = 5940 seconds - this one should orbit at least once or twice
    m = 1000  # satellite mass [kg]
  
    pos0 = np.array([7e6, 0, 0])       # 7000 km from Earth center
    vel0 = np.array([0, 7500, 0])      # roughly LEO orbital speed

    dt = 1.0
    steps = 20000

    positions, velocities = simulate_orbit(pos0, vel0, m, dt, steps)

    animate_orbit(positions)
