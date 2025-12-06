import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

#constants

G = 6.67430e-11  # gravitational constant
M_earth = 5.972e24 # earth mass
omega_earth = 7.2921159e-5   # rad/s (Earth rotation rate)

omega_vec = np.array([0.0, 0.0, omega_earth])  # rotation about +z (Earth convention)


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



# 2 ways to add, one is way simpler. Either make earth rotate by adding a
# a rotational frame, or just add a tangential velocity if the initial position
# is at the earths surface. Meaning that we compute the sum of the xyz
# components and if they sum up to the earths radius, then we add a tangential
# velocity component based on the earths rotation. 





def gravitational_acceleration(pos):
    """Return the gravitational acceleration vector at position pos."""
    r = np.linalg.norm(pos) # returns absolute distance from origin.
    return -G * M_earth * pos / r**3

def launch_site_rotational_velocity(pos_launch, omega_vec=omega_vec):
    # v = Omega x r
    return np.cross(omega_vec, pos_launch)  #cross product




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

    # choose launch latitude (deg) and altitude (m)
    lat_deg = 0.0               # 0 = equator, 90 = north pole
    altitude = 0.0              # altitude above Earth's surface
    R_earth = 6371000.0         # meters (approx)
    m = 1000  # satellite mass [kg]

    # compute initial position in ECEF-like coordinates (simple)
    lat = np.deg2rad(lat_deg)
    # place launch on prime meridian for simplicity: longitude = 0
    x = (R_earth + altitude) * np.cos(lat)
    y = 0.0
    z = (R_earth + altitude) * np.sin(lat)
    pos0 = np.array([x, y, z])

    # base launch velocity you impart relative to ground (e.g. pointing eastward)
    # Example: initial burn gives 9500 m/s in local east direction (you choose)
    # Here we create a local East unit vector (for lon=0):
    east = np.array([ -np.sin(0.0), np.cos(0.0), 0.0 ])  # lon=0 -> east points +y
    
    burn_speed = 9500.0    # m/s (example)
    vel_rel_ground = burn_speed * east

    # add Earth's rotation velocity at the launch site
    v_rot = launch_site_rotational_velocity(pos0)   # Omega x r

    # inertial initial velocity = ground-relative burn + rotation velocity
    vel0_inertial = vel_rel_ground + v_rot

    print("launch lat:", lat_deg, "deg")
    print("v_rot (m/s):", v_rot)
    print("initial inertial speed (m/s):", np.linalg.norm(vel0_inertial))


    dt = 1.0
    steps = 20000

    positions, velocities = simulate_orbit(pos0, vel0_inertial, m, dt, steps)

    animate_orbit(positions)

