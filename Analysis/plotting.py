"""
Functions for plotting and animating simulation trajectories.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import animation

from Main.telemetry import Logger


def plot_trajectory_3d(log: Logger, r_earth: float):
    """Static 3D plot of trajectory around a spherical Earth."""
    positions = np.array(log.r)
    times = np.array(log.t_sim)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Earth sphere (light, semi-transparent) and wireframe for context
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x = r_earth * np.outer(np.cos(u), np.sin(v))
    y = r_earth * np.outer(np.sin(u), np.sin(v))
    z = r_earth * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, cmap="Blues", alpha=0.1, linewidth=0, antialiased=False)
    ax.plot_wireframe(x, y, z, color="lightblue", alpha=0.2, linewidth=0.3)

    # Trajectory with time-based color gradient
    if positions.shape[0] > 1:
        segments = np.stack([positions[:-1], positions[1:]], axis=1)
        t_norm = (times - times.min()) / max(times.ptp(), 1e-9)
        colors = plt.cm.plasma(t_norm[:-1])
        lc = Line3DCollection(segments, colors=colors, linewidths=2, label="Trajectory")
        ax.add_collection3d(lc)
    else:
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color="tab:red", label="Trajectory", lw=2)
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color="green", s=30, label="Launch")
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color="black", s=30, label="Final")

    # Symmetric limits based on max radial distance
    r_max = max(np.linalg.norm(p) for p in positions)
    lim = 1.05 * max(r_earth, r_max)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=35)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("3D Trajectory")
    plt.legend()
    plt.tight_layout()
    plt.show()


def animate_trajectory(log: Logger, r_earth: float):
    """Simple 3D animation of the trajectory."""
    positions = np.array(log.r)
    if positions.shape[0] < 2:
        return

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Earth sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x = r_earth * np.outer(np.cos(u), np.sin(v))
    y = r_earth * np.outer(np.sin(u), np.sin(v))
    z = r_earth * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, cmap="Blues", alpha=0.05, linewidth=0, antialiased=False)
    ax.plot_wireframe(x, y, z, color="lightblue", alpha=0.2, linewidth=0.3)

    traj_line, = ax.plot([], [], [], color="tab:red", lw=2)
    point, = ax.plot([], [], [], "o", color="black", markersize=5)

    r_max = max(np.linalg.norm(p) for p in positions)
    lim = 1.05 * max(r_earth, r_max)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=35)

    def init():
        traj_line.set_data([], [])
        traj_line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        return traj_line, point

    def update(frame):
        traj_line.set_data(positions[: frame + 1, 0], positions[: frame + 1, 1])
        traj_line.set_3d_properties(positions[: frame + 1, 2])
        point.set_data([positions[frame, 0]], [positions[frame, 1]])
        point.set_3d_properties([positions[frame, 2]])
        return traj_line, point

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(positions),
        init_func=init,
        interval=50,
        blit=True,
        repeat=False,
    )
    ax.set_title("Trajectory Animation")
    plt.show()