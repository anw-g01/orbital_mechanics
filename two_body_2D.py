import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
plt.rcParams["font.size"] = 9
plt.rcParams["font.family"] = "monospace"
plt.rcParams["lines.linewidth"] = 1
from tqdm import tqdm
import time
import datetime
from constants import *


def initialise_vectors(steps: int, r0: float, v0: float) -> tuple:
    """Initialise position and velocity vectors with initial conditions."""
    r = np.zeros((steps, 2))        # positions
    v = np.zeros((steps, 2))        # velocities
    r[0] = np.array([r0, 0])        # step 1: Moon starts on the x-axis
    v[0] = np.array([0.0, v0])      # step 1: Moon's velocity is along +y axis
    return r, v


def euler_method(steps: int, dt: float, r0: float, v0: float) -> tuple:
    """Solve ODE using Euler's method."""
    r, v = initialise_vectors(steps, r0, v0)
    t1 = time.time()
    for i in range(steps - 1):
        r_i, v_i = r[i], v[i]
        r_mag = np.linalg.norm(r_i)
        a = -G * M_EARTH * r_i / r_mag**3
        v[i + 1] = v_i + a * dt
        r[i + 1] = r_i + v_i * dt
    t2 = time.time()
    delta = t2 - t1
    rate = (steps / delta) / 1000
    print(f"Euler: {delta:.2f}s for {steps:,} steps ({rate:.1f}K steps/sec)")
    return r, v


def verlet_method(steps: int, dt: float, r0: float, v0: float) -> tuple:
    """Solve ODE using Verlet's method."""
    r, v = initialise_vectors(steps, r0, v0)
    t1 = time.time()
    for i in range(steps - 1):
        r_i, v_i = r[i], v[i]
        r_mag = np.linalg.norm(r_i)
        a_i = -G * M_EARTH * r_i / r_mag**3     # current acceleration
        # update position:
        r[i + 1] = r_i + v_i * dt + 0.5 * a_i * dt**2
        # compute new acceleration from new position
        r_mag = np.linalg.norm(r[i + 1])
        a_next = -G * M_EARTH * r[i + 1] / r_mag**3
        # update velocity (with average acceleration)
        v[i + 1] = v_i + 0.5 * (a_i + a_next) * dt
    t2 = time.time()
    delta = t2 - t1
    rate = (steps / delta) / 1000
    print(f"Verlet: {delta:.2f}s for {steps:,} steps ({rate:.1f}K steps/sec)")
    return r, v


def plot_orbit(
    r0: float = 3.844e8,                      # (m) initial Moon-Earth distance
    v0: float = 1022,                         # (m/s) initial Moon orbital speed
    time_step_mins: float = 120,                     # (s) time step (default 120 mins)
    time_periods: float = 1.3,                   # no. of time periods (lunar orbits)
    euler: bool = False,
    verlet: bool = True,
    init_moon: bool = False,
    figure_size: tuple = (10, 10),
    figure_title: str = "Moon Orbit Around Earth",
    earth_markersize: int = 40,
    moon_markersize: int = 11,
    earth_colour: str = "tab:blue",
    moon_colour: str = "tab:red",
    moon_orbit_colour: str = "tab:red",
    add_axis_limits: bool = True,
    max_axis_extent_pct: float = 1.1,
    show_legend: bool = True,
) -> tuple:

    T = 27.3 * 24 * 3600  # Orbital period of the Moon (s)
    dt = 60 * time_step_mins
    print(f"simulating {time_periods} time period(s)... ")
    steps = int(time_periods * T / dt)

    # --- SETUP FIGURE --- #
    fig, ax = plt.subplots(figsize=figure_size)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal")
    ax.set_title(figure_title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # --- SIMULATE TRAJECTORY --- #
    if euler:
        r, v = euler_method(steps, dt, r0, v0)
        x, y = r[:, 0], r[:, 1]
        ax.plot(x, y, linestyle="--", linewidth=0.75, color="tab:red", label="Moon Orbit (Euler)")
    if verlet:
        r, v = verlet_method(steps, dt, r0, v0)
        x, y = r[:, 0], r[:, 1]
        ax.plot(x, y, linestyle="-", linewidth=0.75, color=moon_orbit_colour, label="Moon Orbit (Verlet)")

    # --- ADD MARKERS --- #
    ax.plot(0, 0, marker="o", markersize=earth_markersize, color=earth_colour)  # Earth
    if init_moon:
        ax.plot(r0, 0, marker="o", markersize=moon_markersize, color=moon_colour, alpha=0.5)  # Initial Moon
    ax.plot(x[-1], y[-1], marker="o", markersize=moon_markersize, color=moon_colour)  # Final Moon

    # --- AXIS LIMITS & LEGEND --- #
    if add_axis_limits:
        max_limit = max_axis_extent_pct * np.max(np.abs(r))
        ax.set_xlim(-max_limit, max_limit)
        ax.set_ylim(-max_limit, max_limit)
    if show_legend:
        ax.legend()
    plt.show()

    return r, v


if __name__ == "__main__":

    # Euler vs Verlet comparison:
    r, v = plot_orbit(
        time_step_mins=10,                # dt = 10 minutes (time step)
        time_periods=2,                   
        euler=True,
        verlet=True,
        figure_size=(10, 10),
        earth_markersize=40,
        moon_markersize=11,
        earth_colour="tab:blue",
        moon_colour="tab:grey",
        moon_orbit_colour="tab:green",
        add_axis_limits=True,
        max_axis_extent_pct=1.1,
        show_legend=True,
    )

    # Moon Orbit around Earth:
    r, v = plot_orbit(
        time_step_mins=60,
        time_periods=1,                   # no. of time periods (lunar orbits)
        figure_size=(10, 10),
        earth_markersize=40,
        moon_markersize=11,
        earth_colour="tab:blue",
        moon_colour="tab:grey",
        moon_orbit_colour="tab:grey",
        add_axis_limits=True,
        max_axis_extent_pct=1.5,
        show_legend=True,
    )

    # Higher eccentricity elliptical orbit:
    r, v = plot_orbit(
        v0=1200,                            # faster initial orbital velocity
        time_step_mins=240,
        time_periods=4.75,
        figure_size=(12, 12),
        figure_title="Elliptical Orbit",
        earth_markersize=40,
        moon_markersize=11,
        earth_colour="tab:blue",
        moon_colour="tab:red",              # red "Moon"
        moon_orbit_colour="tab:red",
        add_axis_limits=False,
        show_legend=False,
    )