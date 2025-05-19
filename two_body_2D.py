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
from tqdm_pbar import tqdmFA


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
    r0: float = 3.844e8,                # initial Moon-Earth distance (m)
    v0: float = 1022,                   # initial Moon orbital speed (m/s)
    time_step_mins: float = 120,        # time step (default 120 mins) (s)
    time_periods: float = 1.3,          # no. of time periods (lunar orbits)
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


def animate(
    r0: float = 3.844e8,
    v0: float = 1022,
    time_step_mins: int = 120,
    time_periods: float = 1.3,
    trail_length_pct: float = 0.1,
    frames_per_second: int = 60,
    bit_rate: int = 15_000,
    figure_size: tuple = (10, 10),
    figure_title: str = "Moon Orbit Around Earth",
    earth_markersize: int = 40,
    moon_markersize: int = 11,
    earth_colour: str = "tab:blue",
    moon_colour: str = "tab:grey",
    moon_orbit_colour: str = "tab:grey",
    add_axis_limits: bool = True,
    max_axis_extent_pct: float = 1.1,
    show_legend: bool = True,
) -> None:
    interval = int(1000 / frames_per_second)

    # --- SIMULATE ORBIT --- #
    r, v = plot_orbit(
        r0=r0,
        v0=v0,
        time_step_mins=time_step_mins,
        time_periods=time_periods,
        figure_size=figure_size,
        figure_title=figure_title,
        earth_markersize=earth_markersize,
        moon_markersize=moon_markersize,
        earth_colour=earth_colour,
        moon_colour=moon_colour,
        moon_orbit_colour=moon_orbit_colour,
        add_axis_limits=add_axis_limits,
        max_axis_extent_pct=max_axis_extent_pct,
        show_legend=show_legend
    )

    steps = r.shape[0]
    trail_length = int(trail_length_pct * steps)
    total_time = steps * (interval * 1e-3)

    print(f"\n{steps:,} steps @ {frames_per_second} fps (~{interval * 1e-3:.3f} sec/frame)")
    print(f"time step (dt): {time_step_mins:,.2f} mins")
    print(f"animation duration: {total_time / 60:.2f} mins ({total_time:,.1f} sec)\n")
    print("writing frames to file...\n")

    # --- SETUP FIGURE --- #
    fig, ax = plt.subplots(figsize=figure_size)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.set_title(figure_title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ax.plot(0, 0, marker="o", markersize=earth_markersize, color=earth_colour)
    moon_orbit, = ax.plot([], [], linestyle="-", lw=0.75, color=moon_orbit_colour, label="Moon Orbit")
    moon_marker, = ax.plot([], [], marker="o", markersize=moon_markersize, color=moon_colour)

    if add_axis_limits:
        max_extent = max_axis_extent_pct * np.max(np.abs(r))
        ax.set_xlim(-max_extent, max_extent)
        ax.set_ylim(-max_extent, max_extent)
    if show_legend:
        ax.legend()

    # --- PROGRESS BAR --- #
    pbar = tqdmFA(total=steps)

    # --- ANIMATION FUNCTIONS --- #
    def init():
        moon_orbit.set_data([], [])
        moon_marker.set_data([], [])
        return moon_orbit, moon_marker

    def update(frame: int):
        i0 = max(0, frame - trail_length)                   # index a no. of datapoints behind current frame
        x, y = r[i0:frame + 1, 0], r[i0:frame + 1, 1]       # create a "trail" behind the orbiting body
        moon_orbit.set_data(x, y)
        moon_marker.set_data([x[-1]], [y[-1]])
        pbar.update(1)
        return moon_orbit, moon_marker

    ani = FuncAnimation(
        fig=fig,
        func=update,
        frames=range(steps),
        init_func=init,
        interval=interval,
        repeat=False,
        blit=True,
    )

    # --- SAVE ANIMATION --- #
    file_name = (
        f"2D_orbit_{frames_per_second}fps_"
        f"{time_periods:.1f}T_{time_step_mins}mins_dt_"
        f"{v0}ms-1_v0_{bit_rate}kbps.mp4"
    )

    writer = FFMpegWriter(
        fps=frames_per_second,
        bitrate=bit_rate,
        metadata=dict(artist="anw"),
    )

    ani.save(filename=file_name, writer=writer)

    # --- REPORT --- #
    elapsed = int(pbar.format_dict["elapsed"])
    t = datetime.timedelta(seconds=elapsed)
    print(f"\n\ntotal elapsed time: {t}")

    avg_iter_per_sec = steps / t.total_seconds()
    if 1 / avg_iter_per_sec < 1:
        avg_rate = f"{1 / avg_iter_per_sec * 1e3:.0f} ms/frame"
    else:
        avg_rate = f"{1 / avg_iter_per_sec:.2f} sec/frame"
    print(f"{avg_iter_per_sec:.1f} frames/sec processed ({avg_rate})")

    return None


if __name__ == "__main__":

    # ----- PLOT ORBITS ----- #

    # Euler vs Verlet comparison:
    r, v = plot_orbit(
        time_step_mins=10,    # dt = 10 minutes (time step)
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

    # # Moon Orbit around Earth:
    r, v = plot_orbit(
        time_step_mins=60,
        time_periods=1,                 
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

    # # Higher eccentricity elliptical orbit:
    r, v = plot_orbit(
        v0=1200,    # faster initial orbital velocity
        time_step_mins=240,
        time_periods=4.75,
        figure_size=(12, 12),
        figure_title="Elliptical Orbit",
        earth_markersize=40,
        moon_markersize=11,
        earth_colour="tab:blue",
        moon_colour="tab:red",    # red "Moon"
        moon_orbit_colour="tab:red",
        add_axis_limits=False,
        show_legend=False,
    )

    # ----- ANIMATE ORBITAL MOTION ----- #

    # Moon Orbit around Earth:
    animate(
        time_step_mins=120,
        time_periods=1.1,
        figure_size=(10, 10),
        figure_title="Moon Orbit Around Earth",
        earth_markersize=40,
        moon_markersize=11,
        earth_colour="tab:blue",
        moon_colour="tab:grey",
        moon_orbit_colour="tab:grey",
        add_axis_limits=True,
        max_axis_extent_pct=1.1,    # axes 10% larger than the maximum orbit radius
        show_legend=True,
        bit_rate=20_000,
    )

    # Higher eccentricity elliptical orbit:
    animate(
        v0=1275,                            # faster initial orbital velocity for the Moon
        time_step_mins=240,
        time_periods=6,
        figure_size=(12, 12),
        figure_title="Elliptical Orbit",
        earth_markersize=40,
        moon_markersize=11,
        earth_colour="tab:blue",
        moon_colour="tab:red",
        moon_orbit_colour="tab:red",
        add_axis_limits=True,
        show_legend=False,
        bit_rate=20_000,
    )

    