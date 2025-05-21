import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
plt.rcParams["font.size"] = 9
plt.rcParams["font.family"] = "monospace"
plt.rcParams["lines.linewidth"] = 1
plt.rcParams["grid.color"] = (0.5, 0.5, 0.5, 0.1)    # global [RGB + alpha] override
from tqdm import tqdm
import time
import datetime
from constants import *
from tqdm_pbar import tqdmFA


def initialise_vectors(steps: int, r0: float, v0: float, dim: int = 2, i: float = i_MOON) -> tuple:
    """Initialise position and velocity vectors with initial conditions."""
    r = np.zeros((steps, dim))          # positions
    v = np.zeros((steps, dim))          # velocities
    if dim == 2:
        r[0] = np.array([r0, 0])        # Moon starts on the x-axis
        v[0] = np.array([0.0, v0])      # Moon's velocity is along +y axis
    else:   # dim == 3
        i = np.radians(i)               # inclination angle converted to radians
        r[0] = np.array([r0 * np.cos(i), 0, r0 * np.sin(i)])    # x, y, z
        v[0] = np.array([0, v0 * np.cos(i), v0 * np.sin(i)])    # x, y, z
    return r, v


def euler_method(steps: int, dt: float, r0: float, v0: float, dim: int = 2, i: float = i_MOON) -> tuple:
    """Solve ODE using Euler's method."""
    r, v = initialise_vectors(steps, r0, v0, dim, i)
    t1 = time.time()
    for i in range(steps - 1):
        r_i, v_i = r[i], v[i]
        r_mag = np.linalg.norm(r_i)     # magnitude of all components
        a = -G * M_EARTH * r_i / r_mag**3
        v[i + 1] = v_i + a * dt
        r[i + 1] = r_i + v_i * dt
    t2 = time.time()
    delta = t2 - t1
    rate = (steps / delta) / 1000
    print(f"Euler: {delta:.2f}s for {steps:,} steps ({rate:.1f}K steps/sec)")
    return r, v


def verlet_method(steps: int, dt: float, r0: float, v0: float, dim: int = 2, i: float = i_MOON) -> tuple:
    """Solve ODE using Verlet's method."""
    r, v = initialise_vectors(steps, r0, v0, dim, i)
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


def plot_orbit2d(
    r0: float = D_EARTH_MOON,       # average Earth-Moon distance (m)
    v0: float = V_MOON,             # orbital speed of Moon (m/s)
    time_step_mins: float = 120,                     
    time_periods: float = 1.3,                  
    euler: bool = False,
    verlet: bool = True,
    figure_size: tuple = (10, 10),
    figure_title: str = None,
    earth_markersize: int = 40,
    moon_markersize: int = 11,
    earth_colour: str = "tab:blue",
    moon_colour: str = "tab:grey",
    moon_orbit_colour: str = "tab:grey",
    x_axis_limits: tuple = None,
    y_axis_limits: tuple = None,
    max_axis_extent: float = 1.1,
    show_legend: bool = True,
    to_scale: bool = False
) -> tuple:

    T = 27.3 * 24 * 3600  # Orbital period of the Moon (s)
    dt = 60 * time_step_mins
    print(f"simulating {time_periods} time period(s)... ")
    steps = int(time_periods * T / dt)

    # --- SETUP FIGURE --- #
    fig, ax = plt.subplots(figsize=figure_size)
    if figure_title:
        ax.set_title(figure_title)
    ax.grid(True, alpha=0.15)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # --- SIMULATE TRAJECTORY --- #
    line_width = 0.5 if to_scale else 0.75  # smaller orbit trail linewidth for smaller (accurate sized) model
    if euler:
        r, v = euler_method(steps, dt, r0, v0)
        x, y = r[:, 0], r[:, 1]
        ax.plot(x, y, linestyle="--", linewidth=line_width, color="tab:red", label="Moon Orbit (Euler)")
    if verlet:
        r, v = verlet_method(steps, dt, r0, v0)
        x, y = r[:, 0], r[:, 1]
        ax.plot(x, y, linestyle="-", linewidth=line_width, color=moon_orbit_colour, label="Moon Orbit (Verlet)")

    # --- ADD MARKERS --- #
    if to_scale:
        earth = Circle((0, 0), radius=R_EARTH, color=earth_colour, zorder=10)
        ax.add_patch(earth)
        moon_final = Circle((x[-1], y[-1]), radius=R_MOON, color=moon_colour, zorder=10)
        ax.add_patch(moon_final)
    else:
        ax.plot(0, 0, marker="o", markersize=earth_markersize, color=earth_colour)    # Earth
        ax.plot(x[-1], y[-1], marker="o", markersize=moon_markersize, color=moon_colour)    # Final Moon

    # --- AXIS LIMITS & LEGEND --- #
    # independent overrides (if only one set of limits is provided):
    if x_axis_limits:
        ax.set_xlim(x_axis_limits)
    else:
        x_extent = max_axis_extent * np.max(np.abs(x))
        ax.set_xlim(-x_extent, x_extent)

    if y_axis_limits:
        ax.set_ylim(y_axis_limits)
    else:
        y_extent = max_axis_extent * np.max(np.abs(y))
        ax.set_ylim(-y_extent, y_extent)
    if show_legend:
        ax.legend()
    plt.show()

    return r, v


def animate2d(
    r0: float = D_EARTH_MOON,
    v0: float = V_MOON,
    time_step_mins: int = 120,
    time_periods: float = 1.3,
    trail_length_pct: float = 10,
    frames_per_second: int = 60,
    bit_rate: int = 20_000,
    figure_size: tuple = (10, 10),
    figure_title: str = None,
    earth_markersize: int = 40,
    moon_markersize: int = 11,
    earth_colour: str = "tab:blue",
    moon_colour: str = "tab:grey",
    moon_orbit_colour: str = "tab:grey",
    x_axis_limits: tuple = None,
    y_axis_limits: tuple = None,
    max_axis_extent: float = 1.1,
    show_legend: bool = True,
    to_scale: bool = False,
    create_gif: bool = False
) -> None:
    interval = int(1000 / frames_per_second)    # time between frames (FuncAnimation input parameter in ms)

    # --- SIMULATE ORBIT --- #
    print(f"\ngenerating 2D plot with matplotlib...\n")
    r, v = plot_orbit2d(
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
        x_axis_limits=x_axis_limits,
        y_axis_limits=y_axis_limits,
        max_axis_extent=max_axis_extent,
        show_legend=show_legend,
        to_scale=to_scale
    )

    steps = r.shape[0]
    trail_length = int(trail_length_pct / 100 * steps)
    total_time = steps * (interval * 1e-3)

    print(f"\n{steps:,} steps @ {frames_per_second} fps (~{interval * 1e-3:.3f} sec/frame)")
    print(f"time step (dt): {time_step_mins:,.2f} mins")
    print(f"animation duration: {total_time / 60:.2f} mins ({total_time:,.1f} sec)\n")
    file_ext = "gif" if create_gif else "mp4"
    print(f"writing {steps} frames to {file_ext.upper()}...\n")

    # --- SETUP FIGURE --- #
    fig, ax = plt.subplots(figsize=figure_size)
    if figure_title:
        ax.set_title(figure_title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # Earth marker (remains fixed):
    if to_scale:
        earth = Circle((0, 0), radius=R_EARTH, color=earth_colour, zorder=10)
        ax.add_patch(earth)
    else:
        ax.plot(0, 0, marker="o", markersize=earth_markersize, color=earth_colour)

    # initialise moon marker element:
    line_width = 0.3 if to_scale else 0.75
    if to_scale:
        x0, y0 = r[0, 0], r[0, 1]
        moon_marker = Circle((x0, y0), radius=R_MOON, color=moon_colour, zorder=10)
        ax.add_patch(moon_marker)
    else:
        moon_marker, = ax.plot([], [], marker="o", markersize=moon_markersize, color=moon_colour)
    # initialise moon orbit trail element:
    moon_orbit, = ax.plot([], [], linestyle="-", lw=line_width, color=moon_orbit_colour, label="Moon Orbit")

    # axis limits: independent overrides (if only one set of limits is provided):
    x, y = r[:, 0], r[:, 1]
    if x_axis_limits:
        ax.set_xlim(x_axis_limits)
    else:
        x_extent = max_axis_extent * np.max(np.abs(x))
        ax.set_xlim(-x_extent, x_extent)

    if y_axis_limits:
        ax.set_ylim(y_axis_limits)
    else:
        y_extent = max_axis_extent * np.max(np.abs(y))
        ax.set_ylim(-y_extent, y_extent)
    if show_legend:
        ax.legend()

    # --- PROGRESS BAR --- #
    pbar = tqdmFA(total=steps)

    # --- ANIMATION FUNCTIONS --- #
    def init():
        moon_orbit.set_data([], [])
        if not to_scale:
            moon_marker.set_data([], [])
        else:
            moon_marker.center = (x0, y0)
        return moon_orbit, moon_marker

    def update(frame: int):
        i0 = max(0, frame - trail_length)    # starting index of trail datapoint
        x, y = r[i0:frame + 1, 0], r[i0:frame + 1, 1]
        moon_orbit.set_data(x, y)
        # update moon marker:
        if to_scale:
            moon_marker.center = (x[-1], y[-1])
        else:
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
        f"2D_orbit_"
        f"dt={time_step_mins}mins_"
        f"T={time_periods:.2f}T_"
        f"elev={a}_azim={b}_"
        f"axes={max_axis_extent}_"
        f"v0={v0}ms-1_"
        f".{file_ext}"
    )

    if not create_gif:
        writer = FFMpegWriter(
            fps=frames_per_second,
            bitrate=bit_rate,
        )
    else:
        writer = PillowWriter(
            fps=frames_per_second,
        )
    dpi = 100 if create_gif else 200
    ani.save(filename=file_name, writer=writer, dpi=dpi)
    pbar.close()    # close progress bar

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


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale - one possible solution to 
    Matplotlib's ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    """
    x_lim, y_lim, z_lim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    x_range, y_range, z_range = x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0]
    # all axes will be scaled to the same range as the largest one:
    L = max(x_range, y_range, z_range)  
    x_mid, y_mid, z_mid = np.mean(x_lim), np.mean(y_lim), np.mean(z_lim)    # same as (a + b) / 2
    ax.set_xlim3d([x_mid - L / 2, x_mid + L / 2])
    ax.set_ylim3d([y_mid - L / 2, y_mid + L / 2])
    ax.set_zlim3d([z_mid - L / 2, z_mid + L / 2])


def draw_sphere(ax, c: tuple, r: float, color: str, res: int = 50):
    u = np.linspace(0, 2 * np.pi, res)   # azimuthal angles
    v = np.linspace(0, np.pi, res)       # polar angles
    x = c[0] + r * np.outer(np.cos(u), np.sin(v))
    y = c[1] + r * np.outer(np.sin(u), np.sin(v))
    z = c[2] + r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, linewidth=0, antialiased=False)


def plot_orbit3d(
    r0: float = D_EARTH_MOON,       # average Earth-Moon distance (m)
    v0: float = V_MOON,             # orbital speed of Moon (m/s)
    i: float = i_MOON,
    time_step_mins: float = 120,                     
    time_periods: float = 1.3,                  
    figure_size: tuple = (10, 10),
    figure_title: str = None,
    earth_markersize: int = 3000,
    moon_markersize: int = 100,
    earth_colour: str = "tab:blue",
    moon_colour: str = "tab:grey",
    moon_orbit_colour: str = "tab:grey",
    max_axis_extent: float = 1.1,
    show_legend: bool = True,
    to_scale: bool = False,
    view_angles: tuple = (30, -60)  # default elev/azim angles
) -> tuple:

    T = 27.3 * 24 * 3600  # Orbital period of the Moon (s)
    dt = 60 * time_step_mins
    print(f"simulating {time_periods} time period(s)... ")
    steps = int(time_periods * T / dt)

    # --- SETUP 3D FIGURE --- #
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111, projection="3d")
    if figure_title:
        ax.set_title(figure_title)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

    # --- SIMULATE TRAJECTORY --- #
    r, v = verlet_method(steps, dt, r0, v0, dim=3, i=i)
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    ax.plot3D(x, y, z, color=moon_orbit_colour, linewidth=0.75, label="Moon Orbit", zorder=10)

    # --- ADD MARKERS --- #
    if not to_scale:
        ax.plot(0, 0, 0, marker="o", markersize=earth_markersize, color=earth_colour)    
        ax.plot(x[-1], y[-1], z[-1], marker="o", markersize=moon_markersize, color=moon_colour)    
        # ax.scatter(0, 0, 0, color=earth_colour, s=earth_markersize, zorder=1)               # Earth position
        # ax.scatter(x[-1], y[-1], z[-1], s=moon_markersize, color=moon_colour, zorder=3)     # final Moon position
    else:
        draw_sphere(ax, c=(0, 0, 0), r=R_EARTH, color=earth_colour)
        draw_sphere(ax, c=(x[-1], y[-1], z[-1]), r=R_MOON, color=moon_colour)
        
    # --- AXIS LIMITS & LEGEND --- #
    max_limit = max_axis_extent * np.max(np.abs(r))
    ax.set_xlim3d(-max_limit, max_limit)
    ax.set_ylim3d(-max_limit, max_limit)
    ax.set_zlim3d(-max_limit, max_limit)
    if show_legend:
        ax.legend()
    
    a, b = view_angles
    ax.view_init(elev=a, azim=b)     # default view: elev=30, azim=-60
    plt.tight_layout()
    plt.show()

    return r, v


def animate3d(
    r0: float = D_EARTH_MOON,       # average Earth-Moon distance (m)
    v0: float = V_MOON,             # orbital speed of Moon (m/s)
    i: float = i_MOON,
    time_step_mins: float = 120,                     
    time_periods: float = 1.3,          
    trail_length_pct: float = 10,
    frames_per_second: int = 60,
    bit_rate: int = 20_000,        
    figure_size: tuple = (10, 10),
    figure_title: str = None,
    earth_markersize: int = 3000,
    moon_markersize: int = 100,
    earth_colour: str = "tab:blue",
    moon_colour: str = "tab:grey",
    moon_orbit_colour: str = "tab:grey",
    max_axis_extent: float = 1.1,
    show_legend: bool = True,
    view_angles: tuple = (30, -60),  # default elev/azim angles
    rotate_camera: bool = False,
    dpi: int = 100,
    create_gif: bool = False
) -> None:
    interval = int(1000 / frames_per_second)    # time between frames (FuncAnimation input parameter in ms)

    # --- SIMULATE ORBIT --- #
    print(f"\ngenerating 3D plot with matplotlib...\n")
    r, v = plot_orbit3d(
        r0=r0,
        v0=v0,
        i=i,
        time_step_mins=time_step_mins,
        time_periods=time_periods,
        figure_size=figure_size,
        figure_title=figure_title,
        earth_markersize=earth_markersize,
        moon_markersize=moon_markersize,
        earth_colour=earth_colour,
        moon_colour=moon_colour,
        moon_orbit_colour=moon_orbit_colour,
        max_axis_extent=max_axis_extent,
        show_legend=show_legend,
        view_angles=view_angles
    )

    steps = r.shape[0]
    trail_length = int(trail_length_pct / 100 * steps)
    total_time = steps * (interval * 1e-3)

    print(f"\n{steps:,} steps @ {frames_per_second} fps (~{interval * 1e-3:.3f} sec/frame)")
    print(f"time step (dt): {time_step_mins:,.2f} mins")
    print(f"animation duration: {total_time / 60:.2f} mins ({total_time:,.1f} sec)\n")
    file_ext = "gif" if create_gif else "mp4"
    print(f"writing {steps} frames to {file_ext.upper()}...\n")

    # --- SETUP FIGURE --- #
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111, projection="3d")
    if figure_title:
        ax.set_title(figure_title)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

    # --- AXIS LIMITS & LEGEND --- #
    max_limit = max_axis_extent * np.max(np.abs(r))
    ax.set_xlim3d(-max_limit, max_limit)
    ax.set_ylim3d(-max_limit, max_limit)
    ax.set_zlim3d(-max_limit, max_limit)

    a, b = view_angles               # starting view angles
    ax.view_init(elev=a, azim=b)     # default view: elev=30, azim=-60

    # Earth position marker (remains fixed):
    ax.plot(0, 0, 0, marker="o", markersize=earth_markersize, color=earth_colour)
    # ax.scatter(0, 0, 0, color=earth_colour, s=earth_markersize)
    # initialise moon orbit and marker elements::
    moon_orbit, = ax.plot([], [], [], linestyle="-", lw=0.75, color=moon_orbit_colour, label="Moon Orbit")
    moon_marker, = ax.plot([], [], [], marker="o", markersize=moon_markersize, color=moon_colour)    # final Moon position
    # moon_marker = ax.scatter([], [], [], marker="o", s=moon_markersize, color=moon_colour, label="Moon")
    if show_legend:
        ax.legend()
    
    # --- ANIMATE FUNCTIONS --- #
    def init():
        moon_orbit.set_data_3d([], [], [])
        moon_marker.set_data_3d([], [], [])
        # moon_marker._offsets3d = ([], [], [])
        return moon_orbit, moon_marker

    def update(frame: int):
        i0 = max(0, frame - trail_length)    # starting index of trail datapoint
        x, y, z = r[i0:frame + 1, 0], r[i0:frame + 1, 1], r[i0:frame + 1, 2]
        # update moon orbit trail:
        moon_orbit.set_data_3d(x, y, z)
        # update moon marker:
        moon_marker.set_data_3d([x[-1]], [y[-1]], [z[-1]])
        # moon_marker._offsets3d = ([x[-1]], [y[-1]], [z[-1]])
        # rotate camera view (elevation & azimuth angle):
        if rotate_camera:
            elev = (a + 0.005 * frame) % 360  # slowly spin elevation
            azim = (b - 0.04 * frame) % 360  # slowly spin azimuth
            ax.view_init(elev=elev, azim=azim)
        pbar.update(1)     # update progress bar
        return moon_orbit, moon_marker

    pbar = tqdmFA(total=steps)     # progress bar for logging
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
        f"3D_orbit_"
        f"dt={time_step_mins}mins_"
        f"T={time_periods:.2f}T_"
        f"elev={a}_azim={b}_"
        f"axes={max_axis_extent}_"
        f"v0={v0}ms-1_"
        f"rot={rotate_camera}_"
        f".{file_ext}"
    )

    if not create_gif:
        writer = FFMpegWriter(
            fps=frames_per_second,
            bitrate=bit_rate,
        )
    else:
        writer = PillowWriter(
            fps=frames_per_second,
        )
    dpi = 100 if create_gif else dpi
    ani.save(filename=file_name, writer=writer, dpi=dpi)
    pbar.close()    # close progress bar

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

    # ----- 2D ANIMATION EXAMPLES ----- #

    # Moon Orbit around Earth:
    animate2d(
        time_step_mins=120,
        time_periods=1.1,
        figure_size=(10, 10),
        figure_title="Moon Orbit Around Fixed Earth (TO SCALE)",
        earth_markersize=40,
        moon_markersize=11,
        max_axis_extent_pct=1.05,    # axes 10% larger than the maximum orbit radius
        to_scale=True,
        trail_length_pct=5
    )

    # higher eccentricity elliptical orbit:
    animate2d(
        v0=V_MOON + 278,    # faster initial orbital velocity for the Moon
        time_step_mins=300,
        time_periods=5.5,
        figure_size=(12, 12),
        figure_title="Elliptical Orbit",
        earth_markersize=40,
        moon_markersize=11,
        earth_colour="tab:blue",
        moon_colour="tab:red",
        moon_orbit_colour="tab:red",
        x_axis_limits=(-1.8e9, 5e8),
        trail_length_pct=5,    # test a shorter 5% orbit trail length
        show_legend=False,
    )

    # ----- 3D ANIMATION EXAMPLES ----- #

    # Moon orbit around Earth:
    animate3d(
        time_step_mins=120,
        time_periods=1.02,    # no. of time periods (lunar orbits)
        figure_size=(12, 12),
        figure_title="Moon Orbit Around Fixed Earth (NOT TO SCALE)",
        earth_markersize=35,
        moon_markersize=12,
        earth_colour="tab:blue",
        moon_colour="tab:grey",
        moon_orbit_colour="tab:grey",
        max_axis_extent=0.75,
        trail_length_pct=8,
        show_legend=True,
        view_angles=(20, -40),     # default angles: elev=30, azim=-60
        # rotate_camera=True,
        dpi=200
    )

    animate3d(
        v0=V_MOON + 150,        # faster initial orbital velocity for the Moon
        i=25,                   # inclination angle
        time_step_mins=180,
        time_periods=4.5,      # no. of time periods (lunar orbits)
        figure_size=(12, 12),
        earth_markersize=30,
        moon_markersize=10,
        earth_colour="tab:orange",
        moon_colour="tab:purple",
        moon_orbit_colour="tab:purple",
        max_axis_extent=0.75,
        trail_length_pct=8,
        view_angles=(20, -40),  # default angles: elev=30, azim=-60
        show_legend=False,
        rotate_camera=True,
        dpi=200
    )