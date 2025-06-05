import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator
plt.rcParams["font.size"] = 8
plt.rcParams["font.family"] = "monospace"
plt.rcParams["lines.linewidth"] = 1
from matplotlib.animation import FuncAnimation, FFMpegWriter
import datetime
from constants import *
from tqdm_pbar import tqdmFA


def two_body_system(
    m1: float = M_EARTH,
    m2: float = M_MOON,
    d: float = D_EARTH_MOON,
    v0: float = V_MOON,
    i_deg: float = i_MOON,
    T_days: float = 27.321,
    steps: int = 1000,
    ode_method: str = "RK45",
    rtol: float = 1e-3,
    atol: float = 1e-6
) -> tuple:

    # initial position vectors of mass 1 and mass 2:
    i_rad = np.radians(i_deg)   # convert inclination angle to radians
    r2_0 = np.array([d * np.cos(i_rad), 0.0, d * np.sin(i_rad)])      
    r1_0 = -m2/m1 * r2_0        # position vector of Earth (with barycentre at origin)                                     

    # initial velocity vectors:
    v0 = np.array([0.0, v0, 0.0])
    v1_0 = -m2/(m1 + m2) * v0
    v2_0 = m1/(m1 + m2) * v0

    # initial state vector, [r_e, v_e, r_m, v_m]:
    Z0 = np.concatenate([
        r1_0,
        v1_0,
        r2_0,
        v2_0
    ])

    def func(t, Z):
        r1, v1, r2, v2 = np.split(Z, 4)             # unpack state vector (4 equal sub-arrays)
        r = r2 - r1                                 # relative position vector of Moon w.r.t. Earth
        r_mag = np.linalg.norm(r)                   # magnitude (Earth-Moon distance)
        F = ( G*m1*m2 / r_mag**3 ) * r              # force vector (Newton's law of gravitation)
        a1 = F / m1                                 # acceleration of Earth
        a2 = -F / m2                                # acceleration of Moon (equal and opposite)
        return np.concatenate([v1, a1, v2, a2])     # return the time derivative of the state vector

    # ----- EVALUATION & SOLVE ----- #
    orbital_period = 2 * np.pi * np.sqrt(d**3 / (G * (m1 + m2)))    # orbital period of the Earth-Moon system
    T = T_days * 24 * 3600                                          # one lunar orbit (s)
    t_span = (0, T) if T > 0 else (0, orbital_period) 
    t_eval = np.linspace(t_span[0], t_span[1], steps)               # time points at which to store the solution
    dt = t_eval[1] - t_eval[0]                                      # time eval step size

    print(f"\nrunning ODE solver ({ode_method=})...")
    print(f"using {rtol=:.0e}, {atol=:.0e} (default: rtol=1e-3, atol=1e-6)")
    print(f"t_eval=({t_eval[0]:.0f}, {t_eval[-1]:,.0f}), {steps=:,}, dt≈{dt:.2f}")
    sol = solve_ivp(func, t_span, Z0, t_eval=t_eval, method=ode_method, rtol=rtol, atol=atol)

    # ----- EXTRACT RESULTS ----- #
    t, Z = sol.t, sol.y
    success = sol.success
    print(f"\nsolver success: {success} ({sol.nfev:,} nfev)")
    print(f"t.shape: {t.shape}, Z.shape: {Z.shape}\n")
    p = [m1, m2, d, v0, Z0, T]    # parameters for plotting and/or analysis

    return t, Z, p


def plot_orbits2d(
    t, Z, p,
    m1_colour: str = "tab:blue",
    m1_trail_colour: str = "tab:blue",
    m2_colour: str = "tab:grey",
    m2_trail_colour: str = "tab:grey",
    m1_markersize: int = 1000,
    m2_markersize: int = 100,
    figure_size: tuple = (10, 10),
    figure_title: str = None,
    line_width: float = 0.75,
    grid_alpha: float = 0.15,
    x_axis_max_ticks: int = 5,
    y_axis_max_ticks: int = 5,
    x_axis_limits: tuple = None,
    y_axis_limits: tuple = None,
    max_axis_extent: float = 1.05,
    show_legend: bool = False,
    to_scale: bool = False,
    radius1: float = R_EARTH,
    body1_label: str = None,
    radius2: float = R_MOON,
    body2_label: str = None,
    show_barycentre: bool = False
) -> None:

    r1, v1, r2, v2 = np.vsplit(Z, 4)    # unpack state vector (split along rows)
    x1, y1 = r1[0, :], r1[1, :]         # unpack body 1 2D coordinates
    x2, y2 = r2[0, :], r2[1, :]         # unpack body 2 2D coordinates  

    # ----- FIGURE ----- #
    fig, ax = plt.subplots(figsize=figure_size)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MaxNLocator(x_axis_max_ticks))
    ax.yaxis.set_major_locator(MaxNLocator(y_axis_max_ticks))
    if figure_title:
        ax.set_title(figure_title)
    ax.set_xlabel(r"$x$ ($m$)")
    ax.set_ylabel(r"$y$ ($m$)")
    # grids and dashed lines
    ax.grid(True, alpha=grid_alpha)
    if show_barycentre:     # barycentre lies at the origin
        dashed_alpha, dashed_linewidth = 0.1, 0.8
        hline = ax.axhline(0, linestyle="--", color="black", alpha=dashed_alpha, linewidth=dashed_linewidth, zorder=1)
        vline = ax.axvline(0, linestyle="--", color="black", alpha=dashed_alpha, linewidth=dashed_linewidth, zorder=1)
        hline.set_dashes([10, 10]), vline.set_dashes([10, 10])
        ax.scatter(0, 0, marker="x", s=25, color="tab:red", label="barycentre", alpha=0.4, zorder=10)

    # --- PLOT EARTH & MOON ORBITS --- #
    line_width = 0.65 if to_scale else line_width
    ax.plot(x1, y1, color=m1_trail_colour, linewidth=line_width)
    ax.plot(x2, y2, color=m2_trail_colour, linewidth=line_width)

    # --- ADD MARKERS --- #
    if to_scale:    # show Earth & Moon to scale
        body1 = Circle((x1[-1], y1[-1]), radius=radius1, color=m1_colour, label=body1_label, zorder=5)
        body2 = Circle((x2[-1], y2[-1]), radius=radius2, color=m2_colour, label=body2_label, zorder=5)
        ax.add_patch(body1), ax.add_patch(body2)
    else:
        ax.scatter(x1[-1], y1[-1], color=m1_colour, s=m1_markersize)
        ax.scatter(x2[-1], y2[-1], color=m2_colour, s=m2_markersize)

    # --- AXIS LIMITS & LEGEND --- #
    # independent overrides (if only one set of limits is provided):
    if x_axis_limits:
        ax.set_xlim(x_axis_limits)
    else:
        x_all = np.concatenate([x1, x2])
        x_extent = max_axis_extent * np.max(np.abs(x_all))
        ax.set_xlim(-x_extent, x_extent)
    if y_axis_limits:
        ax.set_ylim(y_axis_limits)
    else:
        y_all = np.concatenate([y1, y2])
        y_extent = max_axis_extent * np.max(np.abs(y_all))
        ax.set_ylim(-y_extent, y_extent)
    if show_legend:
        ax.legend()
    plt.show()


def animate2d(
    t, Z, p,
    m1_colour: str = "tab:blue",
    m1_trail_colour: str = "tab:blue",
    m2_colour: str = "tab:grey",
    m2_trail_colour: str = "tab:grey",
    m1_markersize: int = 1000,
    m2_markersize: int = 100,
    figure_size: tuple = (10, 10),
    figure_title: str = None,
    line_width: float = 0.75,
    grid_alpha: float = 0.15,
    x_axis_max_ticks: int = 5,
    y_axis_max_ticks: int = 5,
    x_axis_limits: tuple = None,
    y_axis_limits: tuple = None,
    max_axis_extent: float = 1.05,
    show_legend: bool = False,
    to_scale: bool = False,
    radius1: float = R_EARTH,
    body1_label: str = None,
    radius2: float = R_MOON,
    body2_label: str = None,
    show_barycentre: bool = False,
    # params for animation writing:
    frames_per_second: int = 60,
    trail_length_pct: float = 10,  
    dots_per_inch: int = 200,
) -> None:

    print(f"\nPlotting total solution of animation...\n")
    print("<CLOSE FIGURE WINDOW TO START ANIMATION WRITING>\n")

    plot_orbits2d(
            t, Z, p,
            m1_colour=m1_colour, m1_trail_colour=m1_trail_colour,
            m2_colour=m2_colour, m2_trail_colour=m2_trail_colour,
            m1_markersize=m1_markersize, m2_markersize=m2_markersize,
            figure_size=figure_size, figure_title=figure_title,
            line_width=line_width, grid_alpha=grid_alpha,
            x_axis_max_ticks=x_axis_max_ticks, y_axis_max_ticks=y_axis_max_ticks,
            x_axis_limits=x_axis_limits, y_axis_limits=y_axis_limits, max_axis_extent=max_axis_extent,
            show_legend=show_legend, to_scale=to_scale,
            radius1=radius1, body1_label=body1_label, radius2=radius2, body2_label=body2_label,
            show_barycentre=show_barycentre
        )
    interval = int(1000 / frames_per_second)                # convert FPS to milliseconds
    steps = len(t)                                          # total number of time steps from the ODE solver
    trail_length = int((trail_length_pct / 100) * steps) 
    print(f"\n{steps:,} steps @ {frames_per_second} fps (~{interval * 1e-3:.3f} sec/frame)")
    print(f"writing {steps} frames to MP4...\n")

    # ----- EXTRACT COORDINATES ----- #
    r1, v1, r2, v2 = np.vsplit(Z, 4)    # unpack state vector (split along rows)
    x1, y1 = r1[0, :], r1[1, :]         # unpack body 1 2D coordinates
    x2, y2 = r2[0, :], r2[1, :]         # unpack body 2 2D coordinates  

    # ----- FIGURE ----- #
    fig, ax = plt.subplots(figsize=figure_size)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MaxNLocator(x_axis_max_ticks))
    ax.yaxis.set_major_locator(MaxNLocator(y_axis_max_ticks))
    if figure_title:
        ax.set_title(figure_title)
    ax.set_xlabel(r"$x$ ($m$)")
    ax.set_ylabel(r"$y$ ($m$)")
    # grids and dashed lines
    ax.grid(True, alpha=grid_alpha)
    if show_barycentre:     # barycentre lies at the origin
        dashed_alpha, dashed_linewidth = 0.1, 0.8
        hline = ax.axhline(0, linestyle="--", color="black", alpha=dashed_alpha, linewidth=dashed_linewidth, zorder=1)
        vline = ax.axvline(0, linestyle="--", color="black", alpha=dashed_alpha, linewidth=dashed_linewidth, zorder=1)
        hline.set_dashes([10, 10]), vline.set_dashes([10, 10])
        ax.scatter(0, 0, marker="x", s=25, color="tab:red", label="barycentre", alpha=0.4, zorder=10)

    # --- PLOT ELEMENTS (TO BE UPDATED IN ANIMATION) --- #
    if to_scale:    # show Earth & Moon to scale
        body1 = Circle((x1[0], y1[0]), radius=radius1, color=m1_colour, label=body1_label, zorder=5)
        body2 = Circle((x2[0], y2[0]), radius=radius2, color=m2_colour, label=body2_label, zorder=5)
        ax.add_patch(body1), ax.add_patch(body2)
    else:
        m1_marker = ax.scatter([], [], color=m1_colour, s=m1_markersize, zorder=5)
        m2_marker = ax.scatter([], [], color=m2_colour, s=m2_markersize, zorder=5)
    line_width = 0.65 if to_scale else line_width
    m1_orbit, = ax.plot([], [], color=m1_trail_colour, linewidth=line_width)
    m2_orbit, = ax.plot([], [], color=m2_trail_colour, linewidth=line_width)

    # --- AXIS LIMITS & LEGEND --- #
    # independent overrides (if only one set of limits is provided):
    if x_axis_limits:
        ax.set_xlim(x_axis_limits)
    else:
        x_all = np.concatenate([x1, x2])
        x_extent = max_axis_extent * np.max(np.abs(x_all))
        ax.set_xlim(-x_extent, x_extent)
    if y_axis_limits:
        ax.set_ylim(y_axis_limits)
    else:
        y_all = np.concatenate([y1, y2])
        y_extent = max_axis_extent * np.max(np.abs(y_all))
        ax.set_ylim(-y_extent, y_extent)
    if show_legend:
        ax.legend()

    # --- PROGRESS BAR --- #
    pbar = tqdmFA(total=len(t))

    # ----- ANIMATION FUNCTION SETUP ----- #
    def init():
        m1_orbit.set_data([], []), m2_orbit.set_data([], [])
        if to_scale:
            body1.center, body2.center = (x1[0], y1[0]), (x2[0], y2[0])
            return m1_orbit, m2_orbit, body1, body2
        else:
            m1_marker.set_offsets((x1[0], y1[0])), m2_marker.set_offsets((x2[0], y2[0]))
            return m1_orbit, m2_orbit, m1_marker, m2_marker

    def update(frame):
        # update orbit trails
        i0 = max(0, frame - trail_length)  # start index for the trail
        m1_orbit.set_data(x1[i0: frame], y1[i0: frame])
        m2_orbit.set_data(x2[i0: frame], y2[i0: frame])
        # update marker positions
        if to_scale:
            body1.center, body2.center = (x1[frame], y1[frame]), (x2[frame], y2[frame])
            pbar.update(1)
            return m1_orbit, m2_orbit, body1, body2
        else:
            m1_marker.set_offsets((x1[frame], y1[frame]))
            m2_marker.set_offsets((x2[frame], y2[frame]))
            pbar.update(1)
            return m1_orbit, m2_orbit, m1_marker, m2_marker

    ani = FuncAnimation(
        fig=fig,
        func=update,
        frames=range(len(t)),
        init_func=init,
        interval=interval,
        blit=True,
    )

    writer = FFMpegWriter(
        fps=frames_per_second,
        bitrate=30_000,
    )

    file_name = (
        "2D_barycentric_"
        f"{figure_title.replace(' ', '_')}_"
        f"{to_scale=}_"
        f"{frames_per_second}fps_"
        f"{trail_length_pct}%trail_"
        f"{len(t):,}steps_"
        f"{body1_label.lower()}_{body2_label.lower()}_"
        ".mp4"
    )
    ani.save(filename=file_name, writer=writer, dpi=dots_per_inch)
    pbar.close()    # close progress bar
    print(f"\nanimation saved as '{file_name}.mp4'")

    # --- REPORT --- #
    elapsed = int(pbar.format_dict["elapsed"])
    t = datetime.timedelta(seconds=elapsed)
    print(f"\ntotal elapsed time: {t}")

    avg_iter_per_sec = steps / t.total_seconds()
    if 1 / avg_iter_per_sec < 1:
        avg_rate = f"{1 / avg_iter_per_sec * 1e3:.0f} ms/frame"
    else:
        avg_rate = f"{1 / avg_iter_per_sec:.2f} sec/frame"
    print(f"{avg_iter_per_sec:.1f} frames/sec processed ({avg_rate})")

    return None


if __name__ == "__main__":

    # ----- SIMULATE PLUTO-CHARON SYSTEM ----- #

    t, Z, p = two_body_system(
        m1=M_PLUTO, m2=M_CHARON,        
        d=D_PLUTO_CHARON, v0=V_CHARON,      
        i_deg=i_CHARON, T_days=T_PLUTO_CHARON * 2.5, 
        rtol=1e-6, steps=1000
    )

    animate2d(
        t, Z, p,
        m1_colour="tab:brown", m1_trail_colour="tab:brown",
        m2_colour="tab:olive", m2_trail_colour="tab:olive",
        figure_title="Pluto-Charon System (TO SCALE)",
        show_barycentre=True,
        to_scale=True,
        radius1=R_PLUTO, body1_label="Pluto",
        radius2=R_CHARON, body2_label="Charon",
        show_legend=True, max_axis_extent=1.15,
        # animation writing params:
        trail_length_pct=10,
        dots_per_inch=100
    )

    import sys; sys.exit()

    # ----- SIMULATE EARTH-MOON SYSTEM (DEFAULT PARAMETERS) ----- #

    t, Z, p = two_body_system(steps=5000, rtol=1e-6)

    plot_orbits2d(
        t, Z, p,
        to_scale=True,
        figure_title="Earth-Moon System (TO SCALE)",
    )

    # ----- HIGHLY EXAGGERATED EARTH-MOON SYSTEM ----- #

    t, Z, p = two_body_system(
        m1=M_EARTH * 0.2,
        d=D_EARTH_MOON * 0.1,
        T_days=3,
        rtol=1e-9,
        steps=1000
    )

    plot_orbits2d(
        t, Z, p,
        figure_title="EXAGGERATED Earth-Moon System (NOT TO SCALE)",
        show_barycentre=True,
        show_legend=True,
        max_axis_extent=1.15,
        x_axis_limits=(-1.5e7, 4e7)
    )