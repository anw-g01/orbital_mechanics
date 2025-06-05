import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator
plt.rcParams["font.size"] = 8
plt.rcParams["font.family"] = "monospace"
plt.rcParams["lines.linewidth"] = 1
from constants import *


def two_body_system(
        m1: float = M_EARTH,
        m2: float = M_MOON,
        d: float = D_EARTH_MOON,
        v0: float = V_MOON,
        i_deg: float = i_MOON,
        T_days: float = 27.321,
        steps: int = 1000,
        ode_method: str = "RK45"
    ) -> tuple:

    # initial position vectors of mass 1 and mass 2:
    i_rad = np.radians(i_deg)   # convert inclination angle to radians
    r2_0 = np.array([d * np.cos(i_rad), 0.0, d * np.sin(i_rad)])      
    r1_0 = -m2/m1 * r2_0                                                          

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
        r = r2 - r1                                   # relative position vector of Moon w.r.t. Earth
        r_mag = np.linalg.norm(r)                       # magnitude (Earth-Moon distance)
        F = ( G*m1*m2 / r_mag**3 ) * r         # force vector (Newton's law of gravitation)
        a1 = F / m1                               # acceleration of Earth
        a2 = -F / m2                               # acceleration of Moon (equal and opposite)
        return np.concatenate([v1, a1, v2, a2])     # return the time derivative of the state vector

    # ----- EVALUATION & SOLVE ----- #
    orbital_period = 2 * np.pi * np.sqrt(d**3 / (G * (m1 + m2)))  # orbital period of the Earth-Moon system
    T = T_days * 24 * 3600                                  # one lunar orbit (s)
    t_span = (0, T) if T > 0 else (0, orbital_period) 
    t_eval = np.linspace(t_span[0], t_span[1], steps)       # time points at which to store the solution
    dt = t_eval[1] - t_eval[0]                              # time eval step size

    print(f"\nrunning ODE solver ({ode_method=})...")
    print(f"t_eval=({t_eval[0]:.0f}, {t_eval[-1]}), {steps=:,}, dt≈{dt:.2f}")#
    sol = solve_ivp(func, t_span, Z0, t_eval=t_eval, method=ode_method)

    # ----- EXTRACT RESULTS ----- #
    t, Z = sol.t, sol.y
    success = sol.success
    print(f"\nsolver success: {success} ({sol.nfev:,} nfev)")
    print(f"t.shape: {t.shape}, Z.shape: {Z.shape}\n")
    # also return vector of any essential parameters:
    p = [m1, m2, d, v0, Z0, T]  

    return t, Z, p


def plot_orbits2d(
    t, Z, p,
    in_degrees: bool = True,
    m1_colour: str = "tab:blue",
    m1_trail_colour: str = "tab:blue",
    m2_colour: str = "tab:grey",
    m2_trail_colour: str = "tab:grey",
    m1_markersize: int = 500,
    m2_markersize: int = 100,
    figure_size: tuple = (8, 8),
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
    radius2: float = R_MOON,
    show_barycentre: bool = False
    ) -> None:

    r1, v1, r2, v2 = np.vsplit(Z, 4)    # unpack state vector (split along rows)
    x1, y1 = r1[0, :], r1[1, :]         # unpack Earth's 2D coordinates
    x2, y2 = r2[0, :], r2[1, :]         # unpack Moon's 2D coordinates  

    # ----- FIGURE ----- #
    fig, ax = plt.subplots(figsize=figure_size)
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
        hline = ax.axhline(0, linestyle="--", color="black", alpha=dashed_alpha, linewidth=dashed_linewidth)
        vline = ax.axvline(0, linestyle="--", color="black", alpha=dashed_alpha, linewidth=dashed_linewidth)
        hline.set_dashes([10, 10]), vline.set_dashes([10, 10])
        ax.scatter(0, 0, marker="x", s=25, color="tab:red", label="barycentre", zorder=10)

    # --- PLOT EARTH & MOON ORBITS --- #
    line_width = 0.5 if to_scale else line_width
    ax.plot(x1, y1, color=m1_trail_colour, linewidth=line_width)
    ax.plot(x2, y2, color=m2_trail_colour, linewidth=line_width)

    # --- ADD MARKERS --- #
    if to_scale:    # show Earth & Moon to scale
        body1 = Circle((x1[-1], y1[-1]), radius=radius1, color=m1_colour)
        body2 = Circle((x2[-1], y2[-1]), radius=radius2, color=m2_colour)
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



if __name__ == "__main__":


    # ----- SIMULATE EARTH-MOON SYSTEM (DEFAULT PARAMETERS) ----- #

    t, Z, p = two_body_system(steps=5000)

    plot_orbits2d(
        t, Z, p,
        to_scale=True,
        figure_title="Earth-Moon System (TO SCALE)",
        show_barycentre
    )