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
    p = [Z0, T]  

    return t, Z, p


def plot_orbits2d(
    t, Z, p,
    in_degrees: bool = True,
    earth_colour: str = "tab:blue",
    earth_trail_colour: str = "tab:blue",
    moon_colour: str = "tab:grey",
    moon_trail_colour: str = "tab:grey",
    earth_markersize: int = 100,
    moon_markersize: int = 40,
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
    to_scale: bool = False
    ) -> None:

    r_e, v_e, r_m, v_m = np.vsplit(Z, 4)    # unpack state vector (split along rows)
    x_e, y_e = r_e[0, :], r_e[1, :]         # unpack Earth's 2D coordinates
    x_m, y_m = r_m[0, :], r_m[1, :]         # unpack Moon's 2D coordinates  

    r_e0, v_e0, r_m0, v_m0 = np.split(p[0], 4)           # unpack initial state vector

    # ----- FIGURE ----- #
    fig, ax = plt.subplots(figsize=figure_size)
    ax.grid(True, alpha=grid_alpha)
    ax.xaxis.set_major_locator(MaxNLocator(x_axis_max_ticks))
    ax.yaxis.set_major_locator(MaxNLocator(y_axis_max_ticks))
    if figure_title:
        ax.set_title(figure_title)
    ax.set_xlabel(r"$x$ ($m$)")
    ax.set_ylabel(r"$y$ ($m$)")

    # --- PLOT EARTH & MOON ORBITS --- #
    line_width = 0.5 if to_scale else line_width
    ax.plot(x_e, y_e, color=earth_trail_colour, linewidth=line_width)
    ax.plot(x_m, y_m, color=moon_trail_colour, linewidth=line_width)

    # --- ADD MARKERS --- #
    if to_scale:
        earth = Circle((x_e[-1], y_e[-1]), radius=R_EARTH, color=earth_colour, label="Earth")
        moon = Circle((x_m[-1], y_m[-1]), radius=R_MOON, color=moon_colour, label="Moon")
        ax.add_patch(earth), ax.add_patch(moon)
    else:
        ax.scatter(x_m[-1], y_m[-1], color=moon_colour, s=moon_markersize, label="Moon")
        ax.scatter(x_e[-1], y_e[-1], color=earth_colour, s=earth_markersize, label="Earth")

    # --- AXIS LIMITS & LEGEND --- #
    # independent overrides (if only one set of limits is provided):
    if x_axis_limits:
        ax.set_xlim(x_axis_limits)
    else:
        x_all = np.concatenate([x_e, x_m])
        x_extent = max_axis_extent * np.max(np.abs(x_all))
        ax.set_xlim(-x_extent, x_extent)
    if y_axis_limits:
        ax.set_ylim(y_axis_limits)
    else:
        y_all = np.concatenate([y_e, y_m])
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
    )