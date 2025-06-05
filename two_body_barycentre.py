import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, FFMpegWriter
plt.rcParams["font.size"] = 9
plt.rcParams["font.family"] = "monospace"
plt.rcParams["lines.linewidth"] = 1
import time
import datetime
from constants import *
from tqdm_pbar import tqdmFA


def earth_moon_system(
        T_days: float = 27.321,
        steps: int = 1000,
        method: str = "RK4"
    ) -> tuple:
    """
    Simulate the Earth-Moon system using a two-body problem approach.
    The simulation uses the barycentric coordinates of the Earth and Moon.
    """
    
    i_rad = np.radians(i_MOON)

    # initial position vectors of Earth and Moon:
    r_m0 = np.array([D_EARTH_MOON * np.cos(i_rad), 0.0, D_EARTH_MOON * np.sin(i_rad)])      
    r_e0 = -M_MOON/M_EARTH * r_m0                                                          

    # initial velocity vectors of Earth and Moon:
    v0 = np.array([0.0, V_MOON, 0.0])
    v_e0 = -M_MOON/(M_EARTH + M_MOON) * v0
    v_m0 = M_EARTH/(M_EARTH + M_MOON) * v0

    # initial state vector, [r_e, v_e, r_m, v_m]:
    Z0 = np.concatenate([
        r_e0,
        v_e0,
        r_m0,
        v_m0
    ])

    def func(t, Z):
        r_e, v_e, r_m, v_m = np.split(Z, 4)             # unpack state vector (4 equal sub-arrays)
        r = r_m - r_e                                   # relative position vector of Moon w.r.t. Earth
        r_mag = np.linalg.norm(r)                       # magnitude (Earth-Moon distance)
        F = ( G*M_EARTH*M_MOON / r_mag**3 ) * r         # force vector (Newton's law of gravitation)
        a_e = F / M_EARTH                               # acceleration of Earth
        a_m = -F / M_MOON                               # acceleration of Moon (equal and opposite)
        return np.concatenate([v_e, a_e, v_m, a_m])     # return the time derivative of the state vector

    # ----- EVALUATION & SOLVE ----- #
    orbital_period = 2 * np.pi * np.sqrt(D_EARTH_MOON**3 / (G * (M_EARTH + M_MOON)))  # orbital period of the Earth-Moon system
    T = T_days * 24 * 3600                                  # one lunar orbit (s)
    t_span = (0, T) if T > 0 else (0, orbital_period) 
    t_eval = np.linspace(t_span[0], t_span[1], steps)       # time points at which to store the solution
    dt = t_eval[1] - t_eval[0]                              # time eval step size

    print(f"\nrunning ODE solver ({method=})...")
    print(f"t_eval=({t_eval[0]:.0f}, {t_eval[-1]}), {steps=:,}, dt≈{dt:.2f}")#
    sol = solve_ivp(func, t_span, Z0, t_eval=t_eval, method="RK45")

    # ----- EXTRACT RESULTS ----- #
    t, Z = sol.t, sol.y
    success = sol.success
    print(f"\nsolver success: {success} ({sol.nfev:,} nfev)")
    print(f"t.shape: {t.shape}, Z.shape: {Z.shape}\n")
    # also return vector of any essential parameters:
    p = [Z0, T]  

    return t, Z, p