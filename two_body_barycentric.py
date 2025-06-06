import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation, FFMpegWriter
import datetime
from constants import *
from tqdm_pbar import tqdmFA
from config import SystemParams, PlotConfig
from typing import Optional, Tuple
# configure matplotlib defaults
plt.rcParams.update({
    "font.size": 8,
    "font.family": "monospace",
    "lines.linewidth": 1
})


class TwoBodySystem:
    """
    A complete two-body gravitational system simulator with plotting and animation capabilities.
    
    This class handles the numerical integration of two gravitating bodies around their barycenter, 
    with comprehensive visualisation options including static plots and animations in both 2D and 3D.
    """

    def __init__(
        self, 
        params: Optional[SystemParams] = None, 
        config: Optional[PlotConfig] = None
    ) -> None:
        self.params = params if params else SystemParams()
        self.config = config if config else PlotConfig()
        self.t = None
        self.Z = None
        self._solve()    # solve the system of ODEs upon instantation to cache t and Z

    def _solve(self) -> None:
        """Numerically solve the ODEs for the two-body system using the provided parameters."""
        # extract system parameters from the dataclass:
        m1,m2, d, v0 = self.params.m1, self.params.m2, self.params.d, self.params.v0
        i_deg, T_days, steps = self.params.i_deg, self.params.T_days, self.params.steps
        ode_method, rtol, atol = self.params.ode_method, self.params.rtol, self.params.atol
        # initial position vectors of mass 1 and mass 2:
        i_rad = np.radians(i_deg)   # convert inclination angle to radians
        r2_0 = np.array([d * np.cos(i_rad), 0.0, d * np.sin(i_rad)])      
        r1_0 = -m2/m1 * r2_0        # position vector of Earth (with barycentre at origin)                                     
        # initial velocity vectors:
        v0 = np.array([0.0, v0, 0.0])
        v1_0 = -m2/(m1 + m2) * v0
        v2_0 = m1/(m1 + m2) * v0
        # initial state vector, [r_e, v_e, r_m, v_m]:
        Z0 = np.concatenate([r1_0, v1_0, r2_0, v2_0])
        
        def func(t, Z):
            """Computes the time derivative of the state vector."""
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
        # quick simulation report:
        print(f"\nrunning ODE solver ({ode_method=})...")
        print(f"using {rtol=:.0e}, {atol=:.0e} (default: rtol=1e-3, atol=1e-6)")
        print(f"t_eval=({t_eval[0]:.0f}, {t_eval[-1]:,.0f}), {steps=:,}, dt≈{dt:.2f}")
        sol = solve_ivp(func, t_span, Z0, t_eval=t_eval, method=ode_method, rtol=rtol, atol=atol)
        
        # ----- EXTRACT RESULTS ----- #
        t, Z = sol.t, sol.y
        success = sol.success
        print(f"\nsolver success: {success} ({sol.nfev:,} nfev)")
        print(f"t.shape: {t.shape}, Z.shape: {Z.shape}\n")
        self.t, self.Z = t, Z

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Public interface to solve or re-solve the system and return the solution.

        Returns:
            `t` (`np.ndarray`): time points at which the solution is evaluated.
            `Z` (`np.ndarray`): state vector containing positions and velocities of both bodies.
        """
        if self.t is None or self.Z is None:
            return self._solve()
        return self.t, self.Z   # return existing cached solution if available

    def plot_orbits2d(self) -> None:
        """Plot the complete orbits of the two bodies in 2D, using parameters defined in `self.config`."""
        cf = self.config    # shorthand alias for the PlotConfig instance
        
        if self.t is None or self.Z is None:
            print("Empty solutions, re-running ODE solver...")
            t, Z = self.solve()
        else:
            t, Z = self.t, self.Z

        r1, v1, r2, v2 = np.vsplit(Z, 4)    # unpack state vector (split along rows)
        x1, y1 = r1[0, :], r1[1, :]         # unpack body 1 2D coordinates
        x2, y2 = r2[0, :], r2[1, :]         # unpack body 2 2D coordinates  
        
        # ----- FIGURE SETUP ----- #
        fig, ax = plt.subplots(figsize=cf.figure_size)
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(MaxNLocator(cf.x_axis_max_ticks))
        ax.yaxis.set_major_locator(MaxNLocator(cf.y_axis_max_ticks))
        if cf.figure_title:
            ax.set_title(cf.figure_title)
        ax.set_xlabel(r"$x$ ($m$)")
        ax.set_ylabel(r"$y$ ($m$)")
        # grids and dashed lines
        ax.grid(True, alpha=cf.grid_alpha)
        hline = ax.axhline(0, linestyle="--", color="black", alpha=cf.dashed_line_alpha, linewidth=cf.dashed_line_width, zorder=1)
        vline = ax.axvline(0, linestyle="--", color="black", alpha=cf.dashed_line_alpha, linewidth=cf.dashed_line_width, zorder=1)
        hline.set_dashes([10, 10]), vline.set_dashes([10, 10])
        if cf.show_bc:     # barycentre lies at the origin
            ax.scatter(0, 0, marker="x", s=cf.bc_markersize, color=cf.bc_colour, label=cf.bc_legend_label, alpha=cf.bc_alpha, zorder=10)

        # --- PLOT FULL ORBIT TRAILS --- #
        ax.plot(x1, y1, color=cf.body1_trail_colour, linewidth=cf.line_width)
        ax.plot(x2, y2, color=cf.body2_trail_colour, linewidth=cf.line_width)

        # --- ADD MARKERS --- #
        if cf.to_scale:    # show bodies to scale
            body1 = Circle((x1[-1], y1[-1]), radius=cf.body1_radius, color=cf.body1_colour, label=cf.body1_legend_label, zorder=5)
            body2 = Circle((x2[-1], y2[-1]), radius=cf.body2_radius, color=cf.body2_colour, label=cf.body2_legend_label, zorder=5)
            ax.add_patch(body1), ax.add_patch(body2)
        else:
            ax.scatter(x1[-1], y1[-1], color=cf.body1_colour, s=cf.body1_markersize)
            ax.scatter(x2[-1], y2[-1], color=cf.body2_colour, s=cf.body2_markersize)

        # --- AXIS LIMITS & LEGEND --- #
        if cf.x_axis_limits:
            ax.set_xlim(cf.x_axis_limits)
        else:
            x_all = np.concatenate([x1, x2])
            x_extent = cf.max_axis_extent * np.max(np.abs(x_all))
            ax.set_xlim(-x_extent, x_extent)
        if cf.y_axis_limits:
            ax.set_ylim(cf.y_axis_limits)
        else:
            y_all = np.concatenate([y1, y2])
            y_extent = cf.max_axis_extent * np.max(np.abs(y_all))
            ax.set_ylim(-y_extent, y_extent)
        if cf.show_legend:
            ax.legend()
        plt.show()

    def animate2d(
        self, 
        trail_length_pct: float = 10, 
        fps: int = 60, dpi: int = 200, 
        bitrate: int = 50_000
    ) -> None:
        """
        Animate the 2D orbits of the two-body system, by writing an MP4 file.

        Args:
            `trail_length_pct` (`float`): Percentage of the total number of steps to use as the orbit trail length.
            `fps` (`int`): Frames per second for the animation.
            `dpi` (`int`): Dots per inch for the saved animation file.
        """

        cf = self.config    # shorthand alias for the PlotConfig instance
    
        if self.t is None or self.Z is None:
            print("Empty solutions, re-running ODE solver...")
            t, Z = self.solve()
        else:
            t, Z = self.t, self.Z

        # show the static complete orbit trails (final positions) first:
        print(f"\nplotting final positions with complete orbit trails...")
        print("<CLOSE FIGURE WINDOW TO START ANIMATION WRITING>\n")
        self.plot_orbits2d()    # figure window must be closed before continuing 

        # key animation statistics
        interval = int(1000 / fps)     # convert FPS to milliseconds
        steps = len(t)                                      # total number of time steps from the ODE solver
        trail_length = int((trail_length_pct / 100) * steps) 
        print(f"\n{steps:,} steps @ {fps} fps (~{interval * 1e-3:.3f} sec/frame)")
        print(f"writing {steps} frames to MP4...\n")

        # ----- EXTRACT COORDINATES ----- #
        r1, v1, r2, v2 = np.vsplit(Z, 4)    # unpack state vector (split along rows)
        x1, y1 = r1[0, :], r1[1, :]         # unpack body 1 2D coordinates
        x2, y2 = r2[0, :], r2[1, :]         # unpack body 2 2D coordinates  

        # ----- FIGURE SETUP ----- #
        fig, ax = plt.subplots(figsize=cf.figure_size)
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(MaxNLocator(cf.x_axis_max_ticks))
        ax.yaxis.set_major_locator(MaxNLocator(cf.y_axis_max_ticks))
        if cf.figure_title:
            ax.set_title(cf.figure_title)
        ax.set_xlabel(r"$x$ ($m$)")
        ax.set_ylabel(r"$y$ ($m$)")
        # grids and dashed lines
        ax.grid(True, alpha=cf.grid_alpha)
        hline = ax.axhline(0, linestyle="--", color="black", alpha=cf.dashed_line_alpha, linewidth=cf.dashed_line_width, zorder=1)
        vline = ax.axvline(0, linestyle="--", color="black", alpha=cf.dashed_line_alpha, linewidth=cf.dashed_line_width, zorder=1)
        hline.set_dashes([10, 10]), vline.set_dashes([10, 10])
        if cf.show_bc:     # barycentre lies at the origin
            ax.scatter(0, 0, marker="x", s=cf.bc_markersize, color=cf.bc_colour, label=cf.bc_legend_label, alpha=cf.bc_alpha, zorder=10)

        # --- PLOT ELEMENTS (TO BE UPDATED IN ANIMATION) --- #
        if cf.to_scale:    # show planetary bodies to scale
            body1_marker = Circle((x1[0], y1[0]), radius=cf.body1_radius, color=cf.body1_colour, label=cf.body1_legend_label, zorder=5)
            body2_marker = Circle((x2[0], y2[0]), radius=cf.body2_radius, color=cf.body2_colour, label=cf.body2_legend_label, zorder=5)
            ax.add_patch(body1_marker), ax.add_patch(body2_marker)
        else:
            body1_marker = ax.scatter([], [], color=cf.body1_colour, s=cf.body1_markersize, zorder=5)
            body2_marker = ax.scatter([], [], color=cf.body2_colour, s=cf.body2_markersize, zorder=5)
        body1_orbit, = ax.plot([], [], color=cf.body1_trail_colour, linewidth=cf.line_width)
        body2_orbit, = ax.plot([], [], color=cf.body2_trail_colour, linewidth=cf.line_width)

        # --- AXIS LIMITS & LEGEND --- #
        if cf.x_axis_limits:
            ax.set_xlim(cf.x_axis_limits)
        else:
            x_all = np.concatenate([x1, x2])
            x_extent = cf.max_axis_extent * np.max(np.abs(x_all))
            ax.set_xlim(-x_extent, x_extent)
        if cf.y_axis_limits:
            ax.set_ylim(cf.y_axis_limits)
        else:
            y_all = np.concatenate([y1, y2])
            y_extent = cf.max_axis_extent * np.max(np.abs(y_all))
            ax.set_ylim(-y_extent, y_extent)
        if cf.show_legend:
            ax.legend()

        # --- PROGRESS BAR --- #
        pbar = tqdmFA(total=len(t))

        # ----- ANIMATION FUNCTION SETUP ----- #
        def init():
            body1_orbit.set_data([], []), body2_orbit.set_data([], [])
            if cf.to_scale:
                body1_marker.center = (x1[0], y1[0])
                body2_marker.center = (x2[0], y2[0])
            else:
                body1_marker.set_offsets([[x1[0], y1[0]]])
                body2_marker.set_offsets([[x2[0], y2[0]]])
            return body1_orbit, body2_orbit, body1_marker, body2_marker

        def update(frame):
            """Update orbit trails and marker positions."""
            i0 = max(0, frame - trail_length)    # start index for the trail
            body1_orbit.set_data(x1[i0: frame + 1], y1[i0: frame + 1])    # update orbit trail
            body1_orbit.set_data(x2[i0: frame + 1], y2[i0: frame + 1])
            if cf.to_scale:
                body1_marker.center = (x1[frame], y1[frame])    # update matplotlib.patches.Circle position
                body2_marker.center = (x2[frame], y2[frame])
            else:
                body1_marker.set_offsets([[x1[frame], y1[frame]]])    # update scatter marker position
                body2_marker.set_offsets([[x2[frame], y2[frame]]])
            pbar.update(1)    # update tqdm progress bar
            return body1_orbit, body2_orbit, body1_marker, body2_marker

        # --- WRITE AND SAVE TO FILE --- #
        ani = FuncAnimation(fig, update, frames=range(steps), init_func=init, interval=interval, blit=True)
        writer = FFMpegWriter(fps=fps, bitrate=bitrate)
        # generate a file name:
        name1, name2 = cf.body1_legend_label, cf.body2_legend_label
        time_period = self.params.T_days
        file_name = (
            f"2D_baryc_{name1.lower()}-{name2.lower()}_"
            f"{time_period=:.1f}_"
            f"{trail_length_pct:.0f}%trail_"
            f"{cf.to_scale=}_"
            f"{steps=:,.0f}_"
            f".mp4"
        )
        ani.save(filename=file_name, writer=writer, dpi=dpi)
        pbar.close()    # close progress bar
        print(f"\nanimation saved as '{file_name}'")

        # --- QUICK FRAME WRITING REPORT --- #
        elapsed = int(pbar.format_dict["elapsed"])
        t_elapsed = datetime.timedelta(seconds=elapsed)
        print(f"\ntotal elapsed time: {t_elapsed}")
        avg_iter_per_sec = steps / t_elapsed.total_seconds()
        if 1 / avg_iter_per_sec < 1:
            avg_rate = f"{1 / avg_iter_per_sec * 1e3:.0f} ms/frame"
        else:
            avg_rate = f"{1 / avg_iter_per_sec:.2f} sec/frame"
        print(f"{avg_iter_per_sec:.1f} frames/sec processed ({avg_rate})")   