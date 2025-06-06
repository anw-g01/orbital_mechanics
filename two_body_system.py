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
    "font.size": 9,
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

    def _create_figure2d(
        self, 
        x_coords: Tuple[np.ndarray, np.ndarray], 
        y_coords: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Setup a 2D base figure with axes for plotting the two-body system.
        
        Args:
            `x_coords` (`Tuple[np.ndarray, np.ndarray]`): x-coordinates of the two bodies.
            `y_coords` (`Tuple[np.ndarray, np.ndarray]`): y-coordinates of the two bodies.

        Returns:
            `fig` (`plt.Figure`): the created figure.
            `ax` (`plt.Axes`): the axes for plotting.
        """
        cf = self.config    # shorthand alias for the PlotConfig instance
        
        # ----- FIGURE SETUP ----- #
        fig, ax = plt.subplots(figsize=cf.figure_size)    # create a figure with specified size
        if cf.figure_title:
            ax.set_title(cf.figure_title)
        ax.xaxis.set_major_locator(MaxNLocator(cf.x_axis_max_ticks))
        ax.yaxis.set_major_locator(MaxNLocator(cf.y_axis_max_ticks))
        ax.set_xlabel(r"$x$ ($m$)")
        ax.set_ylabel(r"$y$ ($m$)")

        # ----- GRIDS AND DASHED LINES ----- #
        ax.grid(True, alpha=cf.grid_alpha)
        hline = ax.axhline(0, linestyle="--", color="black", alpha=cf.dashed_line_alpha, linewidth=cf.dashed_line_width, zorder=1)
        vline = ax.axvline(0, linestyle="--", color="black", alpha=cf.dashed_line_alpha, linewidth=cf.dashed_line_width, zorder=1)
        hline.set_dashes([10, 10])
        vline.set_dashes([10, 10])

        # show barycentre marker (lies exactly at the origin):
        if cf.show_bc:     
            ax.scatter(0, 0, marker="x", s=cf.bc_markersize, color=cf.bc_colour, label=cf.bc_legend_label, alpha=cf.bc_alpha, zorder=10)
        
        # --- AXIS LIMITS & LEGEND --- #
        if cf.x_axis_limits:
            ax.set_xlim(cf.x_axis_limits)
        else:
            x_all = np.concatenate(x_coords)    # (x1, x2)
            x_extent = cf.max_axis_extent * np.max(np.abs(x_all))
            ax.set_xlim(-x_extent, x_extent)
        if cf.y_axis_limits:
            ax.set_ylim(cf.y_axis_limits)
        else:
            y_all = np.concatenate(y_coords)    # (y1, y2)
            y_extent = cf.max_axis_extent * np.max(np.abs(y_all))
            ax.set_ylim(-y_extent, y_extent)
        ax.set_aspect("equal")    # ensure equal aspect ratio after setting limits
        return fig, ax

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
        
        # ----- MAIN FIGURE ----- #
        fig, ax = self._create_figure2d((x1, x2), (y1, y2))   # create a 2D base figure with axes

        # --- PLOT FULL ORBIT TRAILS --- #
        ax.plot(x1, y1, color=cf.body1_trail_colour, linewidth=cf.line_width)
        ax.plot(x2, y2, color=cf.body2_trail_colour, linewidth=cf.line_width)

        # --- ADD MARKERS --- #
        if cf.to_scale:    # show bodies to scale
            body1 = Circle((x1[-1], y1[-1]), radius=cf.body1_radius, color=cf.body1_colour, label=cf.body1_legend_label, zorder=5)
            body2 = Circle((x2[-1], y2[-1]), radius=cf.body2_radius, color=cf.body2_colour, label=cf.body2_legend_label, zorder=5)
            ax.add_patch(body1), ax.add_patch(body2)
        else:
            ax.scatter(x1[-1], y1[-1], color=cf.body1_colour, s=cf.body1_markersize, zorder=5)
            ax.scatter(x2[-1], y2[-1], color=cf.body2_colour, s=cf.body2_markersize, zorder=5)

        if cf.show_legend:
            ax.legend()
        plt.show()

    def animate2d(
        self, 
        trail_length_pct: float = 10, 
        trail_length_factor: float = 3,
        fps: int = 60, dpi: int = 200, 
        bitrate: int = 50_000
    ) -> None:
        """
        Animate the 2D orbits of the two-body system, by writing an MP4 file.

        Args:
            `trail_length_pct` (`float`): percentage of the total number of steps to use as the orbit trail length.
            `trail_length_factor` (`float`): factor by which to extend the trail length for the first body.
            `fps` (`int`): frames per second for the animation.
            `dpi` (`int`): dots per inch for the saved animation file.
        """

        cf = self.config    # shorthand alias for the PlotConfig instance
    
        if self.t is None or self.Z is None:
            print("Empty solutions, re-running ODE solver...")
            t, Z = self.solve()
        else:
            t, Z = self.t, self.Z

        # show the static complete orbit trails (final positions) first:
        print(f"plotting final positions with complete orbit trails...")
        print("<CLOSE FIGURE WINDOW TO START ANIMATION WRITING>")
        self.plot_orbits2d()    # figure window must be closed before continuing 

        # key animation statistics
        interval = int(1000 / fps)     # convert FPS to milliseconds
        steps = len(t)                                      # total number of time steps from the ODE solver
        trail_length = int((trail_length_pct / 100) * steps) 
        print(f"\n{steps:,} steps @ {fps} fps (~{interval * 1e-3:.3f} sec/frame) and {dpi} DPI")
        duration = steps / fps
        sec_per_orbit = self.params.T_days * 24 * 3600 / steps    # seconds per orbit step
        print(f"total video duration: {duration:.2f} sec ({duration / 60:.1f} min)")
        print(f"writing {steps} frames to MP4...\n")

        # ----- EXTRACT COORDINATES ----- #
        r1, v1, r2, v2 = np.vsplit(Z, 4)    # unpack state vector (split along rows)
        x1, y1 = r1[0, :], r1[1, :]         # unpack body 1 2D coordinates
        x2, y2 = r2[0, :], r2[1, :]         # unpack body 2 2D coordinates  

        # ----- FIGURE SETUP ----- #
        fig, ax = self._create_figure2d((x1, x2), (y1, y2))   # create a 2D base figure with axes

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

        if cf.show_legend:
            ax.legend()

        # --- PROGRESS BAR --- #
        pbar = tqdmFA(total=steps, fps=fps)

        # ----- ANIMATION FUNCTION SETUP ----- #
        def init():
            body1_orbit.set_data([], [])
            body2_orbit.set_data([], [])
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
            i0_1 = max(0, frame - int(trail_length * trail_length_factor))    # longer trail for body 1
            body1_orbit.set_data(x1[i0_1: frame + 1], y1[i0_1: frame + 1])      # update orbit trail
            body2_orbit.set_data(x2[i0: frame + 1], y2[i0: frame + 1])
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
        if not name1 or not name2:
            name1, name2 = "body1", "body2"
        time_period = self.params.T_days
        file_name = (
            f"2D_{name1.lower()}-{name2.lower()}_"
            f"{dpi=}_"
            f"{trail_length_pct:.0f}%trail(factor={trail_length_factor})_"
            f"{steps=:,.0f}_"
            f"T_days={time_period:.1f}_"
            f"{cf.to_scale=}_"
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

# ----- EXAMPLE USAGE OF THE CLASS ----- #

def pluto_charon_system() -> None:
    """Simulate and animate the Pluto-Charon two-body system with realistic parameters."""
    pluto_charon = TwoBodySystem(
        params=SystemParams(
            m1=M_PLUTO, 
            m2=M_CHARON, 
            d=D_PLUTO_CHARON, 
            v0=V_CHARON, 
            i_deg=i_CHARON, 
            T_days=T_PLUTO_CHARON * 1.175 * 4,
            rtol=1e-9, steps=1000
        ),
        config=PlotConfig(
            body1_radius=R_PLUTO, 
            body2_radius=R_CHARON,
            figure_size=(10, 10),
            figure_title="Pluto-Charon System (TO SCALE)",
            body1_legend_label="Pluto", 
            body2_legend_label="Charon",
            body1_colour="tab:brown", 
            body1_trail_colour="tab:brown",
            body2_colour="tab:olive", 
            body2_trail_colour="tab:olive",
            max_axis_extent=1.15,
            show_legend=True, 
            to_scale=True, 
            show_bc=True
        )
    )
    # pluto_charon.plot_orbits2d()
    pluto_charon.animate2d(
        trail_length_pct=2,
        trail_length_factor=3,
        dpi=200
    )


def earth_moon_system(exaggerated: bool = False) -> None:
    """
    Simulate and animate the Earth-Moon two-body system with realistic parameters.
    
    Args:
        `exaggerated` (`bool`): whether to exaggerate the orbital paths in the animation.
    """
    if not exaggerated:
        earth_moon = TwoBodySystem(
            params=SystemParams(
                rtol=1e-9, 
                steps=1000,
                T_days=27.321 * 1.03 * 3,  
            ),
            config=PlotConfig(
                figure_size=(10, 10),
                figure_title="Earth-Moon System (TO SCALE)",
                body1_legend_label="Earth", 
                body2_legend_label="Moon",
                max_axis_extent=1.05,
                line_width=0.5,
                to_scale=True, 
                show_bc=False
            )
        )
        # earth_moon.plot_orbits2d()
        earth_moon.animate2d(
            trail_length_pct=2,
            trail_length_factor=4,
            dpi=200
        )
    else:
        earth_moon = TwoBodySystem(
            params=SystemParams(
                m1=M_EARTH * 0.4, m2=M_MOON * 1, 
                d=D_EARTH_MOON * 0.1, v0=V_MOON * 1, 
                i_deg=i_MOON, 
                T_days=0.62 * 3,
                rtol=1e-6, 
                steps=750
            ),
            config=PlotConfig(
                figure_size=(10, 10),
                figure_title="EXAGGERATED Earth-Moon System (NOT TO SCALE)",
                body1_legend_label="Earth'", body2_legend_label="Moon'",
                body1_colour="tab:blue", body1_trail_colour="tab:blue",
                body2_colour="tab:red", body2_trail_colour="tab:red",
                max_axis_extent=1.4, 
                x_axis_limits=(-1e7, 4.5e7),
                show_legend=True, 
                to_scale=False, 
                show_bc=True, bc_alpha=0.8
            )
        )
        earth_moon.animate2d(
            trail_length_pct=2,
            trail_length_factor=,
            dpi=200
        )


def equal_mass_system() -> None:
    """Simulate and animate a two-body system with equal masses."""
    mass = M_EARTH              # use Earth mass for both bodies
    radius = R_EARTH            # use Earth radius for both bodies (if to scale)
    distance = D_EARTH_MOON     # use Earth-Moon distance
    orbits = 3                  # number of orbits (roughly) to simulate
    equal_mass = TwoBodySystem(
        params=SystemParams(
            m1=mass, m2=mass, d=distance, 
            v0=600, 
            i_deg=10, 
            T_days=26 * orbits,
            rtol=1e-9, 
            steps=750
        ),
        config=PlotConfig(
            body1_radius=radius, body2_radius=radius,
            body1_colour="tab:red", body1_trail_colour="tab:red",
            body2_colour="tab:green", body2_trail_colour="tab:green",
            figure_size=(10, 10), figure_title="Equal Mass Two-Body System",
            max_axis_extent=1.1, y_axis_limits=(-2.5e8, 2.5e8),
            to_scale=True, show_legend=True,
            show_bc=True, bc_alpha=0.8, bc_colour="tab:blue", bc_markersize=50            
        )
    )
    # equal_mass.plot_orbits2d()
    equal_mass.animate2d(
        trail_length_pct=5,
        trail_length_factor=1,
        dpi=300
    )