import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation, FFMpegWriter
import datetime
from constants import *
from tqdm_pbar import tqdmFA
from config import SystemParams, PlotConfig, PlotConfig3D
from typing import Optional, Tuple, Union
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# configure matplotlib defaults
plt.rcParams.update({
    "font.size": 8,
    "font.family": "monospace",
    "lines.linewidth": 1,
    "grid.color": (0.5, 0.5, 0.5, 0.1)    # 3D grid alpha 
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
        config: Optional[PlotConfig] = None,
        config_3d: Optional[PlotConfig3D] = None
    ) -> None:
        self.params = params if params else SystemParams()
        self.config = config if config else PlotConfig()
        self.config_3d = config_3d if config_3d else PlotConfig3D()
        self.t = None
        self.Z = None
        self._solve()    # solve the system of ODEs upon instantation to cache t and Z

    def _solve(self) -> None:
        """Numerically solve the ODEs for the two-body system using the provided parameters."""
        p = self.params    # shorthand alias for the SystemParams instance
        # extract system parameters from the dataclass:
        m1,m2, d, v0 = p.m1, p.m2, p.d, p.v0
        i_deg, T_days, steps = p.i_deg, p.T_days, p.steps
        ode_method, rtol, atol = p.ode_method, p.rtol, p.atol
        # initial position vectors of mass 1 and mass 2:
        i_rad = np.radians(i_deg)   # convert inclination angle to radians
        r2_0 = np.array([d * np.cos(i_rad), 0.0, d * np.sin(i_rad)])      
        r1_0 = -m2/m1 * r2_0        # position vector of Earth (with barycentre at origin)                                     
        # initial velocity vectors:
        v0 = np.array([0.0, v0, 0])
        v1_0 = -m2/(m1 + m2) * v0
        v2_0 = m1/(m1 + m2) * v0
        # initial state vector, [r_e, v_e, r_m, v_m]:
        Z0 = np.concatenate([r1_0, v1_0, r2_0, v2_0])

        def _func(t: np.ndarray, Z: np.ndarray) -> np.ndarray:
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
        sol = solve_ivp(_func, t_span, Z0, t_eval=t_eval, method=ode_method, rtol=rtol, atol=atol)
        
        # ----- EXTRACT RESULTS ----- #
        t, Z = sol.t, sol.y
        success = sol.success
        print(f"\nsolver success: {success} ({sol.nfev:,} nfev)")
        print(f"t.shape: {t.shape}, Z.shape: {Z.shape}")
        self.t, self.Z = t, Z

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """Public interface to solve or re-solve the system and return the solution."""
        if self.t is None or self.Z is None:
            self._solve()
        return self.t, self.Z   # return existing cached solution if available

    def _check_solve(self) -> None:
        """Check if the ODE solver has been run, and if not, run it."""
        if self.t is None or self.Z is None:
            print("Empty solutions, re-running ODE solver...")
            self.t, self.Z = self.solve()    # call the solve method to ensure t and Z are populated

    def _project_to_orbital_plane(
        self, 
        r1: np.ndarray, 
        r2: np.ndarray,
        verbose: bool = False
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Project 3D coordinates onto the orbital plane for head-on 2D viewing.
        Assuming the orbital plane is NOT the xy-plane (for highly inclined orbits).
        """
        if verbose:
            i = self.params.i_deg    # orbital inclination in degrees
            print(f"\norbital inclination: {i:.1f}° ({np.radians(i):.2f} rad)")
            print(f"projecting 3D coordinates to orbital plane for head-on 2D viewing...")

        # use initial positions to define the plane
        r_rel0 = r2[:, 0] - r1[:, 0]    # initial relative position vector
        r_rel1 = r2[:, 1] - r1[:, 1]    # second time step relative position vector

        # calculate normal vector to the orbital plane:
        norm = np.cross(r_rel0, r_rel1)     # cross product
        norm /= np.linalg.norm(norm)        # normalise the normal vector

        # create two orthogonal vectors in the orbital plane:
        u1 = r_rel0 / np.linalg.norm(r_rel0)    # 1. initial relative position (normalised) in the plane
        u2 = np.cross(norm, u1)                 # 2. orthogonal to u1 and norm vector
        u2 /= np.linalg.norm(u2)                # normalise u2

        # project the 3D coordinates onto the orbital plane, using dot product 
        # (with each time step) to project onto the plane, defined by u1 and u2:
        x1, y1 = np.dot(u1, r1), np.dot(u2, r1)     # project body 1 coordinates
        x2, y2 = np.dot(u1, r2), np.dot(u2, r2)     # project body 2 coordinates

        return (x1, y1), (x2, y2)

    def _create_figure2d(
        self, 
        x_coords: Tuple[np.ndarray, np.ndarray], 
        y_coords: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Setup a 2D base figure with axes for plotting the two-body system."""
        cf = self.config    # shorthand alias for the PlotConfig instance
        
        # ----- FIGURE SETUP ----- #
        fig, ax = plt.subplots(figsize=cf.figure_size)    # create a figure with specified size
        ax.set_aspect("equal")    # ensure equal aspect ratio after setting limits
        if cf.figure_title:
            ax.set_title(cf.figure_title, fontdict={"fontsize": cf.title_fontsize})
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
        if cf.display_baryc:     
            ax.scatter(0, 0, marker="x", s=cf.baryc_markersize, color=cf.baryc_colour, label=cf.baryc_legend_label, alpha=cf.baryc_alpha, zorder=10)
        
        # --- AXIS LIMITS --- #
        if cf.x_axis_limits:
            ax.set_xlim(cf.x_axis_limits)
        else:
            x_all = np.concatenate(x_coords)    # (x1, x2)
            x_extent = cf.max_axis_extent2d * np.max(np.abs(x_all))
            ax.set_xlim(-x_extent, x_extent)
        if cf.y_axis_limits:
            ax.set_ylim(cf.y_axis_limits)
        else:
            y_all = np.concatenate(y_coords)    # (y1, y2)
            y_extent = cf.max_axis_extent2d * np.max(np.abs(y_all))
            ax.set_ylim(-y_extent, y_extent)

        # --- LEGEND --- #
        if cf.display_legend:
            # create proxy markers for legend (independent of actual plotted objects)
            body1_legend = mlines.Line2D([], [], color=cf.body1_colour, marker='o', linestyle='None', markersize=6, label=cf.body1_legend_label)
            body2_legend = mlines.Line2D([], [], color=cf.body2_colour, marker='o', linestyle='None', markersize=6, label=cf.body2_legend_label)
            baryc_legend = mlines.Line2D([], [], color=cf.baryc_colour, marker='x', linestyle='None', markersize=6, label=cf.baryc_legend_label)
            ax.legend(handles=[body1_legend, body2_legend, baryc_legend])
        
        return fig, ax

    def plot_orbits2d(self) -> None:
        """Plot the complete orbits of the two bodies in 2D, using parameters defined in `self.config`."""
        
        cf = self.config    # shorthand alias for the PlotConfig instance
        t, Z = self.t, self.Z
        r1, _, r2, _ = np.vsplit(Z, 4)    # unpack state vector (split along rows)

        if self.params.head_on_view:    
            # project to orbital plane for head-on 2D viewing:
            (x1, y1), (x2, y2) = self._project_to_orbital_plane(r1, r2, verbose=True)
        else:
            # unpack 2D coordinates directly (assumes r1 and r2 are already in 2D):
            x1, y1 = r1[0, :], r1[1, :]         # unpack body 1 2D coordinates
            x2, y2 = r2[0, :], r2[1, :]         # unpack body 2 2D coordinates

        # ----- MAIN FIGURE ----- #
        fig, ax = self._create_figure2d((x1, x2), (y1, y2))   # create a 2D base figure with axes

        # --- PLOT FULL ORBIT TRAILS --- #
        ax.plot(x1, y1, color=cf.body1_trail_colour, linewidth=cf.line_width)
        ax.plot(x2, y2, color=cf.body2_trail_colour, linewidth=cf.line_width)

        # --- ADD MARKERS --- #
        if cf.to_scale:    # show bodies to scale
            body1 = mpatches.Circle((x1[-1], y1[-1]), radius=cf.body1_radius, color=cf.body1_colour, label=cf.body1_legend_label, zorder=5)
            body2 = mpatches.Circle((x2[-1], y2[-1]), radius=cf.body2_radius, color=cf.body2_colour, label=cf.body2_legend_label, zorder=5)
            ax.add_patch(body1), ax.add_patch(body2)
        else:
            ax.scatter(x1[-1], y1[-1], color=cf.body1_colour, s=cf.body1_markersize, zorder=5)
            ax.scatter(x2[-1], y2[-1], color=cf.body2_colour, s=cf.body2_markersize, zorder=5)

        # --- TEXT: CURRENT TIME STEP (DAYS) --- #
        if cf.display_time:
            self.t_days = t / (24 * 60 * 60)
            xpos, ypos = cf.time_text_pos
            ax.text(
                xpos, ypos,   # position in axes coordinates (0, 0) bottom left, (1, 1) top right
                f"T = {self.t_days[-1]:.{cf.time_dp}f} days", 
                transform=ax.transAxes,    # map coordinates from axes to figure coordinates
                fontsize=cf.title_fontsize - 1,
            )     

        fig.set_tight_layout(True)    # adjust layout to fit all elements
        plt.show()

    def animate2d(
        self, 
        show_plot_first: bool = True,
        fps: int = 60, 
        bitrate: int = 50_000,
        dpi: int = 200,
    ) -> None:
        """Animate the 2D orbits of the two-body system. Writes to an MP4 file."""

        cf = self.config    # shorthand alias for the PlotConfig instance
        self._check_solve()    
        t, Z = self.t, self.Z

        # show the static complete orbit trails (final positions):
        if show_plot_first:
            print(f"\nplotting final positions with complete orbit trails...")
            print("<CLOSE FIGURE WINDOW TO START ANIMATION WRITING>")
            self.plot_orbits2d()    # figure window must be closed before continuing 

        # key animation statistics
        interval = int(1000 / fps)     # convert FPS to milliseconds
        steps = len(t)                                      # total number of time steps from the ODE solver
        trail_length = int((cf.trail_length_pct / 100) * steps) 
        print(f"\n# ---------- 2D ANIMATION ---------- #")
        print(f"{steps:,} steps @ {fps} fps (~{interval * 1e-3:.3f} sec/frame) and {dpi} DPI")
        duration = steps / fps
        print(f"total video duration: {duration:.2f} sec ({duration / 60:.1f} min)")
        print(f"writing {steps} frames to MP4...\n")

        # ----- EXTRACT COORDINATES ----- #
        r1, _, r2, _ = np.vsplit(Z, 4)    # unpack state vector (split along rows)
        # project to orbital plane if head-on view is enabled:
        if self.params.head_on_view:    
            (x1, y1), (x2, y2) = self._project_to_orbital_plane(r1, r2)
        else:
            # unpack 2D coordinates directly (assumes r1 and r2 are already in 2D):
            x1, y1 = r1[0, :], r1[1, :]         # unpack body 1 2D coordinates
            x2, y2 = r2[0, :], r2[1, :]         # unpack body 2 2D coordinates

        # ----- FIGURE SETUP ----- #
        fig, ax = self._create_figure2d((x1, x2), (y1, y2))   # create a 2D base figure with axes

        # --- PLOT ELEMENTS (TO BE UPDATED IN ANIMATION) --- #
        if cf.to_scale:    # show planetary bodies to scale
            body1_marker = mpatches.Circle((x1[0], y1[0]), radius=cf.body1_radius, color=cf.body1_colour, label=cf.body1_legend_label, zorder=5)
            body2_marker = mpatches.Circle((x2[0], y2[0]), radius=cf.body2_radius, color=cf.body2_colour, label=cf.body2_legend_label, zorder=5)
            ax.add_patch(body1_marker), ax.add_patch(body2_marker)
        else:
            body1_marker = ax.scatter([], [], color=cf.body1_colour, s=cf.body1_markersize, zorder=5)
            body2_marker = ax.scatter([], [], color=cf.body2_colour, s=cf.body2_markersize, zorder=5)
        body1_orbit, = ax.plot([], [], color=cf.body1_trail_colour, linewidth=cf.line_width)
        body2_orbit, = ax.plot([], [], color=cf.body2_trail_colour, linewidth=cf.line_width)
        if cf.display_time:
            xpos, ypos = cf.time_text_pos
            time_text = ax.text(xpos, ypos, "", transform=ax.transAxes, fontsize=cf.title_fontsize-1)

        # --- PROGRESS BAR --- #
        pbar = tqdmFA(total=steps, fps=fps)

        # ----- ANIMATION FUNCTION SETUP ----- #
        def _init() -> tuple:
            body1_orbit.set_data([], [])
            body2_orbit.set_data([], [])
            if cf.to_scale:
                body1_marker.center = (x1[0], y1[0])
                body2_marker.center = (x2[0], y2[0])
            else:
                body1_marker.set_offsets([[x1[0], y1[0]]])
                body2_marker.set_offsets([[x2[0], y2[0]]])
            time_text.set_text(f"T = {self.t_days[0]:.1f} days")    # set initial time text
            return body1_orbit, body2_orbit, body1_marker, body2_marker

        def _update(frame) -> tuple:
            # update orbit trails:
            i0 = max(0, frame - trail_length)    # start index for the trail
            i0_1 = max(0, frame - int(trail_length * cf.trail_length_factor))      # longer trail for body 1
            body1_orbit.set_data(x1[i0_1: frame + 1], y1[i0_1: frame + 1])      # update orbit trail
            body2_orbit.set_data(x2[i0: frame + 1], y2[i0: frame + 1])
            # update markers:
            if cf.to_scale:
                body1_marker.center = (x1[frame], y1[frame])    # update matplotlib.patches.Circle position
                body2_marker.center = (x2[frame], y2[frame])
            else:
                body1_marker.set_offsets([[x1[frame], y1[frame]]])    # update scatter marker position
                body2_marker.set_offsets([[x2[frame], y2[frame]]])
            # update current time step text display if toggled:
            if cf.display_time:
                time_text.set_text(f"T = {self.t_days[frame]:.{cf.time_dp}f} days")
            pbar.update(1)    # update tqdm progress bar
            return body1_orbit, body2_orbit, body1_marker, body2_marker

        # --- WRITE AND SAVE TO FILE --- #
        ani = FuncAnimation(fig, _update, frames=range(steps), init_func=_init, interval=interval, blit=True)
        writer = FFMpegWriter(fps=fps, bitrate=bitrate)
        # generate a file name:
        name1, name2 = cf.body1_legend_label, cf.body2_legend_label
        if not name1 or not name2:
            name1, name2 = "body1", "body2"
        time_period = self.params.T_days
        file_name = (
            f"2D_{name1.lower()}-{name2.lower()}_"
            f"{dpi=}_"
            f"{cf.trail_length_pct:.0f}%trail(factor={cf.trail_length_factor})_"
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

    def create_figure3d(
        self, 
        x_coords: Tuple[np.ndarray, np.ndarray],
        y_coords: Tuple[np.ndarray, np.ndarray],
        z_coords: Tuple[np.ndarray, np.ndarray],
    ) -> Tuple[plt.Figure, Axes3D]:
        """Setup a 3D base figure with axes for plotting the two-body system."""
        cf, cf_3d = self.config, self.config_3d    # shorthand alias for the PlotConfig instances

        # ----- 3D FIGURE SETUP ----- #
        fig = plt.figure(figsize=cf_3d.figure_size)
        ax = fig.add_subplot(111, projection="3d")
        if cf_3d.figure_title:
            ax.set_title(cf_3d.figure_title, fontdict={"fontsize": cf.title_fontsize})
        ax.set_xlabel(r"$x$ ($m$)")
        ax.set_ylabel(r"$y$ ($m$)")
        ax.set_zlabel(r"$z$ ($m$)")
        ax.xaxis.set_major_locator(MaxNLocator(cf.x_axis_max_ticks))
        ax.yaxis.set_major_locator(MaxNLocator(cf.y_axis_max_ticks))
        ax.zaxis.set_major_locator(MaxNLocator(cf_3d.z_axis_max_ticks))
        ax.set_box_aspect([1, 1, 1])    # set equal aspect ratio for all axes

        # --- AXIS LIMITS --- #
        all = np.concatenate((x_coords, y_coords, z_coords))
        max_extent = cf_3d.max_axis_extent3d * np.max(np.abs(all))    # calculate max extent based on coordinates
        ax.set_xlim3d(-max_extent, max_extent)
        ax.set_ylim3d(-max_extent, max_extent)
        ax.set_zlim3d(-max_extent, max_extent)  # equal limits for equal aspect ratio 

        # --- DASHED LINES --- #
        if cf_3d.draw_dashes3d:
            x_dash = ax.plot([-max_extent, max_extent], [0, 0], [0, 0], linestyle="--", color="black", alpha=cf.dashed_line_alpha, linewidth=cf.dashed_line_width, zorder=1)
            y_dash = ax.plot([0, 0], [-max_extent, max_extent], [0, 0], linestyle="--", color="black", alpha=cf.dashed_line_alpha, linewidth=cf.dashed_line_width, zorder=1)
            z_dash = ax.plot([0, 0], [0, 0], [-max_extent, max_extent], linestyle="--", color="black", alpha=cf.dashed_line_alpha, linewidth=cf.dashed_line_width, zorder=1)
            x_dash[0].set_dashes([10, 10]), y_dash[0].set_dashes([10, 10]), z_dash[0].set_dashes([10, 10])

        # show barycentre marker (lies exactly at the origin):
        if cf_3d.display_baryc:     
            ax.scatter(0, 0, 0, marker="x", s=cf_3d.baryc_markersize, color=cf.baryc_colour, label=cf.baryc_legend_label, alpha=cf.baryc_alpha)
    
        if cf_3d.display_legend:
            # Create proxy markers for legend (independent of actual plotted objects)
            body1_legend = mlines.Line2D([], [], color=cf.body1_colour, marker='o', linestyle='None', markersize=6, label=cf.body1_legend_label)
            body2_legend = mlines.Line2D([], [], color=cf.body2_colour, marker='o', linestyle='None', markersize=6, label=cf.body2_legend_label)
            baryc_legend = mlines.Line2D([], [], color=cf.baryc_colour, marker='x', linestyle='None', markersize=6, label=cf.baryc_legend_label)
            ax.legend(handles=[body1_legend, body2_legend, baryc_legend])

        # --- CAMERA VIEW ANGLE (ELEVATION & AZIMUTH) --- #
        ax.view_init(elev=cf_3d.elev_start, azim=cf_3d.azim_start)    # set initial starting angles

        return fig, ax

    def plot_orbits3d(self) -> None:
        """Plot the complete orbits of the two bodies in 3D."""
        cf, cf_3d = self.config, self.config_3d    
        t, Z = self.t, self.Z

        r1, _, r2, _ = np.vsplit(Z, 4)            # unpack state vector (split along rows)
        x1, y1, z1 = r1[0, :], r1[1, :], r1[2, :]   # unpack body 1 3D coordinates
        x2, y2, z2 = r2[0, :], r2[1, :], r2[2, :]   # unpack body 2 3D coordinates

        # ----- MAIN 3D FIGURE ----- #
        fig, ax = self.create_figure3d((x1, x2), (y1, y2), (z1, z2))    # create a 3D base figure with axes

        # --- PLOT FULL ORBIT TRAILS --- #
        ax.plot(x1, y1, z1, color=cf.body1_trail_colour, linewidth=cf.line_width)
        ax.plot(x2, y2, z2, color=cf.body2_trail_colour, linewidth=cf.line_width)

        # --- ADD MARKERS --- #
        if cf_3d.markers_to_relative_scale:    # show bodies to scale
            # relative scale of marker2 size based on defined marker1 size
            size2 = cf_3d.body1_markersize * (cf.body2_radius / cf.body1_radius) ** 2    # scale by the square of the radius ratio
        else:
            size2 = cf_3d.body1_markersize
        ax.scatter(x1[-1], y1[-1], z1[-1], color=cf.body1_colour, s=cf_3d.body1_markersize, label=cf.body1_legend_label, zorder=5)
        ax.scatter(x2[-1], y2[-1], z2[-1], color=cf.body2_colour, s=size2, label=cf.body2_legend_label, zorder=5)

        # --- TEXT: CURRENT TIME STEP (DAYS) --- #
        if cf_3d.display_time:
            self.t_days = t / (24 * 60 * 60)
            xpos, ypos = cf_3d.time_text_pos
            ax.text2D(
                xpos, ypos,   # position in axes coordinates (0, 0) bottom left, (1, 1) top right
                f"T = {self.t_days[-1]:.{cf.time_dp}f} days", 
                transform=ax.transAxes,    # map coordinates from axes to figure coordinates
                fontsize=cf.title_fontsize - 1,
            )

        fig.set_tight_layout(True)    # adjust layout to fit all elements
        plt.show()

    def animate3d(
        self, 
        show_plot_first: bool = True,
        fps: int = 60, 
        dpi: int = 200, 
        bitrate: int = 50_000,
    ) -> None:
        """Animate the 3D orbits of the two-body system. Writes to an MP4 file."""
        
        cf, cf_3d = self.config, self.config_3d    # shorthand alias for the PlotConfig instance
        self._check_solve()    
        t, Z = self.t, self.Z

        # show the static complete orbit trails (final positions) first:
        if show_plot_first:
            print(f"\nplotting final positions with complete orbit trails...")
            print("<CLOSE FIGURE WINDOW TO START ANIMATION WRITING>")
            self.plot_orbits3d()    # figure window must be closed before continuing 

        # key animation statistics
        interval = int(1000 / fps)      # convert FPS to milliseconds
        steps = len(t)                  # total number of time steps from the ODE solver
        trail_length = int((cf.trail_length_pct / 100) * steps) 
        print(f"\n# ---------- 3D ANIMATION ---------- #")
        print(f"{steps:,} steps @ {fps} fps (~{interval * 1e-3:.3f} sec/frame), {dpi=}")
        duration = steps / fps
        print(f"total video duration: {duration:.2f} sec ({duration / 60:.1f} min)")
        print(f"writing {steps} frames to MP4...\n")

        # ----- EXTRACT COORDINATES ----- #
        r1, v1, r2, v2 = np.vsplit(Z, 4)            # unpack state vector (split along rows)
        x1, y1, z1 = r1[0, :], r1[1, :], r1[2, :]   # unpack body 1 3D coordinates
        x2, y2, z2 = r2[0, :], r2[1, :], r2[2, :]   # unpack body 2 3D coordinates

        # ----- BASE 3D FIGURE SETUP ----- #
        fig, ax = self.create_figure3d((x1, x2), (y1, y2), (z1, z2)) 

        # --- PLOT ELEMENTS (TO BE UPDATED IN ANIMATION) --- #
        if cf_3d.markers_to_relative_scale:    # show planetary bodies to scale
            # relative scale of marker2 size based on defined marker1 size
            size2 = cf_3d.body1_markersize * (cf.body2_radius / cf.body1_radius) ** 2    # scale by the square of the radius ratio
        else:
            size2 = cf_3d.body1_markersize
        body1_marker = ax.scatter(x1[0], y1[0], z1[0], color=cf.body1_colour, s=cf_3d.body1_markersize, label=cf.body1_legend_label, zorder=5)
        body2_marker = ax.scatter(x2[0], y2[0], z2[0], color=cf.body2_colour, s=size2, label=cf.body2_legend_label, zorder=5)
        body1_orbit, = ax.plot([], [], [], color=cf.body1_trail_colour, linewidth=cf.line_width)
        body2_orbit, = ax.plot([], [], [], color=cf.body2_trail_colour, linewidth=cf.line_width)
        if cf_3d.display_time:
            xpos, ypos = cf_3d.time_text_pos
            time_text = ax.text2D(xpos, ypos, "", transform=ax.transAxes, fontsize=cf.title_fontsize-1)

        # --- PROGRESS BAR --- #
        pbar = tqdmFA(total=steps, fps=fps)

        # ----- ANIMATION FUNCTION SETUP ----- #
        def _init() -> tuple:
            body1_orbit.set_data([], []), body2_orbit.set_data([], [])
            if cf_3d.markers_to_relative_scale:    
                body1_marker.center = (x1[0], y1[0], z1[0])
                body2_marker.center = (x2[0], y2[0], z2[0])
            else:
                body1_marker._offsets3d = ([x1[0]], [y1[0]], [z1[0]])
                body2_marker._offsets3d = ([x2[0]], [y2[0]], [z2[0]])
            if cf_3d.display_time:
                time_text.set_text(f"T = {self.t_days[0]:.{cf.time_dp}f} days")            # set initial time text
            ax.view_init(elev=cf_3d.elev_start, azim=cf_3d.azim_start)      # set initial camera angles
            return body1_orbit, body2_orbit, body1_marker, body2_marker

        def _update(frame) -> tuple:
            # update orbit trails:
            i0 = max(0, frame - trail_length)                                       # start index for the trail
            i0_1 = max(0, frame - int(trail_length * cf.trail_length_factor))       # longer trail for body 1
            body1_orbit.set_data_3d(x1[i0_1:frame], y1[i0_1:frame], z1[i0_1:frame])    # update orbit trail
            body2_orbit.set_data_3d(x2[i0:frame], y2[i0:frame], z2[i0:frame])
            # update markers:
            body1_marker._offsets3d = ([x1[frame]], [y1[frame]], [z1[frame]])
            body2_marker._offsets3d = ([x2[frame]], [y2[frame]], [z2[frame]])
            # camera panning:
            if cf_3d.camera_pan:
                a0, a1 = cf_3d.azim_start, cf_3d.azim_end
                e0, e1 = cf_3d.elev_start, cf_3d.elev_end
                # update only if the end angles are specified:
                a_next = a0 + (a1 - a0) * frame / steps if a1 is not None else a0     # 'not None' otherwise 0 is False
                e_next = e0 + (e1 - e0) * frame / steps if e1 is not None else e0
                ax.view_init(elev=e_next, azim=a_next)
            # update current time step text display if toggled:
            if cf_3d.display_time:
                time_text.set_text(f"T = {self.t_days[frame]:.{cf.time_dp}f} days")
            pbar.update(1)
            return body1_orbit, body2_orbit, body1_marker, body2_marker

        # --- WRITE AND SAVE TO FILE --- #
        ani = FuncAnimation(fig, _update, frames=range(steps), init_func=_init, interval=interval, blit=True)
        writer = FFMpegWriter(fps=fps, bitrate=bitrate)
        # generate a file name:
        name1, name2 = cf.body1_legend_label, cf.body2_legend_label
        if not name1 or not name2:
            name1, name2 = "body1", "body2"
        time_period = self.params.T_days
        file_name = (
            f"3D_{name1.lower()}-{name2.lower()}_"
            f"{dpi=}_"
            f"(e0={cf_3d.elev_start},a0={cf_3d.azim_start})to"
            f"(ef={cf_3d.elev_end},af={cf_3d.azim_end})_"
            f"{cf.trail_length_pct:.0f}%trail(factor={cf.trail_length_factor})_"
            f"{steps=:,.0f}_"
            f"T_days={time_period:.1f}_"
            f"to_scale={cf_3d.markers_to_relative_scale}_"
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


def pluto_charon_system() -> None:
    """Simulate and animate the Pluto-Charon two-body system with realistic parameters."""
    pluto_charon = TwoBodySystem(
        params=SystemParams(
            m1=M_PLUTO, 
            m2=M_CHARON, 
            d=D_PLUTO_CHARON, 
            v0=V_CHARON, 
            i_deg=360 - i_CHARON,
            head_on_view=True,      # head-on view for 2D plot (project to orbital plane)
            T_days=T_PLUTO_CHARON * 1.166 * 4,
            rtol=1e-12, atol=1e-9, steps=600,
            ode_method="RK45",    # use a high-order ODE solver for better accuracy
        ),
        config=PlotConfig(
            body1_radius=R_PLUTO, 
            body2_radius=R_CHARON,
            figure_size=(8, 8),
            figure_title="2D Pluto-Charon System - Flat Orbit (TO SCALE)",
            title_fontsize=11,
            body1_legend_label="Pluto", 
            body2_legend_label="Charon",
            body1_colour="tab:brown", 
            body1_trail_colour="tab:brown",
            body2_colour="tab:olive", 
            body2_trail_colour="tab:olive",
            baryc_colour="tab:blue",    # barycentre colour
            display_baryc=True,
            max_axis_extent2d=1.1,
            trail_length_pct=6,
            trail_length_factor=2.5,
            display_legend=True, 
            to_scale=True, 
        )
    )
    # setup 3D plot configuration dataclass:
    pluto_charon.config_3d = PlotConfig3D(
        markers_to_relative_scale=True,         
        body1_markersize=500,    # size of body2 is scaled if markers_to_relative_scale=True
        max_axis_extent3d=1,
        # -- camera panning during animation -- #
        elev_start=20, azim_start=-75,
        camera_pan=True,
        elev_end=10, azim_end=-30,
        # -- title and legend -- #
        # figure_title="3D Pluto-Charon System",
        figure_size=(10, 10),
        display_legend=False
    )

    # plot only the complete orbits in 2D and 3D:
    pluto_charon.plot_orbits2d()        # 2D
    pluto_charon.plot_orbits3d()        # 3D

    # create 2D and 3D animations:
    pluto_charon.animate2d(dpi=100, show_plot_first=False)     # 2D
    pluto_charon.animate3d(dpi=100, show_plot_first=False)     # 3D


def equal_mass_system() -> None:
    """Simulate and animate a two-body system with equal masses."""
    mass = M_EARTH              # use Earth mass for both bodies
    radius = R_EARTH            # use Earth radius for both bodies (if to scale)
    distance = D_EARTH_MOON     # use Earth-Moon distance
    equal_mass = TwoBodySystem(
        params=SystemParams(
            m1=mass, m2=mass, d=distance, 
            v0=600, 
            i_deg=10, 
            T_days=26 * 3,
            rtol=1e-9, 
            steps=600
        ),
        config=PlotConfig(
            body1_radius=radius * 1.5, body2_radius=radius * 1.5,
            body1_colour="tab:red", body1_trail_colour="tab:red",
            body2_colour="tab:green", body2_trail_colour="tab:green",
            figure_size=(10, 10), max_axis_extent2d=1.1, y_axis_limits=(-2.5e8, 2.5e8),
            to_scale=True, display_legend=False,
            display_baryc=True, baryc_colour="tab:blue",          
            trail_length_pct=8, 
            trail_length_factor=1,
            time_dp=0,    # no decimal places for time text
        )
    )
    # setup 3D plot configuration dataclass:
    equal_mass.config_3d = PlotConfig3D(
        markers_to_relative_scale=True,         # not recommened to use spheres for drawing (set False)
        body1_markersize=200,                   # size of body2 is scaled if markers_to_relative_scale=True
        max_axis_extent3d=1,
        # camera panning during animation:
        elev_start=20, azim_start=-30,
        camera_pan=True,
        elev_end=50, azim_end=-75,
        # title and legend:
        figure_size=(10, 10),
        display_legend=False,
    )

    # plot only the complete orbits in 2D and 3D:
    # equal_mass.plot_orbits2d()        # 2D
    # equal_mass.plot_orbits3d()        # 3D

    # create 2D and 3D animations:
    equal_mass.animate2d(dpi=300)    # create animation with 2D figure
    equal_mass.animate3d(dpi=300)    # create animation with 3D figure


def earth_moon_system(exaggerated: bool = False) -> None:
    """Simulate and animate the Earth-Moon two-body system with realistic (optional exaggeration toggle) parameters."""
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
                max_axis_extent2d=1.05,
                line_width=0.5,
                to_scale=True, 
                display_baryc=False,
                trail_length_pct=2,
                trail_length_factor=4,
            )
        )
        earth_moon.animate2d(dpi=200)
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
                max_axis_extent2d=1.4, 
                x_axis_limits=(-1e7, 4.5e7),
                display_legend=True, 
                to_scale=False, 
                display_baryc=True, baryc_alpha=0.8,
                trail_length_pct=2,
                trail_length_factor=5

            )
        )
        earth_moon.animate2d(dpi=200)    # create animation with 2D figure