import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec
import datetime
from constants import *
from tqdm_pbar import tqdmFA
from config import SystemParams, PlotConfig, PlotConfig3D
from typing import Optional, Tuple
from ode_systems import two_body_ode
import inspect
# configure matplotlib defaults:
plt.rcParams.update({
    "font.size": 12.5,
    "font.family": "Latin Modern Roman",
    "mathtext.fontset": "cm",
    "lines.linewidth": 1,
    "xtick.labelsize": 9,    # smaller ticker markers
    "ytick.labelsize": 9,
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
        self.body1_name = self.config.body1_legend_label if self.config.body1_legend_label else "Body1"
        self.body2_name = self.config.body2_legend_label if self.config.body2_legend_label else "Body2"
        self.t = None
        self.r1 = None    # position vector of body 1 (e.g., Earth)
        self.r2 = None    # position vector of body 2 (e.g., Moon)
        # body 1 & 2 coordinates:
        self.x1 = None
        self.y1 = None
        self.z1 = None
        self.x2 = None
        self.y2 = None
        self.z2 = None
        # velocity vectors:
        self.v1 = None
        self.v2 = None
        # body 1 & 2 coordinates projected onto the orbital plane:
        self.x1_proj = None
        self.y1_proj = None
        self.x2_proj = None
        self.y2_proj = None
        # automatically solve the system upon initialisation and populate attributes:
        self._solve()    
        self.steps = len(self.t)    # number of time steps from the ODE solver
        # attributes for animation writing - see _animation_params():
        self.used_steps = None      # number of frames to write to MP4
        self.trail_length = None    # trail length of masses in frames
        self.interval = None        # seconds per frame (1000/fps milliseconds)
        self.file_prefix = None

    def inspect_class(self) -> None:
        """Inspect and print the attributes and methods of the TwoBodySystem instance."""
        print("\ninspecting all class attributes...\n")
        for attr, value in vars(self).items():
            if isinstance(value, np.ndarray):
                print(f"{attr}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"{attr}: {value}")

        print("\ninspecting all class methods...\n")
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        for name, method in methods:
            if name.startswith("__"):
                continue  # Skip dunder methods
            sig = inspect.signature(method)
            print(f"{name}{sig}")

    def _solve(self) -> None:
        """Numerically solve the ODEs for the two-body system using the provided parameters."""

        # get the ODE system function f(t, Z):
        func, Z0 = two_body_ode(self.params)

        # extract system parameters from the dataclass:
        p = self.params
        m1, m2, d = p.m1, p.m2, p.d
        T_days, steps = p.T_days, p.steps
        ode_method, rtol, atol = p.ode_method, p.rtol, p.atol
        
        # ----- EVALUATION & SOLVE ----- #
        orbital_period = 2 * np.pi * np.sqrt(d**3 / (G * (m1 + m2)))    # orbital period of the Earth-Moon system
        T = T_days * 24 * 3600                                          # one lunar orbit (s)
        t_span = (0, T) if T > 0 else (0, orbital_period) 
        t_eval = np.linspace(t_span[0], t_span[1], steps)               # time points at which to store the solution
        dt = t_eval[1] - t_eval[0]                                      # time eval step size
        # print logging report:
        print(f"\n# ======== {self.body1_name}-{self.body2_name} Two-Body System ======== #")
        print(f"running ODE solver ({ode_method=})...")
        print(f"using {rtol=:.0e}, {atol=:.0e} (default: rtol=1e-3, atol=1e-6)")
        print(f"t_eval=({t_eval[0]:.0f}, {t_eval[-1]:,.0f}), {steps=:,}, dt≈{dt:.2f}")
        sol = solve_ivp(func, t_span, Z0, t_eval=t_eval, method=ode_method, rtol=rtol, atol=atol)
        
        # ----- EXTRACT OUTPUTS ----- #
        t, Z = sol.t, sol.y    # unpack time and state vector from the solution
        print(f"\nsolver success: {sol.success} ({sol.nfev:,} nfev)")
        print(f"t.shape: {t.shape}, Z.shape: {Z.shape}")
        print("storing all result arrays as attributes...")

        # ----- STORE OUTPUT ARRAYS AS ATTRIBUTES ----- #
        self.t = t
        self.r1, self.v1, self.r2, self.v2 = np.split(Z, 4)    
        # unpack and store coordinates for body 1 and body 2:
        self.x1, self.y1, self.z1 = self.r1[0, :], self.r1[1, :], self.r1[2, :]
        self.x2, self.y2, self.z2 = self.r2[0, :], self.r2[1, :], self.r2[2, :]
        # project the 3D coordinates onto the orbital plane for head-on 2D viewing:
        (self.x1_proj, self.y1_proj), (self.x2_proj, self.y2_proj) = self._project_to_orbital_plane(self.r1, self.r2)

    def _project_to_orbital_plane(
        self, 
        r1: np.ndarray, 
        r2: np.ndarray,
        verbose: bool = False,
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

        if self.params.rotate_proj_90cw:
            # rotate the coordinates 90 degrees counter-clockwise around the z-axis
            return (y1, -x1), (y2, -x2)    # swap x and y coordinates with a sign change

        return (x1, y1), (x2, y2)

    def _create_figure2d(self, fig: plt.Figure, gs: GridSpec) -> plt.Axes:
        """Helper function to populate the base figure and axes used for plotting orbits in 2D."""
        cf = self.config    
        ax = fig.add_subplot(gs[0, 0])    
        if cf.figure_title:
            ax.set_title(cf.figure_title)
        ax.xaxis.set_major_locator(MaxNLocator(cf.x_axis_max_ticks))
        ax.yaxis.set_major_locator(MaxNLocator(cf.y_axis_max_ticks))
        ax.set_xlabel(r"$x$ ($m$)")
        ax.set_ylabel(r"$y$ ($m$)")

        # dashed cross lines:
        ax.grid(True, alpha=cf.grid_alpha)
        hline = ax.axhline(0, linestyle="--", color="black", alpha=cf.dashed_line_alpha, linewidth=cf.dashed_line_width, zorder=1)
        vline = ax.axvline(0, linestyle="--", color="black", alpha=cf.dashed_line_alpha, linewidth=cf.dashed_line_width, zorder=1)
        hline.set_dashes([10, 10])
        vline.set_dashes([10, 10])

        # show barycentre marker (origin):
        if cf.display_baryc:     
            ax.scatter(0, 0, marker="x", s=cf.baryc_markersize, color=cf.baryc_colour, label=cf.baryc_legend_label, alpha=cf.baryc_alpha, zorder=10)
        
        # set axis limits based on projected coordinates or original 2D coordinates:
        if self.params.head_on_view:
            # use projected coordinates to calculate axis limits:
            x1, y1, x2, y2 = self.x1_proj, self.y1_proj, self.x2_proj, self.y2_proj
        else:
            # unpack 2D coordinates normally:
            x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2    

        # set axis limits and aspect ratio:
        if cf.x_axis_limits:
            ax.set_xlim(cf.x_axis_limits)
        else:
            x_all = np.concatenate((x1, x2))
            x_extent = cf.max_axis_extent2d * np.max(np.abs(x_all))
            ax.set_xlim(-x_extent, x_extent)
        if cf.y_axis_limits:
            ax.set_ylim(cf.y_axis_limits)
        else:
            y_all = np.concatenate((y1, y2))
            y_extent = cf.max_axis_extent2d * np.max(np.abs(y_all))
            ax.set_ylim(-y_extent, y_extent)
        ax.set_aspect("equal")    

        # proxy markers for legend:
        if cf.display_legend:
            body1_legend = mlines.Line2D([], [], color=cf.body1_colour, marker='o', linestyle='None', markersize=6, label=cf.body1_legend_label)
            body2_legend = mlines.Line2D([], [], color=cf.body2_colour, marker='o', linestyle='None', markersize=6, label=cf.body2_legend_label)
            baryc_legend = mlines.Line2D([], [], color=cf.baryc_colour, marker='x', linestyle='None', markersize=6, label=cf.baryc_legend_label)
            ax.legend(handles=[body1_legend, body2_legend, baryc_legend], fontsize=cf.legend_fontsize, markerscale=cf.legend_markerscale, handletextpad=0.2)
        
        return ax

    def _plot_orbits2d(self, fig: plt.Figure, gs: GridSpec) -> plt.Axes:
        """Helper method to populate a figure and axes with 2D orbits."""
        cf = self.config    # shorthand alias for the PlotConfig instance
        if self.params.head_on_view:
            # use projected coordinates for head-on viewing:
            x1, y1 = self.x1_proj, self.y1_proj    
            x2, y2 = self.x2_proj, self.y2_proj
        else:
            # use original 2D coordinates:
            x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2   

        # create a 2D base figure with axes:
        ax = self._create_figure2d(fig, gs)    

        # plot 2D orbit trajectories:
        ax.plot(x1, y1, color=cf.body1_trail_colour, linewidth=cf.line_width)
        ax.plot(x2, y2, color=cf.body2_trail_colour, linewidth=cf.line_width)

        # display final positions of the bodies:
        if cf.to_scale:    
            # draw planetary bodies to scale:
            body1 = mpatches.Circle((x1[-1], y1[-1]), radius=cf.body1_radius, color=cf.body1_colour, label=cf.body1_legend_label, zorder=5)
            body2 = mpatches.Circle((x2[-1], y2[-1]), radius=cf.body2_radius, color=cf.body2_colour, label=cf.body2_legend_label, zorder=5)
            ax.add_patch(body1), ax.add_patch(body2)
        else:
            ax.scatter(x1[-1], y1[-1], color=cf.body1_colour, s=cf.body1_markersize, zorder=5)
            ax.scatter(x2[-1], y2[-1], color=cf.body2_colour, s=cf.body2_markersize, zorder=5)

        # display time step in days if toggled:
        if cf.display_time:
            t_days = self.t / (24 * 60 * 60)
            xpos, ypos = cf.time_text_pos
            ax.text(
                xpos, ypos,   # position in axes coordinates (0, 0) bottom left, (1, 1) top right
                f"T = {t_days[-1]:.{cf.time_dp}f} days", 
                transform=ax.transAxes,    # map coordinates from axes to figure coordinates
            )     
        return ax
    
    def plot_orbits2d(self) -> Tuple[plt.Figure, GridSpec, plt.Axes]:
        """Plot the complete orbits of the two bodies in 2D."""
        fig = plt.figure(figsize=self.config .figure_size)
        gs = GridSpec(nrows=1, ncols=1, figure=fig)    # create a single subplot grid
        ax = self._plot_orbits2d(fig, gs)
        plt.tight_layout(), plt.show()    
        return fig, gs, ax

    def _get_filename(self) -> str:
        """Generate a filename for animation MP4 files based on system and configuration parameters."""
        p, cf = self.params, self.config
        name1, name2 = self.body1_name.lower(), self.body2_name.lower()
        prefix = f"{self.file_prefix}_" if self.file_prefix else ""
        return (
            f"{prefix}"
            f"{name1}-{name2}_"
            f"dpi={cf.dpi}_"
            f"{cf.trail_length_pct:.0f}%trail(factor={cf.trail_length_factor})_"
            f"{self.steps=:,.0f}_"
            f"T_days={p.T_days:.1f}_"
            f"vid_dur={cf.video_duration:.0f}_"
            f"{cf.to_scale=}"
            f".mp4"
        )

    def _animation_params(self, dim: str) -> None:
        """Set animation parameters as attributes based on the configuration."""
        cf = self.config    # shorthand alias for the PlotConfig instance

        # calculate key animation parameters:
        self.used_steps = int(cf.fps * cf.video_duration)    # number of steps for the animation
        self.trail_length = int((cf.trail_length_pct / 100) * self.used_steps)    # trail length in frames
        self.interval = 1000 / cf.fps    # milliseconds per frame 
        
        # check if the number of steps is sufficient for the video duration:
        if self.used_steps > self.steps:
            raise ValueError(f"{self.used_steps:,} frames required for a {cf.video_duration:.1f} sec video at {cf.fps} FPS, but only {self.steps:,.0f} frames ({self.steps / cf.fps:.1f} sec) available. Increase `steps` or reduce `video_duration`.")
        
        # print animation parameters and stats:
        name1, name2 = self.body1_name, self.body2_name
        print(f"\n# ------- {dim} Animation of {name1}-{name2} System ------- #")
        print(f"using {self.used_steps}/{len(self.t)} time steps => {cf.video_duration:.1f} sec video duration")
        print(f"{self.used_steps:,} frames @ {cf.fps} fps (~{self.interval * 1e-3:.3f} sec/frame), dpi={cf.dpi}")
        print(f"writing {self.used_steps} frames to MP4...\n")

    def _write_to_mp4(self, ani: FuncAnimation, pbar: tqdmFA) -> None:
        """Write the animation to an MP4 file using the configured parameters."""
        cf = self.config    
        
        # start animation:
        writer = FFMpegWriter(fps=cf.fps, bitrate=cf.bit_rate)
        file_name = self._get_filename()    # generate filename for the animation
        ani.save(filename=file_name, writer=writer, dpi=cf.dpi)
        pbar.close()    # close progress bar
        print(f"\nanimation saved as '{file_name}'")

        # quick report:
        elapsed = int(pbar.format_dict["elapsed"])
        time = datetime.timedelta(seconds=elapsed)
        print(f"\ntotal elapsed time: {time}")
        avg_iter_per_sec = self.used_steps / time.total_seconds()
        if 1 / avg_iter_per_sec < 1:
            avg_rate = f"{1 / avg_iter_per_sec * 1e3:.0f} ms/frame"
        else:
            avg_rate = f"{1 / avg_iter_per_sec:.2f} sec/frame"
        print(f"{avg_iter_per_sec:.1f} frames/sec processed ({avg_rate})")

    def animate2d(self, show_plot_first: bool = True) -> None:
        """Animate the 2D orbits of the two-body system. Writes to an MP4 file."""
        cf = self.config

        # show the static complete orbit trails (final positions):
        if show_plot_first:
            print(f"\nplotting final positions with complete orbit trails...")
            print("<CLOSE FIGURE WINDOW TO START ANIMATION WRITING>")
            self.plot_orbits2d()    # figure window must be closed before continuing     

        # set axis limits based on projected coordinates or original 2D coordinates:
        if self.params.head_on_view:
            # use projected coordinates to calculate axis limits:
            x1, y1, x2, y2 = self.x1_proj, self.y1_proj, self.x2_proj, self.y2_proj
        else:
            # unpack 2D coordinates normally:
            x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2  
        
        # set animation parameters and print key stats:
        self._animation_params(dim="2D")

        # base 2D figure and axes setup:
        fig = plt.figure(figsize=self.config_3d.figure_size)
        gs = GridSpec(nrows=1, ncols=1, figure=fig)
        ax = self._create_figure2d(fig, gs)

        # initialise plot elements:
        if cf.to_scale:    # show planetary bodies to scale
            body1_marker = mpatches.Circle((x1[0], y1[0]), radius=cf.body1_radius, color=cf.body1_colour, label=cf.body1_legend_label, zorder=5)
            body2_marker = mpatches.Circle((x2[0], y2[0]), radius=cf.body2_radius, color=cf.body2_colour, label=cf.body2_legend_label, zorder=5)
            ax.add_patch(body1_marker), ax.add_patch(body2_marker)
        else:
            body1_marker = ax.scatter([], [], color=cf.body1_colour, s=cf.body1_markersize, zorder=5)
            body2_marker = ax.scatter([], [], color=cf.body2_colour, s=cf.body2_markersize, zorder=5)
        body1_trail, = ax.plot([], [], color=cf.body1_trail_colour, linewidth=cf.line_width)
        body2_trail, = ax.plot([], [], color=cf.body2_trail_colour, linewidth=cf.line_width)
        if cf.display_time:
            t_days = self.t / (24 * 60 * 60)    # convert time to days
            xpos, ypos = cf.time_text_pos
            time_text = ax.text(xpos, ypos, "", transform=ax.transAxes)

        # ----- ANIMATION FUNCTIONS ----- #

        def init() -> tuple:
            body1_trail.set_data([], [])
            body2_trail.set_data([], [])
            if cf.to_scale:
                body1_marker.center = (x1[0], y1[0])
                body2_marker.center = (x2[0], y2[0])
            else:
                body1_marker.set_offsets([[x1[0], y1[0]]])
                body2_marker.set_offsets([[x2[0], y2[0]]])
            if cf.display_time:
                time_text.set_text(f"T = {t_days[0]:.1f} days")    # set initial time text
            return body1_trail, body2_trail, body1_marker, body2_marker

        def update(frame) -> tuple:
            # update orbit trails:
            i0 = max(0, frame - self.trail_length)    # start index for the trail
            i0_1 = max(0, frame - int(self.trail_length * cf.trail_length_factor))      # longer trail for body 1
            body1_trail.set_data(x1[i0_1: frame + 1], y1[i0_1: frame + 1])      # update orbit trail
            body2_trail.set_data(x2[i0: frame + 1], y2[i0: frame + 1])
            # update markers:
            if cf.to_scale:
                body1_marker.center = (x1[frame], y1[frame])    # update matplotlib.patches.Circle position
                body2_marker.center = (x2[frame], y2[frame])
            else:
                body1_marker.set_offsets([[x1[frame], y1[frame]]])    # update scatter marker position
                body2_marker.set_offsets([[x2[frame], y2[frame]]])
            # update current time step text display if toggled:
            if cf.display_time:
                time_text.set_text(f"T = {t_days[frame]:.{cf.time_dp}f} days")
            pbar.update(1)    # update tqdm progress bar
            return body1_trail, body2_trail, body1_marker, body2_marker

        # create and save animation MP4 file:
        pbar = tqdmFA(total=self.steps, fps=cf.fps)    # initialise custom tqdm progress bar
        ani = FuncAnimation(fig, update, frames=range(self.used_steps), init_func=init, blit=True)
        self.file_prefix = "2D"    # prefix for the filename
        self._write_to_mp4(ani, pbar)    # write the animation to an MP4 file

    def _create_figure3d(self, fig: plt.Figure, gs: GridSpec) -> plt.Axes:
        """Helper function to populate the base figure and axes used for plotting orbits in 3D."""
        cf, cf_3d = self.config, self.config_3d    # shorthand alias for the PlotConfig instances
        ax = fig.add_subplot(gs[0, 0], projection="3d")
        if cf_3d.figure_title:
            ax.set_title(cf_3d.figure_title)
        ax.xaxis.set_major_locator(MaxNLocator(cf_3d.num_axis_max_ticks))
        ax.yaxis.set_major_locator(MaxNLocator(cf_3d.num_axis_max_ticks))
        ax.zaxis.set_major_locator(MaxNLocator(cf_3d.num_axis_max_ticks))
        ax.set_xlabel(r"$x$ ($m$)")
        ax.set_ylabel(r"$y$ ($m$)")
        ax.set_zlabel(r"$z$ ($m$)")

        # axis limits and equal aspect ratio:
        ax.set_box_aspect([1, 1, 1])    
        all = np.concatenate((self.x1, self.y1, self.z1, self.x2, self.y2, self.z2))    # combine all coordinates
        max_extent = cf_3d.max_axis_extent3d * np.max(np.abs(all))    # calculate max extent based on coordinates
        ax.set_xlim3d(-max_extent, max_extent)
        ax.set_ylim3d(-max_extent, max_extent)
        ax.set_zlim3d(-max_extent, max_extent)

        # dashed cross lines:
        if cf_3d.draw_dashes3d:
            x_dash = ax.plot([-max_extent, max_extent], [0, 0], [0, 0], linestyle="--", color="black", alpha=cf.dashed_line_alpha, linewidth=cf.dashed_line_width, zorder=1)
            y_dash = ax.plot([0, 0], [-max_extent, max_extent], [0, 0], linestyle="--", color="black", alpha=cf.dashed_line_alpha, linewidth=cf.dashed_line_width, zorder=1)
            z_dash = ax.plot([0, 0], [0, 0], [-max_extent, max_extent], linestyle="--", color="black", alpha=cf.dashed_line_alpha, linewidth=cf.dashed_line_width, zorder=1)
            x_dash[0].set_dashes([10, 10]), y_dash[0].set_dashes([10, 10]), z_dash[0].set_dashes([10, 10])

        # show barycentre marker (origin):
        if cf_3d.display_baryc:     
            ax.scatter(0, 0, 0, marker="x", s=cf_3d.baryc_markersize, color=cf.baryc_colour, label=cf.baryc_legend_label, alpha=cf.baryc_alpha)
    
        # proxy markers for legend:
        if cf_3d.display_legend:
            body1_legend = mlines.Line2D([], [], color=cf.body1_colour, marker='o', linestyle='None', markersize=6, label=cf.body1_legend_label)
            body2_legend = mlines.Line2D([], [], color=cf.body2_colour, marker='o', linestyle='None', markersize=6, label=cf.body2_legend_label)
            baryc_legend = mlines.Line2D([], [], color=cf.baryc_colour, marker='x', linestyle='None', markersize=6, label=cf.baryc_legend_label)
            ax.legend(handles=[body1_legend, body2_legend, baryc_legend], fontsize=cf.legend_fontsize, markerscale=cf.legend_markerscale, handletextpad=0.2)

        # set initial camera view angle (elevation & azimuth):
        ax.view_init(elev=cf_3d.elev_start, azim=cf_3d.azim_start)    # set initial starting angles

        return ax

    def _plot_orbits3d(self, fig: plt.Figure, gs: GridSpec) -> plt.Axes:
        """Helper method to populate a figure and axes with 3D orbits."""
        cf, cf_3d = self.config, self.config_3d  

        # create a 3D base figure with axes:
        ax = self._create_figure3d(fig, gs)    

        # plot 3D orbit trajectories:
        ax.plot(self.x1, self.y1, self.z1, color=cf.body1_trail_colour, linewidth=cf.line_width)
        ax.plot(self.x2, self.y2, self.z2, color=cf.body2_trail_colour, linewidth=cf.line_width)

        # display final positions of the bodies:
        if cf_3d.markers_to_relative_scale:    # show bodies to scale
            # relative scale of marker2 size based on defined marker1 size
            size2 = cf_3d.body1_markersize * (cf.body2_radius / cf.body1_radius) ** 2    # scale by the square of the radius ratio
        else:
            size2 = cf_3d.body2_markersize
        ax.scatter(self.x1[-1], self.y1[-1], self.z1[-1], color=cf.body1_colour, s=cf_3d.body1_markersize, label=cf.body1_legend_label, zorder=5)
        ax.scatter(self.x2[-1], self.y2[-1], self.z2[-1], color=cf.body2_colour, s=size2, label=cf.body2_legend_label, zorder=5)

        # display time step in days if toggled:
        if cf_3d.display_time:
            t_days = self.t / (24 * 60 * 60)
            xpos, ypos = cf_3d.time_text_pos
            ax.text2D(
                xpos, ypos,   # position in axes coordinates (0, 0) bottom left, (1, 1) top right
                f"T = {t_days[-1]:.{cf.time_dp}f} days", 
                transform=ax.transAxes,    # map coordinates from axes to figure coordinates
            )
        return ax

    def plot_orbits3d(self) -> Tuple[plt.Figure, GridSpec, plt.Axes]:
        """Plot the complete orbits of the two bodies in 3D."""
        fig = plt.figure(figsize=self.config_3d.figure_size)
        gs = GridSpec(nrows=1, ncols=1, figure=fig)    # create a single subplot grid
        ax = self._plot_orbits3d(fig, gs)
        plt.tight_layout(), plt.show()
        return fig, gs, ax

    def plot_orbits(self) -> plt.Figure:
        """
        Plot side-by-side both 2D and 3D figures of the complete two-body system orbits.

        Params:
            - `left_2d`: bool, whether to place the 2D plot on the left side.
            - `ratio`: float, width ratio of the 3D plot relative to the 2D plot.
        """
        cf = self.config
        fig = plt.figure(figsize=self.config.dashboard_figure_size, constrained_layout=True)

        # create a grid layout for the figure:
        if cf.left_2d:
            outer = GridSpec(1, 2, width_ratios=[1, cf.width_ratio], wspace=0.1, figure=fig)
            gs_2d = outer[0].subgridspec(nrows=1, ncols=1)
            gs_3d = outer[1].subgridspec(nrows=1, ncols=1)
        else:
            outer = GridSpec(1, 2, width_ratios=[cf.width_ratio, 1], wspace=0.1, figure=fig)
            gs_3d = outer[0].subgridspec(nrows=1, ncols=1)
            gs_2d = outer[1].subgridspec(nrows=1, ncols=1)
        
        # generate subplots with existing methods:
        ax_3d = self._plot_orbits3d(fig, gs_3d)
        ax_2d = self._plot_orbits2d(fig, gs_2d)

        plt.show()

        return fig, (ax_3d, ax_2d)

    def animate3d(self, show_plot_first: bool = True) -> None:
        """Animate the 3D orbits of the two-body system. Writes to an MP4 file."""
        cf, cf_3d = self.config, self.config_3d    # shorthand alias for the PlotConfig instance

        # show the static complete orbit trails (final positions):
        if show_plot_first:
            print(f"\nplotting final positions with complete orbit trails...")
            print("<CLOSE FIGURE WINDOW TO START ANIMATION WRITING>")
            self.plot_orbits3d()    # figure window must be closed before continuing 

        # set animation parameters and print key stats:
        self._animation_params(dim="3D")

        # base 2D figure and axes setup:
        fig = plt.figure(figsize=self.config_3d.figure_size)
        gs = GridSpec(nrows=1, ncols=1, figure=fig)
        ax = self._create_figure3d(fig, gs)

        # initialise plot elements:
        if cf_3d.markers_to_relative_scale:    # show planetary bodies to scale
            # relative scale of marker2 size based on defined marker1 size
            size2 = cf_3d.body1_markersize * (cf.body2_radius / cf.body1_radius) ** 2    # scale by the square of the radius ratio
        else:
            size2 = cf_3d.body1_markersize
        body1_marker = ax.scatter(self.x1[0], self.y1[0], self.z1[0], color=cf.body1_colour, s=cf_3d.body1_markersize, label=cf.body1_legend_label, zorder=5)
        body2_marker = ax.scatter(self.x2[0], self.y2[0], self.z2[0], color=cf.body2_colour, s=size2, label=cf.body2_legend_label, zorder=5)
        body1_orbit, = ax.plot([], [], [], color=cf.body1_trail_colour, linewidth=cf.line_width)
        body2_orbit, = ax.plot([], [], [], color=cf.body2_trail_colour, linewidth=cf.line_width)
        if cf_3d.display_time:
            t_days = self.t / (24 * 60 * 60)    # convert time to days
            xpos, ypos = cf_3d.time_text_pos
            time_text = ax.text2D(xpos, ypos, "", transform=ax.transAxes)

        # ----- ANIMATION FUNCTIONS ----- #

        def init() -> tuple:
            body1_orbit.set_data([], []), body2_orbit.set_data([], [])
            body1_marker._offsets3d = ([self.x1[0]], [self.y1[0]], [self.z1[0]])
            body2_marker._offsets3d = ([self.x2[0]], [self.y2[0]], [self.z2[0]])
            if cf_3d.display_time:
                time_text.set_text(f"T = {t_days[0]:.{cf.time_dp}f} days")    # set initial time text
            ax.view_init(elev=cf_3d.elev_start, azim=cf_3d.azim_start)    # set initial camera angles
            return body1_orbit, body2_orbit, body1_marker, body2_marker

        def update(frame) -> tuple:
            # update orbit trails:
            i0 = max(0, frame - self.trail_length)    # start index for the trail
            i0_1 = max(0, frame - int(self.trail_length * cf.trail_length_factor))    # longer trail for body 1
            body1_orbit.set_data_3d(self.x1[i0_1:frame + 1], self.y1[i0_1:frame + 1], self.z1[i0_1:frame + 1])    # update orbit trail
            body2_orbit.set_data_3d(self.x2[i0:frame + 1], self.y2[i0:frame + 1], self.z2[i0:frame + 1])
            # update markers:
            body1_marker._offsets3d = ([self.x1[frame]], [self.y1[frame]], [self.z1[frame]])
            body2_marker._offsets3d = ([self.x2[frame]], [self.y2[frame]], [self.z2[frame]])
            # camera panning:
            if cf_3d.camera_pan:
                a0, a1 = cf_3d.azim_start, cf_3d.azim_end
                e0, e1 = cf_3d.elev_start, cf_3d.elev_end
                # update only if the end angles are specified:
                a_next = a0 + (a1 - a0) * frame / self.steps if a1 is not None else a0     # 'not None' otherwise 0 is False
                e_next = e0 + (e1 - e0) * frame / self.steps if e1 is not None else e0
                ax.view_init(elev=e_next, azim=a_next)
            # update current time step text display if toggled:
            if cf_3d.display_time:
                time_text.set_text(f"T = {t_days[frame]:.{cf.time_dp}f} days")
            pbar.update(1)
            return body1_orbit, body2_orbit, body1_marker, body2_marker

        # create and save animation MP4 file:
        pbar = tqdmFA(total=self.steps, fps=cf.fps)    # initialise custom tqdm progress bar
        ani = FuncAnimation(fig, update, frames=range(self.used_steps), init_func=init, blit=True)
        self.file_prefix = "3D"    # prefix for the filename
        self._write_to_mp4(ani, pbar)    # write the animation to an MP4 file

    def animate(self, show_plot_first: bool = True) -> None:
        """Animate side-by-side both 2D and 3D figures of the complete two-body system orbits."""
        cf, cf_3d = self.config, self.config_3d
        if show_plot_first:
            print(f"\nplotting final positions with complete orbit trails...")
            print("<CLOSE FIGURE WINDOW TO START ANIMATION WRITING>")
            self.plot_orbits()

        # determine coordinates to use for 2D plot
        if self.params.head_on_view:
            # use projected coordinates for 2D head-on viewing:
            x1_plot2d, y1_plot2d = self.x1_proj, self.y1_proj    
            x2_plot2d, y2_plot2d = self.x2_proj, self.y2_proj
        else:
            # use original 2D coordinates:
            x1_plot2d, y1_plot2d, x2_plot2d, y2_plot2d = self.x1, self.y1, self.x2, self.y2  

        # set animation parameters and print key stats:

        # create figure with side-by-side layout
        fig = plt.figure(figsize=self.config.dashboard_figure_size, constrained_layout=True)
        
        # create grid layout matching plot_orbits method:
        if cf.left_2d:
            self._animation_params(dim="| 2D + 3D |")
            outer = GridSpec(1, 2, width_ratios=[1, cf.width_ratio], wspace=cf.wspace, figure=fig)
            gs_2d = outer[0].subgridspec(nrows=1, ncols=1)
            gs_3d = outer[1].subgridspec(nrows=1, ncols=1)
        else:
            self._animation_params(dim="| 3D + 2D |")
            outer = GridSpec(1, 2, width_ratios=[cf.width_ratio, 1], wspace=cf.wspace, figure=fig)
            gs_3d = outer[0].subgridspec(nrows=1, ncols=1)
            gs_2d = outer[1].subgridspec(nrows=1, ncols=1)

        # populate base 2D & 3D figures:
        ax_2d = self._create_figure2d(fig, gs_2d)    # 2D axes
        ax_3d = self._create_figure3d(fig, gs_3d)    # 3D axes 

        # ----- INITIALIZE PLOT ELEMENTS ----- #
        # 2D markers and orbit trails:
        if cf.to_scale:    
            body1_marker2d = mpatches.Circle((x1_plot2d[0], y1_plot2d[0]), radius=cf.body1_radius, color=cf.body1_colour, label=cf.body1_legend_label, zorder=5)
            body2_marker2d = mpatches.Circle((x2_plot2d[0], y2_plot2d[0]), radius=cf.body2_radius, color=cf.body2_colour, label=cf.body2_legend_label, zorder=5)
            ax_2d.add_patch(body1_marker2d), ax_2d.add_patch(body2_marker2d)
        else:
            body1_marker2d = ax_2d.scatter([], [], color=cf.body1_colour, s=cf.body1_markersize, zorder=5)
            body2_marker2d = ax_2d.scatter([], [], color=cf.body2_colour, s=cf.body2_markersize, zorder=5)
        body1_trail2d, = ax_2d.plot([], [], color=cf.body1_trail_colour, linewidth=cf.line_width)
        body2_trail2d, = ax_2d.plot([], [], color=cf.body2_trail_colour, linewidth=cf.line_width)
        # 3D markers and orbit trails:
        if cf_3d.markers_to_relative_scale:
            # relative scale of marker2 size based on defined marker1 size
            size2_3d = cf_3d.body1_markersize * (cf.body2_radius / cf.body1_radius) ** 2    # scale by the square of the radius ratio
        else:
            size2_3d = cf_3d.body1_markersize
        body1_marker3d = ax_3d.scatter(self.x1[0], self.y1[0], self.z1[0], color=cf.body1_colour, s=cf_3d.body1_markersize, label=cf.body1_legend_label, zorder=5)
        body2_marker3d = ax_3d.scatter(self.x2[0], self.y2[0], self.z2[0], color=cf.body2_colour, s=size2_3d, label=cf.body2_legend_label, zorder=5)
        body1_trail3d, = ax_3d.plot([], [], [], color=cf.body1_trail_colour, linewidth=cf.line_width)
        body2_trail3d, = ax_3d.plot([], [], [], color=cf.body2_trail_colour, linewidth=cf.line_width)

        # initialise time period display text if toggled:
        time_text2d, time_text3d = None, None
        if cf.display_time or cf_3d.display_time:
            t_days = self.t / (24 * 60 * 60)    # convert time to days
            if cf.display_time:    # for 2D plot
                xpos, ypos = cf.time_text_pos
                time_text2d = ax_2d.text(xpos, ypos, "", transform=ax_2d.transAxes)
            if cf_3d.display_time:    # for 3D plot
                xpos, ypos = cf_3d.time_text_pos
                time_text3d = ax_3d.text2D(xpos, ypos, "", transform=ax_3d.transAxes)
        
        # ----- ANIMATION FUNCTIONS ----- #

        def init() -> tuple:
            body1_trail2d.set_data([], [])
            body2_trail2d.set_data([], [])
            body1_trail3d.set_data([], [])
            body2_trail3d.set_data([], [])
            if cf.to_scale:
                body1_marker2d.center = (x1_plot2d[0], y1_plot2d[0])
                body2_marker2d.center = (x2_plot2d[0], y2_plot2d[0])
            else:
                body1_marker2d.set_offsets([[x1_plot2d[0], y1_plot2d[0]]])
                body2_marker2d.set_offsets([[x2_plot2d[0], y2_plot2d[0]]])
            body1_marker3d._offsets3d = ([self.x1[0]], [self.y1[0]], [self.z1[0]])
            body2_marker3d._offsets3d = ([self.x2[0]], [self.y2[0]], [self.z2[0]])
            if time_text2d:
                time_text2d.set_text(f"T = {t_days[0]:.{cf.time_dp}f} days")
            if time_text3d:
                time_text3d.set_text(f"T = {t_days[0]:.{cf.time_dp}f} days")
            # set initial 3D camera angles for the 3D plot:
            ax_3d.view_init(elev=cf_3d.elev_start, azim=cf_3d.azim_start)    
            return body1_trail2d, body2_trail2d, body1_marker2d, body2_marker2d, body1_trail3d, body2_trail3d, body1_marker3d, body2_marker3d
        
        def update(frame) -> tuple:
            
            # update orbit trails:
            i0 = max(0, frame - self.trail_length)    # start index for the trail
            i0_1 = max(0, frame - int(self.trail_length * cf.trail_length_factor))    # longer trail for body 1
            body1_trail2d.set_data(x1_plot2d[i0_1: frame + 1], y1_plot2d[i0_1: frame + 1])   
            body2_trail2d.set_data(x2_plot2d[i0: frame + 1], y2_plot2d[i0: frame + 1])
            body1_trail3d.set_data_3d(self.x1[i0_1: frame + 1], self.y1[i0_1: frame + 1], self.z1[i0_1: frame + 1])
            body2_trail3d.set_data_3d(self.x2[i0: frame + 1], self.y2[i0: frame + 1], self.z2[i0: frame + 1])
            
            # update markers:
            if cf.to_scale:
                body1_marker2d.center = (x1_plot2d[frame], y1_plot2d[frame])
                body2_marker2d.center = (x2_plot2d[frame], y2_plot2d[frame])
            else:
                body1_marker2d.set_offsets([[x1_plot2d[frame], y1_plot2d[frame]]])
                body2_marker2d.set_offsets([[x2_plot2d[frame], y2_plot2d[frame]]])
            body1_marker3d._offsets3d = ([self.x1[frame]], [self.y1[frame]], [self.z1[frame]])
            body2_marker3d._offsets3d = ([self.x2[frame]], [self.y2[frame]], [self.z2[frame]])

            # update 3D camera panning (if toggled):
            if cf_3d.camera_pan:
                a0, a1 = cf_3d.azim_start, cf_3d.azim_end
                e0, e1 = cf_3d.elev_start, cf_3d.elev_end
                # update only if the end angles are specified:
                a_next = a0 + (a1 - a0) * frame / self.steps if a1 is not None else a0
                e_next = e0 + (e1 - e0) * frame / self.steps if e1 is not None else e0
                ax_3d.view_init(elev=e_next, azim=a_next)

            # update current time step text display (if toggled):
            if time_text2d:
                time_text2d.set_text(f"T = {t_days[frame]:.{cf.time_dp}f} days")
            if time_text3d:
                time_text3d.set_text(f"T = {t_days[frame]:.{cf.time_dp}f} days")

            pbar.update(1)    # update tqdm progress bar
            return body1_trail2d, body2_trail2d, body1_marker2d, body2_marker2d, body1_trail3d, body2_trail3d, body1_marker3d, body2_marker3d

        pbar = tqdmFA(total=self.steps, fps=cf.fps)    # initialise custom tqdm progress bar
        ani = FuncAnimation(fig, update, frames=range(self.used_steps), init_func=init, blit=True)
        self.file_prefix = "2D+3D"    # prefix for the filename
        self._write_to_mp4(ani, pbar)    # write the animation to an MP4 file


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
            rotate_proj_90cw=True,  # rotate 2D projection 90 degrees (matches 3D  alignment)
            ode_method="RK45",      # use a high-order ODE solver for better accuracy
            rtol=1e-12, 
            atol=1e-9, 
            T_days=T_PLUTO_CHARON * 1.166 * 3,
            steps=600,
        ),
        config=PlotConfig(
            body1_radius=R_PLUTO, 
            body2_radius=R_CHARON,
            figure_size=(8, 8),
            figure_title="2D Pluto-Charon System – Orbital Plane (TO SCALE)",
            body1_legend_label="Pluto", 
            body2_legend_label="Charon",
            body1_colour="tab:brown", 
            body1_trail_colour="tab:brown",
            body2_colour="tab:olive", 
            body2_trail_colour="tab:olive",
            baryc_colour="tab:blue", 
            display_baryc=True,
            max_axis_extent2d=1.1,
            trail_length_pct=6,
            trail_length_factor=2.5,
            display_legend=True, 
            to_scale=True, 
            # MP4 animation parameters:
            dpi=250,        
        )
    )
    # setup 3D plot configuration dataclass:
    pluto_charon.config_3d = PlotConfig3D(
        markers_to_relative_scale=True,         
        body1_markersize=500,    # size of body2 is scaled if markers_to_relative_scale=True
        # -- camera panning during animation -- #
        elev_start=20, azim_start=-75,
        camera_pan=True,
        elev_end=10, azim_end=-20,
        # figure_title="3D Pluto-Charon System",
        figure_size=(10, 10),
        display_legend=False
    )

    # plot both 2D and 3D orbits side by side:
    # pluto_charon.plot_orbits()    

    # animate both 2D and 3D orbits in a single MP4 file:
    pluto_charon.animate(show_plot_first=False)

    # separately create 2D and 3D animations :
    # pluto_charon.animate2d(show_plot_first=False)     
    # pluto_charon.animate3d(show_plot_first=False)   
    import sys; sys.exit("\nEARLY EXIT")


def earth_moon_system(exaggerated: bool = False) -> None:
    """Simulate and animate the Earth-Moon two-body system with realistic (optional exaggeration toggle) parameters."""
    if not exaggerated:
        earth_moon = TwoBodySystem(
            params=SystemParams(
                T_days=27.321 * 1.0283 * 3,  
                rtol=1e-12, 
                steps=600,
            ),
            config=PlotConfig(
                figure_size=(10, 10),
                # figure_title="Earth-Moon System - Orbital Plane (TO SCALE)",
                time_dp=0,
                body1_legend_label="Earth", 
                body2_legend_label="Moon",
                max_axis_extent2d=1.05,
                line_width=0.4,
                to_scale=True, 
                display_baryc=False,    # no barycentre marker in 2D
                trail_length_pct=5,
                trail_length_factor=3,
            )
        )
        # setup 3D plot configuration dataclass:
        earth_moon.config_3d = PlotConfig3D(
            markers_to_relative_scale=True,         
            body1_markersize=50,    # size of body2 is scaled if markers_to_relative_scale=True
            max_axis_extent3d=1,
            # -- camera panning during animation -- #
            elev_start=0, azim_start=-60,
            camera_pan=True,
            elev_end=40, azim_end=-40,
            # -- title and legend -- #
            figure_size=(10, 10),
            display_legend=False,
            display_baryc=False,    # no barycentre marker in 3D
        )

        # create animation with both 2D and 3D orbits in a single MP4 file:
        earth_moon.animate()    
    else:
        earth_moon = TwoBodySystem(
            params=SystemParams(
                m1=M_EARTH * 0.4, m2=M_MOON * 1, 
                d=D_EARTH_MOON * 0.1, v0=V_MOON * 1, 
                i_deg=i_MOON, 
                T_days=0.62 * 3,
                rtol=1e-6, 
                steps=600
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
        earth_moon.config_3d = PlotConfig3D(
            markers_to_relative_scale=False,
            body1_markersize=600,    # size of body2 is scaled if markers_to_relative_scale=True
            body2_markersize=50,
        )

        # create animation with both 2D and 3D orbits in a single MP4 file:
        earth_moon.animate()


def equal_mass_system() -> None:
    """Simulate and animate a two-body system with equal masses."""
    mass = M_EARTH              # use Earth mass for both bodies
    radius = R_EARTH            # use Earth radius for both bodies (if to scale)
    distance = D_EARTH_MOON     # use Earth-Moon distance
    equal_mass = TwoBodySystem(
        params=SystemParams(
            m1=mass, m2=mass, d=distance, 
            v0=600, 
            i_deg=20, 
            T_days=25.85 * 3,
            rtol=1e-12,
            steps=600
        ),
        config=PlotConfig(
            body1_radius=radius * 1.5, 
            body2_radius=radius * 1.5,
            body1_colour="tab:red", 
            body1_trail_colour="tab:red",
            body2_colour="tab:green", 
            body2_trail_colour="tab:green",
            figure_size=(10, 10), 
            max_axis_extent2d=1.1, 
            y_axis_limits=(-4e8, 4e8),
            to_scale=True, 
            display_legend=False,
            display_baryc=True, 
            baryc_colour="tab:blue",          
            trail_length_pct=8, 
            trail_length_factor=1,
            time_dp=0,    # no decimal places for time text
        )
    )
    # setup 3D plot configuration dataclass:
    equal_mass.config_3d = PlotConfig3D(
        markers_to_relative_scale=True,         # not recommened to use spheres for drawing (set False)
        body1_markersize=150,                   # size of body2 is scaled if markers_to_relative_scale=True
        max_axis_extent3d=1,
        # camera panning during animation:
        elev_start=20, azim_start=-30,
        camera_pan=True,
        elev_end=50, azim_end=-75,
        # title and legend:
        figure_size=(10, 10),
        display_legend=False,
        time_text_pos = (0.12, 0.88)
    )

    # create animation with both 2D and 3D orbits in a single MP4 file:
    equal_mass.animate()