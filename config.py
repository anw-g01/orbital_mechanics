from dataclasses import dataclass
from typing import Tuple, Optional
from constants import *


@dataclass
class SystemParams:
    """Phyiscal parameters for a two-body system simulation."""
    m1: float = M_EARTH
    m2: float = M_MOON
    d: float = D_EARTH_MOON
    v0: float = V_MOON
    i_deg: float = i_MOON
    head_on_view: bool = True
    # ODE solver parameters
    T_days: float = 27.321
    steps: int = 1000
    ode_method: str = "RK45"
    rtol: float = 1e-3
    atol: float = 1e-6


@dataclass
class PlotConfig:
    """Configuration for plotting a two-body system."""
    # figure properties:
    figure_size: Tuple[int, int] = (10, 10)
    figure_title: Optional[str]  = None
    title_fontsize: int = 10
    grid_alpha: float = 0.15
    dashed_line_alpha: float = 0.1
    dashed_line_width: float = 0.8
    # axis configuration:
    x_axis_max_ticks: int = 3
    y_axis_max_ticks: int = 3
    x_axis_limits: Optional[Tuple[float, float]] = None
    y_axis_limits: Optional[Tuple[float, float]] = None
    max_axis_extent2d: float = 1.05
    # colours and styling:
    body1_colour: str = "tab:blue"
    body1_trail_colour: str = "tab:blue"
    body2_colour: str = "tab:grey"
    body2_trail_colour: str = "tab:grey"
    body1_markersize: int = 1000
    body2_markersize: int = 100
    line_width: float = 0.7
    trail_length_pct: int = 10  
    trail_length_factor: float = 3.0
    # display options:
    display_time: bool = True
    time_text_pos: Tuple[float, float] = (0.04, 0.95)
    time_dp: int = 1
    display_legend: bool = False
    display_baryc: bool = False
    baryc_colour: str = "tab:red"
    baryc_legend_label: str = "Barycentre"
    baryc_alpha: float = 0.8
    baryc_markersize: int = 12
    # planetary body properties (defaults for Earth-Moon system):
    to_scale: bool = False
    body1_radius: float = R_EARTH
    body1_legend_label: Optional[str] = None
    body2_radius: float = R_MOON
    body2_legend_label: Optional[str] = None


@dataclass
class PlotConfig3D():
    """Extra configuration parameters for 3D plots (on top of the existing 2D plot configurations) for a two-body system."""
    # figure properties:
    figure_title: Optional[str] = None
    figure_size: Tuple[int, int] = (10, 10)
    body1_markersize: int = 600
    body2_markersize: int = 200
    # axis configuration:
    max_axis_extent3d: float = 1
    z_axis_limits: Optional[Tuple[float, float]] = None
    z_axis_max_ticks: int = 3
    draw_dashes3d: bool = True
    # display options:
    display_time: bool = True
    time_text_pos: Tuple[float, float] = (0.1, 0.88)
    display_legend: bool = False
    display_baryc: bool = True
    baryc_markersize: int = 10
    # if markers are to scale:
    markers_to_relative_scale: bool = False
    sphere_alpha: float = 0.8
    sphere_res: int = 50
    # animation parameters:
    elev_start: int = 30
    azim_start: int = -60
    camera_pan: bool = False
    elev_end: Optional[int] = None
    azim_end: Optional[int] = None