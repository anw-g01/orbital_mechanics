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
    # ODE solver parameters
    T_days: float = 27.321
    steps: int = 1000
    ode_method: str = "RK45"
    rtol: float = 1e-3
    atol: float = 1e-6


@dataclass
class PlotConfig:
    """Configuration for plotting a two-body system."""
    # figure properties
    figure_size: Tuple[int, int] = (10, 10)
    figure_title: Optional[str]  = None
    grid_alpha: float = 0.15
    dashed_line_alpha: float = 0.1
    dashed_line_width: float = 0.8
    # axis configuration
    x_axis_max_ticks: int = 5
    y_axis_max_ticks: int = 5
    x_axis_limits: Optional[Tuple[float, float]] = None
    y_axis_limits: Optional[Tuple[float, float]] = None
    max_axis_extent: float = 1.05
    # colours and styling
    body1_colour: str = "tab:blue"
    body1_trail_colour: str = "tab:blue"
    body2_colour: str = "tab:grey"
    body2_trail_colour: str = "tab:grey"
    body1_markersize: int = 1000
    body2_markersize: int = 100
    line_width: float = 0.7
    # display options
    show_legend: bool = False
    show_bc: bool = False
    bc_colour: str = "tab:red"
    bc_legend_label: str = "barycentre"
    bc_alpha: float = 0.4
    bc_markersize: int = 25
    # planetary body properties (defaults for Earth-Moon system)
    to_scale: bool = False
    body1_radius: float = R_EARTH
    body1_legend_label: Optional[str] = None
    body2_radius: float = R_MOON
    body2_legend_label: Optional[str] = None
