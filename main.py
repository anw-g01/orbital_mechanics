from constants import *
from two_body_fixed_host import plot_orbit3d, animate3d


if __name__ == "__main__":

    # r, v = plot_orbit3d(
    #     v0=V_MOON + 278,    # faster initial orbital velocity for the Moon
    #     i=20,               # inclination angle 
    #     time_step_mins=600,
    #     time_periods=4.5,                   # no. of time periods (lunar orbits)
    #     figure_size=(12, 12),
    #     earth_markersize=2000,
    #     moon_markersize=100,
    #     earth_colour="tab:blue",
    #     moon_colour="tab:red",
    #     moon_orbit_colour="tab:red",
    #     max_axis_extent=0.5,
    #     show_legend=False,
    #     # to_scale=True,
    #     view_angles=(20, -50)     # default angles: elev=30, azim=-60
    # )

        # animate3d(
    #     time_step_mins=120,
    #     time_periods=1.02,    # no. of time periods (lunar orbits)
    #     figure_size=(12, 12),
    #     figure_title="Moon Orbit Around Fixed Earth (NOT TO SCALE)",
    #     earth_markersize=35,
    #     moon_markersize=12,
    #     earth_colour="tab:blue",
    #     moon_colour="tab:grey",
    #     moon_orbit_colour="tab:grey",
    #     max_axis_extent=0.75,
    #     trail_length_pct=8,
    #     show_legend=True,
    #     view_angles=(20, -40),     # default angles: elev=30, azim=-60
    #     # rotate_camera=True,
    #     dpi=200
    # )