from constants import *
from two_body_barycentric import TwoBodySystem
from config import SystemParams, PlotConfig

if __name__ == "__main__":

    pluto_charon = TwoBodySystem(
        params=SystemParams(
            m1=M_PLUTO, 
            m2=M_CHARON, 
            d=D_PLUTO_CHARON, 
            v0=V_CHARON, 
            i_deg=i_CHARON, 
            T_days=T_PLUTO_CHARON * 4,
            rtol=1e-6, steps=1000
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
            show_legend=True, 
            to_scale=True, 
            show_bc=True
        )
    )

    # pluto_charon.plot_orbits2d()

    pluto_charon.animate2d(trail_length_pct=8, dpi=100)