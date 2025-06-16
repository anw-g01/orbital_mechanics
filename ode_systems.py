import numpy as np
from typing import Tuple, Callable
from config import SystemParams
from constants import G


def two_body_ode(p: SystemParams)-> Tuple[Callable[[float, np.ndarray], np.ndarray], np.ndarray]:
    """ODE system of equations for the two-body problem."""

    # extract system parameters from the dataclass:
    m1, m2, d, v0, i_deg = p.m1, p.m2, p.d, p.v0, p.i_deg

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

    def func(t: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Computes the time derivative of the state vector."""
        r1, v1, r2, v2 = np.split(Z, 4)             # unpack state vector (4 equal sub-arrays)
        r = r2 - r1                                 # relative position vector of Moon w.r.t. Earth
        r_mag = np.linalg.norm(r)                   # magnitude (Earth-Moon distance)
        F = ( G*m1*m2 / r_mag**3 ) * r              # force vector (Newton's law of gravitation)
        a1 = F / m1                                 # acceleration of Earth
        a2 = -F / m2                                # acceleration of Moon (equal and opposite)
        return np.concatenate([v1, a1, v2, a2])     # return the time derivative of the state vector
    
    return func, Z0