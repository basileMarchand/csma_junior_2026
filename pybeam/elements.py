import numpy as np 

INTEG_RULES = {
    "beam_2": np.polynomial.legendre.leggauss(2)
}

SHAPE_FUNCTION = {
    "beam_linear": lambda xi: 0.5*np.array([-(xi-1), xi+1])
}

DSHAPE_FUNCTION = {
    "beam_linear": lambda xi: 0.5*np.array([-1, +1])
}