import numpy as np
from numpy import linalg as LA

class Matfactor():
    """ Base class for matrix factorization.

    Parameters
    ----------
    type: string
        String refering the factorization method used.
    """
    def __init__(self, type):
        self.type = type

    def reduce(self, *args, **kwargs):
        pass

class EVD(Matfactor):
    """ Class used for Eigen value decomposition.

    Parameters
    ----------
    mat: numpy 2D array
        Matrix which needs to be eigen value decomposed.

    Returns
    -------
    w: numpy array
        The eigen values array
    v: numpy 2D array
        The eigen vectors array    
    """
    def reduce(self, mat):
        return(LA.eig(mat))
   

