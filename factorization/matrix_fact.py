import numpy as np
from numpy import linalg as LA

class Matfactor():
    def __init__(self, type):
        self.type = type

    def reduce(self, *args, **kwargs):
        pass

class EVD(Matfactor):
    def reduce(self, mat):
        return(LA.eig(mat))
   

