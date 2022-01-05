import numpy as np
from abc import ABC, abstractmethod

class MatrixReduce:
    """ Base class for matrix reduce.

    Parameters
    ----------
    type: string
        String refering to the matrix reduce method used.
    """
    def __init__(self, type):
        self.type = type

    @abstractmethod
    def reduce(self, *args, **kwargs):
        pass

    """ Returns the matrix reduction type set for the class

    Returns
    -------
    type: string
        String refering to the matrix reduce method used. 
    """
    def get_type(self):
        return self.type


class SubMatrixReduce(MatrixReduce):
    """ A class for generating submatrixes from large sparse matrix.

    Source: M. Lass, S. Mohr, H. Wiebeler, T. Kühne, C. Plessl,
        "A Massively Parallel Algorithm for the Approximate Calculation of Inverse p-th Roots of Large 
         Sparse Matrices"

    Parameters
    ----------
    inp_matrix: NumPy 2D array
        The large sparse array to which submatrix reduction is performed.

    Returns
    -------
    lst_of_submatx: List
        List of submatrixes.
    """       
    def reduce(self, inp_matrix):
        lst_of_submatx = []
        for i in range(inp_matrix.shape[1]):
            non_zeros_index = []
            for j in range(inp_matrix.shape[0]):
                if inp_matrix[j][i] != 0:
                    non_zeros_index.append(j)
            length = len(non_zeros_index)
            #print(length)
            submatrix = np.zeros((length, length))
            non_zeros = np.asarray(non_zeros_index)
            for k in range(0, length):
                for l in range(0, length):
                submatrix[k, l] = inp_matrix[non_zeros[k], non_zeros[l]]
            #print("submatrrix", submatrix)
            lst_of_submatx.append(submatrix)
        return lst_of_submatx

