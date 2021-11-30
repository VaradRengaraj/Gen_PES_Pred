import numpy as np

class MatrixReduce:

   def __init__(self, type):
      self.type = type

   #@abstractmethod
   def reduce(self, *args, **kwargs):
      pass

   def get_type(self):
      return self.type


class SubMatrixReduce(MatrixReduce):

   def reduce(self, inp_matrix):
      #print("submatreduce")
      lst_of_submatx = []
      for i in range(inp_matrix.shape[1]):
         non_zeros_index = []
         #lst_of_submatx = []
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
         #lst_of_submatx.append(submatrix)
         #print("submatrrix", submatrix)
         #submatrix = submatrix.flatten()
         #print("submatrix_f", submatrix)
         lst_of_submatx.append(submatrix)
         #del non_zeros_index
         #del submatrix
         #del non_zeros
      return lst_of_submatx

   def reducewithcharge(self, inp_matrix, charge_mat):
      print("submatreduce")
      lst_of_submatx = []
      charge_desc = np.zeros(1)
      for i in range(inp_matrix.shape[1]):
         non_zeros_index = []
         #lst_of_submatx = []
         for j in range(inp_matrix.shape[0]):
            if inp_matrix[j][i] != 0:
               non_zeros_index.append(j)
         length = len(non_zeros_index)
         print(length)
         submatrix = np.zeros((length, length))
         non_zeros = np.asarray(non_zeros_index)
         for k in range(0, length):
            for l in range(0, length):
               submatrix[k, l] = inp_matrix[non_zeros[k], non_zeros[l]]
         #lst_of_submatx.append(submatrix)
         print("submatrrix", submatrix)
         charge_desc = charge_mat[i]*(i+1)
         submatrix = np.concatenate([submatrix.flatten(),charge_desc.reshape(1,)])
         print("submatrix_f", submatrix)
         lst_of_submatx.append(submatrix)
         #del non_zeros_index
         #del submatrix
         #del non_zeros
      return lst_of_submatx

class MatrixReduceSimp(MatrixReduce):
   
   def reduce(self, inp_matrix):
      lst_data = []
#      temp = np.zeros(inp_matrix.shape[0]).reshape(9,)
      for i in range(0, inp_matrix.shape[1]):
         temp = np.zeros(inp_matrix.shape[0])
         for j in range(0, inp_matrix.shape[0]):
            temp[j] = inp_matrix[j, i]
         lst_data.append(temp)
      return lst_data

