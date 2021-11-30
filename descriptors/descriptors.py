#from abc import ABC, abstractmethod
import numpy as np
import math as mt

class Descriptor():
   def __init__(self, type):
      self.type = type

   #@abstractmethod
   def create(self, *args, **kwargs):
      pass  
 
   def get_type(self):
      return self.type


class CoulombDesc(Descriptor):

   def create(self, num_atoms, num_mols, data, charge, cutoff_limit, bc, cutoff=False):

      total_elements = num_atoms*num_mols
      self.coulomb = np.zeros((total_elements, total_elements))

      for i in range(0, total_elements):
         for j in range(i, total_elements):
            if i == j:
               self.coulomb[i, j] = 0.5*pow(charge[i], 2.4)
               #print("i = j", self.coulomb[i, j])
            else:
               bond_len = mt.sqrt(pow(((data[i, 0]-data[j, 0])-bc[0]*np.around((data[i, 0]-data[j, 0])/bc[0])), 2)+
                       pow(((data[i, 1]-data[j, 1])-bc[1]*np.around((data[i, 1]-data[j, 1])/bc[1])), 2)+
                       pow(((data[i, 2]-data[j, 2])-bc[2]*np.around((data[i, 2]-data[j, 2])/bc[2])), 2))
               #bond_len = np.linalg.norm(data[i,:]-data[j,:])
               #print(" r & bond_len", r, bond_len)
               if cutoff == True:
                  if ((charge[i]*charge[j])/bond_len) <= cutoff_limit:
                     self.coulomb[i, j] = self.coulomb[j, i] = 0
                  else:
                     self.coulomb[i, j] = self.coulomb[j, i] = (charge[i]*charge[j])/bond_len
               else:
                  self.coulomb[i, j] = self.coulomb[j, i] = (charge[i]*charge[j])/bond_len
               #print("i != j", self.coulomb[i, j])
               #print("i != j", r)

      return self.coulomb

