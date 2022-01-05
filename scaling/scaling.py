import numpy as np
from abc import ABC,abstractmethod
import os
from pathlib import Path
import h5py

class Scaling(ABC):
    """ Base class for feature scaling.

    Parameters
    ----------
    type: string
        String refering to the scaling method to be used.    
    """ 
    def __init__(self, type):
        self.type = type

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    """ Function returns to the type of the scaling used.

    Returns
    -------
    type: string
        String refering to the scaling method that is set.
    """
    def get_type(self):
        return self.type


class StandScaler(Scaling):
    """ A Class to perform standard scaling of the features. 
   
    Parameters
    ----------
    filepath: string
        HDF5 file path where the feature datasets are stored.

    Returns
    -------
    mean: numpy array
        array containing the mean values of the features.
    std: numpy array
        array containing the standard deviation values of the features.
    """
    def run(self, filepath):
        # read the hdf5 files and calculate the mean and variance matrices
        path = os.getcwd()
        dirpath = os.path.join(path, filepath)
        print(dirpath)

        if not os.path.exists(dirpath):
            raise RuntimeError('Dataset folder not found')

        p = Path(filepath)
        assert(p.is_dir())
        files = sorted(p.glob('*.hdf5'))

        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        hdf5_file = files[0]
        hf = h5py.File(hdf5_file, 'r')
        hf_dataset_list = list(hf.keys())
        data_frames_list = [s for s in hf_dataset_list if "frame" in s]
        frc_frames_list = [s for s in hf_dataset_list if "frc" in s]
        ener_list = [s for s in hf_dataset_list if "ener" in s]

        if len(data_frames_list) < 1 or len(frc_frames_list) < 1 or len(ener_list) < 1:
            raise RuntimeError('Required data frames or force frames or energy frames is missing')

        #just find the size of the frame files. Based on which the mean and variance 2D arrays are created
        shape = np.array(hf['.']['frame_0']).shape
        print(shape)
        mean = np.zeros((shape[0], shape[1]))
        var = np.zeros((shape[0], shape[1]))
        
        for frame in data_frames_list:
            data = np.array(hf['.'][frame])
            mean += data
        
        mean /= len(data_frames_list)

        for frame in data_frames_list:
            data = np.array(hf['.'][frame])
            var += pow((data - mean), 2)

        var /= len(data_frames_list)
        data = np.array(hf['.'][frame])
        #print("normalized", (data-mean)/pow(var,0.5))
        if 0:
            for frame in data_frames_list:
                data = np.array(hf['.'][frame])
                print((data-mean)/pow(var, 0.5))

        return mean, pow(var,0.5) 
        
