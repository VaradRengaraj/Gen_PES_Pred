import os
import yaml
import numpy as np
import random
from descriptors.descriptors import CoulombDesc
from reduction.matrixreduction import SubMatrixReduce
from factorization.matrix_fact import EVD
import h5py
from itertools import zip_longest
import torch.utils.data as Data
from dataloader.custdataset import CustomDataset
from pathlib import Path
from models.neuralnetwork import NeuralNetwork
from scaling.scaling import StandScaler

class ML_AS_EF():
    def __init__(self, configfile=None):

        dirpath = os.path.join(os.getcwd(), "config")
        filepath = os.path.join(dirpath, configfile)

        if not os.path.exists(filepath):
            msg = "Config file doesn't exist"
            raise TypeError(msg)

        filep = self.filecheck(filepath)
        config = yaml.safe_load(filep)
       
        if config["general"]["hdf5_path"] is not None and isinstance(config["general"]["hdf5_path"], str):
            self.hdf5_dirpath = os.path.join(os.getcwd(), config["general"]["hdf5_path"])
            self.hdf5_dirname = config["general"]["hdf5_path"]
        else:
            msg = "Hdf5 file path doesn't exist"
            raise TypeError(msg)

        if config["general"]["hdf5_file_name"] is not None and isinstance(config["general"]["hdf5_file_name"], str):
            self.hdf5_filepath = os.path.join(self.hdf5_dirpath, config["general"]["hdf5_file_name"])
        else:
            msg = "hdf5 filename doesn't exist"
            raise TypeError(msg)

        self.descriptor_settings = {}
        self.algorithm_settings = {}
           
        if config["general"]["num_frames"] is not None and isinstance(config["general"]["num_frames"], int):
            self.num_frames = config["general"]["num_frames"]
        else:
            msg = "Config should have number of frames"
            raise TypeError(msg)

        if not os.path.exists(self.hdf5_filepath):
        # create the directory where the hdf5 dataset is stored
        # The settings necessary to generate the hdf5 dataset is saved.
            if not os.path.exists(self.hdf5_dirpath):
                try:
                    os.mkdir(self.hdf5_dirpath)
                except OSError:
                    print ("Creation of the directory %s failed" % "dataset")
                else:
                    print ("Successfully created the directory %s" % "dataset")
       
            self.descriptor_settings['hdf5_filepath'] = self.hdf5_filepath
            descriptors_supported = ['coulomb','COULOMB', 'sine', 'SINE']

            if config["descriptor"]["name"] is not None:
                if config["descriptor"]["name"] not in descriptors_supported:
                    msg = "The descriptor mentioned in the config file is not supported"
                    raise NotImplementedError(msg)
                else:
                    self.descriptor_settings['descriptor'] = config["descriptor"]["name"]
            else:
                self.descriptor_settings['descriptor'] = "coulomb"

            if config["descriptor"]["feat_vect_size"] is not None and isinstance(config["descriptor"]["feat_vect_size"], int):
                self.descriptor_settings['feat_vect_size'] = config["descriptor"]["feat_vect_size"]
            else:
                msg = "Config does not have size of the feature vector"
                raise TypeError(msg)
           
            if config["general"]["num_atoms"] is not None and isinstance(config["general"]["num_atoms"], int):
                self.descriptor_settings['num_atoms'] = config["general"]["num_atoms"]
            else:
                msg = "Config does not have number of atoms"
                raise TypeError(msg)

            if config["general"]["num_mols"] is not None and isinstance(config["general"]["num_mols"], int):
                self.descriptor_settings['num_mols'] = config["general"]["num_mols"]
            else:
                msg = "Config does not have number of mols"
                raise TypeError(msg)
        
            self.descriptor_settings['total_atoms'] = self.descriptor_settings['num_atoms']*self.descriptor_settings['num_mols']

            if config["general"]["input_file_format"]["path"] is not None and \
                         isinstance(config["general"]["input_file_format"]["path"], str):
                self.descriptor_settings['input_dirpath'] = os.path.join(os.getcwd(), config["general"]["input_file_format"]["path"])
            else:
                msg = "Config should have input directory name"
                raise TypeError(msg)

            if config["general"]["input_file_format"]["pos_file_name"] is not None and \
                         isinstance(config["general"]["input_file_format"]["pos_file_name"], str):
                self.descriptor_settings['input_pos_fname'] = os.path.join(self.descriptor_settings['input_dirpath'], 
                                        config["general"]["input_file_format"]["pos_file_name"]) 
            else:
                msg = "Config should have position file name"
                raise TypeError(msg)
           
            if config["general"]["input_file_format"]["frc_file_name"] is not None and \
                         isinstance(config["general"]["input_file_format"]["frc_file_name"], str):
                self.descriptor_settings['input_frc_fname'] = os.path.join(self.descriptor_settings['input_dirpath'], 
                                        config["general"]["input_file_format"]["frc_file_name"])                    
            else:
                msg = "Config should have force file name"
                raise TypeError(msg)

            if config["general"]["input_file_format"]["ener_file_name"] is not None and \
                         isinstance(config["general"]["input_file_format"]["ener_file_name"], str):
                self.descriptor_settings['input_ener_fname'] = os.path.join(self.descriptor_settings['input_dirpath'], 
                                         config["general"]["input_file_format"]["ener_file_name"])                  
            else:
                msg = "Config should have energy file name"
                raise TypeError(msg)

            input_file_format_supported = ['rpmd','RPMD']
         
            if config["general"]["input_file_format"]["type"] is not None:
                if config["general"]["input_file_format"]["type"] not in input_file_format_supported:
                    msg = "Input file format not supported"
                    raise TypeError(msg)
                else:
                    self.descriptor_settings['extra_lines'] = 2
            
            #if config["general"]["num_frames"] is not None and isinstance(config["general"]["num_frames"], int):
            #    self.descriptor_settings['num_frames'] = config["general"]["num_frames"]
            #else:
            #    msg = "Config should have number of frames"
            #    raise TypeError(msg)

            if config["general"]["sim_type"]["method"] is not None and \
                       config["general"]["sim_type"]["method"] == 'pbc':
                self.descriptor_settings['pbc'] = config["general"]["sim_type"]["pbc_boundary"]

            for i in range(self.descriptor_settings['num_atoms']):
                atom = "atom_%d" % (i+1)
                if config["general"]["charge_values"][atom] is not None and isinstance(config["general"]["charge_values"][atom], int):
                    self.descriptor_settings[atom] = config["general"]["charge_values"][atom]
                else:
                    msg = "Charge values not present"
                    raise TypeError(msg)
        
        if 1:
        # dump all the yaml parsed descriptor informations      
            for k,v in self.descriptor_settings.items():
                print(k, v)

        algorithm_supported = ['Neural Network', 'NN', 'nn']

        if config["training"]["type"] is not None and isinstance(config["training"]["type"],str):
            if config["training"]["type"] not in algorithm_supported:
                    msg = "Training algorithm not supported"
                    raise TypeError(msg)
            else:
             # create a neural network dict with basic settings
                self.algorithm_setttings = {'type': 'NN',
                                             'batch_size': 30,
                                             'scaling': False,
                                             'scaling_type': None,
                                             'train_size': 70, #percent
                                             'validation_size': 30, #percent
                                             'hidden_layers': [20,20],
                                             'activation': ['Tanh', 'Tanh', 'Linear'],
                                             'optimizer': 'sgd',
                                             'epochs': 200,
                                             'device': 'cuda'}
        # update default algorithm settings with values from config
        self.algorithm_settings.update(config["training"]["NN"])
  
        if 1:
        #dump all the yaml parsed nn information
            for k, v in self.algorithm_settings.items():
                print(k, v)
    
    def filecheck(self, fn):
        try:
            fd = open(fn, "r")
            return fd
        except IOError:
            return 0

    def grouper(self, iterable, n, fillvalue=None):
        args = [iter(iterable)] * n
        return zip_longest(*args, fillvalue=fillvalue)
    
    def feature_frames(self, hf, fd_pos, charge_array):
        counter = 0
        lst_total_frames = []
        lst_find_max = []
        total_atoms = self.descriptor_settings['total_atoms']
        extra_lines = self.descriptor_settings['extra_lines']
        bc = np.zeros([3])    
        bc[0] = bc[1] = bc[2] = self.descriptor_settings['pbc']
        no_atoms = self.descriptor_settings['num_atoms']
        no_mols = self.descriptor_settings['num_mols']
        num_frames = self.num_frames
        size_feat = self.descriptor_settings['feat_vect_size']

        for lines in self.grouper(fd_pos, total_atoms+extra_lines, ''):

            if len(lines) < total_atoms+extra_lines:
                raise ValueError(
                     "Reached end of the position file"
                )
            #print("lines")
            #print(lines)

            lst2 = []
            lst = []

            for j in range(extra_lines, len(lines)):
                sttr = ' '.join(lines[j].split()).split(' ', 1)[1]
                lst = [float(s) for s in sttr.split(' ')]
                lst2.append(lst)

            #print("np")
            frame_pos = np.asarray(lst2)
            print(frame_pos)

            col = CoulombDesc("coulomb")
            colmat = col.create(no_atoms, no_mols, frame_pos, charge_array, 1.5, bc, True)
            submatreduce = SubMatrixReduce("submatreduce")
            lst_submats = submatreduce.reduce(colmat)
            matreduce = EVD("evd")

            for i in range(0, len(lst_submats)):

                if 0:
                    from numpy import loadtxt, savetxt
                    # save array
                    #filename = "data_%d" % i+1
                    file_name = "data_%d" % (i+1)
                    savetxt(file_name, lst_submats[i], delimiter=',')

                ei, vec = matreduce.reduce(lst_submats[i])
                print("ei", ei)
                print("vec", vec)
                ei[::-1].sort()
                ei = np.real(ei)

                if ei.size < size_feat:
                    lst_submats[i] = np.pad(ei[:size_feat], [(0,size_feat-ei.size)], mode='constant', constant_values=0)
                else:
                    lst_submats[i] = ei[:size_feat]

            ar = np.asarray(lst_submats)
            ar = ar.reshape(no_atoms*no_mols, size_feat)
            dataset_name = "frame_%d" % counter
            hf.create_dataset(dataset_name, data=ar, shape=(total_atoms,size_feat), compression='gzip', chunks=True)

            #print(lst2)
            if 0:
                lst_total_frames.append(lst_submats)

            counter += 1
            if counter == num_frames:
                break

    def force_frames(self, hf, fd_frc):
        total_atoms = self.descriptor_settings['total_atoms']
        extra_lines = self.descriptor_settings['extra_lines']
        no_atoms = self.descriptor_settings['num_atoms']
        no_mols = self.descriptor_settings['num_mols']
        num_frames = self.num_frames
        counter = 0

        for lines in self.grouper(fd_frc, total_atoms+extra_lines, ''):

            if len(lines) < total_atoms+extra_lines:
                raise ValueError(
                     "Reached end of the forces file"
                )

            lst2 = []
            lst = []

            for j in range(extra_lines, len(lines)):
                sttr = ' '.join(lines[j].split()).split(' ', 1)[1]
                lst = [float(s) for s in sttr.split(' ')]
                lst2.append(lst)

            forces_frame = np.asarray(lst2)
            #print(forces_frame)
            #print("frcs_frame_shape", forces_frame.shape)

            if 0:
                lst_f = lst_total_frames[counter]

            #print("lst_f len", len(lst_f))
            dataset_name = "frc_%d" % counter
            hf.create_dataset(dataset_name, data=forces_frame, shape=(total_atoms,3), compression='gzip', chunks=True)
            if 0:
                for j in range(len(lst_f)):
                    temp = lst_f[j]
                    print("temp.shape", temp.shape)
                    new = np.concatenate((temp, forces_frame[j,:].reshape(1,3)), axis=1)
                    lst_f[j] = new

                lst_total_frames[counter] = lst_f

            counter += 1

            if counter == num_frames:
                break

    def ener_frames(self, hf, fd_ener):
        counter = 0
        ener_dataset = "ener"
        num_frames = self.num_frames

        for lines in self.grouper(fd_ener, 1, ''):

            if len(lines) < 1:
                raise ValueError(
                 "Reached end of the energy file"
                )

            lst = []
            #for j in range(0, len(lines)):
            #print(lines[0])
            sttr = ' '.join(lines[0].split())
            lst = [float(s) for s in sttr.split(' ')]
            ener_val = np.asarray(lst)

            if 0:
                lst_f = lst_total_frames[counter]
            #ener_frame = np.zeros(len(lst_f))
            #ener_frame = ener_frame.fill(ener_val[3])
            #ener_frame = ener_frame.reshape(len(lst_f),1)

            enner_data = ener_val[3].reshape(1,1)

            if counter == 0:
                hf.create_dataset(ener_dataset, data=enner_data, shape=(1,1), maxshape=(10000,1), compression='gzip', chunks=True)
            else:
                hf[ener_dataset].resize((hf[ener_dataset].shape[0] + enner_data.shape[0]), axis=0)
                hf[ener_dataset][-enner_data.shape[0]:] = enner_data

            if 0:
                for j in range(len(lst_f)):
                    temp = lst_f[j]
                    new = np.concatenate((temp, ener_val[3].reshape(1,1)), axis=1)
                    lst_f[j] = new

                lst_total_frames[counter] = lst_f

            counter += 1

            if counter == num_frames:
                break

    def find_no_element(self, filename):

        dirpath = os.path.join(os.getcwd(), filename)

        if not os.path.exists(dirpath):
            raise RuntimeError('Dataset folder not found')

        p = Path(dirpath)
        assert(p.is_dir())
        files = sorted(p.glob('*.hdf5'))

        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        hdf5_file = files[0]
        hf = h5py.File(hdf5_file, 'r')
        hf_dataset_list = list(hf.keys())
        frc_frames_list = [s for s in hf_dataset_list if "frc" in s]

        if len(frc_frames_list) != 0:
            return np.array(hf['.'][frc_frames_list[0]]).shape[0]
        else:
            return 0

    def run(self):
    # check if the descriptor dict is empty, if empty hdf5 dataset generation is not done.
        if self.descriptor_settings:
            fd_pos = self.filecheck(self.descriptor_settings['input_pos_fname'])
            fd_frc = self.filecheck(self.descriptor_settings['input_frc_fname'])
            fd_ener = self.filecheck(self.descriptor_settings['input_ener_fname'])

            charge_array = np.zeros(self.descriptor_settings['num_atoms']*self.descriptor_settings['num_mols'])
            charge_arr_org = np.zeros(self.descriptor_settings['num_atoms'])
            
            for i in range(self.descriptor_settings['num_atoms']):
                atom = "atom_%d" % (i+1)
                charge_arr_org[i] = self.descriptor_settings[atom]

            j = 0
    
            for i in range(0, self.descriptor_settings['num_atoms']*self.descriptor_settings['num_mols']):
                charge_array[i] = charge_arr_org[j]
                j += 1
                if j == self.descriptor_settings['num_atoms']:
                    j = 0

            if fd_pos == 0 or fd_frc == 0 or fd_ener == 0:
                raise ValueError(
                      "Error: File does not appear to exist"
                )

            hf = h5py.File(self.descriptor_settings['hdf5_filepath'], 'a')
            self.feature_frames(hf, fd_pos, charge_array)
            self.force_frames(hf, fd_frc)
            self.ener_frames(hf, fd_ener)

    # train the model, presently only nn is supported. 
    # create training and validation lists
        #print((self.algorithm_settings['train_size']/100)*self.num_frames)
        print("num_frames",self.num_frames)
        print("dataset_path",self.hdf5_dirpath)

        lst_trng = random.sample(range(0,1000), 750)
        lst_val = list(set(list(range(0, self.num_frames))) - set(lst_trng))
        print(lst_trng)
        print(lst_val)

        if 0:
            scal = StandScaler("StdScaler")
            m,s = scal.run("dataset3")

        dict_dataset = {'trng':lst_trng,'val':lst_val}
        trng_set = CustomDataset(dict_dataset['trng'], self.hdf5_dirname,load_data=False, data_cache_size=20, transform=None, mean=None, std=None)
        val_set = CustomDataset(dict_dataset['val'], self.hdf5_dirname,load_data=False, data_cache_size=20, transform=None, mean=None, std=None)
        loader_params = {'batch_size': 40, 'shuffle': False}
        trng_loader = Data.DataLoader(trng_set, **loader_params)
        val_loader = Data.DataLoader(val_set, **loader_params)

        # find the total number of elements from the dataset
        num_elements = self.find_no_element(self.hdf5_dirname)

        print("hidden", self.algorithm_settings['hidden_layers'])
        print("activation", self.algorithm_settings['activation'])
        print("optimizer", self.algorithm_settings['optimizer'])
        # create NN instance
        model = {}
        model['hiddenlayers'] = self.algorithm_settings['hidden_layers']
        model['activation'] = self.algorithm_settings['activation']
        model['optimizer'] = {}
        model['optimizer']['method'] = self.algorithm_settings['optimizer']
        model['optimizer']['parameters'] = {}
        nn = NeuralNetwork(num_elements, model['hiddenlayers'], model['activation'], model['optimizer'], random_seed=10, batch_size=None, epoch=500, validate=True, force_matching=False)
        nn.train(trng_loader, model['optimizer'], ValData=val_loader)


