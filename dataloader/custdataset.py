import torch
import torch.utils.data as Data
import random
import numpy as np
import h5py
import os
from pathlib import Path
import numpy

class CustomDataset(Data.Dataset):

    def __init__(self, frame_list, filepath, load_data, data_cache_size = 50, transform=None, mean=None, std=None):
        super().__init__()
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform
        self.frame_list = frame_list
        self.load_data = load_data
        self.last_updated = 0
        self.init_cache = False
        self.mean = mean
        self.std = std
        #self.max_frame_size = max_frame_size

        # Validate the file path and check if the frames, forces and energy file is present
        # Also check if the num of frames matches the num of frc frames.

        dirpath = os.path.join(os.getcwd(), filepath)
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

        if len(data_frames_list) != len(frc_frames_list):
            raise RuntimeError('Num of data frames is not equal to the num of force frames')

        if 0:
            lst_frame_size = []
            for frame in data_frames_list:
                tmp_arr = np.array(hf['.'][frame])
                lst_frame_size.append(tmp_arr.shape[1])
            self.max_frame_size = max(lst_frame_size)
            #print("max_frame_size", self.max_frame_size)

        # Prepare initial cache based on the frame list/ load the entire date if load_data=True
        hf.close()
        self.file_name = hdf5_file
        self._prepare_initial_data(self.file_name, self.load_data)
        self.init_cache = True

    def _prepare_initial_data(self, filename, load_data):        

        hf = h5py.File(filename, 'r')
        
        if load_data:
            prepare_frames_list = self.frame_list
        else:
            prepare_frames_list = self.frame_list[:self.data_cache_size]

        for key in list(self.data_cache.keys()):
            del self.data_cache[key]

        for frame in prepare_frames_list:
            lst_tmp = []
            dataset_name = "frame_%d" % frame
            data = np.array(hf['.'][dataset_name])
            #lst_tmp.append(data)
            #lst_tmp.append(unused)
            #print("data.shape", data.shape)
            if self.transform == "StandSc":
                data = (data-self.mean)/self.std
            self.data_cache[dataset_name] = data

            lst_tmp2 = []
            frc_name = "frc_%d" %frame
            data = np.array(hf['.'][frc_name])
            #lst_tmp2.append(data)
            #lst_tmp2.append(unused)
            #print("force.shape", data.shape)
            self.data_cache[frc_name] = data    

            tmp = np.zeros((data.shape[0], 1))
            tmp.fill(np.array(hf['.']["ener"])[frame].item())
            #print("ener.shape", tmp.shape)
            ener_name = "ener_%d" %frame
            self.data_cache[ener_name] = tmp

        if not load_data:
            if len(self.frame_list) < self.data_cache_size:
                self.last_updated = len(self.frame_list)
            elif self.data_cache_size == 1:
                self.last_updated = 1
            else:
                self.last_updated = self.data_cache_size
        
        hf.close()       

    def _load_cache(self, filename):

        hf = h5py.File(filename, 'r')
        
        if self.data_cache_size == 1:
            prepare_frames_list = self.frame_list[self.last_updated:(self.last_updated+1)]
            print("prepare_frames_list", prepare_frames_list)
        else:
            prepare_frames_list = self.frame_list[self.last_updated:int(self.last_updated+int(self.data_cache_size/2))]

        for frame in prepare_frames_list:
            lst_tmp = []
            dataset_name = "frame_%d" % frame
            data = np.array(hf['.'][dataset_name])
            #lst_tmp.append(data)
            #lst_tmp.append(unused)
            if self.transform == "StandSc":
                data = (data-self.mean)/self.std                    

            self.data_cache[dataset_name] = data

            lst_tmp2 = []
            frc_name = "frc_%d" %frame
            data = np.array(hf['.'][frc_name])
            #lst_tmp2.append(data)
            #lst_tmp2.append(unused)
            self.data_cache[frc_name] = data

            tmp = np.zeros((data.shape[0], 1))
            tmp.fill(np.array(hf['.']["ener"])[frame].item())
            ener_name = "ener_%d" %frame
            self.data_cache[ener_name] = tmp

        if self.data_cache_size == 1:
            self.last_updated += 1
        else:
            self.last_updated += int(self.data_cache_size/2)
        
        hf.close()

    def __len__(self):
        return len(self.frame_list)

    
    def __getitem__(self, index):

        if index == 0:
        # reset the cache and build it new. Do this if only the frame list's length is larger than cache size
            if len(self.frame_list) > self.data_cache_size and self.init_cache == False:
                #print("comes here --1")
                self._prepare_initial_data(self.file_name, self.load_data)
            else:
                #print("comes here --2")
                self.init_cache = False
            self.entries_used = 0

        else:
        # return the requested data and update the cache if necessary    
        # if the size of frame list is smaller than the cache size, additional cache load is not necessary
            if len(self.frame_list) > self.data_cache_size and self.data_cache_size > 1:    
                if self.entries_used == self.data_cache_size/2:
        #            # delete the first used data_cache_size/2 entries from the cache.
        #            #print("cache_size/2", self.data_cache_size/2)
                    for key in list(self.data_cache.keys())[:(int(self.data_cache_size/2)*3)]:
                        del self.data_cache[key]
                    self._load_cache(self.file_name)
                    self.entries_used = 0           
           
        self.entries_used += 1
        file_no = self.frame_list[index]
        dataset_name = "frame_%d" % file_no
        frc_name = "frc_%d" % file_no
        ener_name = "ener_%d" % file_no
        feat_arr = torch.from_numpy(self.data_cache[dataset_name]).to('cuda:0')
        frc_arr = torch.from_numpy(self.data_cache[frc_name]).to('cuda:0')
        ener_arr = torch.from_numpy(self.data_cache[ener_name]).to('cuda:0')

        # special case when cache_size is equal to 1
        if self.data_cache_size == 1:
            for key in list(self.data_cache.keys()):
                del self.data_cache[key] 
            self._load_cache(self.file_name)
       
        #print("frc_arr", frc_arr)
        #print("frc_arr.shape", frc_arr.shape)
        #print("feat_arr.shape", feat_arr.shape)
        #print("ener_arr", ener_arr)

        #print("feat_arr.shape[1]", feat_arr.shape[1])
        # check if the feat arr size is less than max_feat_arr size, if so append zeros and return
        if 0:
            if type(feat_arr) is numpy.ndarray:
                if feat_arr.shape[1] < self.max_frame_size:
                    temp = np.zeros((feat_arr.shape[0], self.max_frame_size - feat_arr.shape[1]))
                    temp1 = np.concatenate((feat_arr, temp), axis=1)
                else:
                    temp1 = feat_arr
            else:
                raise RuntimeError('The feature array in hdf5 file is not in the numpy format')

        # return the features, frces and the energy
        #feature_arr = torch.from_numpy(temp1)
        #print("ener_arr", ener_arr)
        return (feat_arr, frc_arr, ener_arr)
        #return feature_arr

