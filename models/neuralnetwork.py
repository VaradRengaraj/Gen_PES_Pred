import torch.utils.data as Data
import torch
from models.optimizers.regressor import Regressor
import time
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from numpy import savetxt
import numpy as np

class NeuralNetwork():
    """ Neural network(NN) definition class, neural network is created, trained. GPU friendly. Typically instanstiated
    after the model is chosen as neural network. Uses Regressor class, for the optimizer facility.

    Parameters
    ----------
    elements: int
        Int containing the number of elements in the system.
    hiddenlayers: list or dict
        List containing the number of nodes in each hidden layer.
    activation: list
        List containing the activation function names for each hidden layer.
    optimizer: dict
        Dict whose parameters are optimizer method and the parameters used.
    random_seed: int
        Int containing the random seed value.
    batch_size: int
        Int containing the batch size used for NN training. This is not used now. 
        Batch size comes automatically from dataloader.
    epoch: int
        Int containing the number of NN training cycles.
    validate: bool
        Boolean to inform if we are using validation dataset.
    force_matching: bool
        Boolean to inform if we are training forces also.  
    """


    def __init__(self, elements, hiddenlayers, activation, optimizer, random_seed, batch_size, epoch, validate, force_matching):
        self.elements = elements
        print(f"No of elements   : {self.elements}")
        self._hiddenlayers = hiddenlayers

        # Hidden layer list is saved.
        if isinstance(hiddenlayers, list):
            hl = {}
            for element in range(self.elements):
                hl[element] = hiddenlayers + [1]
            self.hiddenlayers = hl
        elif isinstance(hiddenlayers, dict):
            for key, value in hiddenlayers.items():
                hiddenlayers[key] = value + [1]
            self.hiddenlayers = hl
        else:
            msg = f"Don't recognize {type(hiddenlayers)}. " +\
                  f"Please refer to documentations!"
            raise TypeError(msg)

        # Set-up activation
        self.activation = {}
        activation_modes = ['Tanh', 'Sigmoid', 'Linear', 'ReLU',
                            'PReLU', 'ReLU6', 'RReLU', 'SELU', 'CELU',
                            'Softplus', 'Softshrink', 'Softsign', 'Tanhshrink',
                            'Softmin', 'Softmax', 'LogSoftmax', 'LogSigmoid',
                            'LeakyReLU', 'Hardtanh', 'Hardshrink', 'ELU',]

        if isinstance(activation, str):
            for e in self.elements:
                self.activation[e] = [activation] * \
                                     len(self.hiddenlayers[e])
        elif isinstance(activation, list):
            for element in range(self.elements):
                self.activation[element] = activation
        else:
            # Users construct their own activations.
            self.activation = activation

        for element in range(self.elements):
            print(f"act   : {self.activation[element]}")

        # Check if each of the activation functions is implemented.
        for e in range(self.elements):
            for act in self.activation[e]:
                if act not in activation_modes:
                    msg = f"{act} is not implemented. " +\
                          f"Please choose from {activation_modes}."
                    raise NotImplementedError(msg)
            assert len(self.activation[e]) == len(self.hiddenlayers[e]),\
            "The length of the activation function is inconsistent "+\
            "with the length of the hidden layers."

        if random_seed:
            torch.manual_seed(random_seed)

        self.batch_size = batch_size
        self.epoch = epoch
        self.validate = validate
        self.force_matching = force_matching
  
        if torch.cuda.is_available():       
            self.device = "cuda:0"
        else:
            self.device = "cpu"

    def train(self, TrainData, optimizer, ValData=None):
        """ function to train the NN.
        
        Parameters
        ----------
        TrainData: DataLoader obj
            DataLoader instance used to get training dataset.
        Optimizer: Dict
            Dict containing the model and parameters list.
        ValData: DataLoader obj
            DataLoader instance used to get validation dataset.
        """
        if(isinstance(TrainData, Data.DataLoader)):
            train_features, train_frces, train_labels = next(iter(TrainData))
            self.no_of_descriptors = train_features.shape[2]
            print("no_of_desc", self.no_of_descriptors)
        else:
            msg = f"Don't recognize {type(TrainData)}. " +\
                  f"Please refer to documentations!"
            raise TypeError(msg)

        if(ValData):
            if not (isinstance(ValData, Data.DataLoader)):
                msg = f"Don't recognize {type(ValData)}. " +\
                      f"Please refer to documentations!"
                raise TypeError(msg)

        # Calculate the total number of entries in the validation and 
        # training dataset.
        trng_size = len(TrainData.dataset)
        val_size = len(ValData.dataset)

        # If batch_size is None and optimizer is Adam or SGD, 
        # then batch_size equals total structures.
        if optimizer['method'] in ['sgd', 'SGD', 'Adam', 'adam', 'ADAM']:
            if self.batch_size == None:
                self.batch_size = self.elements
                print("batchsize", self.batch_size)

        # Calculate total number of parameters.
        self.total_parameters = 0
        for element in range(self.elements):
            for i, hl in enumerate(self.hiddenlayers[element]):
                if i == 0:
                    self.total_parameters += (self.no_of_descriptors+1)*hl
                else:
                    self.total_parameters += (self.hiddenlayers[element][i-1]+1)*hl

        print(f"No of descriptors   : {self.no_of_descriptors}")
        print(f"Total number of parameters   : {self.total_parameters}")

        self.models = {}
        for element in range(self.elements): # Number of models depend on species
            m = 'torch.nn.Sequential('
            for i, act in enumerate(self.activation[element]):
                if i == 0:
                    m += f'torch.nn.Linear({self.no_of_descriptors}, \
                               {self.hiddenlayers[element][i]}), '
                else:
                    m += f'torch.nn.Linear({self.hiddenlayers[element][i-1]}, \
                               {self.hiddenlayers[element][i]}), '

                if act == 'Linear':
                    continue
                else:
                    m += f'torch.nn.{act}(), '
            m += f')'

            self.models[element] = eval(m).double().to(self.device)

        self.regressor = Regressor(optimizer['method'], optimizer['parameters'])
        self.optimizer = self.regressor.regress(models=self.models)

        if 0:
            scheduler = StepLR(self.optimizer, step_size=25, gamma=0.1)

        print(f"No of descriptors  : {self.no_of_descriptors}")
        print(f"No of parameters   : {self.total_parameters}")
        print(f"No of epochs       : {self.epoch}")
        print(f"Optimizer          : {optimizer['method']}")
        print(f"Batch_size         : {self.batch_size}\n")

        # Run Neural Network Potential Training
        t0 = time.time()
        val_batch = None
        #current = 0
        lst_tr = []
        lst_val = []

        for epoch in range(300): #range(self.epoch):
            print("\nEpoch {:4d}: ".format(epoch+1))
            current = 0
            num_batches = 0
            avg_trng_loss = 0.

            for batch in TrainData:
                ener_loss, ener_mae = self.calculate_loss(self.models, batch)
                ener_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                current += batch[0].shape[0]
                num_batches += 1
                avg_trng_loss += ener_mae 
                print(" Training Loss MSE: {:10.6f}     Training Loss MAE: {:10.4f} [{:5d}/{:5d}]".\
                        format(ener_loss, ener_mae, current, trng_size))
                  
            if torch.cuda.is_available():
                avg_trng_loss = avg_trng_loss.detach().to("cpu").numpy()
            else:
                avg_trng_loss = avg_trng_loss.detach().numpy()

            print("Average Training Loss {:10.4f} [{:5d}/{:5d}]".\
                        format(avg_trng_loss/num_batches, (epoch+1), 10))
            lst_tr.append(avg_trng_loss/num_batches)
            current = 0
            num_batches = 0
            avg_val_loss = 0.

            if self.validate:
                with torch.no_grad():
                    for val_batch in ValData:
                        val_ener_loss, val_ener_mae = self.calculate_loss(self.models, val_batch)
                        num_batches += 1
                        avg_val_loss += val_ener_mae
                        current += val_batch[0].shape[0] 
                        print(" Validation Loss MSE: {:10.6f}     Validation Loss MAE: {:10.4f}  [{:5d}/{:5d}]".\
                            format(val_ener_loss, val_ener_mae, current, val_size))

                    print(" Average validation loss MAE {:10.4f} [{:5d}/{:5d}]".\
                        format(avg_val_loss/num_batches, (epoch+1), 10))                   
        
            #scheduler.step()
            if torch.cuda.is_available():
                avg_val_loss = avg_val_loss.detach().to("cpu").numpy()
            else:
                avg_val_loss = avg_val_loss.detach().numpy()

            lst_val.append(avg_val_loss/num_batches) 

        t1 = time.time()
        print("\nThe training time: {:.2f} s".format(t1-t0))
        np1 = np.asarray(lst_tr)
        np2 = np.asarray(lst_val)
        savetxt('trng_loss', np1, delimiter=',')
        savetxt('val_loss', np2, delimiter=',')

    def calculate_loss(self, models, batch):
        energy_loss = 0.
        energy_mae = 0.
        features = batch[0]
        frces = batch[1]
        energy = batch[2]

        #print("features shape", features.shape)
        #print("frces shape", frces.shape)
        #print("energy shape", energy.shape)
        _Energy = 0
        Energy = energy[0::1,0,:]
        #print("Energy", Energy)

        for element, model in models.items():
            if self.validate:
                model.eval()

            #temp = Variable(features[0::1,element,:])
            temp = features[0::1,element,:]
            #print("feature-->", temp)
            #print("temp.shape", temp.shape)
            _x = temp.requires_grad_()
            #_energy = model(_x).sum() 
            _energy = model(_x)
            #print("_enery shape", _energy.shape)
            #print("_energy", _energy)
            _Energy += _energy

        #print("_Energy shape", _Energy.shape)
        #print("_Energy", _Energy)
        #print("Energy", Energy)
        loss_func = torch.nn.MSELoss()  
        mae_func = torch.nn.L1Loss()
        energy_loss = loss_func(_Energy, Energy)
        energy_mae = mae_func(_Energy, Energy)
        #print("energy_loss, energy_mae", energy_loss, energy_mae)  
        return energy_loss, energy_mae
