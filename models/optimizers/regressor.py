
class Regressor:
    """ This class defines the optimizers

    Parameters
    ----------
    method: string
        Name of the optimization method, e.g.: 'SGD'.
    user_kwargs: dict
        The arguments typically coming from config file. 
    """
    def __init__(self, method, user_kwargs):
        self.method = method
        
        kwargs = {'lr': 1,}
        
        if self.method in ['SGD', 'sgd']:
            from torch.optim import SGD as optimizer
            _kwargs = {'lr': 0.001,
                       'momentum': 0.,
                       'dampening': 0.,
                       'weight_decay': 0,
                       'nesterov': False}
        
        elif self.method in ['Adam', 'ADAM', 'adam']:
            from torch.optim import Adam as optimizer
            _kwargs = {'lr': 0.001,
                       'betas': (0.9, 0.999), 
                       'eps': 1e-08,
                       'weight_decay': 0, 
                       'amsgrad': False}

        else:
            msg = f"The {method} is not implemented yet."
            raise NotImplementedError(msg)
            
        kwargs.update(_kwargs)
        
        if user_kwargs is not None:
            kwargs.update(user_kwargs)

        self.optimizer = optimizer
        self.kwargs = kwargs


    def regress(self, models):
        """ Returns the optimizer to models.

        Parameters
        ----------
        models: object
            Class representing the regression model.

        Returns
        -------
        regressor
            PyTorch optimizer.
        """
        try:
            params = models['model'].parameters()
        except:
            params = [p for model in models.values() for p in model.parameters()]

        if self.method in ['SGD', 'sgd']:
            regressor = self.optimizer(params,
                                       lr=self.kwargs['lr'],
                                       momentum=self.kwargs['momentum'],
                                       dampening=self.kwargs['dampening'],
                                       weight_decay=self.kwargs['weight_decay'],
                                       nesterov=self.kwargs['nesterov'])

        elif self.method in ['adam', 'ADAM', 'Adam']:
            regressor = self.optimizer(params,
                                       lr=self.kwargs['lr'],
                                       betas=self.kwargs['betas'],
                                       eps=self.kwargs['eps'],
                                       weight_decay=self.kwargs['weight_decay'],
                                       amsgrad=self.kwargs['amsgrad'])

        return regressor
