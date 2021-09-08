import torch

from ..estimators.erm import PyTorchHypothesisSpace
from ..utils import lazy_kwarg_init

class SGDLearner(PyTorchHypothesisSpace):

    def __init__(self, num_classes=4, lr=1e-2, 
        momentum=0.9, batch_size=250, epochs=150, lr_step=100, 
        gamma=0.1, features_lr=2e-5, features_epochs=0, finetune=False):

        optimizer = lazy_kwarg_init(torch.optim.SGD,
            lr=lr, momentum=momentum)
        scheduler = torch.optim.lr_scheduler.StepLR
        scheduler = lazy_kwarg_init(scheduler,
            step_size=lr_step, gamma=gamma)

        super().__init__(num_classes, epochs, optimizer, 
            scheduler, batch_size, finetune)
        
        if self.finetune:
            self.foptimizer = lazy_kwarg_init(
                torch.optim.SGD, lr=features_lr, 
                momentum=momentum)
            if features_epochs is not None:
                self.fepochs = features_epochs
                # otherwise, super sets to epochs
    
    def __call__(self):
        raise NotImplementedError('SGDLearner is abstract.'
            ' Please, implement __call__ in a dervied class.')