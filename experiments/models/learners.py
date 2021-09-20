import torch

from ..utils import lazy_kwarg_init

class PyTorchHypothesisSpace:

    def __init__(self, num_classes, epochs, optimizer,      
        scheduler, batch_size, finetune):
        self.num_classes = num_classes
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size
        loader = torch.utils.data.DataLoader
        self.dataloader = lazy_kwarg_init(loader,
            batch_size=batch_size, shuffle=True)
        self.test_dataloader = lazy_kwarg_init(loader,
            batch_size=batch_size, shuffle=False, drop_last=False)
        self.finetune = finetune
        if self.finetune:
            # (better way?) by default, use same optimizer
            self.foptimizer = optimizer
            self.fepochs = epochs
    
    def __call__(self):
        raise NotImplementedError('PyTorchHypothesisSpace'
            ' is abstract. Please, implement __call__ in '
            ' a dervied class.')

    def loss_fn(self, yhat, y, w=None, **kwargs):
        w = w if w is not None else torch.ones_like(y).float()
        if self.num_classes > 1:
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        else:
            _loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
            loss_fn = lambda yhat, y: _loss_fn(yhat, y.float())
        l = loss_fn(yhat, y)
        return (l * w).sum() / w.sum()

    def errors(self, yhat, y):
        if self.num_classes > 1:
            errors = lambda yhat, y: (yhat.argmax(dim=1)!=y).sum()
        else:
            errors = lambda yhat, y: ((yhat > 0).long()!=y).sum()
        return errors(yhat, y)

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