import torch
import torchvision as tv

from .learners import SGDLearner

class ResNetHypothesisSpace(SGDLearner):

    def __init__(self, size, *args, features_lr=1e-4,     
        features_epochs=None, nonlinear_head=False, 
        batch_size=64, **kwargs):
        self.nonlinear_head = nonlinear_head
        if size == 18:
            self.resnet = tv.models.resnet18
        elif size == 50:
            self.resnet = tv.models.resnet50
        else:
            raise NotImplementedError(
                f'Resnet{size} not implemented.')
        super().__init__(*args, features_lr=features_lr, 
            features_epochs=features_epochs, finetune=True, 
            batch_size=batch_size, **kwargs)
    
    def __call__(self):
        resnet = self.resnet(pretrained=True)
        num_inputs = resnet.fc.in_features
        resnet.fc = torch.nn.Identity()
        model = torch.nn.Sequential()
        model.add_module('f', resnet)
        model.add_module('d', torch.nn.Identity())
        head = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 256),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.num_classes)
        ) if self.nonlinear_head else \
            torch.nn.Linear(num_inputs, self.num_classes)
        model.add_module('c', head)
        return model

class ResNet18HypothesisSpace(ResNetHypothesisSpace):

    def __init__(self, *args, **kwargs):

        super().__init__(18, *args, **kwargs)

class ResNet50HypothesisSpace(ResNetHypothesisSpace):

    def __init__(self, *args, **kwargs):

        super().__init__(50, *args, **kwargs)