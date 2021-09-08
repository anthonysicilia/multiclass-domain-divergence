import torch

from .learners import SGDLearner

class DigitsHypothesisSpace(SGDLearner):

    """
    A simple 4-layer CNN
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        if self.finetune:
            raise ValueError('DigitsHypothesisSpace '
                'cannot be finetuned.')
    
    def __call__(self):
        return torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(1),
            torch.nn.Linear(9216, 128),
            torch.nn.Linear(128, self.num_classes)
        )