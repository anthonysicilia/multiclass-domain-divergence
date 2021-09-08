import torch

from .learners import SGDLearner

class LinearHypothesisSpace(SGDLearner):

    def __init__(self, *args, num_inputs=768, **kwargs):

        super().__init__(*args, **kwargs)
        if self.finetune:
            raise ValueError('LinearHypothesisSpace '
                'cannot be finetuned.')
        self.num_inputs = num_inputs
    
    def __call__(self):
        return torch.nn.Linear(self.num_inputs, self.num_classes)

class NonLinearHypothesisSpace(SGDLearner):

    def __init__(self, *args, num_inputs=768, **kwargs):

        super().__init__(*args, **kwargs)
        if self.finetune:
            raise ValueError('NonLinearHypothesisSpace '
                'cannot be finetuned.')
        self.num_inputs = num_inputs
    
    def __call__(self):
        return torch.nn.Sequential(
            torch.nn.Linear(self.num_inputs, 256),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.num_classes)
        )