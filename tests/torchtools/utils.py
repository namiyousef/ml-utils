import torch
from torch.utils.data import Dataset
class LinearModel(torch.nn.Module):

    def __init__(self, n_input_features, n_output_features):
        super().__init__()
        self.linear = torch.nn.Linear(n_input_features, n_output_features, bias=False)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


class HuggingFaceModel(torch.nn.Module):
    pass


class MultiTaskModel(torch.nn.Module):
    pass

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):

        return dict(
            inputs=self.X[index],
            targets=self.y[index],
            metadata=0,
        )

class HuggingFaceDatasetMock():
    pass

class ModelMock:

    def __init__(self):
        pass

    def __call__(self, batch_dict):
        batch_dict['loss'] = None
        batch_dict['outputs'] = None
        return batch_dict

    def to(self, device):
        return self

    def train(self):
        pass

    def eval(self):
        pass

class LossMock:
    def __init__(self):
        pass

    def __call__(self):
        pass

    def backward(self):
        pass

    def item(self):
        return 0

    def to(self, device):
        return self


class OptimizerMock:

    def __init__(self):
        pass

    def step(self):
        pass

    def zero_grad(self):
        pass

class SchedulerMock:

    def __init__(self):
        pass

    def step(self):
        pass