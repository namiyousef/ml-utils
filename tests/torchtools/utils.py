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
            instance_ids=torch.tensor(index),
            metadata=0,
        )

class HuggingFaceDatasetMock():
    pass

class ModelMock:

    def __init__(self):
        pass

    def __call__(self, inputs):
        output = -inputs
        return output

    def to(self, device):
        return self

    def train(self):
        pass

    def eval(self):
        pass

class LossMock:
    def __init__(self):
        pass

    def __call__(self, inputs, targets):
        self.loss = torch.mean(targets - inputs)
        return self.loss
    def backward(self):
        pass

    def item(self):
        return self.loss.item()

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


class TorchMetricMock:
    def __init__(self, torch_metric):
        self.torch_metric = torch_metric
        if self.torch_metric.__dict__['reduction'] == 'none':
            self.instance_metric = True

    def forward(self, batch):
        scores = self.torch_metric(batch['inputs'], batch['outputs'])
        if self.instance_metric:
            return batch['instance_ids'], scores
        else:
            return scores


def prepare_torch_metric(metric):
    metric._forward = metric.forward
    metric.forward = lambda batch:  (batch['instance_ids'], metric._forward(batch['inputs'], batch['targets'])) if metric.__dict__['reduction'] == 'none' else metric._forward(batch['inputs'], batch['targets'])
    return metric
