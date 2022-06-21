import torch
from torch.utils.data import Dataset
class LinearModel(torch.nn.Module):

    def __init__(self, n_input_features, n_output_features):
        super().__init__()
        self.linear = torch.nn.Linear(n_input_features, n_output_features)

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