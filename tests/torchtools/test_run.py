# TODO consider changing the name of this
from mlutils.torchtools.run import BaseTrainer
from tests.torchtools.utils import LinearModel, SimpleDataset
import unittest
import torch
from torch.utils.data import DataLoader
# TODO figure out pip install for testing?


class TestHuggingFaceTrainer(unittest.TestCase):
    pass

class TestBaseTrainer(unittest.TestCase):

    def setUp(self) -> None:
        self.X = torch.arange(1, 101).reshape(-1, 2)
        self.weights = torch.tensor([2, 3])
        self.y = self.X @ self.weights
        self.dataset = SimpleDataset(self.X, self.y)
        self.model = LinearModel(1, 1)
        self.optimizer = torch.optim.AdamW(self.model.parameters())
        self.scheduler = None
        self.loss = None
        self.metrics = None

    def test_train(self):
        # expected to converge to 0
        trainer = BaseTrainer(self.model, self.optimizer, self.loss, self.scheduler, self.metrics, debug=True)

        data_loader = DataLoader(self.dataset, batch_size=20)
        epochs = 2

        trainer.train(data_loader, epochs, val_loader=data_loader)

    pass


class TestMultiTaskTrainer(unittest.TestCase):
    pass