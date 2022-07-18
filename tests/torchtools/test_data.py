import unittest
import torch
import random


from mlutils.torchtools.data import get_curriculum_dataloader

from tests.torchtools.utils import SimpleDataset


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
date_format = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s', datefmt=date_format)
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)

class TestData(unittest.TestCase):
    def setUp(self) -> None:
        self.length = 10
        self.batch_size = 2
        self.X = torch.ones((self.length, 2))
        self.y = torch.zeros(self.length)
    def test_curriculum_loader(self):
        dataset = SimpleDataset(self.X, self.y)
        logger.info('Test 1 - test curriculum basic')
        difficulty_indices = [
            [9, 8, 7], [6, 5], [4, 3, 2], [1, 0]
        ]
        loader = get_curriculum_dataloader(
            dataset, difficulty_indices, batch_size=self.batch_size, drop_last=False, shuffle=False
        )
        for i, batch in enumerate(loader):

            assert (batch['instance_ids'] == torch.tensor(
               [self.length - (j+1) for j in list(range(i*self.batch_size, (i+1)*self.batch_size))]
                    )).all()


        logger.info('Test 2 - test curriculum shuffle')
        seed = 0
        loader = get_curriculum_dataloader(
            dataset, difficulty_indices, batch_size=self.batch_size, drop_last=False, shuffle=seed
        )


        difficulty_indices_shuffled = difficulty_indices.copy()
        for i in range(len(difficulty_indices_shuffled)):
            random.Random(seed).shuffle(difficulty_indices_shuffled[i])  # need to run twice...
            random.Random(seed).shuffle(difficulty_indices_shuffled[i])

        difficulty_indices_shuffled = [index for indices in difficulty_indices_shuffled for index in indices]

        for i, batch in enumerate(loader):
            assert (batch['instance_ids'] == torch.tensor(difficulty_indices_shuffled[i*self.batch_size: (i+1)*self.batch_size])).all()


