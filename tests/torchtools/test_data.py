import unittest
import torch
import random
from copy import deepcopy

from mlutils.torchtools.data import get_curriculum_dataloader, get_probabilistic_curriculum_dataloader

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
        self.epochs = 2
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

    def test_probabilisitic_curriculum_loader(self):

        dataset = SimpleDataset(self.X, self.y)
        difficulty_indices = [
            [9, 8, 7], [6, 5], [4, 3, 2], [1, 0]
        ]
        logger.info('Test 1 - test curriculum no shuffling')
        epochs = 6

        loader = get_probabilistic_curriculum_dataloader(
            dataset,
            difficulty_indices,
            batch_size=self.batch_size,
            drop_last=False,
            num_phases_after_curriculum=epochs - len(difficulty_indices),
            intra_shard_shuffle=False,
            inter_shard_shuffle=False
        )
        expected_visible_data = []
        for epoch in range(epochs):
            if epoch < len(difficulty_indices):
                expected_visible_data += difficulty_indices[epoch]
            for i, batch in enumerate(loader):
                instance_ids = batch['instance_ids']
                expected_instance_ids = torch.tensor([
                    expected_visible_data[j] for j in range(i*self.batch_size, min((i+1)*self.batch_size, len(expected_visible_data)))
                ])
                assert (instance_ids == expected_instance_ids).all()

        logger.info('Test 2 - test with intra batch shuffling')
        seed = 0
        loader = get_probabilistic_curriculum_dataloader(
            dataset,
            difficulty_indices,
            batch_size=self.batch_size,
            drop_last=False,
            num_phases_after_curriculum=epochs - len(difficulty_indices),
            intra_shard_shuffle=seed,
            inter_shard_shuffle=False
        )

        copy_indices = deepcopy(difficulty_indices)
        shards = []
        for epoch in range(epochs):
            print(f'Phase: {epoch}')
            if epoch < len(copy_indices):
                shards.append(copy_indices[epoch])
                for i in range(len(shards)):
                    random.Random(seed).shuffle(shards[i])

                expected_visible_data = [index for shard in shards for index in shard]
            else:
                expected_visible_data = [index for shard in shards for index in shard]
                random.Random(seed).shuffle(expected_visible_data)
            for i, batch in enumerate(loader):
                instance_ids = batch['instance_ids']
                expected_instance_ids = torch.tensor([
                    expected_visible_data[j] for j in
                    range(i * self.batch_size, min((i + 1) * self.batch_size, len(expected_visible_data)))
                ])
                assert (instance_ids == expected_instance_ids).all()

        logger.info('Test 3 - test with inter batch shuffling')
        seed = 0
        loader = get_probabilistic_curriculum_dataloader(
            dataset,
            difficulty_indices,
            batch_size=self.batch_size,
            drop_last=False,
            num_phases_after_curriculum=epochs - len(difficulty_indices),
            intra_shard_shuffle=False,
            inter_shard_shuffle=seed
        )

        copy_indices = deepcopy(difficulty_indices)
        shards = []
        for epoch in range(epochs):
            print(f'Phase: {epoch}')
            if epoch < len(copy_indices):
                shards.append(copy_indices[epoch])
                random.Random(seed).shuffle(shards)

                expected_visible_data = [index for shard in shards for index in shard]
            else:
                expected_visible_data = [index for shard in shards for index in shard]
            for i, batch in enumerate(loader):
                instance_ids = batch['instance_ids']
                expected_instance_ids = torch.tensor([
                    expected_visible_data[j] for j in
                    range(i * self.batch_size, min((i + 1) * self.batch_size, len(expected_visible_data)))
                ])
                assert (instance_ids == expected_instance_ids).all()


        logger.info('Test 4 - test all shuffling')

        seed = 0
        loader = get_probabilistic_curriculum_dataloader(
            dataset,
            difficulty_indices,
            batch_size=self.batch_size,
            drop_last=False,
            num_phases_after_curriculum=epochs - len(difficulty_indices),
            intra_shard_shuffle=seed,
            inter_shard_shuffle=seed
        )

        copy_indices = deepcopy(difficulty_indices)
        shards = []
        for epoch in range(epochs):
            print(f'Phase: {epoch}')
            if epoch < len(copy_indices):
                shards.append(copy_indices[epoch])
                random.Random(seed).shuffle(shards)

                for i in range(len(shards)):
                    random.Random(seed).shuffle(shards[i])

                expected_visible_data = [index for shard in shards for index in shard]
            else:
                expected_visible_data = [index for shard in shards for index in shard]
                random.Random(seed).shuffle(expected_visible_data)
            for i, batch in enumerate(loader):
                instance_ids = batch['instance_ids']
                expected_instance_ids = torch.tensor([
                    expected_visible_data[j] for j in
                    range(i * self.batch_size, min((i + 1) * self.batch_size, len(expected_visible_data)))
                ])
                assert (instance_ids == expected_instance_ids).all()
