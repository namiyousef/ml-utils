from torch.utils.data import DataLoader, BatchSampler, Sampler
import random
import torch
import warnings

class CurriculumSampler(Sampler):
    """Sampling class to create curriculum learning batches from a given dataset and associated diffiulty indices
    E.g. if data has indices [1, 2, 3, 4, 5] but the difficulties of the instances are: [[2, 5, 4], [3, 1]]
    Then would return batches (bs=2)
    [2, 5]
    [4, 3]
    [1]
    If shuffling enabled, then will shuffle the inner indices, not the difficulty boundaries
    e.g.
    [5, 2] # shuffled
    [4, 3] # not shuffled because 4 is easier than 3, they come from diff groups
    [1]
    
    :param dataset: dataset you want to use for curriculum learning
    :type dataset: torch.utils.data.Dataset
    :param batch_size: batch_size
    :type batch_size: batch_size
    :param difficulty_indices: difficulty indices # TODO need to think about higher dimensinoal or lower dimensional difficulties
    :type difficulty_indices: List[List]
    :param shuffle: shuffle flag
    :type shuffle: bool
    :returns: generator object of indices
    """
    def __init__(self, dataset, batch_size, difficulty_indices, shuffle=False):
        super().__init__(dataset)
        self.dataset_length = len(dataset)
        self.batch_size = batch_size
        self.n_batches = self.dataset_length / self.batch_size

        if isinstance(shuffle, bool):
            if shuffle:
                print('We are shuffling with no seed')
                for i in range(len(difficulty_indices)):
                    random.shuffle(difficulty_indices[i])
        elif isinstance(shuffle, int):
            print('we are shuffling with seed')
            for i in range(len(difficulty_indices)):
                random.Random(shuffle).shuffle(difficulty_indices[i])

        self.difficulty_indices = [index for indices in difficulty_indices for index in indices]

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        for batch_num in range(int(self.n_batches)):
            for index in range(batch_num * self.batch_size, (batch_num + 1) * self.batch_size):
                yield self.difficulty_indices[index]

        if int(self.n_batches) < self.n_batches:
            for index in range(int(self.n_batches) * self.batch_size, self.dataset_length):
                yield self.difficulty_indices[index]

def get_curriculum_dataloader(dataset, difficulty_indices, batch_size, drop_last=False, shuffle=False):
    """
    Returns DataLoader for CurriculumDataset
    :param dataset:
    :param difficulty_indices:
    :param batch_size:
    :param drop_last:
    :param shuffle:
    :return:
    """
    return DataLoader(
        dataset, batch_size=None,  # must be disabled when using samplers
        sampler=BatchSampler(
            CurriculumSampler(dataset, batch_size, difficulty_indices, shuffle=shuffle),
            batch_size=batch_size, drop_last=drop_last)
    )



class RandomBatchSampler(Sampler):
    """Sampling class to create random sequential batches from a given dataset
    E.g. if data is [1,2,3,4] with bs=2. Then first batch, [[1,2], [3,4]] then shuffle batches -> [[3,4],[1,2]]
    This is useful for cases when you are interested in 'weak shuffling'
    :param dataset: dataset you want to batch
    :type dataset: torch.utils.data.Dataset
    :param batch_size: batch size
    :type batch_size: int
    :returns: generator object of shuffled batch indices
    """
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_batches = self.dataset_length / self.batch_size
        self.batch_ids = torch.randperm(int(self.n_batches))

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        for id in self.batch_ids:
            idx = torch.arange(id * self.batch_size, (id + 1) * self.batch_size)
            for index in idx:
                yield int(index)
        if int(self.n_batches) < self.n_batches:
            idx = torch.arange(int(self.n_batches) * self.batch_size, self.dataset_length)
            for index in idx:
                yield int(index)


def get_random_batch_dataloader(dataset, batch_size=32, drop_last=False):
    """Implements fast loading by taking advantage of .h5 dataset
    The .h5 dataset has a speed bottleneck that scales (roughly) linearly with the number
    of calls made to it. This is because when queries are made to it, a search is made to find
    the data item at that index. However, once the start index has been found, taking the next items
    does not require any more significant computation. So indexing data[start_index: start_index+batch_size]
    is almost the same as just data[start_index]. The fast loading scheme takes advantage of this. However,
    because the goal is NOT to load the entirety of the data in memory at once, weak shuffling is used instead of
    strong shuffling.
    :param dataset: a dataset that loads data from .h5 files
    :type dataset: torch.utils.data.Dataset
    :param batch_size: size of data to batch
    :type batch_size: int
    :param drop_last: flag to indicate if last batch will be dropped (if size < batch_size)
    :type drop_last: bool
    :returns: dataloading that queries from data using shuffled batches
    :rtype: torch.utils.data.DataLoader
    """
    return DataLoader(
        dataset, batch_size=None,  # must be disabled when using samplers
        sampler=BatchSampler(RandomBatchSampler(dataset, batch_size), batch_size=batch_size, drop_last=drop_last)
    )

class ProbabilisticCurriculumSampler(Sampler):
    """Sampling class to generate curriculum learning samplers from a given dataset and associated difficulty indices.
    This is largely based on the following paper: https://arxiv.org/pdf/1811.00739.pdf.
    The implementation at the moment is "default".
    A caveat is in our case, we don't have any further bucketing after the shards are created. Instead we sample
    uniformly from each shard.
    Consider difficulty_indices [[9, 8, 7], [6, 5, 4], [3, 2, 1], [0]] and batch_size=2
    In the case of no shuffle entirely, this is simply a batch sampler, thus:
    batch 0: [9, 8]
    batch 1: [7] etc..

    In the case of intra_shuffle, at each "phase" we will have each shard shuffled, so
    phase 1: visible data [9, 8, 7] --> shuffled [9, 7, 8]
    batch 0: [9, 7]
    batch 1: [8]

    phase 2: visible data [9, 8, 7], [6, 5, 4] --> shuffled [9, 7, 8], [4, 5, 6]
    batch 0: [9, 8]
    batch 1: [7, 6]
    batch 2: [5, 4]

    In the case of inter_shuffle, we simply shuffle the shards before any intra shuffling (if enabled). So if both
    are on, then:

    phase 1:[9, 8, 7] --> [9, 7, 8]
    batch 0: [9, 8], batch 1: [7]

    phase 2: [9, 8, 7], [6, 5, 4] --> shuffle shards [6, 5, 4], [9, 8, 7] --> shuffle within shards [4, 5, 6], [9, 7, 8]
    batch 0: [6, 5]
    batch 1: [4, 9]
    batch 2: [8, 7]

    NOTE: this is designed to run with simple training paradigms (e.g. with num_epochs). However, epochs is not always the same
    as "phase". In this case, number of epochs must be set to be num_shards + num_phases_after_curriculum. This has implications
    for how some metrics are stored and calculated (e.g. instance metrics) and need to be accounted for elsewhere.

    :param dataset: dataset you want to batch
    :type dataset: torch.utils.data.Dataset
    :param batch_size: batch size
    :type batch_size: int
    :param difficulty_indices: ids of instances ordered by difficulty and grouped into shards
    :type difficulty_indices: List[List[int]]
    :param num_phases_after_curriculum: number of epochs to train for after
    :type num_phases_after_curriculum: int
    :param intra_shard_shuffle: flag to enable shuffling within each shard at each phase (including after curriculum)
    :type intra_shard_shuffle: bool or int (for seed)
    :param inter_shard_shuffle: flag to enable shuffling of shards within each curriculum phase
    :type inter_shard_shuffle: bool or int (for seed)

    :returns: generator object

    """
    def __init__(
            self,
            dataset,
            batch_size,
            difficulty_indices,
            num_phases_after_curriculum,
            sampling_weights,
            intra_shard_shuffle=True,
            inter_shard_shuffle=True
    ):
        # TODO shuffle/no shuffle?
        # TODO strategy?

        super().__init__(dataset)
        self.dataset_length = len(dataset)
        self.batch_size = batch_size
        self.num_shards = len(difficulty_indices)
        self.num_phases_after_curriculum = num_phases_after_curriculum
        self.shard_id = 0
        self.shards = []
        self.visible_data_length = 0
        self.difficulty_indices = difficulty_indices
        self.intra_shard_shuffle = intra_shard_shuffle
        self.inter_shard_shuffle = inter_shard_shuffle

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        if self.shard_id < self.num_shards:
            shard = self.difficulty_indices[self.shard_id]
            self.shards.append(shard)
            self.visible_data_length += len(shard)

            if isinstance(self.intra_shard_shuffle, bool):
                if self.intra_shard_shuffle:
                    for i in range(len(self.shards)):
                        random.shuffle(self.shards[i])
            elif isinstance(self.intra_shard_shuffle, int):
                for i in range(len(self.shards)):
                    random.Random(self.intra_shard_shuffle).shuffle(self.shards[i])

            if isinstance(self.inter_shard_shuffle, bool):
                if self.inter_shard_shuffle:
                    random.shuffle(self.shards)
            elif isinstance(self.inter_shard_shuffle, int):
                random.Random(self.inter_shard_shuffle).shuffle(self.shards)

            self.n_batches = self.visible_data_length / self.batch_size
            visible_data = [index for indices in self.shards for index in indices]


        else:
            # behaviour: we are flattening based on the shards as opposed to the last order of visible data
            # does this matter at all? In the case of no shuffle maybe?
            visible_data = [index for indices in self.shards for index in indices]
            if isinstance(self.intra_shard_shuffle, bool):
                if self.intra_shard_shuffle:
                    random.shuffle(visible_data)


            elif isinstance(self.intra_shard_shuffle, int):
                random.Random(self.intra_shard_shuffle).shuffle(visible_data)

        for batch_num in range(int(self.n_batches)):
            for index in range(batch_num * self.batch_size, (batch_num + 1) * self.batch_size):
                yield visible_data[index]

        if int(self.n_batches) < self.n_batches:
            for index in range(int(self.n_batches) * self.batch_size, self.visible_data_length):
                yield visible_data[index]

        self.shard_id += 1


def get_probabilistic_curriculum_dataloader(
        dataset,
        difficulty_indices,
        num_phases_after_curriculum=2,
        sampling_weights=None,
        intra_shard_shuffle=True,
        inter_shard_shuffle=True,
        batch_size=32,
        drop_last=False
):
    assert num_phases_after_curriculum >= 0, 'num_phases_after_curriculum must be greater than or equal to 0'
    if sampling_weights:
        warnings.warn('You have provided sampling weights as an argument but this is currently ignored', UserWarning, stacklevel=2)
    return DataLoader(
        dataset, batch_size=None,  # must be disabled when using samplers
        sampler=BatchSampler(
            ProbabilisticCurriculumSampler(
                dataset,
                batch_size,
                difficulty_indices,
                num_phases_after_curriculum,
                sampling_weights,
                intra_shard_shuffle,
                inter_shard_shuffle
            ),
            batch_size=batch_size,
            drop_last=drop_last
        )
    )
