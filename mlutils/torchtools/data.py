from torch.utils.data import DataLoader, BatchSampler, Sampler
import random
import torch

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

        if shuffle:
            for i in range(len(difficulty_indices)):
                random.shuffle(difficulty_indices[i])
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
            CurriculumSampler(dataset, batch_size, difficulty_indices, shuffle),
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