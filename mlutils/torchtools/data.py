from torch.utils.data import DataLoader, Dataset, BatchSampler, Sampler
import random

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

