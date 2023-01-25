from mlutils.external_apis.microsoft import batch_by_size, batch_by_size_min_buckets
import unittest

class TestMicrosoftTranslate(unittest.TestCase):

    def test_autobatch(self):

        limit = 1000
        # test sizes less than limit
        size_mapping = {0: 100, 1: 100, 2: 700}
        batched_items = batch_by_size(size_mapping, limit)

        assert batched_items == [{'idx': list(size_mapping.keys()), 'total_size': sum(size_mapping.values())}]
        
        # test sizes greater than limit
        size_mapping = {0: 900, 1: 200, 2: 600, 3: 100, 4: 200}
        batched_items = batch_by_size(size_mapping, limit)
        assert batched_items == [
            {'idx': [0], 'total_size': 900},
            {'idx': [1, 2, 3], 'total_size': 900},
            {'idx': [4], 'total_size': 200}
        ]

        # test sizes = limit
        size_mapping = {0: 900, 1: 100, 2: 100}
        batched_items = batch_by_size(size_mapping, limit)
        assert batched_items == [
            {'idx': [0, 1], 'total_size': 1000},
            {'idx': [2], 'total_size': 100}
        ]

    def test_autobatch_max(self,):

        limit = 1000

        # test: initial exceeds limit
        size_mapping = {0: 1001, 6: 200, 1: 900, 2: 100, 3: 400, 4: 600, 5: 800}
        batched_items = batch_by_size_min_buckets(size_mapping, limit)
        
        # sort batch items for comparison
        ground_truth = [{'idx': [0], 'total_size': 1001}, {'idx': [6, 5], 'total_size': 1000}, {'idx': [1, 2], 'total_size': 1000}, {'idx': [3, 4], 'total_size': 1000}]
        ground_truth_mapping = {tuple(sorted(batch_dict['idx'])): batch_dict['total_size'] for batch_dict in  ground_truth}

        batched_items_mapping = {tuple(sorted(batch_dict['idx'])): batch_dict['total_size'] for batch_dict in  batched_items}

        assert ground_truth_mapping == batched_items_mapping

        
if __name__ == '__main__':
    unittest.main()

    