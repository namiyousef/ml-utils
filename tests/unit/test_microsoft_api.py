from mlutils.external_apis.microsoft import batch_by_size
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



if __name__ == '__main__':
    unittest.main()