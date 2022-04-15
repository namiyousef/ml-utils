import unittest
from mlutils.torchtools.metrics import Metric, FScore
from sklearn.metrics import f1_score
import torch
import math


binary_single_batch = {
    'input': torch.tensor([1,0,1,0,1,0]),
    'target': torch.tensor([1,0,1,0,1,0]),
    'expected': 1
}


class TestMetrics(unittest.TestCase):
    def test_metric(self):

        # empty
        class TestMetric(Metric):
            def __init__(self):
                super(TestMetric, self).__init__()
            def _forward(self, outputs, targets, **kwargs):
                return outputs, targets

        test_metric = TestMetric()

        # test single batch input
        single_batch = torch.tensor([1,2,3,4,5])

        outputs, targets = test_metric(single_batch, single_batch)
        assert (outputs == single_batch).all() and (targets == single_batch).all()

        # test multiple batch input
        multiple_batch = torch.tensor([[1,2,3,4,5], [1,2,3,4,5]])
        outputs, targets = test_metric(multiple_batch, multiple_batch)
        assert (outputs == multiple_batch.flatten()).all() and (targets == multiple_batch.flatten()).all()

        # test argmax
        single_batch_probas = torch.tensor([[0.5, 0.3, 0.2]])
        single_batch = torch.tensor([0])
        outputs, targets = test_metric(single_batch_probas, single_batch)
        assert (outputs == torch.argmax(single_batch_probas, dim=-1)).all() and (targets == single_batch).all()

        # multiple batch
        multiple_batch_probas = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0, 0.9]])
        multiple_batch = torch.tensor([0, 2])
        outputs, targets = test_metric(multiple_batch_probas, multiple_batch)
        assert (outputs == torch.argmax(multiple_batch_probas, dim=-1)).all() and (targets == multiple_batch).all()


    def test_recall(self):
        pass

    def test_precision(self):
        pass

    def test_fscore(self):
        metric = FScore()

        output = torch.tensor([1, 0, 1, 0, 1, 0])
        target = output
        score = metric(output, target)
        expected_score = f1_score(target, output)

        assert score.item() == expected_score


        # multiple classes
        metric = FScore(average='macro') # TODO maybe need to redefine n_classes cuz unclear?
        output = torch.tensor([1,2,2,3,3,3,4,4,4,4,5,5,5,5,5])
        target = torch.tensor([1,2,2,3,3,3,4,4,4,4,5,5,5,5,5])
        score = metric(output, target)
        expected_score = f1_score(target, output, average='macro')

        assert score.item() == expected_score

        output = torch.tensor([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5])
        target = torch.tensor([1, 0, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5])
        score = metric(output, target)
        expected_score = f1_score(target, output, average='macro')

        # double check f1 to see if working correctly??
        assert math.isclose(score.item(), expected_score, abs_tol=0.01)

        # test single batch inputs

        # test multi batch inputs

        # test diff betas

        # test multi class problem


if __name__ == '__main__':
    unittest.main()