import torch
# TODO deprecate this and use torchmetrics instead

def _get_tensor_metadata(tensor):
    shape = tensor.shape
    dim = len(shape)
    dtype = tensor.dtype
    return shape, dim, dtype


class Metric(torch.nn.Module):
    """ Base class """
    def __init__(self, reduction='none', ignore_index=-100):
        super(Metric, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction # TODO so far does nothing, but thing about cases when you won't require
        # means, but might requires totals instead, but also cases where you might want outputs to be at batch level,
        # etc. (batch,) where each item is the metric for that batch

    def forward(self, outputs, targets):

        outputs_shape, outputs_dim, outputs_dtype = _get_tensor_metadata(outputs)
        targets_shape, targets_dim, targets_dtype = _get_tensor_metadata(targets)
        if outputs_dim == 1:
            assert outputs_dim == targets_dim
            if outputs_dtype != torch.int64:
                outputs = torch.where(outputs >= 0.5, 1, 0)

            ids_ignore = targets == self.ignore_index
            filtered_outputs = outputs[~ids_ignore]
            filtered_targets = targets[~ids_ignore]
            return self._forward(filtered_outputs, filtered_targets)

        else:
            outputs_shape, outputs_dim, outputs_dtype = _get_tensor_metadata(outputs)

            if (outputs_dim - targets_dim) == 1:
                # TODO this does not work for probabilistic metrics
                outputs = torch.argmax(outputs, dim=-1)
                outputs_shape, outputs_dim, outputs_dtype = _get_tensor_metadata(outputs)

            targets = targets.flatten()
            assert outputs_dim == targets_dim

            outputs = outputs.flatten()

            ids_ignore = targets == self.ignore_index
            filtered_outputs = outputs[~ids_ignore]
            filtered_targets = targets[~ids_ignore]

            return self._forward(filtered_outputs, filtered_targets)


class MultiAccuracy(torch.nn.Module):
    """Calculates accuracy for multiclass inputs (batchsize, feature length) by determining the most likely class
    using argmax -> (batchsize,) and then comparing with targets which are also (batchsize,)
    """
    def __init__(self):
        super(MultiAccuracy, self).__init__()

    def forward(self, outputs, targets):
        if outputs.shape != targets.shape:
            outputs = torch.argmax(outputs, dim=-1)
        return torch.sum(outputs == targets, dim=-1) / targets.shape[-1]




class Precision(Metric):
    """Implements the Precision score for a binary classification problem with integer classes {0, 1}
    """
    def __init__(self):
        super(Precision, self).__init__()

    def _forward(self, outputs, targets):

        # filter only positive predictions
        ids_pos = targets == 1

        pos_targets = targets[ids_pos]
        pos_outputs = outputs[ids_pos]
        neg_targets = targets[~ids_pos]
        neg_outputs = outputs[~ids_pos]
        tp = torch.sum(pos_outputs == pos_targets)
        fp = torch.sum(neg_targets != neg_outputs)
        return tp/(tp+fp)

class Recall(Metric):
    """Implements the Recall score for a binary classification problem with pixel values between [0, 1]
    """
    def __init__(self):
        super(Recall, self).__init__()

    def _forward(self, outputs, targets):

        # filter only positive predictions
        ids_pos = targets == 1

        pos_targets = targets[ids_pos]
        pos_outputs = outputs[ids_pos]

        tp = torch.sum(pos_outputs == pos_targets)
        fn = torch.sum(pos_targets != pos_outputs)
        recall = tp / (tp + fn)
        return recall


class FScore(Metric):
    """
    Calculates
    """

    def __init__(self, beta=1, average='binary'):
        super(FScore, self).__init__()
        self.beta = beta
        self.average = average

    def _forward(self, outputs, targets, **kwargs):
        # TODO needs to inherit num classes from here!
        unique_classes = torch.unique(targets)
        if self.average == 'binary':
            assert len(unique_classes) <= 2, f'Found more than 2 classes. Please set "average" to correct value'
            # not equal because you might get a case where it happens
            # that all the batch is of a certain class only
            # calculate precision and recall
            recall = Recall()(outputs, targets)
            precision = Precision()(outputs, targets)
            f_score = (1+self.beta ** 2) * (precision * recall) / ((self.beta ** 2) * precision + recall)
            if torch.isnan(f_score):
                f_score = 0.0
            return f_score

        elif self.average == 'macro':


            metric = FScore(beta=self.beta, average='binary')  # binary f1score on each item
            f_scores = torch.zeros(len(unique_classes), dtype=torch.float32)
            for i, unique_class in enumerate(unique_classes):
                binary_outputs = torch.where(outputs == unique_class, 1, 0)
                binary_targets = torch.where(targets == unique_class, 1, 0)
                f_scores[i] = metric(binary_outputs, binary_targets)
            return torch.mean(f_scores)
