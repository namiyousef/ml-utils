from colabtools.utils import move_to_device, get_gpu_utilization
from colabtools.config import DEVICE
import torch
import gc
import time

# TODO need to figure out how to do the logging properly
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


def call_model():
    pass


class BaseTrainer:
    def __init__(self, model, optimizer, loss=None, scheduler=None, metrics=None, debug=False):
        # TODO think about how metrics would work for the case of multitask learning

        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.scheduler = scheduler
        self.metrics = metrics

        self.debug = debug

        # TODO do you want the loss packages with or without the model? With huggingface it comes with, e.g. in the return, with normal nn tpyically you treat it
        # as a separate parameter?
        # todo how to package model and loss together?


    def train(self, train_loader, epochs, val_loader=None, verbose=0):
        # design choice: we don't specify any data related parameters here. The choice is left to the user to define them according to their dataset

        for epoch_id in range(epochs):
            self.model.train()
            for batch_id, batch in enumerate(train_loader):

                batch = self.prepare_batch(batch)

                # TODO send batch_metadata to GPU, also think about how to choose which items to send and which not to
                # can probably change this to preprocess gradient, then can have process gradient!
                self.optimizer.zero_grad()


                model_output_dict = self.compute_loss(batch)


                if self.metrics:
                    self.compute_metrics(model_output_dict)


                # TODO metric calculation and storage in metadata
                # TODO metric calculation for gradient??

                self.process_gradient() # what if process gradient requires inputs, targets, metadata, even the loss value? e.g. things that aren't stored in the model, how we will pass that in?

                model_output_dict['loss'].backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                del model_output_dict # metadata, loss (but need to figure out: a) if deleting loss deletes computational graph, b) if metadata needs to be deleted at all?). If a), then need to change design
                gc.collect()
                torch.cuda.empty_cache()

                if self.debug:
                    break

            if val_loader:
                self.test(val_loader)


    def test(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            pass

        if self.debug:
            pass
        pass


    def save(self):
        pass

    def _log(self):
        # default logging behaviour
        pass

    def process_gradient(self):
        return None
        # option for clipping
        #torch.nn.utils.clip_grad_norm_(
        #   parameters=self.model.parameters(), max_norm=MAX_NORM
        #) # one might want to provide this as an input parameter
        pass
    def _call_model(self):
        pass

    # TODO need to think about options where you want to do processing on the gradient!

    def _save(self):
        pass # default save behavioru
    pass

    def prepare_batch(self, batch):
        batch['inputs'] = move_to_device(batch['inputs'])
        batch['targets'] = move_to_device(batch['targets'])

        # what is the purpose of metadata?
        #batch['metadata'] = move_to_device()
        # these are things that you explicitly don't want on the GPU. Everything else should go within inputs/targets
        # the question now becomes, will you even need metadata?

        return batch

    def compute_metrics(self):
        pass

    def compute_loss(self, model_output_dict):
        # TODO advantage of having this as a separate function is that you can automatically time using a timer wrapper
        # TODO need to think about if hugginface will be split or not, this has implications for the warnings / errors
        if self.loss:
            # e.g. standard models
            outputs = self.model(model_output_dict['inputs'])
            loss = self.loss(outputs, model_output_dict['targets'])
        else:
            # TODO what about huggingface + multitask?
            # e.g. HuggingFace models
            loss, outputs = self.model(
                labels=model_output_dict['targets'],
                input_ids=model_output_dict['inputs']['input_ids'],
                attention_mask=model_output_dict['inputs']['attention_mask'],
                return_dict=False
            )

        model_output_dict['outputs'] = outputs

        return model_output_dict



def HuggingFaceTrainer(BaseTrainer):
    def __init__():
        super().__init__()
        pass

def MultiTaskTrainer(BaseTrainer):
    def __init__():
        super().__init__()
        pass


# TODO metric collection specify if at batch or epoch! Also metric should technically be able to use anything from the model
# TODO move_to_device needs to be made much smarter and more automatic


def _call_huggingface_model(model, inputs, targets, **kwargs):
    # TODO need to upgrade for more general case
    output_dict = model(
        labels=targets,
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
    )
    return output_dict

def train(model, optimizer, train_loader, epochs, val_loader=[], verbose=1, metrics=[]):

    for epoch in range(epochs):
        model.train()
        start_epoch_message = f'EPOCH {epoch + 1} STARTED'
        print(start_epoch_message)
        print(f'{"-" * len(start_epoch_message)}')
        start_epoch = time.time()

        start_load = time.time()
        training_loss = 0

        for i, (inputs, targets) in enumerate(train_loader):
            start_train = time.time()
            inputs = move_to_device(inputs, DEVICE)
            targets = move_to_device(targets, DEVICE)

            optimizer.zero_grad()

            # TODO need to refactor this to be flexible for any input/output pair. Output should be a dict, input should be inputs, targets
            # need to think about case where you have over/under defined inputs and targets. Also move_to_device should be smart enough to know which items to tag as GPU or not.
            # by default should move everything to GPU, unless there is an 'ignore' tag! Think about how to do this.
            loss, outputs = model(
                labels=targets,
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                return_dict=False
            )
            gpu_util = 'NO GPU' if DEVICE == 'cpu' else get_gpu_utilization()
            training_loss += loss.item()

            # backward pass

            # torch.nn.utils.clip_grad_norm_(
            #    parameters=model.parameters(), #max_norm=MAX_NORM
            # )

            loss.backward()
            optimizer.step()

            # metrics
            for metric in metrics:
                score = metric(outputs, targets)
                scores['scores'][metric.__class__.__name__].append(score.item())

            confidence = update_confidence(confidence, epoch, inputs, outputs, targets)
            del targets, inputs, loss, outputs
            gc.collect()
            torch.cuda.empty_cache()

            end_train = time.time()

            if verbose > 1:
                print(
                    f'Batch {i + 1} complete. '
                    f'Time taken: load({start_train - start_load:.3g}),'
                    f'train({end_train - start_train:.3g}),'
                    f'total({end_train - start_load:.3g}). '
                    f'GPU util. after train: {gpu_util}. '
                    f'Metrics: {" ".join([f"{metric_name}({score_list[-1]:.3g})" for metric_name, score_list in scores.get("scores", {}).items()])}'
                )
            start_load = time.time()
            break

        for metric in metrics:
            score = scores['scores'][metric.__class__.__name__][:i + 1]
            avg_score = sum(score) / len(score)
            scores['epoch_scores'][metric.__class__.__name__].append(avg_score)
            scores['epoch_batch_ids'][metric.__class__.__name__].append(i)

        print_message = f'Epoch {epoch + 1}/{epochs} complete. ' \
                        f'Time taken: {start_load - start_epoch:.3g}. ' \
                        f'Loss: {training_loss / (i + 1): .3g}. ' \
                        f'Metrics: {" ".join([f"{metric_name}({score_list[-1]:.3g})" for metric_name, score_list in scores.get("epoch_scores", {}).items()])}'

        if verbose:
            print(f'{"-" * len(print_message)}')
            print(print_message)
            print(f'{"-" * len(print_message)}')

        # if epoch % save_freq == 0:
        #    encoded_model_name = encode_model_name(model_name, epoch+1)
        #    save_path = f'models/{encoded_model_name}'
        #    model.save_pretrained(save_path)
        #    print(f'Model saved at epoch {epoch+1} at: {save_path}')
    var, mean = torch.var_mean(confidence, dim=1, unbiased=False)
    metadata[lang] = {'var': var.to('cpu'), 'mean': mean.to('cpu')}

    del var, mean, confidence, model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

def test():
    pass

metadata = {}