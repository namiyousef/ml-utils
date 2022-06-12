from colabtools.utils import move_to_device, get_gpu_utilization
from colabtools.config import DEVICE
import torch
import gc
import time

def call_model():
    pass

class Trainer():
    def _log(self):
        # default logging behaviour
        pass
    def _process_gradient(self):
        # option for clipping
        pass
    def _call_model(self):
        pass

    # TODO need to think about options where you want to do processing on the gradient!

    def _save(self):
        pass # default save behavioru
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