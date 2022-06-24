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

# in terms of storage, we want the same plotting capability
# so for a multi task problem, we will require the metrics we given as a dict with lists for each task, but ultimately in terms of storage they will be the same. The only difference will be that
# they will have the task name in the parameters
# the way the metrics are stored will also be affected by the type of reduction used. We'll mandate that metrics from this library must use the same "reduction" term as in PyTorch

class BaseTrainer:

    def __init__(self, model, optimizer, loss=None, scheduler=None, metrics=None, debug=False):

        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.scheduler = scheduler
        self.metrics = metrics
        self.debug = debug

        self.history = dict(
            loss=dict(
                params=self._get_metric_params(self.loss) if self.loss else dict(name='default_model_loss'),
            )
        )
        if self.metrics:
            # for multi task setting, initialize metrics dict should return a flattened version of the metrics. The task name should be in the params
            # compute metrics will then have to also be modified for the multi task case to consider this data structure change
            self.metric_map = self.get_metric_map()
            metrics_dict = self._initialize_metrics_dict()
            self.history['metrics'] = metrics_dict


    # verbose = 0, only shows warnings and errors
    # verbose = 1, shows info at epoch level
    # verbose = 2, shows info at batch level
    # debug = True, shows the times as well. Note: need to think of replacing the break clause for the debug!
    # design choice: we don't specify any data related parameters here. The choice is left to the user to define them according to their dataset
    # TODO need to think about stratification

    def _update_time_series_history(self, time_series_output_dict, split, collect_time_series_every_n_steps, step,
                                    epoch_id):
        if time_series_output_dict:
            for metric_id, metric_score in time_series_output_dict.items():
                self.history['metrics']['time_series'][metric_id][split]['epoch'][epoch_id] += metric_score
                if collect_time_series_every_n_steps and not step % collect_time_series_every_n_steps:
                    step_id = step // collect_time_series_every_n_steps
                    self.history['metrics']['time_series'][metric_id][split]['step'][step_id] = metric_score
        
    # TODO need to use to('cpu') and detach()!
    # TODO need to remove gradient computation for metrics
    def _update_instance_metrics_history(self, instance_merics_output_dict, split, epoch_id, collect_instance_metrics_every_n_steps=None, data_loader=None, include_current=True):
        if collect_instance_metrics_every_n_steps:
            raise NotImplementedError('No support for instance metric collection at arbitrary steps')
        for metric_id, (instance_ids, instance_scores) in instance_merics_output_dict.items():
            self.history['metrics']['instance_metrics'][metric_id][split][instance_ids, epoch_id] = instance_scores

    def _update_time_series_history_epoch(self, batch_id, epoch_id, collect_time_series_every_n_steps):
        self.history['loss']['train']['epoch'][epoch_id] /= (batch_id + 1)
        if collect_time_series_every_n_steps:
            self.history['loss']['train']['epoch_step_ids'][epoch_id] = (epoch_id+1) * (batch_id+1) - 1

    def _log_batch_complete(self, verbose, batch_id, avg_batch_loss, metadata):
        if verbose > 1:
            logger.info({"msg": f"Batch {batch_id + 1} complete", "loss": avg_batch_loss})
            if self.debug:
                logger.debug(
                    {"msg": f"Batch {batch_id + 1} metadata", **{k: f'{v:.3g}' for k, v in metadata.items()}}
                )

    def _log_epoch_complete(self, verbose, split, epoch_id):
        if verbose > 0:
            logger.info({"msg": f"Epoch {epoch_id + 1} of split <{split}> complete"})
            if self.debug:
                logger.debug(
                    {"msg": f"Epoch {epoch_id + 1} metadata", }
                )
    def train(self,
              train_loader,
              epochs,
              val_loader=None,
              verbose=0,
              epoch_start_id=0,
              collect_time_series_every_n_steps=None,
              ):
        # TODO need to add method for starting from a certain dict! and printing should follow from that
        _num_train_samples, _num_train_steps, _num_train_collect_steps = self._initialize_train_parameters(
            train_loader, epochs, collect_time_series_every_n_steps
        )

        # collect_time_series_every_n_steps only applies for time_series models, it is used to generate the empty_history_dict
        self.history['loss']['train'] = self._create_empty_time_series_dict(
            num_epochs=epochs,
            num_steps=_num_train_collect_steps
        )
        self.history['axes'] = dict(
            epoch_step_id=[], # can calculate this in advance...
        )
        if collect_time_series_every_n_steps:
            self.history['axes']['train'] = dict(
                step_id=[]
            )


        if self.metrics:
            self._initialize_time_series_store('train',
                                               num_epochs=epochs,
                                               num_steps=_num_train_collect_steps
                                               )
            self._initialize_instance_metrics_store('train',
                                                    num_instances=_num_train_samples,
                                                    num_features=epochs
                                                    )

        if val_loader:
            _num_validation_samples, _num_validation_steps, _collect_val_time_series_every_n_steps = self._initialize_val_parameters(
                val_loader,
                collect_time_series_every_n_steps=collect_time_series_every_n_steps,
                num_train_steps=_num_train_steps,
                num_epochs=epochs
            )

            self.history['loss']['validation'] = self._create_empty_time_series_dict(
                num_epochs=epochs,
                num_steps=_num_train_collect_steps
            )
            if collect_time_series_every_n_steps:
                self.history['axes']['validation'] = dict(
                        # map
                        step_id=[],  # the step at which each epoch is saved
                    )


            if self.metrics:
                self._initialize_time_series_store('validation',
                                                   num_epochs=epochs,
                                                   num_steps=_num_train_collect_steps
                                                   )
                self._initialize_instance_metrics_store('validation',
                                                        num_instances=_num_validation_samples,
                                                        num_features=epochs
                                                        )
        
        steps = 0
        for epoch_id in range(epoch_start_id, epochs):
            training_metadata = {}

            if verbose > 0:
                logger.info(f'Starting training for Epoch {epoch_id+1}...')

            self.model.train()
            _start_load_time = time.time()
            for batch_id, batch in enumerate(train_loader):
                batch = self.prepare_batch(batch)

                # TODO send batch_metadata to GPU, also think about how to choose which items to send and which not to
                # can probably change this to preprocess gradient, then can have process gradient!
                self.optimizer.zero_grad()

                _start_train_time = time.time()
                model_output_dict = self.compute_loss(batch)
                _end_train_time = time.time()

                training_metadata['train_time'] = _end_train_time - _start_train_time

                avg_batch_loss = model_output_dict['loss'].item()

                self.history['loss']['train']['epoch'][epoch_id] += avg_batch_loss

                if not steps % collect_time_series_every_n_steps:
                    step_id = steps // collect_time_series_every_n_steps
                    logger.info(f'Logging step result at step_id={step_id}')
                    self.history['loss']['train']['step'][step_id] = avg_batch_loss # TODO see if failure occurs because of diff devices here


                if self.metrics:
                    _start_compute_metric_time = time.time()
                    metrics_output_dict = self.compute_metrics(model_output_dict)
                    time_series_output_dict = metrics_output_dict.get('time_series', None)
                    instance_metrics_output_dict = metrics_output_dict.get('instance_metrics', None)
                    self._update_time_series_history(time_series_output_dict, 'train', collect_time_series_every_n_steps, batch_id, epoch_id)
                    self._update_instance_metrics_history(instance_metrics_output_dict, 'train', epoch_id)


                    _end_compute_metric_time = time.time()
                    training_metadata['compute_metrics_time'] = _end_compute_metric_time - _start_compute_metric_time

                # TODO metric calculation and storage in metadata
                # TODO metric calculation for gradient??
                _start_optimize_time = time.time()
                self.process_gradient() # what if process gradient requires inputs, targets, metadata, even the loss value? e.g. things that aren't stored in the model, how we will pass that in?



                model_output_dict['loss'].backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                del model_output_dict # metadata, loss (but need to figure out: a) if deleting loss deletes computational graph, b) if metadata needs to be deleted at all?). If a), then need to change design
                gc.collect()
                torch.cuda.empty_cache()
                _end_optimize_time = time.time()
                training_metadata['optimize_time'] = _end_optimize_time - _start_optimize_time
                training_metadata['total_batch_time'] = _end_optimize_time - _start_load_time
                _start_load_time = time.time()

                self._log_batch_complete(verbose, batch_id, avg_batch_loss, training_metadata)

                steps += 1
                if self.debug:
                    break


            self._update_time_series_history_epoch(batch_id, epoch_id, collect_time_series_every_n_steps)


            self._log_epoch_complete(verbose, 'train', epoch_id)

            if val_loader:
                logger.info(f'Starting validation for Epoch {epoch_id+1}...')
                self.test(val_loader, epoch_id=epoch_id, collect_time_series_every_n_steps=_collect_val_time_series_every_n_steps, verbose=verbose)
                self._log_epoch_complete(verbose, 'validation', epoch_id)

        return self.model

    def test(self, test_loader, verbose=0, epoch_id=None, collect_time_series_every_n_steps=False):
        #steps = len(self.history['axes']['validation']['step_id']) if collect_time_series_every_n_steps else 0
        steps = epoch_id * len(test_loader)
        test_metadata = {}
        if epoch_id is None:
            # TODO need to check this
            self.history['loss']['test'] = self._create_empty_time_series_dict(collect_time_series_every_n_steps=collect_time_series_every_n_steps)
            if self.metrics:
                for metric_id, metric in enumerate(self.metrics):
                    self.history['metrics'][metric_id]['test'] = self._create_empty_time_series_dict(
                        collect_time_series_every_n_steps=collect_time_series_every_n_steps)
            epoch_id = 0



        # todo needs to differentiate between validation and test!
        self.model.eval()
        _num_test_samples = len(test_loader.dataset)
        with torch.no_grad():
            _start_load_time = time.time()
            for batch_id, batch in enumerate(test_loader):

                batch = self.prepare_batch(batch)
                _start_val_time = time.time()
                model_output_dict = self.compute_loss(batch)
                _end_val_time = time.time()
                test_metadata['train_time'] = _end_val_time - _start_val_time


                avg_batch_loss = model_output_dict['loss'].item()
                self.history['loss']['validation']['epoch'][epoch_id] += avg_batch_loss
                if not steps % collect_time_series_every_n_steps:
                    step_id = steps // collect_time_series_every_n_steps
                    print(steps, collect_time_series_every_n_steps, step_id)
                    self.history['loss']['validation']['step'][step_id] = avg_batch_loss
                    self.history['axes']['validation']['step_id'].append(step_id)
                    print(avg_batch_loss, len(self.history['axes']['validation']['step_id']))
                    print(self.history['loss']['validation']['step'])



                if self.metrics:
                    _start_compute_metric_time = time.time()
                    metrics_output_dict = self.compute_metrics(model_output_dict)
                    time_series_output_dict = metrics_output_dict.get('time_series', None)
                    instance_metrics_output_dict = metrics_output_dict.get('instance_metrics', None)
                    self._update_time_series_history(time_series_output_dict, 'validation',
                                                     collect_time_series_every_n_steps, batch_id, epoch_id)
                    self._update_instance_metrics_history(instance_metrics_output_dict, 'validation', epoch_id)

                    _end_compute_metric_time = time.time()
                    test_metadata['compute_metrics_time'] = _end_compute_metric_time - _start_compute_metric_time



                    self.compute_metrics(model_output_dict)

                del model_output_dict
                gc.collect()
                torch.cuda.empty_cache()
                steps += 1
                if self.debug:
                    break

    def get_metric_map(self):
        # TODO this has to change for multi task, based on the metrics!
        metric_map = {i: metric for i, metric in enumerate(self.metrics)}
        return metric_map

    def _initialize_metrics_dict(self):
        # TODO need to change according to the new structure, now based on metric_map
        time_series_params, instance_metrics_params = self._get_metric_components(self.metric_map)
        metrics_dict = {}
        if time_series_params:
            metrics_dict['time_series'] = time_series_params
        if instance_metrics_params:
            metrics_dict['instance_metrics'] = instance_metrics_params
        return metrics_dict

    def save(self):
        pass

    def process_gradient(self):
        # TODO need to think about options where you want to do processing on the gradient!

        return None
        # option for clipping
        #torch.nn.utils.clip_grad_norm_(
        #   parameters=self.model.parameters(), max_norm=MAX_NORM
        #) # one might want to provide this as an input parameter
        pass

    def prepare_batch(self, batch):
        batch['inputs'] = move_to_device(batch['inputs'])
        batch['targets'] = move_to_device(batch['targets'])

        # what is the purpose of metadata?
        #batch['metadata'] = move_to_device()
        # these are things that you explicitly don't want on the GPU. Everything else should go within inputs/targets
        # the question now becomes, will you even need metadata?

        return batch

    def compute_metrics(self, model_output_dict):
        # TODO may need to re-think data structure. Need to think of an intelligent way of doing this automatically....
        # it is difficult for metrics that we don't create ourselves. Instead need wrappers for them, but ensuring that:
        # the dict still stays of the same metric structure (the name will have to inherit/replace the pytorch name)
        with torch.no_grad():
            time_series_dict = self.history['metrics'].get('time_series', None)
            instance_metrics_dict = self.history['metrics'].get('instance_metrics', None)
            metrics_output_dict = {}
            if time_series_dict:
                time_series_score_dict = {}
                for metric_id in time_series_dict.keys():
                    metric = self.metric_map[metric_id]
                    score = metric(model_output_dict)
                    time_series_score_dict[metric_id] = score
                metrics_output_dict['time_series'] = time_series_score_dict

            if instance_metrics_dict:
                instance_metrics_score_dict = {}
                for metric_id in instance_metrics_dict.keys():
                    metric = self.metric_map[metric_id]
                    instance_ids, instance_scores = metric(model_output_dict)
                    instance_metrics_score_dict[metric_id] = (instance_ids, instance_scores)
                metrics_output_dict['instance_metrics'] = instance_metrics_score_dict

        return metrics_output_dict

    def compute_loss(self, batch):
        # TODO advantage of having this as a separate function is that you can automatically time using a timer wrapper
        # TODO need to think about if hugginface will be split or not, this has implications for the warnings / errors
        if self.loss:
            # e.g. standard models
            outputs = self.model(batch['inputs'])
            loss = self.loss(outputs, batch['targets'])
        else:
            # TODO what about huggingface + multitask?
            # e.g. HuggingFace models
            loss, outputs = self.model(
                labels=batch['targets'],
                input_ids=batch['inputs']['input_ids'],
                attention_mask=batch['inputs']['attention_mask'],
                return_dict=False
            )

        batch['loss'] = loss
        batch['outputs'] = outputs

        return batch


    def _get_metric_components(self, metric_map):
        time_series = {}
        instance_metrics = {}
        for metric_id, metric in metric_map.items():
            metric_params = self._get_metric_params(metric)
            metric_params_dict = dict(params=metric_params)
            if metric_params['reduction'] == 'none':
                instance_metrics[metric_id] = metric_params_dict
            else:
                time_series[metric_id] = metric_params_dict
        return time_series, instance_metrics


    # cases: loss always you ignore reduction, so maybe can have an is_loss parameter
    # cases: metric with no reduction, collect_time_series_every_n_steps=True
    # cases: metric with reduction, collect_batch-dats=True
    # and also for collect batch_data False
    # for instance metrics we have: shape(num_d_points, epochs), shape(num_d_points, batch) (this means you would actually have to re-run through the model!!!!)
    # for normal metrics we have: shape(epochs, 1) shape(batch, 1)

    # maybe should make a distinction: timeseries vs. instance_metrics. Time series expected to be of the form: batch/epoch num against aggregate score
    # instance_metric should be of the form: instance_id vs. batch_id/epoch_id with each value being a score. We typically need these reduced at the end (so maybe need metric post-process)
    def _create_empty_time_series_dict(self, num_epochs, num_steps=None):
        history_dict = {
            'epoch': torch.zeros(num_epochs)
        }
        if num_steps:
            # TODO need a parameter for this, e.g. num_steps
            history_dict['step'] = torch.zeros(num_steps)
            # TODO consider removing this and adding it as a metadata for the timeseries; also will need one for the actual steps (e.g. the true batch_ids!)
            history_dict['epoch_step_ids'] = torch.zeros(num_epochs)

        return history_dict

    def _get_metric_params(self, metric):
        params = {
            'name': metric.__class__.__name__,
            **{param: value for param, value in metric.__dict__.items() if not param.startswith('_')}
        }
        return params

    def _initialize_history_dict(self, split, num_steps=None):
        self.history['loss'][split] = self._create_empty_time_series_dict(collect_time_series_every_n_steps=collect_time_series_every_n_steps)
        if self.metrics:
            for metric_id, metric in enumerate(self.metrics):
                self.history['metrics'][metric_id][split] = self._create_empty_time_series_dict(
                    collect_time_series_every_n_steps=collect_time_series_every_n_steps)
        # TODO logging
        # TODO for multitask all we will do is add a wrapper that uses this, and then flatten it



    # TODO need to re-think how the data structure is working, at the moment it is very confusing. Debugging this will be a nightmare

    # TODO how will it work in the case of multi task models?
    def _initialize_time_series_store(self, split, num_epochs, num_steps=None):
        metrics_dict = self.history['metrics'].get('time_series', None)
        if metrics_dict:
            for metric_id, metric_dict in metrics_dict.items():
                self.history['metrics']['time_series'][metric_id][split] = self._create_empty_time_series_dict(
                    num_epochs=num_epochs,
                    num_steps=num_steps
                )

    def _initialize_instance_metrics_store(self, split, num_instances, num_features):
        # TODO should think about a compute by batch feature, e.g. every batch (or num_steps!) you compute the metrics for all of them (including/excluding) the instances within that same batch. At the moment though, not implemented
        # need to think about post ingestion evaluation
        metrics_dict = self.history['metrics'].get('instance_metrics', None)
        if metrics_dict:
            for metric_id, metric_dict in metrics_dict.items():
                # see if metrics_dict acts as a point to this?
                metrics_dict[metric_id][split] = torch.zeros((num_instances, num_features), dtype=torch.float32)


    def _initialize_train_parameters(self, train_loader, num_epochs, collect_time_series_every_n_steps):
        _num_train_samples = len(train_loader.dataset)
        _num_train_steps = len(train_loader) * num_epochs
        _num_train_collect_steps = _num_train_steps // collect_time_series_every_n_steps if collect_time_series_every_n_steps else None
        return _num_train_samples, _num_train_steps, _num_train_collect_steps

    def _initialize_val_parameters(self, val_loader, collect_time_series_every_n_steps, num_train_steps, num_epochs):
        _num_validation_samples = len(val_loader.dataset)
        _num_validation_steps = len(val_loader) * num_epochs
        if collect_time_series_every_n_steps and num_train_steps:
            # update collect time series to validation value
            val_to_train_ratio = _num_validation_steps / num_train_steps
            _updated_n_steps = int(collect_time_series_every_n_steps * val_to_train_ratio)
            collect_time_series_every_n_steps = max(1, _updated_n_steps)
        return _num_validation_samples, _num_validation_steps, collect_time_series_every_n_steps
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