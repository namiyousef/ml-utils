from colabtools.utils import move_to_device, get_gpu_utilization
from colabtools.config import DEVICE
import torch
import gc
import time

# TODO need to figure out how to do the logging properly
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()  # create console handler and set level to debug
ch.setLevel(logging.DEBUG)
date_format = '%Y-%m-%d %H:%M:%S'  # create formatter
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s', datefmt=date_format)
ch.setFormatter(formatter)  # add formatter to ch
logger.addHandler(ch)  # add ch to logger


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

    def train(self,
              train_loader,
              epochs,
              val_loader=None,
              verbose=0,
              epoch_start_id=0,
              collect_time_series_every_n_steps=None,
              break_after_n_steps=0 # TODO update metric and loss initialisation dicts to reflect this change
              ):
        """Function for training a model

        :param train_loader: train data loader
        :param epochs: number of epochs to train model for
        :param val_loader: validation data loader, if any
        :param verbose: verbosity level. 0 - only shows warnings and errors, 1 - shows info every epoch, 2 - shows info every batch (if debug set to True, then shows time elapsed and GPU utilisation as well)
        :param epoch_start_id: epoch at which to start training
        :param collect_time_series_every_n_steps: frequency of time series collection
        :param break_after_n_steps: if debug set to True, then will break training/validation loop after n steps
        :return:
        """

        # initialize training parameters and history collection
        logger.info('Initializing training...')
        self.model.to(DEVICE)
        # initialize global axes collection
        self.history['axes'] = dict(epoch_step_ids=[])  # saves the step_id at which epoch history calculated

        # get training parameters
        _num_train_samples, _num_train_steps, _num_train_collect_steps = self._initialize_train_parameters(
            train_loader, epochs, collect_time_series_every_n_steps
        )

        # update history data structure
        self._initialize_loss_history('train', epochs, _num_train_collect_steps, collect_time_series_every_n_steps)

        if self.metrics:

            self._initialize_metrics_store('train', epochs, _num_train_collect_steps, _num_train_samples)

        if val_loader:
            _num_validation_samples, _num_validation_steps, _collect_val_time_series_every_n_steps = self._initialize_val_parameters(
                val_loader,
                collect_time_series_every_n_steps=collect_time_series_every_n_steps,
                num_train_steps=_num_train_steps,
                num_epochs=epochs
            )
            self._initialize_loss_history('validation', epochs, _num_train_collect_steps,
                                          _collect_val_time_series_every_n_steps)


            if self.metrics:
                self._initialize_metrics_store('validation', epochs, _num_train_collect_steps,
                                               _num_validation_samples)
        if self.debug:
            logger.debug(f'Training initialization complete. Metrics:{bool(self.metrics)}. Validation:{bool(val_loader)}. Full history dict: {self.history}')


        steps = 0  # counter to count the number of steps
        for epoch_id in range(epoch_start_id, epochs):
            training_metadata = {}  # to store in-train metrics, e.g. time taken

            if verbose > 0:
                logger.info(f'Starting training for Epoch {epoch_id+1}...')

            self.model.train()

            _start_load_time = time.time()
            for batch_id, batch in enumerate(train_loader):

                batch = self.prepare_batch(batch)

                # can probably change this to preprocess gradient, then can have process gradient!
                self.optimizer.zero_grad()

                _start_train_time = time.time()
                model_output_dict = self.compute_loss(batch)
                _end_train_time = time.time()

                training_metadata['train_time'] = _end_train_time - _start_train_time

                avg_batch_loss = model_output_dict['loss'].item()

                self._update_loss_batch('train', epoch_id, avg_batch_loss, steps, collect_time_series_every_n_steps)


                if self.metrics:
                    _start_compute_metric_time = time.time()
                    metrics_output_dict = self.compute_metrics(model_output_dict)
                    _end_compute_metric_time = time.time()
                    training_metadata['compute_metrics_time'] = _end_compute_metric_time - _start_compute_metric_time

                    self._update_metrics_batch('train', metrics_output_dict, steps, epoch_id, collect_time_series_every_n_steps)



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

            self._update_loss_history_epoch('train', batch_id, epoch_id, collect_time_series_every_n_steps, is_train=True)
            self._update_metric_time_series_history_epoch('train', batch_id, epoch_id,
                                                          collect_time_series_every_n_steps)

            self._log_epoch_complete(verbose, 'train', epoch_id)

            if val_loader:
                logger.info(f'Starting validation for Epoch {epoch_id+1}...')
                self.test(val_loader, epoch_id=epoch_id, collect_time_series_every_n_steps=_collect_val_time_series_every_n_steps, verbose=verbose)
                self._log_epoch_complete(verbose, 'validation', epoch_id)

        return self.model

    def test(self, test_loader, verbose=0, epoch_id=None, collect_time_series_every_n_steps=False):
        _num_test_steps = len(test_loader)
        _num_test_samples = len(test_loader.dataset)
        test_metadata = {}

        if epoch_id is None:
            split = 'test'
            epoch_id = 0
            self._initialize_loss_history(split, 1, _num_test_steps, collect_time_series_every_n_steps)
            if self.metrics:
                self._initialize_metrics_store(split, 1, _num_test_steps, _num_test_samples)
        else:
            split = 'validation'

        steps = epoch_id * _num_test_steps


        self.model.eval()
        with torch.no_grad():
            _start_load_time = time.time()
            for batch_id, batch in enumerate(test_loader):

                batch = self.prepare_batch(batch)
                _start_val_time = time.time()
                model_output_dict = self.compute_loss(batch)
                _end_val_time = time.time()
                test_metadata['train_time'] = _end_val_time - _start_val_time


                avg_batch_loss = model_output_dict['loss'].item()
                self._update_loss_batch(split, epoch_id, avg_batch_loss, steps, collect_time_series_every_n_steps)


                if self.metrics:
                    _start_compute_metric_time = time.time()
                    metrics_output_dict = self.compute_metrics(model_output_dict)
                    _end_compute_metric_time = time.time()
                    test_metadata['compute_metrics_time'] = _end_compute_metric_time - _start_compute_metric_time

                    self._update_metrics_batch(split, metrics_output_dict, steps, epoch_id, collect_time_series_every_n_steps)


                del model_output_dict
                gc.collect()
                torch.cuda.empty_cache()

                self._log_batch_complete(verbose, batch_id, avg_batch_loss, test_metadata)

                steps += 1
                if self.debug:
                    break

        self._update_loss_history_epoch(split, batch_id, epoch_id, collect_time_series_every_n_steps)
        self._update_metric_time_series_history_epoch(split, batch_id, epoch_id, collect_time_series_every_n_steps)

        # TODO need to think about stratification

    def _initialize_loss_history(self, split, num_epochs, num_steps, collect_time_series_every_n_steps):
        self.history['loss'][split] = self._create_empty_time_series_dict(
            num_epochs=num_epochs,
            num_steps=num_steps
        )
        if self.debug:
            logger.debug(f'Initialized <{split}> loss history')
        if collect_time_series_every_n_steps:
            self.history['axes'][split] = dict(step_id=[])
            if self.debug:
                logger.debug(f'Initialized <{split}> loss history with step_id')

    def _initialize_metrics_store(self, split, num_epochs, num_steps, num_instances):
        self._initialize_time_series_store(split,
                                           num_epochs=num_epochs,
                                           num_steps=num_steps
                                           )
        self._initialize_instance_metrics_store(split,
                                                num_instances=num_instances,
                                                num_features=num_epochs
                                                )
        if self.debug:
            logger.debug(f'Initialized metrics score for <{split}>')


    def _update_time_series_history(self, time_series_output_dict, split, collect_time_series_every_n_steps, step,
                                    epoch_id):
        if time_series_output_dict:
            for metric_id, metric_score in time_series_output_dict.items():
                self.history['metrics']['time_series'][metric_id][split]['epoch'][epoch_id] += metric_score
                if collect_time_series_every_n_steps and not step % collect_time_series_every_n_steps:
                    step_id = step // collect_time_series_every_n_steps
                    self.history['metrics']['time_series'][metric_id][split]['step'][step_id] = metric_score

    def _update_metric_time_series_history_epoch(self, split, batch_id, epoch_id, collect_time_series_every_n_steps):
        for metric_id, metric_dict in self.history['metrics']['time_series'].items():
            metric_dict[split]['epoch'][epoch_id] /= (batch_id + 1)

        # TODO need to use to('cpu') and detach()!
        # TODO need to remove gradient computation for metrics

    def _update_instance_metrics_history(self, instance_merics_output_dict, split, epoch_id,
                                         collect_instance_metrics_every_n_steps=None, data_loader=None,
                                         include_current=True):
        if collect_instance_metrics_every_n_steps:
            raise NotImplementedError('No support for instance metric collection at arbitrary steps')
        for metric_id, (instance_ids, instance_scores) in instance_merics_output_dict.items():
            self.history['metrics']['instance_metrics'][metric_id][split][instance_ids, epoch_id] = instance_scores
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

    def _update_loss_batch(self, split, epoch_id, avg_batch_loss, steps, collect_time_series_every_n_steps):

        self.history['loss'][split]['epoch'][epoch_id] += avg_batch_loss

        if collect_time_series_every_n_steps and not steps % collect_time_series_every_n_steps:
            step_id = steps // collect_time_series_every_n_steps
            self.history['loss'][split]['step'][
                step_id] = avg_batch_loss  # TODO see if failure occurs because of diff devices here
            self.history['axes'][split]['step_id'].append(steps)

    def _update_metrics_batch(self, split, metrics_output_dict, steps, epoch_id, collect_time_series_every_n_steps):
        time_series_output_dict = metrics_output_dict.get('time_series', None)
        instance_metrics_output_dict = metrics_output_dict.get('instance_metrics', None)
        self._update_time_series_history(
            time_series_output_dict, split, collect_time_series_every_n_steps, steps, epoch_id)
        self._update_instance_metrics_history(
            instance_metrics_output_dict, split, epoch_id)
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
        batch['inputs'] = move_to_device(batch['inputs'], DEVICE)
        batch['targets'] = move_to_device(batch['targets'], DEVICE)

        # what is the purpose of metadata? # TODO send batch_metadata to GPU, also think about how to choose which items to send and which not to
        # these are things that you explicitly don't want on the GPU. Everything else should go within inputs/targets
        # the question now becomes, will you even need metadata?

        return batch
    def _initialize_loss_history(self, split, num_epochs, num_steps, collect_time_series_every_n_steps):
        self.history['loss'][split] = self._create_empty_time_series_dict(
            num_epochs=num_epochs,
            num_steps=num_steps
        )
        if self.debug:
            logger.debug(f'Initialized <{split}> loss history')
        if collect_time_series_every_n_steps:
            self.history['axes'][split] = dict(step_id=[])
            if self.debug:
                logger.debug(f'Initialized <{split}> loss history with step_id')

    def _update_loss_history_epoch(self, split, batch_id, epoch_id, collect_time_series_every_n_steps,
                                          is_train=False):
        self.history['loss'][split]['epoch'][epoch_id] /= (batch_id + 1)
        if collect_time_series_every_n_steps:
            self.history['loss'][split]['epoch_step_ids'][epoch_id] = (epoch_id + 1) * (batch_id + 1) - 1
            if is_train:
                self.history['axes']['epoch_step_ids'].append((epoch_id + 1) * (batch_id + 1) - 1)

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