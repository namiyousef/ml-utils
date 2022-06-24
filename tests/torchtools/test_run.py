# TODO consider changing the name of this
from mlutils.torchtools.run import BaseTrainer
from tests.torchtools.utils import LinearModel, SimpleDataset, ModelMock, SchedulerMock, OptimizerMock, LossMock, TorchMetricMock, prepare_torch_metric
import unittest
import torch
import numpy as np
import random
from torch.utils.data import DataLoader

from mlutils.torchtools.testing import assert_tensor_objects_equal

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

# TODO figure out pip install for testing?
get_public_params = lambda object: {param: val for param, val in object.__dict__.items() if not param.startswith('_')}

get_uncounted_metric_params = lambda metric: dict(
    name=metric.__class__.__name__,
    **get_public_params(metric)
)
get_counted_metric_params = lambda metric, metric_id: dict(
    metric_id=metric_id,
    **get_uncounted_metric_params(metric)
)

class TestHuggingFaceTrainer(unittest.TestCase):
    pass

class TestBaseTrainer(unittest.TestCase):

    def setUp(self) -> None:
        seed = 0
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.n_points = 1000
        self.X = torch.linspace(0, 1, self.n_points, dtype=torch.float32).reshape(-1, 1)
        self.weights = torch.tensor([2], dtype=torch.float32)
        self.y = self.X @ self.weights
        self.dataset = SimpleDataset(self.X, self.y)
        self.model = LinearModel(self.X.shape[-1], 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.02)
        self.scheduler = None
        self.loss = torch.nn.MSELoss()
        self.metrics = None


        self.model_mock = ModelMock()
        self.loss_mock = LossMock()
        self.optimizer_mock = OptimizerMock()
        self.scheduler_mock = SchedulerMock()

    def test_initialize_metrics_dict(self):
        logger.info('Test 1 - test no metrics: output should simply contain loss and associated params')
        trainer = BaseTrainer(
            model=self.model,
            optimizer=self.optimizer,
            loss=self.loss,
            scheduler=self.scheduler,
            metrics=self.metrics
        )
        output_history_dict = trainer.history
        expected_history_dict = dict(
            loss=dict(params=get_uncounted_metric_params(self.loss)),
        )
        assert output_history_dict == expected_history_dict


        logger.info('Test 2 - test time series metric: output should contain extra metrics dict with time series')
        metrics = [torch.nn.L1Loss()]
        trainer = BaseTrainer(
            model=self.model,
            optimizer=self.optimizer,
            loss=self.loss,
            scheduler=self.scheduler,
            metrics=metrics
        )
        output_history_dict = trainer.history
        expected_history_dict = dict(
            loss=dict(params=get_uncounted_metric_params(self.loss)),
            metrics=dict(
                time_series={
                    i: dict(params=get_uncounted_metric_params(metric)) for i, metric in enumerate(metrics)
                },
            ),
        )
        assert output_history_dict == expected_history_dict

        logger.info('Test 3 - test instance metric')
        metrics = [torch.nn.L1Loss(reduction='none')]

        trainer = BaseTrainer(
            model=self.model,
            optimizer=self.optimizer,
            loss=self.loss,
            scheduler=self.scheduler,
            metrics=metrics
        )
        output_history_dict = trainer.history
        expected_history_dict = dict(
            loss=dict(params=get_uncounted_metric_params(self.loss)),
            metrics=dict(
                instance_metrics={
                    i: dict(params=get_uncounted_metric_params(metric)) for i, metric in enumerate(metrics)
                }
            ))

        assert output_history_dict == expected_history_dict

        logger.info('Test 4 - test both types of metrics')

        metrics = [
            torch.nn.L1Loss(),
            torch.nn.L1Loss(reduction='none'),
            torch.nn.L1Loss(reduction='none'),
            torch.nn.L1Loss(),
        ]
        trainer = BaseTrainer(
            model=self.model,
            optimizer=self.optimizer,
            loss=self.loss,
            scheduler=self.scheduler,
            metrics=metrics
        )
        output_history_dict = trainer.history
        expected_history_dict = dict(
            loss=dict(params=get_uncounted_metric_params(self.loss)),
            metrics=dict(
                instance_metrics={
                    1: dict(params=get_uncounted_metric_params(metrics[1])),
                    2: dict(params=get_uncounted_metric_params(metrics[2]))
                },
                time_series={
                    0: dict(params=get_uncounted_metric_params(metrics[0])),
                    3: dict(params=get_uncounted_metric_params(metrics[3]))
                }
            ),
        )

        assert output_history_dict == expected_history_dict

    def test_initialize_train_parameters(self):
        logger.info('Test 1 - test initialize train parameters for collect_time_series_every_n_steps')
        X = torch.ones((100, 2))
        y = torch.ones(100)
        train_set = SimpleDataset(X, y)
        train_loader = DataLoader(train_set, batch_size=8)
        collect_time_series_every_n_steps = 5
        model_mock = ModelMock()
        optimizer_mock = OptimizerMock()
        scheduler_mock = SchedulerMock()
        loss_mock = LossMock()  # think about this for huggingface vs non-hugging face
        trainer = BaseTrainer(
            model_mock, optimizer_mock, loss_mock, scheduler_mock
        )
        _num_train_samples, _num_train_steps, _num_train_collect_steps = trainer._initialize_train_parameters(train_loader, collect_time_series_every_n_steps=collect_time_series_every_n_steps)
        expected_train_samples = len(train_set)
        expected_train_steps = len(train_loader)
        expected_train_collect_steps = 2

        assert expected_train_samples == _num_train_samples
        assert expected_train_steps == _num_train_steps
        assert _num_train_collect_steps == expected_train_collect_steps

        logger.info('Test 2 - test initialize train parameters for no collect_time_series_every_n_steps')

        trainer = BaseTrainer(
            model_mock, optimizer_mock, loss_mock, scheduler_mock
        )
        _num_train_samples, _num_train_steps, _num_train_collect_steps = trainer._initialize_train_parameters(
            train_loader, collect_time_series_every_n_steps=None)
        expected_train_samples = len(train_set)
        expected_train_steps = len(train_loader)
        expected_train_collect_steps = None

        assert expected_train_samples == _num_train_samples
        assert expected_train_steps == _num_train_steps
        assert _num_train_collect_steps == expected_train_collect_steps

        logger.info('Test 3 - test initialize train parameters perfect split')

        trainer = BaseTrainer(
            model_mock, optimizer_mock, loss_mock, scheduler_mock
        )
        _num_train_samples, _num_train_steps, _num_train_collect_steps = trainer._initialize_train_parameters(
            train_loader, collect_time_series_every_n_steps=1)
        expected_train_samples = len(train_set)
        expected_train_steps = len(train_loader)
        expected_train_collect_steps = len(train_loader)

        assert expected_train_samples == _num_train_samples
        assert expected_train_steps == _num_train_steps
        assert _num_train_collect_steps == expected_train_collect_steps

        logger.info('Test 4 - test initialize train parameters collect only once')

        trainer = BaseTrainer(
            model_mock, optimizer_mock, loss_mock, scheduler_mock
        )
        _num_train_samples, _num_train_steps, _num_train_collect_steps = trainer._initialize_train_parameters(
            train_loader, collect_time_series_every_n_steps=len(train_loader))
        expected_train_samples = len(train_set)
        expected_train_steps = len(train_loader)
        expected_train_collect_steps = 1

        assert expected_train_samples == _num_train_samples
        assert expected_train_steps == _num_train_steps
        assert _num_train_collect_steps == expected_train_collect_steps

    def test_create_empty_time_series_dict(self):
        logger.info('Test 1 - test dict created correctly for loss without steps')
        trainer = BaseTrainer(
            model=self.model,
            optimizer=self.optimizer,
            loss=self.loss,
            scheduler=self.scheduler,
            metrics=None
        )
        num_epochs = 5
        output_loss_metric_dict = trainer._create_empty_time_series_dict(
            num_epochs=num_epochs,
            num_steps=None
        )
        expected_loss_metric_dict = dict(
                epoch=torch.zeros(num_epochs)
        )
        assert_tensor_objects_equal(output_loss_metric_dict, expected_loss_metric_dict)

        logger.info('Test 2 - test dict created correctly for loss with steps')

        trainer = BaseTrainer(
            model=self.model,
            optimizer=self.optimizer,
            loss=self.loss,
            scheduler=self.scheduler,
            metrics=None
        )

        num_epochs = 5
        num_steps = 1
        output_loss_metric_dict = trainer._create_empty_time_series_dict(
            num_epochs=num_epochs,
            num_steps=num_steps
        )
        expected_loss_metric_dict = dict(
            epoch=torch.zeros(num_epochs),
            steps=torch.zeros(num_steps),
            epoch_step_ids=torch.zeros(num_epochs)
        )
        assert_tensor_objects_equal(output_loss_metric_dict, expected_loss_metric_dict)


    def test_initialize_time_series_store(self):
        metrics = [torch.nn.L1Loss(reduction='none'), torch.nn.L1Loss()]

        logger.info('Test 1 - test time series store')
        trainer = BaseTrainer(
            self.model_mock, self.optimizer_mock, self.loss_mock, self.scheduler_mock, metrics=metrics
        )

        num_epochs = 5
        split = 'train'
        num_steps = None
        trainer._initialize_time_series_store(split, num_epochs, num_steps)
        output_metric_history = trainer.history['metrics']['time_series']

        metric_id = 1
        expected_metric_history = {
            metric_id: dict(
                params=get_uncounted_metric_params(metrics[metric_id]),
                train=dict(
                    epoch=torch.zeros(num_epochs)
                )
            )
        }

        assert output_metric_history[metric_id]['params'] == expected_metric_history[metric_id]['params']
        assert_tensor_objects_equal(
            output_metric_history[metric_id][split],
            expected_metric_history[metric_id][split]
        )

    def test_initialize_instance_metrics_store(self):
        metrics = [torch.nn.L1Loss(), torch.nn.L1Loss(reduction='none')]

        logger.info('Test 1 - test time series store')
        trainer = BaseTrainer(
            self.model_mock, self.optimizer_mock, self.loss_mock, self.scheduler_mock, metrics=metrics
        )

        num_epochs = 5
        split = 'train'
        num_instances = 10
        trainer._initialize_instance_metrics_store(split, num_instances, num_epochs)
        output_metric_history = trainer.history['metrics']['instance_metrics']

        metric_id = 1
        expected_metric_history = {
            metric_id: dict(
                params=get_uncounted_metric_params(metrics[metric_id]),
                train=torch.zeros((num_instances, num_epochs))
            )
        }

        assert output_metric_history[metric_id]['params'] == expected_metric_history[metric_id]['params']
        assert_tensor_objects_equal(
            output_metric_history[metric_id][split],
            expected_metric_history[metric_id][split]
        )

    def test_initialize_val_parameters(self):
        # TODO need to do edge cases, e.g. just perfectly divisible
        trainer = BaseTrainer(
            self.model_mock, self.optimizer_mock, self.loss_mock, self.scheduler_mock
        )
        X = torch.ones((100, 2))
        y = torch.ones(100)
        val_set = SimpleDataset(X, y)
        val_loader = DataLoader(val_set, batch_size=8)

        logger.info('Test 1 - val train params no custom collection')
        collect_time_series_every_n_steps = None

        _num_val_samples, _num_val_steps, _num_val_collect_steps = trainer._initialize_val_parameters(
            val_loader, collect_time_series_every_n_steps=collect_time_series_every_n_steps, num_train_steps=None

        )

        expected_val_samples = len(val_set)
        expected_val_steps = len(val_loader)
        expected_val_collect_steps = None

        assert _num_val_samples == expected_val_samples
        assert _num_val_steps == expected_val_steps
        assert _num_val_collect_steps == expected_val_collect_steps

        logger.info('Test 2 - val train params scaling: edge case where collection items in train exceeds num steps in val')
        num_train_steps = 200
        collect_time_series_every_n_steps = 10 # this is for train

        _num_val_samples, _num_val_steps, collect_val_ts_every_n_steps = trainer._initialize_val_parameters(
            val_loader, collect_time_series_every_n_steps, num_train_steps=num_train_steps
        )

        expected_val_samples = len(val_set)
        expected_val_steps = len(val_loader)
        expected_val_ts_every_n_steps = 1

        assert _num_val_samples == expected_val_samples
        assert _num_val_steps == expected_val_steps
        assert collect_val_ts_every_n_steps == expected_val_ts_every_n_steps

        logger.info('Test 3 - val train params scaling: normal case')
        num_train_steps = 100
        collect_time_series_every_n_steps = 10  # this is for train

        _num_val_samples, _num_val_steps, collect_val_ts_every_n_steps = trainer._initialize_val_parameters(
            val_loader, collect_time_series_every_n_steps, num_train_steps=num_train_steps
        )

        expected_val_samples = len(val_set)
        expected_val_steps = len(val_loader)
        expected_val_ts_every_n_steps = 1

        assert _num_val_samples == expected_val_samples
        assert _num_val_steps == expected_val_steps
        assert collect_val_ts_every_n_steps == expected_val_ts_every_n_steps

        logger.info('Test 4 - val train params scaling: edge case where val steps > train steps')
        num_train_steps = 10
        collect_time_series_every_n_steps = 10  # this is for train

        _num_val_samples, _num_val_steps, collect_val_ts_every_n_steps = trainer._initialize_val_parameters(
            val_loader, collect_time_series_every_n_steps, num_train_steps=num_train_steps
        )

        expected_val_samples = len(val_set)
        expected_val_steps = len(val_loader)
        expected_val_ts_every_n_steps = 13 # this should be correct

        assert _num_val_samples == expected_val_samples
        assert _num_val_steps == expected_val_steps
        assert collect_val_ts_every_n_steps == expected_val_ts_every_n_steps

        logger.info('Test 5 - collect every step')
        num_train_steps = 10
        collect_time_series_every_n_steps = 1  # this is for train

        _num_val_samples, _num_val_steps, collect_val_ts_every_n_steps = trainer._initialize_val_parameters(
            val_loader, collect_time_series_every_n_steps, num_train_steps=num_train_steps
        )

        expected_val_samples = len(val_set)
        expected_val_steps = len(val_loader)
        expected_val_ts_every_n_steps = 1  # this should be correct

        assert _num_val_samples == expected_val_samples
        assert _num_val_steps == expected_val_steps
        assert collect_val_ts_every_n_steps == expected_val_ts_every_n_steps

    def test_compute_loss(self):

        trainer = BaseTrainer(
            self.model_mock, self.optimizer_mock,  self.loss_mock, self.scheduler_mock
        )
        train_set = self.dataset
        train_loader = DataLoader(train_set, batch_size=32)
        for batch in train_loader:
            expected_batch = batch.copy() # should contain inputs, targets, metadata
            model_output_dict = trainer.compute_loss(
                batch
            )  # should contain: inputs, targets, metadata, loss, outputs
            # manual:
            expected_inputs = expected_batch['inputs']
            expected_targets = expected_batch['targets']
            expected_outputs = self.model_mock(expected_inputs)

            expected_loss = self.loss_mock(expected_outputs, expected_targets)
            expected_output_dict = dict(
                outputs=expected_outputs,
                loss=expected_loss,
                **expected_batch
            )
            for k in model_output_dict:
                assert (model_output_dict[k] == expected_output_dict[k]).all(), f'{model_output_dict[k]}{expected_output_dict[k]}'

            print(model_output_dict)
            assert_tensor_objects_equal(model_output_dict, expected_output_dict)
            break
        pass


    def test_compute_metrics(self):
        metric_1 = torch.nn.L1Loss()
        # TODO need to ensure this does not lose GPU?
        metrics = [
            prepare_torch_metric(torch.nn.L1Loss()),
            prepare_torch_metric(torch.nn.L1Loss(reduction='none')),
            prepare_torch_metric(torch.nn.L1Loss(reduction='sum'))
        ]

        trainer = BaseTrainer(
            self.model_mock, self.optimizer_mock, self.loss_mock, self.scheduler_mock, metrics=metrics
        )

        split = 'train'
        num_epochs = 5
        num_instances = 10
        trainer._initialize_time_series_store(split, num_epochs)
        trainer._initialize_instance_metrics_store(split, num_instances, num_features=num_epochs)

        train_set = self.dataset
        train_loader = DataLoader(train_set, batch_size=32)
        for batch in train_loader:
            model_output_dict = trainer.compute_loss(
                batch
            )
            metrics_output_dict = trainer.compute_metrics(model_output_dict)

            instance_metrics_output_dict = metrics_output_dict['instance_metrics']
            time_series_output_dict = metrics_output_dict['time_series']
            expected_instance_metrics_output_dict = {
                1: metrics[1](model_output_dict)
            }
            expected_time_series_output_dict = {
                0: metrics[0](model_output_dict),
                2: metrics[2](model_output_dict)
            }
            assert_tensor_objects_equal(instance_metrics_output_dict, expected_instance_metrics_output_dict)
            assert_tensor_objects_equal(time_series_output_dict, expected_time_series_output_dict)

    def test_update_update_time_series_history(self):
        metrics = [
            prepare_torch_metric(torch.nn.L1Loss()),
            prepare_torch_metric(torch.nn.L1Loss(reduction='none')),
            prepare_torch_metric(torch.nn.L1Loss(reduction='sum'))
        ]
        logger.info('Test 1 - test ::_update_time_series_history:: without steps')
        trainer = BaseTrainer(
            self.model_mock, self.optimizer_mock, self.loss_mock, self.scheduler_mock, metrics=metrics
        )
        split = 'train'
        num_epochs = 5
        num_instances = 10
        train_set = self.dataset
        train_loader = DataLoader(train_set, batch_size=32)
        trainer._initialize_time_series_store(split, num_epochs)
        trainer._initialize_instance_metrics_store(split, num_instances, num_features=num_epochs)

        expected_time_series_history = {
            0: dict(
                epoch=torch.zeros(num_epochs)
            ),
            2: dict(
                epoch=torch.zeros(num_epochs)
            )
        }

        for epoch_id in range(num_epochs):
            for batch_id, batch in enumerate(train_loader):
                model_output_dict = trainer.compute_loss(
                    batch
                )
                metrics_output_dict = trainer.compute_metrics(model_output_dict)
                trainer._update_time_series_history(metrics_output_dict['time_series'], split,
                                                    collect_time_series_every_n_steps=None, step=batch_id, epoch_id=epoch_id)

                expected_time_series_history[0]['epoch'][epoch_id] += metrics[0](model_output_dict)
                expected_time_series_history[2]['epoch'][epoch_id] += metrics[2](model_output_dict)

        output_time_series_history = trainer.history['metrics']['time_series']
        output_time_series_history = {
            i: d['train'] for i, d in output_time_series_history.items()
        }
        assert_tensor_objects_equal(output_time_series_history, expected_time_series_history)

        logger.info('Test 2 - test ::_update_time_series_history:: without steps')

        trainer = BaseTrainer(
            self.model_mock, self.optimizer_mock, self.loss_mock, self.scheduler_mock, metrics=metrics
        )
        split = 'train'
        num_epochs = 5
        num_instances = 10
        train_set = self.dataset
        train_loader = DataLoader(train_set, batch_size=8)
        trainer._initialize_time_series_store(split, num_epochs, len(train_loader))
        trainer._initialize_instance_metrics_store(split, num_instances, num_features=num_epochs)

        expected_time_series_history = {
            0: dict(
                epoch=torch.zeros(num_epochs),
                step=torch.zeros(len(train_loader)),
                epoch_step_ids=torch.zeros(num_epochs)
            ),
            2: dict(
                epoch=torch.zeros(num_epochs),
                step=torch.zeros(len(train_loader)),
                epoch_step_ids=torch.zeros(num_epochs)
            )
        }
        collect_time_series_every_n_steps = 1

        for epoch_id in range(num_epochs):
            for batch_id, batch in enumerate(train_loader):
                model_output_dict = trainer.compute_loss(
                    batch
                )
                metrics_output_dict = trainer.compute_metrics(model_output_dict)
                trainer._update_time_series_history(metrics_output_dict['time_series'], split,
                                                    collect_time_series_every_n_steps=collect_time_series_every_n_steps, step=batch_id,
                                                    epoch_id=epoch_id)

                expected_time_series_history[0]['epoch'][epoch_id] += metrics[0](model_output_dict)
                if not batch_id % collect_time_series_every_n_steps:
                    step_id = batch_id // collect_time_series_every_n_steps
                    expected_time_series_history[0]['step'][step_id] = metrics[0](model_output_dict)

                expected_time_series_history[2]['epoch'][epoch_id] += metrics[2](model_output_dict)
                if not batch_id % collect_time_series_every_n_steps:
                    step_id = batch_id // collect_time_series_every_n_steps
                    expected_time_series_history[2]['step'][step_id] = metrics[2](model_output_dict)

        output_time_series_history = trainer.history['metrics']['time_series']
        output_time_series_history = {
            i: d['train'] for i, d in output_time_series_history.items()
        }

        assert_tensor_objects_equal(output_time_series_history, expected_time_series_history)

    def test_update_instance_metrics_history(self):
        logger.info('Test 1 - :_update_instance_metrics_history:')

        metrics = [
            prepare_torch_metric(torch.nn.L1Loss()),
            prepare_torch_metric(torch.nn.L1Loss(reduction='none')),
            prepare_torch_metric(torch.nn.L1Loss(reduction='sum'))
        ]
        trainer = BaseTrainer(
            self.model_mock, self.optimizer_mock, self.loss_mock, self.scheduler_mock, metrics=metrics
        )
        split = 'train'
        num_epochs = 5
        train_set = self.dataset
        num_instances = len(train_set)

        train_loader = DataLoader(train_set, batch_size=32)
        trainer._initialize_time_series_store(split, num_epochs)
        trainer._initialize_instance_metrics_store(split, num_instances, num_features=num_epochs)

        expected_instance_metric_history = {
            1: torch.zeros((num_instances, num_epochs))
        }

        for epoch_id in range(num_epochs):
            for batch_id, batch in enumerate(train_loader):
                model_output_dict = trainer.compute_loss(
                    batch
                )
                metrics_output_dict = trainer.compute_metrics(model_output_dict)
                trainer._update_instance_metrics_history(metrics_output_dict['instance_metrics'], split, epoch_id)
                expected_instance_ids, expected_instance_scores = metrics[1](batch)
                expected_instance_metric_history[1][expected_instance_ids, epoch_id] = expected_instance_scores

        output_instance_metrics_history = trainer.history['metrics']['instance_metrics']
        output_instance_metrics_history = {i: d[split] for i, d in output_instance_metrics_history.items()}
        assert_tensor_objects_equal(output_instance_metrics_history, expected_instance_metric_history)


    def test_train(self):
        '''logger.info('Test 1 - test training and history of basic model')
        trainer = BaseTrainer(
            model=self.model,
            optimizer=self.optimizer,
            loss=self.loss,
            scheduler=self.scheduler,
            metrics=self.metrics,
        )

        data_loader = DataLoader(self.dataset, batch_size=128, shuffle=True)
        epochs = 2

        model = trainer.train(data_loader, epochs, verbose=2)
        output_history = trainer.history
        print(output_history)
        assert set(output_history.keys()) == {'loss'}
        assert set(output_history['loss'].keys()) == {'params', 'train'}, set(output_history['loss'].keys())
        assert output_history['loss']['params'] == dict(
                name=self.loss.__class__.__name__,
                **{param: value for param, value in self.loss.__dict__.items() if not param.startswith('_')}
        ), output_history['loss']['params']

        assert set(output_history['loss']['train'].keys()) == {'epoch'}, set(output_history['loss']['train'].keys())
        assert len(output_history['loss']['train']['epoch']) == epochs


        logger.info('Test 2 - test training and history of basic model with batch saving')

        trainer = BaseTrainer(
            model=self.model,
            optimizer=self.optimizer,
            loss=self.loss,
            scheduler=self.scheduler,
            metrics=self.metrics,
        )
        batch_size = 128
        n_batches = math.ceil(self.n_points / batch_size)
        data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        epochs = 2

        model = trainer.train(data_loader, epochs, verbose=2, collect_batch_data=True)


        output_history = trainer.history

        assert set(output_history.keys()) == {'loss'}
        assert set(output_history['loss'].keys()) == {'params', 'train'}, set(output_history['loss'].keys())
        assert output_history['loss']['params'] == dict(
            name=self.loss.__class__.__name__,
            **{param: value for param, value in self.loss.__dict__.items() if not param.startswith('_')}
        ), output_history['loss']['params']

        assert set(output_history['loss']['train'].keys()) == {'epoch', 'batch', 'epoch_batch_ids'}, set(output_history['loss']['train'].keys())
        assert len(output_history['loss']['train']['epoch']) == epochs
        assert len(output_history['loss']['train']['batch']) == n_batches*epochs, len(output_history['loss']['train']['batch'])
        assert output_history['loss']['train']['epoch_batch_ids'] == [7, 15]'''

        '''logger.info('Test 3 - test training and history of basic model with metrics with batch saving')

        metrics = [torch.nn.L1Loss(), torch.nn.L1Loss(reduction='none'), torch.nn.L1Loss(reduction='mean')]

        trainer = BaseTrainer(
            model=self.model,
            optimizer=self.optimizer,
            loss=self.loss,
            scheduler=self.scheduler,
            metrics=metrics,
        )
        batch_size = 128
        n_batches = math.ceil(self.n_points / batch_size)
        data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        epochs = 2

        model = trainer.train(data_loader, epochs, verbose=2, collect_time_series_every_n_steps=5)

        output_history = trainer.history

        assert set(output_history.keys()) == {'loss', 'metrics'}
        expected_metrics = [
            dict(
                params=dict(name=metric.__class__.__name__, **{param: value for param, value in metric.__dict__.items() if not param.startswith('_')}),
                train=dict(
                    epoch=[0]*epochs,
                    batch=[0]*n_batches,
                    epoch_batch_ids=[7, 15]
                )
            ) for metric in metrics
        ]
        for output_metric, expected_metric in zip(output_history['metrics'], expected_metrics):
            assert output_metric.keys() == expected_metric.keys()
            assert output_metric['params'] == expected_metric['params']
            assert output_metric['train'].keys() == expected_metric['train'].keys()
            assert len(output_metric['train']['epoch']) == len(expected_metric['train']['epoch'])
            assert len(output_metric['train']['batch']) == len(expected_metric['train']['batch'])
            assert len(output_metric['train']['epoch_batch_ids']) == expected_metric['train']['epoch_batch_ids']

        logger.info('Test 4 - test training and history of basic model with metrics and validation and batch saving')'''

        '''plt.plot(self.X, self.y, '.')
        plt.plot(self.X, model(self.X).detach().numpy(), '.')
        plt.show()

        print(trainer.history)
        plt.plot(range(len(trainer.history['loss']['train']['batch'])), trainer.history['loss']['train']['batch'])
        plt.plot(trainer.history['loss']['train']['epoch_batch_ids'], trainer.history['loss']['train']['epoch'], 'o')
        plt.show()
        for param in trainer.model.parameters():
            print(param)'''

    '''logger.info('Test - test end to end run BaseTrainer with all parameters')


    X = torch.cat(
        [torch.arange(100).reshape(-1, 1)**i for i in range(3)],
        dim=1
    )
    y = torch.arange(100)
    epochs = 5
    train_set = SimpleDataset(X, y)

    TRAIN_BATCH_SIZE = 20
    VAL_BATCH_SIZE = 20
    train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE)
    val_set = SimpleDataset(X, y)
    val_loader = DataLoader(val_set, batch_size=VAL_BATCH_SIZE)

    model_mock = ModelMock()
    optimizer_mock = OptimizerMock()
    scheduler_mock = SchedulerMock()
    loss_mock = LossMock() # think about this for huggingface vs non-hugging face

    metrics = [torch.nn.L1Loss(), torch.nn.L1Loss(reduction='none')]

    trainer = BaseTrainer(
        model_mock, optimizer_mock, loss_mock, scheduler_mock, metrics,
    )

    trainer.train(train_loader, epochs, val_loader, verbose=2, collect_time_series_every_n_steps=5, scale_validation_to_train=True)'''


    def test_test(self):
        pass
class TestMultiTaskTrainer(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()