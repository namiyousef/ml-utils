# TODO consider changing the name of this
from mlutils.torchtools.run import BaseTrainer
from tests.torchtools.utils import LinearModel, SimpleDataset, ModelMock, SchedulerMock, OptimizerMock, LossMock
import unittest
import torch
import numpy as np
import random
import logging

from mlutils.torchtools.testing import assert_tensor_objects_equal

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

    def test_initialize_metrics_dict(self):
        logging.info('Test 1 - test no metrics: output should simply contain loss and associated params')
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


        logging.info('Test 2 - test time series metric: output should contain extra metrics dict with time series')
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

        logging.info('Test 3 - test instance metric')
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

        logging.info('Test 4 - test both types of metrics')

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
        train_set = None
        train_loader = None
        collect_time_series_every_n_steps = None
        raise NotImplementedError('Need to test for edge cases on the value of collect_time_series_data')


    def test_create_empty_time_series_dict(self):
        logging.info('Test 1 - test dict created correctly for loss without steps')
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

        logging.info('Test 2 - test dict created correctly for loss with steps')

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
        pass

    def test_initialize_instance_metrics_store(self):
        pass
    def test_train(self):
        '''logging.info('Test 1 - test training and history of basic model')
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


        logging.info('Test 2 - test training and history of basic model with batch saving')

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

        '''logging.info('Test 3 - test training and history of basic model with metrics with batch saving')

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

        logging.info('Test 4 - test training and history of basic model with metrics and validation and batch saving')'''

        '''plt.plot(self.X, self.y, '.')
        plt.plot(self.X, model(self.X).detach().numpy(), '.')
        plt.show()

        print(trainer.history)
        plt.plot(range(len(trainer.history['loss']['train']['batch'])), trainer.history['loss']['train']['batch'])
        plt.plot(trainer.history['loss']['train']['epoch_batch_ids'], trainer.history['loss']['train']['epoch'], 'o')
        plt.show()
        for param in trainer.model.parameters():
            print(param)'''

    logging.info('Test - test end to end run BaseTrainer with all parameters')

    epochs = 5
    train_set = None
    train_loader = None
    val_set = None
    val_loader = None

    model_mock = ModelMock()
    optimizer_mock = OptimizerMock()
    scheduler_mock = SchedulerMock()
    loss_mock = LossMock() # think about this for huggingface vs non-hugging face

    metrics = []

    trainer = BaseTrainer(
        model_mock, optimizer_mock, loss_mock, scheduler_mock, metrics,
    )

    trainer.train(train_loader, epochs, val_loader, verbose=2, collect_time_series_every_n_steps=5, scale_validation_to_train=True)


    def test_test(self):
        pass
class TestMultiTaskTrainer(unittest.TestCase):
    pass