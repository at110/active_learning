import os
import unittest
from unittest.mock import MagicMock, patch
import torch
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Tuple
from train import train_epoch, validate, save_best_model, run_training  # Adjust the import path
from monai.metrics import DiceMetric
from monai.transforms import Compose, AsDiscrete


class TestTrainEpoch(unittest.TestCase):
    """
    Unit tests for training, validation, and model saving functions.
    """

    def setUp(self) -> None:
        """
        Setup common mocks and variables for tests.
        """
        self.mock_model: MagicMock = MagicMock(spec=Module)
        self.mock_optimizer: MagicMock = MagicMock(spec=Optimizer)
        self.mock_loss_function: MagicMock = MagicMock(spec=Module)
        
        # Ensure the mock model and loss function return tensors with requires_grad=True
        self.mock_model.return_value = torch.randn(1, 2, 160, 160, 160, requires_grad=True)
        self.mock_loss_function.return_value = torch.tensor(1.0, requires_grad=True)
        
        self.mock_data_loader: MagicMock = MagicMock(spec=DataLoader)
        self.mock_data_loader.__iter__.return_value = iter([
            {"image": torch.rand(1, 2, 160, 160, 160, requires_grad=True), "label": torch.rand(1, 2, 160, 160, 160, requires_grad=True)}
        ])

        self.mock_post_pred: Compose = Compose([AsDiscrete(argmax=True, to_onehot=2)])
        self.mock_post_label: Compose = Compose([AsDiscrete(to_onehot=2)])
        self.mock_dice_metric: MagicMock = MagicMock(spec=DiceMetric)
        self.mock_dice_metric.aggregate.return_value = torch.tensor(0.8)

        self.device: torch.device = torch.device('cpu')
        self.root_dir: str = "./test_saving_dir"
        os.makedirs(self.root_dir, exist_ok=True)

    @patch('torch.optim.Adam')
    def test_train_epoch(self, mock_optimizer: MagicMock) -> None:
        """
        Test the train_epoch function to ensure it processes a batch and updates the model as expected.
        """
        avg_loss = train_epoch(self.mock_model, self.mock_data_loader, self.mock_optimizer, 
                               self.mock_loss_function, self.device)
        
        self.mock_optimizer.zero_grad.assert_called()
        self.mock_loss_function.assert_called()
        self.mock_optimizer.step.assert_called()
        self.assertIsNotNone(avg_loss)
        self.assertTrue(isinstance(avg_loss, float))
    
    @patch.object(AsDiscrete, '__call__', return_value=torch.tensor(1))
    def test_validate(self, mock_as_discrete: MagicMock) -> None:
        """
        Test the validate function to ensure it computes validation loss and metric.
        """
        val_loss, metric = validate(self.mock_model, self.mock_data_loader, self.mock_loss_function, 
                                    self.device, self.mock_post_pred, self.mock_post_label, self.mock_dice_metric)

        self.mock_loss_function.assert_called()
        self.assertIsInstance(val_loss, float)
        self.assertIsInstance(metric, float)
        self.assertGreaterEqual(metric, 0.0)
        self.assertLessEqual(metric, 1.0)

    @patch("torch.save")
    def test_save_best_model_improvement(self, mock_save: MagicMock) -> None:
        """
        Test save_best_model function when there is an improvement in the metric.
        """
        metric = 0.9
        best_metric = 0.8
        epoch = 2

        new_best_metric, new_best_epoch = save_best_model(metric, best_metric, self.mock_model, epoch, self.root_dir)

        mock_save.assert_called()

        self.assertEqual(new_best_metric, metric)
        self.assertEqual(new_best_epoch, epoch)

    @patch("torch.save")
    def test_save_best_model_no_improvement(self, mock_save: MagicMock) -> None:
        """
        Test save_best_model function when there is no improvement in the metric.
        """
        metric = 0.7
        best_metric = 0.8
        epoch = 2

        new_best_metric, new_best_epoch = save_best_model(metric, best_metric, self.mock_model, epoch, self.root_dir)

        mock_save.assert_not_called()
        self.assertEqual(new_best_metric, best_metric)
        self.assertNotEqual(new_best_epoch, epoch)

    def tearDown(self) -> None:
        """
        Cleanup the test directory after tests.
        """
        if os.path.exists(self.root_dir):
            for filename in os.listdir(self.root_dir):
                file_path = os.path.join(self.root_dir, filename)
                os.unlink(file_path)
            os.rmdir(self.root_dir)

    @patch('train.train_epoch')
    @patch('train.validate')
    @patch('train.save_best_model')
    def test_run_training(self, mock_save_best_model: MagicMock, mock_validate: MagicMock, mock_train_epoch: MagicMock) -> None:
        """
        Test the run_training function for its execution over epochs and validation intervals.
        """
        mock_train_epoch.return_value = 0.5
        mock_validate.return_value = (0.4, 0.8)
        mock_save_best_model.return_value = (0.8, 2)
        max_epochs = 2  # Define locally since it's specific to this test
        val_interval = 1  # Define locally since it's specific to this test
        run_training(self.mock_model, self.mock_data_loader, self.mock_data_loader, self.mock_optimizer,
                     self.mock_loss_function, self.device, max_epochs, val_interval, self.root_dir)

        self.assertEqual(mock_train_epoch.call_count, max_epochs)
        self.assertEqual(mock_validate.call_count, max_epochs // val_interval)
        if max_epochs >= val_interval:
            mock_save_best_model.assert_called()

if __name__ == '__main__':
    unittest.main()
