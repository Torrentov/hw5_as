import unittest

import torch

from hw_as.datasets import CustomDirAudioDataset, CustomAudioDataset, ASVSpoof2019Dataset
from hw_as.tests.utils import clear_log_folder_after_use
from hw_as.utils import ROOT_PATH
from hw_as.utils.parse_config import ConfigParser


class TestDataset(unittest.TestCase):
    def test_asvspoof2019_dataset(self):
        config_parser = ConfigParser.get_test_configs()
        with clear_log_folder_after_use(config_parser):
            audio_dir = str(ROOT_PATH / "data" / "LA" / "LA")

            ds = ASVSpoof2019Dataset(
                audio_dir,
                "train",
                config_parser=config_parser,
                limit=10,
            )
            self._assert_training_example_is_good(ds[0])

    def _assert_training_example_is_good(self, training_example: dict, contains_text=True):

        for field, expected_type in [
            ("audio", torch.Tensor),
            ("duration", float),
            ("audio_path", str),
            ("gt_label", int),
            ("speaker_id", str),
            ("system_id", str),
        ]:
            self.assertIn(field, training_example, f"Error during checking field {field}")
            self.assertIsInstance(training_example[field], expected_type,
                                  f"Error during checking field {field}")

        # check waveform dimensions
        batch_dim, audio_dim, = training_example["audio"].size()
        self.assertEqual(batch_dim, 1)
        self.assertGreater(audio_dim, 1)
