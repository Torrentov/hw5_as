import unittest

from tqdm import tqdm

from hw_as.collate_fn.collate import collate_fn
from hw_as.datasets import ASVSpoof2019Dataset
from hw_as.tests.utils import clear_log_folder_after_use
from hw_as.utils.object_loading import get_dataloaders
from hw_as.utils.parse_config import ConfigParser
from hw_as.utils import ROOT_PATH



class TestDataloader(unittest.TestCase):
    def test_collate_fn(self):
        config_parser = ConfigParser.get_test_configs()
        with clear_log_folder_after_use(config_parser):
            audio_dir = str(ROOT_PATH / "data" / "LA" / "LA")

            ds = ASVSpoof2019Dataset(
                audio_dir=audio_dir,
                part="train",
                config_parser=config_parser
            )


            batch_size = 3
            batch = collate_fn([ds[i] for i in range(batch_size)])

            self.assertIn("audio_gt", batch)  # torch.tensor

    def test_dataloaders(self):
        _TOTAL_ITERATIONS = 10
        config_parser = ConfigParser.get_test_configs()
        with clear_log_folder_after_use(config_parser):
            dataloaders = get_dataloaders(config_parser)
            for part in ["train", "dev"]:
                dl = dataloaders[part]
                for i, batch in tqdm(enumerate(iter(dl)), total=_TOTAL_ITERATIONS,
                                     desc=f"Iterating over {part}"):
                    if i >= _TOTAL_ITERATIONS: break
