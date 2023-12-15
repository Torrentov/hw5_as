import logging
from pathlib import Path
from tqdm import tqdm

from hw_as.datasets.custom_audio_dataset import CustomAudioDataset

from hw_as.utils import ROOT_PATH

logger = logging.getLogger(__name__)


class ASVSpoof2019Dataset(CustomAudioDataset):
    def __init__(self, audio_dir, part, *args, **kwargs):
        audio_dir = Path(audio_dir)
        if str(audio_dir)[0] != '/' and str(audio_dir)[0] != '\\':
            audio_dir = ROOT_PATH / audio_dir
        protocols_dir = audio_dir / "ASVspoof2019_LA_cm_protocols"
        if part == "train":
            part_protocol = protocols_dir / f"ASVspoof2019.LA.cm.train.trn.txt"
        else:
            part_protocol = protocols_dir / f"ASVspoof2019.LA.cm.{part}.trl.txt"
        part_dir = audio_dir / f"ASVspoof2019_LA_{part}" / "flac"
        data = []
        with open(part_protocol, "r") as f:
            for line in tqdm(f):
                entry = {}
                line = line.strip('\n')
                speaker_id, audio_file_name, _, system_id, label = line.split()
                entry["speaker_id"] = speaker_id
                entry["system_id"] = system_id
                entry["gt_label"] = int(label == "bonafide")
                entry["path"] = str(part_dir / (audio_file_name + ".flac"))
                if len(entry) > 0:
                    data.append(entry)
        super().__init__(data, *args, **kwargs)
