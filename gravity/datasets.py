from pathlib import Path

from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder

import random

_DATA_DIR = Path(
    "/Users/kadeem/Spaces/Projects/Gravity/data/AVoE/A_support/train/expected"
)


class AVoEMicroDataset(Dataset):
    def __init__(self):
        """Initialize the AVoE dataset."""
        self._trials = random.sample(list(_DATA_DIR.glob("*")), k=8)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.__len__():
            raise IndexError()

        trial_dir = self._trials[idx]
        decoder = VideoDecoder(trial_dir / "rgb.avi")
        video = decoder.get_all_frames().data

        return video

    def __len__(self):
        """Returns the number of video pairs in the dataset."""
        return len(self._trials)
