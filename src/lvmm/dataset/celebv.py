import os
from typing import List

from .dset import Dset
from .utils import video_loader
import random


class CelebvDataset(Dset):
    tag = "video:celebv70k"

    def __init__(self, *args, properties: List = [], **kwargs):
        """
        properties:
            ['action', 'emotion', 'face40_details', 'light_color_temp', 'light_dir', 'light_intensity', 'video_name']
        """
        super().__init__(*args, **kwargs)
        self.properties = properties

    def getitem(self, index):
        meta = self.metas[index]
        # "action": action_text[video_name],
        # "emotion": emotion_text[video_name],
        # "face40_details": face40_details_text[video_name],
        # "light_color_temp": light_color_temp_text[video_name],
        # "light_dir": light_dir_text[video_name],
        # "light_intensity": light_intensity_text[video_name],
        # "video_name": video_name,
        filename = os.path.join(self.data_folder, "celebvtext_6",
                                meta["video_name"])
        frames = video_loader(filename,
                              n_frames=self.n_frames,
                              sequence_length=self.sequence_length)

        properties = [random.choice(meta[p].split("\n")) for p in self.properties]
        return self.post_process(frames, " ".join(properties))
