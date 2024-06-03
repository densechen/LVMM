import os
import random

from .dset import Dset
from .utils import video_loader


class VideoClipDataset(Dset):
    tag = "video:hdvila100m"
    def __init__(self, *args, load_category=False, **kwargs):
        """Initialize a VideoClipDataset. NOTE: The frame rate of this dataset is 3.
        Args:
            metafile: The parquet file contains meta information. 
                If more than one file are given, we will load and concat them.
                If a folder is given, we will load all files ended with `parquet` and concat them.
            load_category: Cat category with prompt.
        """
        super().__init__(*args, **kwargs)
        self.load_category = load_category

    def getitem(self, index):
        meta = self.metas[index]
        video_path = os.path.join(
            self.data_folder,
            meta["video_part_id"],
            "".join(meta["clip_id"].split(".")[:-2]),
            meta["clip_id"],
        )
        try:
            frames = video_loader(video_path, n_frames=self.n_frames, sequence_length=self.sequence_length)
        except Exception:
            return random.choice(self)
        text = meta["text"] if not self.load_category else " ".join([meta["text"],  meta["categories"]])
        return self.post_process(frames, text)
