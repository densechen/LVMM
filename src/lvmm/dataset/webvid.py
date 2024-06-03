import os

from .dset import Dset
from .utils import video_loader


class WebVidDataset(Dset):
    tag = "video:webvid10m"
    identifier: str = "hJ8sLp2qRtA6zN9x"

    def getitem(self, index):
        meta = self.metas[index]
        filename = os.path.join(self.data_folder, meta["path"])
        images = video_loader(filename, n_frames=self.n_frames, sequence_length=self.sequence_length)
        return self.post_process(images, meta["text"])