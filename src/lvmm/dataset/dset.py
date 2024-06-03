import os
import random
import warnings
from glob import glob

import cv2
import jsonlines
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


class Resize(object):

    def __init__(self, desired_size, interpolation):
        self.desired_size = desired_size
        self.interpolation = interpolation

    def __call__(self, im):
        im = np.array(im)
        old_size = im.shape[:2]
        ratio = float(self.desired_size) / max(old_size)
        new_size = tuple(int(x * ratio) for x in old_size)

        im = cv2.resize(im, (new_size[1], new_size[0]),
                        interpolation=self.interpolation)
        delta_w = self.desired_size - new_size[1]
        delta_h = self.desired_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(im,
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    cv2.BORDER_CONSTANT,
                                    value=color)

        return new_im


class Dset(Dataset):
    # format: {image/video}:dataset_name
    tag: str  # indicate which dataset is belongs to.
    identifier: str = ""

    def __init__(
            self,
            metafile,
            data_folder,
            resolution: int = 256,
            n_frames: int = 2,
            sequence_length:
        int = 150,  # if n_frames == sequence_length, sample continues frames.
            # if n_frames == 2, randomly sample two frames with a maximum interval of sequence_length.
            ):

        self.resolution = resolution
        self.n_frames = n_frames
        self.sequence_length = sequence_length
        self.data_folder = data_folder

        self.metas = self.load_meta(metafile)

        self.trms = self.build_transforms()

        self.cache_data = None

    def build_transforms(self):
        # Normalize to [-1, 1]
        trms = []
        trms.append(transforms.Resize(self.resolution))
        trms.append(transforms.CenterCrop(self.resolution))
        # trms.append(Resize(self.resolution, cv2.INTER_AREA))
        trms.append(transforms.ToTensor())
        # trms.append(transforms.Lambda(lambda x: x * 2. - 1.))
        return transforms.Compose(trms)

    def __len__(self):
        return len(self.metas)

    def load_meta(self, metafile):
        """
        Args:
            a single string of folder or jsonl
            a list of jsonl files.
            a list of folder is not supported
        """
        if isinstance(metafile, str):
            if metafile.endswith(".jsonl"):
                metafile = [metafile]
            else:
                # a folder is given
                metafile = glob(os.path.join(metafile, "*.jsonl"))
        rank, world_size = dist.get_rank(), dist.get_world_size()
        print(
            f"[Dset.load_meta] Loading partial chunks. [{rank}|{world_size}]"
        )
        metafile = np.array_split(
            metafile, indices_or_sections=world_size)[rank]

        metas = []
        try:
            for m in tqdm(metafile,
                          desc=f"Loading {self.tag} meta files",
                          disable=rank != 0):
                if not os.path.exists(m):
                    warnings.warn(f"Missing metafile: {m}")
                    continue
                with jsonlines.open(m, mode="r") as reader:
                    metas.extend(reader.iter())
        except KeyboardInterrupt:
            print("[Dset] Stopped.")
        return metas

    def post_process(self, frames, text):
        return {
            "txt": text,
            "frames": torch.stack([self.trms(frame) for frame in frames]),
        }

    def getitem(self, index):
        raise NotImplementedError

    def __getitem__(self, index):
        try:
            return self.getitem(index)
        except Exception:
            return random.choice(self)