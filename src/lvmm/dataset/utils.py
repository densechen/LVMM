import random

import decord
from PIL import Image


def split_list(lst, n):
    lists = [lst[i:i+n] for i in range(0, len(lst), n)]
    # drop the one less than n
    return [l for l in lists if len(l) == n]


def video_loader(filename, n_frames, sequence_length):
    """
    Args:
        n_frames: total frames requires. if not enough, padding with last frame.
        frame_rate: the resampled fps. set None to disable it.
    """
    vr = decord.VideoReader(filename)
    if not len(vr):
        raise ValueError(f"Empty videos: {filename}")
    sequence_length = min(sequence_length, len(vr)-1)
    start_indices = random.randint(0, len(vr)-1-sequence_length)
    if n_frames == sequence_length:
        indices = list(range(start_indices, start_indices+sequence_length))
    elif n_frames == 2:
        indices = [start_indices, start_indices + random.randint(1, sequence_length-1)]
    else:
        raise RuntimeError
    images = [Image.fromarray(vr[idx].asnumpy()).convert("RGB") for idx in indices]
    
    return images
