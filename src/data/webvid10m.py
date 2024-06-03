import argparse
import concurrent.futures
import json
import os

import jsonlines
import numpy as np
import psutil
from tqdm import trange

parser = argparse.ArgumentParser()
# /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/densechen/webvid/data/videos/
parser.add_argument("root", help="Root folder of webvid10m.")
args = parser.parse_args()
print("Loading meta.")
with open(os.path.join(".webvid10m", "webvid_meta.json")) as f:
    video_infos = json.load(f)

# split to 10000, that each file contains 10000 items.
split_clip_ids = np.array_split(video_infos, 10000)


print("Processing.")
os.makedirs(".webvid10m", exist_ok=True)
def check_and_save(idx):
    filename = os.path.join(".webvid10m", f"{idx:05d}.jsonl")
    counts = 0
    with jsonlines.open(filename, "w") as f:
        for index, (path, text) in  enumerate(split_clip_ids[idx]):
            if not os.path.exists(path):
                continue
            path = path.replace(args.root, "")
            try:
                f.write(
                    {   
                        "path": path,
                        "text": text,
                    }
                )
            except Exception as e:
                print(e)
            counts += 1
    return counts
    
with concurrent.futures.ThreadPoolExecutor(psutil.cpu_count()) as executor:
    buffer = []
    for idx in range(len(split_clip_ids)):
        buffer.append(
            executor.submit(check_and_save, idx))
    total_videos = 0
    for task in concurrent.futures.as_completed(buffer):
        total_videos += task.result()
        
print(f"Total videos: {total_videos}")