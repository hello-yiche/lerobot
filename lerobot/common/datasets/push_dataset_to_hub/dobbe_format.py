#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Process raw dobb-e format for pushing to Hugging Face Hub.
"""

import gc
import shutil
import os
from pathlib import Path

import h5py
import json
import numpy as np
import torch
import tqdm
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, save_images_concurrently
from lerobot.common.datasets.utils import (
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames


# def get_cameras(hdf5_data):
#     # ignore depth channel, not currently handled
#     # TODO(rcadene): add depth
#     rgb_cameras = [key for key in hdf5_data["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118
#     return rgb_cameras


def check_format(raw_dir):

    episode_dirs = [path for path in Path(
        raw_dir).iterdir() if path.is_dir()]
    assert len(episode_dirs) != 0

    for episode_dir in episode_dirs:
        compressed_imgs = episode_dir / "compressed_images"
        png_files = list(compressed_imgs.glob("*.png"))
        assert len(png_files) > 0

        compressed_video = episode_dir / "compressed_video_h264.mp4"
        assert compressed_video.exists()

        labels = episode_dir / "labels.json"
        assert labels.exists()


def load_from_raw(raw_dir, out_dir, fps, video, debug):
    # only frames from simulation are uncompressed
    episode_dirs = [path for path in Path(
        raw_dir).iterdir() if path.is_dir()]

    ep_dicts = []
    episode_data_index = {"from": [], "to": []}

    id_from = 0
    for ep_idx, ep_path in tqdm.tqdm(enumerate(episode_dirs), total=len(episode_dirs)):
        # load image frames
        compressed_imgs = ep_path / "compressed_images"
        png_files = list(compressed_imgs.glob("*.png"))
        num_frames = len(png_files)

        # last step of demonstration is considered done
        done = torch.zeros(num_frames, dtype=torch.bool)
        done[-1] = True

        # Dictionary for episode data
        ep_dict = {}

        for camera in ["gripper"]:
            img_key = f"observation.images.{camera}"

            if video:
                video_path = ep_path / "compressed_video.mp4"
                assert video_path.exists()

                fname = f"gripper_episode_{ep_idx:06d}.mp4"
                video_dir = out_dir / "videos"
                video_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(video_path, video_dir / fname)

                ep_dict[img_key] = [
                    {"path": f"videos/{fname}", "timestamp": i / fps} for i in range(num_frames)
                ]
            else:
                ep_dict[img_key] = [
                    PILImage.open(file) for file in png_files]

        labels = ep_path / "labels.json"
        with open(labels, "r") as f:
            labels_dict = json.load(f)
            assert len(labels_dict) == num_frames

            # currently store actions as list of [x, y, z, qw, qx, qy, qz, gripper]
            action = [v['xyz']+v['quats']+[v['gripper']]
                      for k, v in labels_dict.items()]
            action = torch.tensor(action)

            ep_dict["action"] = action
            ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames)
            ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
            ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps
            ep_dict["next.done"] = done

            assert isinstance(ep_idx, int)
            ep_dicts.append(ep_dict)

            episode_data_index["from"].append(id_from)
            episode_data_index["to"].append(id_from + num_frames)

        id_from += num_frames

        # process first episode only
        if debug:
            break

    data_dict = {}
    data_dict = concatenate_episodes(ep_dicts)

    return data_dict, episode_data_index


def to_hf_dataset(data_dict, video=False) -> Dataset:
    features = {}

    keys = [key for key in data_dict if "observation.images." in key]
    for key in keys:
        if video:
            features[key] = VideoFrame()
        else:
            features[key] = Image()

    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(
            dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def from_raw_to_lerobot_format(raw_dir: Path, out_dir, fps=None, video=False, debug=False):
    # sanity check
    check_format(raw_dir)

    if fps is None:
        fps = 30

    data_dir, episode_data_index = load_from_raw(
        raw_dir, out_dir,  fps, video, debug)
    hf_dataset = to_hf_dataset(data_dir, video)

    info = {
        "fps": fps,
        "video": video,
    }
    return hf_dataset, episode_data_index, info
