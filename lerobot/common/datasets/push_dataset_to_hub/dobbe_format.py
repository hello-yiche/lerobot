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
import liblzfse

from lerobot.common.datasets.push_dataset_to_hub.utils import (
    concatenate_episodes,
    save_images_concurrently,
)
from lerobot.common.datasets.utils import (
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames

# Set camera input sizes
IMAGE_SIZE = {"gripper": (240, 320), "head": (320, 240)}


def check_format(raw_dir):

    print("Image sizes set as: ", IMAGE_SIZE)

    episode_dirs = [path for path in Path(raw_dir).iterdir() if path.is_dir()]
    assert len(episode_dirs) != 0

    for episode_dir in episode_dirs:

        # States and actions json file
        labels = episode_dir / "labels.json"
        assert labels.exists(), f"Labels file {labels} wasn't found"

        for camera in ["gripper", "head"]:

            # Check for image folders
            compressed_imgs = episode_dir / f"compressed_{camera}_images"
            if not compressed_imgs.exists():
                print(
                    f"Image folder {compressed_imgs} wasn't found. Only video mode will be supported"
                )

            # Video files
            compressed_video = episode_dir / f"{camera}_compressed_video_h264.mp4"
            assert compressed_video.exists()

            # Depth compressed binary files
            depth_bin_path = episode_dir / f"compressed_np_{camera}_depth_float32.bin"
            assert depth_bin_path.exists()


def unpack_depth(depth_bin, num_frames, size):
    h, w = size
    depths = liblzfse.decompress(depth_bin.read_bytes())
    depths = np.frombuffer(depths, dtype=np.float32).reshape((num_frames, h, w))
    return depths


def clip_and_normalize_depth(depths, camera):
    # Clips and normalizes depths based on camera
    # depths: (num_frames, h, w)
    # camera: "gripper", "head"

    if camera == "gripper":
        depths = np.clip(depths * 1000, 0.0, 255.0)
    elif camera == "head":
        max = 4000.0
        depths = np.clip(depths * 1000, 0.0, max)
        depths *= 255.0 / max
    else:
        raise NotImplementedError("Unsupported camera!")

    # Repeat on three channels, for concatenating with RGB images during training
    # (num_frames, h, w, c)
    depths = np.expand_dims(depths, axis=3)
    depths = np.repeat(depths, 3, axis=3)

    return depths


def load_from_raw(raw_dir, out_dir, fps, video, debug):
    episode_dirs = [path for path in Path(raw_dir).iterdir() if path.is_dir()]

    ep_dicts = []
    episode_data_index = {"from": [], "to": []}

    id_from = 0
    for ep_idx, ep_path in tqdm.tqdm(enumerate(episode_dirs), total=len(episode_dirs)):

        # Dictionary for episode data
        ep_dict = {}
        num_frames = 0

        # Parse observation state and action
        labels = ep_path / "labels.json"
        with open(labels, "r") as f:
            labels_dict = json.load(f)
            num_frames = len(labels_dict)

            actions = [
                [
                    data["actions"]["joint_mobile_base_rotate_by"],
                    data["actions"]["joint_lift"],
                    data["actions"]["joint_arm_l0"],
                    data["actions"]["joint_wrist_roll"],
                    data["actions"]["joint_wrist_pitch"],
                    data["actions"]["joint_wrist_yaw"],
                    data["actions"]["stretch_gripper"],
                ]
                for _, data in labels_dict.items()
            ]

            state = [
                [
                    data["observations"]["joint_mobile_base_rotation"],
                    data["observations"]["joint_lift"],
                    data["observations"]["joint_arm_l0"],
                    data["observations"]["joint_wrist_roll"],
                    data["observations"]["joint_wrist_pitch"],
                    data["observations"]["joint_wrist_yaw"],
                    data["observations"]["stretch_gripper"],
                ]
                for _, data in labels_dict.items()
            ]

            ep_dict["observation.state"] = torch.tensor(state)
            ep_dict["action"] = torch.tensor(actions)

        # Parse observation images
        for camera in ["gripper", "head"]:
            img_key = f"observation.images.{camera}"
            depth_key = f"observation.images.{camera}_depth"

            if video:
                video_path = ep_path / f"{camera}_compressed_video_h264.mp4"

                fname = f"{camera}_episode_{ep_idx:06d}.mp4"
                video_dir = out_dir / "videos"
                video_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(video_path, video_dir / fname)

                ep_dict[img_key] = [
                    {"path": f"videos/{fname}", "timestamp": i / fps}
                    for i in range(num_frames)
                ]
            else:
                # Parse RGB images
                compressed_imgs = ep_path / f"compressed_{camera}_images"
                assert (
                    compressed_imgs.exists()
                ), f"Image folder {compressed_imgs} wasn't found. Only video mode is supported."

                rgb_png = list(compressed_imgs.glob("*.png"))

                images = []
                for file in rgb_png:
                    with PILImage.open(file) as f:
                        images.append(f)

                ep_dict[img_key] = images

            # Depth compressed binary inputs
            depth_bin_path = ep_path / f"compressed_np_{camera}_depth_float32.bin"
            depths = unpack_depth(depth_bin_path, num_frames, IMAGE_SIZE[camera])
            depths = clip_and_normalize_depth(depths, camera)

            ep_dict[depth_key] = [
                PILImage.fromarray(x.astype(np.uint8), "RGB") for x in depths
            ]

        # last step of demonstration is considered done
        done = torch.zeros(num_frames, dtype=torch.bool)
        done[-1] = True

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
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1],
        feature=Value(dtype="float32", id=None),
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def from_raw_to_lerobot_format(
    raw_dir: Path, out_dir, fps=None, video=False, debug=False
):
    # sanity check
    check_format(raw_dir)

    if fps is None:
        fps = 15

    data_dir, episode_data_index = load_from_raw(raw_dir, out_dir, fps, video, debug)
    hf_dataset = to_hf_dataset(data_dir, video)

    info = {
        "fps": fps,
        "video": video,
    }
    return hf_dataset, episode_data_index, info
