import os
import random
import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr
from torchvision import transforms
import pandas as pd

from . import utils as utils
from .transform import create_random_augment

import decord
from decord import VideoReader
from decord import cpu

import math 

def get_video_container(path_to_vid, multi_thread_decode=False, backend="decord"):
    """
    Given the path to the video, return the pyav video container.
    Args:
        path_to_vid (str): path to the video.
        multi_thread_decode (bool): if True, perform multi-thread decoding.
        backend (str): decoder backend, options include `pyav` and
            `torchvision`, default is `pyav`.
    Returns:
        container (container): video container.
    """

    if backend == "decord":
        container = VideoReader(path_to_vid, ctx=cpu(0))
        decord.bridge.set_bridge('torch')
        return container
    else:
        raise NotImplementedError("Unknown backend {}".format(backend))


def get_start_end_idx(
    video_size, clip_size, clip_idx, num_clips, use_offset=False
):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        if use_offset:
            if num_clips == 1:
                # Take the center clip if num_clips is 1.
                start_idx = math.floor(delta / 2)
            else:
                # Uniformly sample the clip with the given index.
                start_idx = clip_idx * math.floor(delta / (num_clips - 1))
        else:
            # Uniformly sample the clip with the given index.
            start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx

def decode(
    container,
    sampling_rate,
    num_frames,
    clip_idx=-1,
    num_clips=10,
    backend="pyav",
    max_spatial_scale=0,
    use_offset=False,
    sparse=False,
    total_frames=None,
    start_index=0
):
    """
    Decode the video and perform temporal sampling.
    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames).
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal
            sampling. If clip_idx is larger than -1, uniformly split the
            video to num_clips clips, and select the
            clip_idx-th video clip.
        num_clips (int): overall number of clips to uniformly
            sample from the given video.
        video_meta (dict): a dict contains VideoMetaData. Details can be find
            at `pytorch/vision/torchvision/io/_video_opt.py`.
        target_fps (int): the input video may have different fps, convert it to
            the target video fps before frame sampling.
        backend (str): decoding backend includes `pyav` and `torchvision`. The
            default one is `pyav`.
        max_spatial_scale (int): keep the aspect ratio and resize the frame so
            that shorter edge size is max_spatial_scale. Only used in
            `torchvision` backend.
    Returns:
        frames (tensor): decoded frames from the video.
    """
    # Currently support two decoders: 1) PyAV, and 2) TorchVision.
    assert clip_idx >= -1, "Not valied clip_idx {}".format(clip_idx)
    try:
        if backend == "decord":
            frames = container
        else:
            raise NotImplementedError(
                "Unknown decoding backend {}".format(backend)
            )
    except Exception as e:
        print("Failed to decode by {} with exception: {}".format(backend, e))
        return None

    # Return None if the frames was not decoded successfully.
    if backend == "decord":
        if frames is None:
            return None

    if backend == "decord":
        if not sparse:
            clip_sz = sampling_rate * num_frames
            start_idx, end_idx = get_start_end_idx(
                len(frames),
                clip_sz,
                clip_idx,
                num_clips,
                use_offset=use_offset,
            )
            index = torch.linspace(start_idx, end_idx, num_frames)
            index = torch.clamp(index, 0, len(frames) - 1).long()
            # tmp_frames = [frames[i.item()] for i in index]
            frames = frames.get_batch(index)
    return frames



class Activitynetqa(torch.utils.data.Dataset):
    """
    Activitynetqa video loader. Construct the Activitynetqa video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, mode, num_retries=1, path_to_datadir=None):

        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Activitynetqa".format(mode)
        self.mode = mode
        self.path_to_datadir = path_to_datadir

        # self._video_meta = {}
        self._num_retries = num_retries

        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = 1

        self._construct_loader()
        self.aug = False
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0

        if self.mode == "train":
            self.aug = True

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.path_to_datadir, "{}.csv".format(self.mode)
        )
        assert g_pathmgr.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        df = pd.read_csv(path_to_file)

        self._path_to_videos = df['video_name'].to_list()
        self._labels = df['answer'].to_list()
        self._question = df['question'].to_list()
        self._spatial_temporal_idx = [0]
        for idx in range(self._num_retries - 1):
            self._path_to_videos = self._path_to_videos *2
            self._labels = self._labels *2
            self._question = self._question *2
            self._spatial_temporal_idx.append(idx)

        self.answer_weight = {}
        for answer in self._labels:
            if answer in self.answer_weight.keys():
                self.answer_weight[answer] += 1/len(self._labels)
            else:
                self.answer_weight[answer] = 1/len(self._labels)




        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load Activitynetqa split {} from {}".format(
            self._split_idx, path_to_file
        )
        

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = 256
            max_scale = 320
            crop_size = 224

        elif self.mode in ["val", "test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // 1
            )
            spatial_sample_index = (1)

            min_scale, max_scale, crop_size = ([224] * 3)
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        sampling_rate = utils.get_random_sampling_rate(
            0,
            16,
        )
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.

        video_container = None
        try:
            video_container = get_video_container(
                self._path_to_videos[index] + '.mp4',
                'decord',
            )
        except Exception as e:
            print('bad')
        # Select a random video if the current video was not able to access. 
        
        frames = decode(
            video_container,
            sampling_rate,
            8,
            temporal_sample_index,
            1,
            backend='decord',
            max_spatial_scale=min_scale,
            use_offset=True,
        )

        # If decoding failed (wrong format, video is too short, and etc),
        # select another video.

        if self.aug:
            frames = self._aug_frame(
                frames,
                spatial_sample_index,
                min_scale,
                max_scale,
                crop_size,
            )


        label = self._labels[index]
        question = self._question[index]
        weight = [self.answer_weight[label]]
        return frames, question, label, weight
    
    def _aug_frame(
        self,
        frames,
        spatial_sample_index,
        min_scale,
        max_scale,
        crop_size,
    ):
        aug_transform = create_random_augment(
            input_size=(frames.size(1), frames.size(2)),
            auto_augment='rand-m7-n4-mstd0.5-inc1',
            interpolation='bicubic',
        )
        # T H W C -> T C H W.
        frames = frames.permute(0, 3, 1, 2)
        list_img = self._frame_to_list_img(frames)
        list_img = aug_transform(list_img)
        frames = self._list_img_to_frames(list_img)
        frames = frames.permute(0, 2, 3, 1)

        frames = utils.tensor_normalize(
            frames, [0.45, 0.45, 0.45], [0.225, 0.225, 0.225]
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )
        relative_scales = (
            None if (self.mode not in ["train"] or len(scl) == 0) else scl
        )
        relative_aspect = (
            None if (self.mode not in ["train"] or len(asp) == 0) else asp
        )
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=True,
            inverse_uniform_sampling=False,
            aspect_ratio=relative_aspect,
            scale=relative_scales,
            motion_shift=False
            if self.mode in ["train"]
            else False,
        )


        return frames

    def _frame_to_list_img(self, frames):
        img_list = [
            transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))
        ]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
