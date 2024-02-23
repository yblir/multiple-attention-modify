import os
import time
import logging
import warnings
import numpy
import torch
import torch.nn as nn
# import torch.multiprocessing as mp
import torch.nn.functional as F
from pathlib2 import Path
from torch.utils.data import DataLoader
# import torch.distributed as dist
from models.MAT import MAT
# from datasets.dataset import DeepfakeDataset
# from AGDA import AGDA
# import cv2
# from utils import dist_average, ACC
from config import train_config
# from pprint import pprint

import numpy as np
from scrfd_opencv_gpu.scrfd_face_detect import SCRFD, get_max_face_box
from decord import VideoReader
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2
from loguru import logger
from albumentations import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print(torch.cuda.is_available())

torch.cuda.set_device(0)
transform = alb.Compose([
    # alb.Resize(224, 224),
    alb.Resize(380, 380),
    alb.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # alb.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # HueSaturationValue(p=0.5),
    # ToGray(p=0.1),
    ToTensorV2(),
], additional_targets={})
mynet = SCRFD('scrfd_opencv_gpu/weights/scrfd_10g_kps.onnx', confThreshold=0.5, nmsThreshold=0.5)

config = train_config("ff-cc", ["efficientnet-b4"])
model = MAT(**config.net_config)
# print(net)
# for k, v in net.state_dict().items():
#     print(k, ",", v.shape)

weight = torch.load("weights/pretrained/ff_c23.pth", map_location="cpu")
model.load_state_dict(weight["state_dict"], strict=False)
# for k, v in weight["state_dict"].items():
#     print(k, ",", v.shape)
# print(model)
model.cuda()
model.eval()


# from torch.utils.tensorboard import SummaryWriter
# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)
# # GPU settings
# assert torch.cuda.is_available()


# torch.autograd.set_detect_anomaly(True)
def load_state(net, ckpt):
    sd = net.state_dict()
    nd = {}
    goodmatch = True
    for i in ckpt:
        if i in sd and sd[i].shape == ckpt[i].shape:
            nd[i] = ckpt[i]
            # print(i)
        else:
            print('fail to load %s' % i)
            goodmatch = False
    net.load_state_dict(nd, strict=False)
    return goodmatch


def video_preprocess(video_path):
    """
    视频抽帧,之后返回抽帧图片人脸numpy矩阵
    Args:
        video_path:
    Returns: list
    """
    vr = VideoReader(video_path)
    if len(vr) < 8:
        raise ValueError("the len of frames fewer than 8")

    # 从视频中提取16帧,用于模型预测
    base_idxs, sampled_idxs = sample_indices_test(vr)

    # 调取服务对抽帧图片提取人脸框, 检测人脸坐标不为空,否则再随机从索引范围内挑选人脸非空图片
    # frames list, 每个元素是人脸框rgb 矩阵

    frames = get_faces_from_selected_frames(vr, base_idxs, sampled_idxs)

    # 对16帧图片做增强
    additional_targets = {}
    tmp_imgs = {"image": frames[0]}
    for i in range(1, len(frames)):
        additional_targets[f"image{i}"] = "image"
        tmp_imgs[f"image{i}"] = frames[i]
    transform.add_targets(additional_targets)

    frames = transform(**tmp_imgs)

    # 排列字典frames 关键字,以从小到大形式排列,保证各帧时间连续性
    frames = [frames[i] for i in sorted(frames, reverse=False)]

    frames = torch.stack(frames)  # T, C, H, W
    # process_img = frames.view(-1, frames.size(2), frames.size(3)).contiguous()  # TC, H, W

    return frames


def sample_indices_test(vr):
    """Frame sampling strategy in test stage.

    Args:
        video_len (int): Video frame count.

    """
    video_len = len(vr)

    base_idxs = np.linspace(0, video_len - 1, 150, dtype=np.int)
    base_idxs_len = len(base_idxs)

    tick = base_idxs_len / float(16)
    offsets = np.array([int(tick / 2.0 + tick * x) for x in range(16)])
    offsets = base_idxs[offsets].tolist()

    return base_idxs, offsets


def get_enclosing_box(img_h, img_w, box, margin):
    """Get the square-shape face bounding box after enlarging by a certain margin.

    Args:
        img_h (int): Image height.
        img_w (int): Image width.
        box (list): [x0, y0, x1, y1] format face bounding box.
        margin (float): The margin to enlarge.

    """
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    max_size = max(w, h)

    cx = x0 + w / 2
    cy = y0 + h / 2
    x0 = cx - max_size / 2
    y0 = cy - max_size / 2
    x1 = cx + max_size / 2
    y1 = cy + max_size / 2

    offset = max_size * (margin - 1) / 2
    x0 = int(max(x0 - offset, 0))
    y0 = int(max(y0 - offset, 0))
    x1 = int(min(x1 + offset, img_w))
    y1 = int(min(y1 + offset, img_h))

    return [x0, y0, x1, y1]


def get_faces_from_selected_frames(vr, base_idxs, sampled_idxs):
    img_h, img_w, _ = vr[0].shape
    frames = vr.get_batch(sampled_idxs).asnumpy()
    imgs = []
    for idx in range(len(frames)):
        img = frames[idx]
        try:
            res = mynet.detect(img)
            if not res:
                raise
        except Exception as _:
            raise ValueError("face detect failure")

        output_box, kpss = get_max_face_box(res)
        x0, y0, x1, y1 = output_box

        x0, y0, x1, y1 = get_enclosing_box(img_h, img_w, [x0, y0, x1, y1], 1.3)
        img = img[y0:y1, x0:x1]

        imgs.append(img)

    return imgs


if __name__ == '__main__':
    # video_path = r"E:\DeepFakeDetection\datasets\FF++\manipulated_sequences\Deepfakes\c23\videos\000_003.mp4"
    # video_path = r"/mnt/e/DeepFakeDetection/datasets/FF++/manipulated_sequences/Deepfakes/c23/videos/000_003.mp4"
    video_path = r"/mnt/e/DeepFakeDetection/datasets/FF++/original_sequences/youtube/c23/videos/000.mp4"
    # video_path=r"/mnt/c/Users/Administrator/Desktop/app_make/xiulian_fake/64e81cf3c85a55003dd3d58a_6e9b9e40973cc192c3c958861ade4740.mp4"
    # video_path = r"/mnt/c/Users/Administrator/Desktop/app_make/doupai_real/1692756725339.mp4"
    # video_path="/mnt/e/interesting/PSdetector-master/FALdetector-master/test_data/app_fake_videos/64ec5b8d72df4f00363fa93b_e6b72e5e015fe2c49e5af04a97dfec53.mp4"
    # video_path = "/mnt/e/DeepFakeDetection/forgeryNet_self/02e544a73a5dd5b20a830d2b578bda00/video.mp4"
    total_fake = 0
    # root_path="/mnt/c/Users/Administrator/Desktop/app_make/xiulian_fake"
    # root_path = "/mnt/e/DeepFakeDetection/datasets/FF++/original_sequences/youtube/c23/videos"
    # root_path="/mnt/e/DeepFakeDetection/datasets/FF++/manipulated_sequences/Deepfakes/c23/videos"
    root_path = "/mnt/e/DeepFakeDetection/datasets/bilibli_clean_scence_have_one_face"
    for video_path in Path(root_path).iterdir():
        try:
            images = video_preprocess(str(video_path))
        except Exception as e:
            logger.error(e)
            continue

        count = 0
        # images = torch.stack(imgs, dim=0)
        with torch.no_grad():
            images = images.cuda().float()
            logits = model(images)
            # real_probs = torch.nn.functional.softmax(logits, dim=1)[:, 1]
            # real_probs = real_probs.cpu().numpy()

            real_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        # real_probs = torch.sigmoid(logits)[:, 1].cpu().numpy()
        fake_count = len(real_probs[real_probs > 0.5])
        if fake_count >= 8:
            total_fake += 1
        output_prob = [round(item, 3) for item in real_probs]
        logger.info(f"{video_path.name},fake_count={fake_count},{output_prob}")

    logger.success(f"total_fake={total_fake}")
