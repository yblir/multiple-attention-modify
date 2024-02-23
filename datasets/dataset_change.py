import os
import sys
import numpy as np
import cv2
import json
import torch
import torch.utils.data as data
import pickle
import random
from collections import OrderedDict
from pathlib2 import Path
from decord import VideoReader, cpu
import torchvision.transforms as transforms
from PIL import Image
from sklearn.utils import shuffle

from albumentations import Compose, RandomBrightnessContrast, RandomCrop, IAAAdditiveGaussianNoise, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, ISONoise, MultiplicativeNoise, Cutout, CoarseDropout, \
    MedianBlur, Blur, GlassBlur, MotionBlur, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, ToSepia, RandomShadow, RandomGamma, \
    Rotate, Resize, RandomContrast, RandomBrightness, RandomBrightnessContrast

from .albu import IsotropicResize, FFT, SR, DCT, CustomRandomCrop


class DeepfakeDataset(data.Dataset):
    def __init__(self,
                 # root,
                 # face_info_path,
                 # method='Deepfakes',
                 # compression='c23',
                 split='train',
                 num_segments=16,
                 # transform=None,
                 sparse_span=150,
                 # dense_sample=0,
                 # test_margin=1.3
                 ):
        """Dataset class for ffpp dataset.

        Args:
            split:train,test,val
            num_segments:每个视频抽帧数量
            其他参数都是前朝余孽,用不到, 懒得删了.
        """
        super().__init__()

        # self.root = root
        # self.face_info_path = face_info_path
        # self.method = method
        # self.compression = compression
        self.split = split
        self.num_segments = num_segments
        # self.transform = transform
        self.sparse_span = sparse_span
        # self.dense_sample = dense_sample
        # self.test_margin = test_margin
        # todo
        self.transform = self.create_transforms(380)
        self.transform2 = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])
        # assert self.compression in ['c23', 'c40']
        # assert self.method in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

        self.parse_dataset_info()

        # self.additional_targets = {}
        # for i in range(1, num_segments):
        #     self.additional_targets[f'image{i}'] = 'image'

    @staticmethod
    def create_transforms(size, mode="train"):
        if mode == "train":
            return Compose([
                # ImageCompression(quality_lower=40, quality_upper=100, p=0.1),
                OneOf([
                    MotionBlur(),
                    GaussianBlur(),
                    ImageCompression(quality_lower=40, quality_upper=100),
                ], p=0.3),
                HorizontalFlip(),
                # GaussNoise(p=0.3),
                OneOf([
                    IAAAdditiveGaussianNoise(),
                    GaussNoise(),
                ], p=0.3),
                ISONoise(p=0.3),
                MultiplicativeNoise(p=0.3),

                # 在增强前, 图片已经进行了不失真resize,所以下面的resize操作都不会工作
                # OneOf([
                #     IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                #     IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA,
                #                     interpolation_up=cv2.INTER_LINEAR),
                #     IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR,
                #                     interpolation_up=cv2.INTER_LINEAR),
                #     # CustomRandomCrop(size=size)
                # ], p=1),
                # Resize(height=size, width=size),
                # PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),

                OneOf([RandomBrightnessContrast(), RandomContrast(), RandomBrightness(), FancyPCA(),
                       HueSaturationValue()], p=0.5),
                OneOf([Cutout(), CoarseDropout()], p=0.05),

                ToGray(p=0.1),
                ToSepia(p=0.05),
                RandomShadow(p=0.05),
                RandomGamma(p=0.1),
                ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT,
                                 p=0.5),
                FFT(mode=0, p=0.05),
                DCT(mode=1, p=0.5)
            ])
        elif mode == "val" or mode == "test":
            return Compose([
                # IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                # PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
                DCT(mode=1, p=1)
            ])
        else:
            raise ValueError("mode must be in [train, val, test]")

    def parse_dataset_info(self):
        """Parse the video dataset information
        """
        # ========================================================================
        # ff++
        # json_data = json.load(open(self.split_json_path, 'r'))
        if self.split == "train":
            json_data = json.load(open("/mnt/e/DeepFakeDetection/datasets/FF++/splits/train.json", 'r'))
        else:
            json_data = json.load(open("/mnt/e/DeepFakeDetection/datasets/FF++/splits/val.json", 'r'))

        self.real_names = []
        self.fake_names = []
        for item in json_data:
            # 071_054, 071是原视频,把054中人脸截出来替换071中人脸
            self.real_names.extend([item[0], item[1]])
            self.fake_names.extend([f'{item[0]}_{item[1]}', f'{item[1]}_{item[0]}'])

        # self.dataset_info 存储的是视频名和该视频real/fake标签, 其中视频名可能是名称,也可能是视频路径,在ff++中存储是视频名
        self.dataset_info = [[x, 'real'] for x in self.real_names]
        # "FaceShift"
        for fake_style in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']:
            for x in self.fake_names:
                self.dataset_info.append([x, fake_style])
        # # ==============================================================================
        # # df-1.0 与ff++一样, 测试抽取文件名也与ff+相同
        # df1_real_names, df1_fake_names = [], []
        # df1_root_path = Path("")
        # for dir in ["end_to_end", "reenact_postprocess"]:
        #     for video_path in (df1_root_path / dir).rglob("*"):
        #         if video_path.suffix != ".mp4":
        #             continue
        #         df1_fake_names.append(str(video_path.name))
        # df1_info = [[x, "df1_fake"] for x in df1_fake_names]
        # self.dataset_info.extend(df1_info)
        # # ========================================================================
        # # dfdc
        # dfdc_real_names, dfdc_fake_names = [], []
        # dfdc_json_data = json.load(open("", 'r'))
        # for k, value in dfdc_json_data.items():
        #     if value["label"] == "REAL":
        #         dfdc_real_names.append(k)
        #     else:
        #         dfdc_fake_names.append(k)
        # dfdc_info = [[x, "dfdc_real"] for x in dfdc_real_names] + [[x, "dfdc_fake"] for x in dfdc_fake_names]
        # self.dataset_info.extend(dfdc_info)
        # # ========================================================================
        # # celeb-df,训练时,两个real文件夹混合在一起
        # celeb_real_names, celeb_fake_names = [], []
        # celeb_root_path = Path("")
        # for dir in ["Celeb-real", "Celeb-synthesis", "YouTube-real"]:
        #     for video_path in (celeb_root_path / dir).rglob("*"):
        #         if video_path.suffix != ".mp4":
        #             continue
        #         if dir in ["Celeb-real", "YouTube-real"]:
        #             celeb_real_names.append(str(video_path.name))
        #         else:
        #             celeb_fake_names.append(str(video_path.name))
        # celeb_info = [[x, "celeb_real"] for x in celeb_real_names] + [[x, "celeb_fake"] for x in celeb_fake_names]
        # self.dataset_info.extend(celeb_info)

        # ========================================================================
        # forgeryNet

        # ========================================================================

        # load face bounding box information.
        # 存储人脸框信息的pkl文件
        self.face_info = pickle.load(
                open("/mnt/e/DeepFakeDetection/multiple-attention-master/weights/ffpp_face_rects.pkl", 'rb'))

        print(f'{self.split} has {len(self.real_names)} real videos and '
              f'{len(self.fake_names)} fake videos, face_info has {len(self.dataset_info)}')

    def sample_indices_train(self, video_len, data_list):
        """Frame sampling strategy in training stage.

        Args:
            video_len (int): Video frame length.

        """
        base_idxs = np.array(range(video_len), np.int32)
        count = 0
        for i, item in enumerate(data_list):
            if not item or len(item) == 0:
                # print("=====================")
                base_idxs = np.delete(base_idxs, count)
                continue
            count += 1
        real_idxs = base_idxs

        # 每个视频切分为150段
        if self.sparse_span:
            base_idxs_index = np.linspace(0, len(base_idxs) - 1, self.sparse_span, dtype=np.int32)
            base_idxs = base_idxs[base_idxs_index]
        base_idxs_len = len(base_idxs)

        def over_sample_strategy(total_len):
            if total_len >= self.num_segments:
                offsets = np.sort(random.sample(range(total_len), self.num_segments))
            else:
                inv_ratio = self.num_segments // total_len
                offsets = []
                for idx in range(total_len):
                    offsets.extend([idx] * inv_ratio)
                tail = [total_len - 1] * (self.num_segments - len(offsets))
                offsets.extend(tail)
                offsets = np.asarray(offsets)
            return offsets

        # def dense_sample(total_len):
        #     # print(f'dense! total_len: {total_len}')
        #     if total_len > self.dense_sample:
        #         start_idx = np.random.randint(0, total_len - self.dense_sample)
        #         average_duration = self.dense_sample // self.num_segments
        #         # assert average_duration > 1
        #         offsets = np.multiply(list(range(self.num_segments)), average_duration) + \
        #                   np.random.randint(average_duration, size=self.num_segments)
        #         offsets += start_idx
        #     else:
        #         offsets = over_sample_strategy(total_len)
        #     # print(f'dense offsets: {offsets}')
        #     return offsets

        def non_dense_sample(total_len):
            average_duration = total_len // self.num_segments
            if average_duration > 1:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + \
                          np.random.randint(average_duration, size=self.num_segments)
            else:
                offsets = over_sample_strategy(total_len)
            return offsets

        # 不使用密集采样
        # if self.dense_sample:
        # offsets = dense_sample(base_idxs_len)
        # if random.random() < 0.5:
        #     offsets = dense_sample(base_idxs_len)
        # else:
        #     offsets = non_dense_sample(base_idxs_len)
        # else:
        #     offsets = non_dense_sample(base_idxs_len)

        offsets = non_dense_sample(base_idxs_len)
        return base_idxs[offsets].tolist(), real_idxs

    def sample_indices_test(self, video_len, data_list):
        """Frame sampling strategy in test stage.

        Args:
            video_len (int): Video frame count.

        """
        base_idxs = np.array(range(video_len), np.int32)
        count = 0
        for i, item in enumerate(data_list):
            if not item or len(item) == 0:
                # print("=====================")
                base_idxs = np.delete(base_idxs, count)
                continue
            count += 1
        if self.sparse_span:
            # base_idxs = np.linspace(0, video_len - 1, self.sparse_span, dtype=np.int)
            base_idxs_index = np.linspace(0, len(base_idxs) - 1, self.sparse_span, dtype=np.int32)
            base_idxs = base_idxs[base_idxs_index]
        base_idxs_len = len(base_idxs)

        # if self.dense_sample:
        #     start_idx = max(base_idxs_len // 2 - self.dense_sample // 2, 0)
        #     end_idx = min(base_idxs_len // 2 + self.dense_sample // 2, base_idxs_len)
        #     base_idxs = base_idxs[start_idx: end_idx]
        #     base_idxs_len = len(base_idxs)

        tick = base_idxs_len / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        offsets = base_idxs[offsets].tolist()

        return offsets

    # todo
    def get_enclosing_box(self, img_h, img_w, box):
        """Get the square-shape face bounding box after enlarging by a certain margin.

        Args:
            img_h (int): Image height.
            img_w (int): Image width.
            box (list): [x0, y0, x1, y1] format face bounding box.
            margin (int): The margin to enlarge.

        """
        x0, y0, x1, y1 = box
        wid, hig = x1 - x0, y1 - y0
        # max_size = max(w, h)
        #
        # cx = x0 + w / 2
        # cy = y0 + h / 2
        # x0 = cx - max_size / 2
        # y0 = cy - max_size / 2
        # x1 = cx + max_size / 2
        # y1 = cy + max_size / 2
        #
        # offset = max_size * (margin - 1) / 2
        # x0 = int(max(x0 - offset, 0))
        # y0 = int(max(y0 - offset, 0))
        # x1 = int(min(x1 + offset, img_w))
        # y1 = int(min(y1 + offset, img_h))

        x0 = max(0, x0 - wid // 2.5)
        y0 = max(0, y0 - wid // 1.5)
        x1 = min(img_w - 1, x1 + wid // 2.5)
        y1 = min(img_h - 1, y1 + wid // 2.5)

        return [x0, y0, x1, y1]

    @staticmethod
    def resize_shorter_side(img, min_length):
        """
        Resize the shorter side of img to min_length while
        preserving the aspect ratio.
        """
        ow, oh = img.size
        # oh, ow, _ = img.shape
        mult = 8
        if ow < oh:
            if ow == min_length and oh % mult == 0:
                return img, (ow, oh)
            w = min_length
            h = int(min_length * oh / ow)
        else:
            if oh == min_length and ow % mult == 0:
                return img, (ow, oh)
            h = min_length
            w = int(min_length * ow / oh)
        return img.resize((w, h), Image.BICUBIC), (w, h)
        # img = cv2.resize(img, (h, w))
        # return img

    def decode_selected_frames(self, vr, sampled_idxs, video_face_info_d):
        """Decode image frames from a given video on the fly.

        Args:
            vr (object):
                Decord VideoReader instance.
            sampled_idxs (list):
                List containing the frames to extract from the given video.
            video_face_info_d (dict):
                Dict containing the face bounding box information of each frame from the given video.
        """
        img_h, img_w, _ = vr[0].shape
        frames = vr.get_batch(sampled_idxs).asnumpy()

        # if self.split == 'train':
        #     margin = random.uniform(1.0, 1.5)
        # else:
        #     margin = self.test_margin

        imgs = []
        for idx in range(len(frames)):
            try:
                img = frames[idx]
                x0, y0, x1, y1 = self.get_enclosing_box(img_h, img_w, video_face_info_d[sampled_idxs[idx]])
                x0, y0, x1, y1 = [int(i) for i in (x0, y0, x1, y1)]
                img = img[y0:y1, x0:x1]
                img = Image.fromarray(img)
                # todo
                # img = self.resize_shorter_side(img, 380)[0]
                imgs.append(img)
            except Exception as e:
                print(e)
                print("no face pkl .....")
        return imgs

    @staticmethod
    def resize_image(image, size, letterbox_image=True):
        iw, ih = image.size
        w, h = size  # w=200, h=300
        if letterbox_image:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', size, (128, 128, 128))  # 新建一张image，第二个参数表示尺寸，第三个参数表示颜色
            # --------------------------------------------------#
            #   image.paste函数表示将一张图片覆盖到另一张图片的指定位置去
            #   a.paste(b, (50,50))   将b的左上顶点贴到a的坐标为（50，50）的位置，左上顶点为(0,0), b超出a的部分会被自动舍弃
            # ---------------------------------------------------#
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 不变形resize，两端填充灰边
            # new_image.paste(image, (0, 0))  # 不变形resize，一端填充灰边
        else:
            new_image = image.resize((w, h), Image.BICUBIC)
        return new_image

    # 509.mp4
    def __getitem__(self, index):
        video_name, video_label = self.dataset_info[index]
        # ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        # video_path = os.path.join(eval(f'self.{video_label}_video_dir'), video_name + '.mp4')
        if video_label == "real":
            video_path = f"/mnt/e/DeepFakeDetection/datasets/FF++/original_sequences/youtube/c23/videos/{video_name}.mp4"
        elif video_label in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']:
            video_path = f"/mnt/e/DeepFakeDetection/datasets/FF++/manipulated_sequences/{video_label}/c23/videos/{video_name}.mp4"
        elif video_label == "dfdc_real":
            pass
        elif video_label == "dfdc_fake":
            pass
        else:
            raise
        video_face_info_d = self.face_info[video_name.split('_')[0]]
        vr = VideoReader(video_path)
        video_len = min(len(vr), len(video_face_info_d))
        if video_name == "509":
            print("可算逮到你了!")
        if self.split == 'train':
            sampled_idxs, real_idxs = self.sample_indices_train(video_len, video_face_info_d[:video_len])
        else:
            sampled_idxs = self.sample_indices_test(video_len, video_face_info_d[:video_len])

        frames = self.decode_selected_frames(vr, sampled_idxs, video_face_info_d)
        # 加灰度条resize到380,380, 在转为numpy格式
        frames = [np.asarray(self.resize_image(item, (380, 380))) for item in frames]
        # make sure the augmentation parameter is applied the same on each frame.
        additional_targets = {}
        tmp_imgs = {"image": frames[0]}
        for i in range(1, len(frames)):
            additional_targets[f"image{i}"] = "image"
            tmp_imgs[f"image{i}"] = frames[i]
        self.transform.add_targets(additional_targets)

        frames = self.transform(**tmp_imgs)

        frames = OrderedDict(sorted(frames.items(), key=lambda x: x[0]))
        frames = list(frames.values())

        pre_imgs = []
        for item in frames:
            pre_imgs.append(self.transform2(item))
        pre_imgs = torch.stack(pre_imgs)  # T, C, H, W

        video_label_int = 0 if video_label == 'real' else 1

        return pre_imgs, torch.tensor([video_label_int] * len(pre_imgs))

    def __len__(self):
        return len(self.dataset_info)
