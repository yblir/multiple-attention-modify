import os
import sys

import cv2
import argparse
import numpy as np
import pickle
from loguru import logger

np.set_printoptions(threshold=np.inf)


class SCRFD():
    def __init__(self, onnxmodel, confThreshold=0.5, nmsThreshold=0.5):
        self.inpWidth = 640
        self.inpHeight = 640
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.net = cv2.dnn.readNet(onnxmodel)
        self.keep_ratio = True
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2

    def resize_image(self, srcimg):
        padh, padw, newh, neww = 0, 0, self.inpHeight, self.inpWidth
        if self.keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.inpHeight, int(self.inpWidth / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                padw = int((self.inpWidth - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, padw, self.inpWidth - neww - padw, cv2.BORDER_CONSTANT,
                                         value=0)  # add border
            else:
                newh, neww = int(self.inpHeight * hw_scale) + 1, self.inpWidth
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                padh = int((self.inpHeight - newh) * 0.5)
                img = cv2.copyMakeBorder(img, padh, self.inpHeight - newh - padh, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_AREA)
        return img, newh, neww, padh, padw

    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1])
            y1 = y1.clamp(min=0, max=max_shape[0])
            x2 = x2.clamp(min=0, max=max_shape[1])
            y2 = y2.clamp(min=0, max=max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def distance2kps(self, points, distance, max_shape=None):
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            if max_shape is not None:
                px = px.clamp(min=0, max=max_shape[1])
                py = py.clamp(min=0, max=max_shape[0])
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    def detect(self, srcimg):
        img, newh, neww, padh, padw = self.resize_image(srcimg)
        blob = cv2.dnn.blobFromImage(img, 1.0 / 128, (self.inpWidth, self.inpHeight), (127.5, 127.5, 127.5),
                                     swapRB=True)
        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs_ = self.net.forward(self.net.getUnconnectedOutLayersNames())
        outs=[]
        for i in range(3):
            outs.extend([outs_[i],outs_[i+3],outs_[i+6]])
        # inference output
        scores_list, bboxes_list, kpss_list = [], [], []
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outs[idx * self.fmc][0]
            # print("outs[0].shape=", outs[0].shape, len(outs), min(outs[0]), max(outs[0]), outs[0][:10])
            bbox_preds = outs[idx * self.fmc + 1][0] * stride
            # print(bbox_preds[0].shape, len(bbox_preds), min(bbox_preds[0]), max(bbox_preds[0]), bbox_preds[0][:10])
            # sys.exit()
            kps_preds = outs[idx * self.fmc + 2][0] * stride
            height = blob.shape[2] // stride
            width = blob.shape[3] // stride
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if self._num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))

            pos_inds = np.where(scores >= self.confThreshold)[0]
            # print(anchor_centers.shape, bbox_preds.shape)
            bboxes = self.distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            kpss = self.distance2kps(anchor_centers, kps_preds)
            # kpss = kps_preds
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

        scores = np.vstack(scores_list).ravel()
        # bboxes = np.vstack(bboxes_list) / det_scale
        # kpss = np.vstack(kpss_list) / det_scale
        bboxes = np.vstack(bboxes_list)
        kpss = np.vstack(kpss_list)
        bboxes[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]
        ratioh, ratiow = srcimg.shape[0] / newh, srcimg.shape[1] / neww
        bboxes[:, 0] = (bboxes[:, 0] - padw) * ratiow
        bboxes[:, 1] = (bboxes[:, 1] - padh) * ratioh
        bboxes[:, 2] = bboxes[:, 2] * ratiow
        bboxes[:, 3] = bboxes[:, 3] * ratioh
        kpss[:, :, 0] = (kpss[:, :, 0] - padw) * ratiow
        kpss[:, :, 1] = (kpss[:, :, 1] - padh) * ratioh
        indices = cv2.dnn.NMSBoxes(bboxes.tolist(), scores.tolist(), self.confThreshold, self.nmsThreshold)

        res = []
        for i in indices:
            # i = i[0]
            xmin, ymin, xamx, ymax = bboxes[i, 0], bboxes[i, 1], \
                                     bboxes[i, 0] + bboxes[i, 2], bboxes[i, 1] + bboxes[i, 3]
            res.append([[xmin, ymin, xamx, ymax], kpss[i]])

        return res


def get_max_face_box(res):
    max_value = None
    max_idx = None

    for idx, (box, _) in enumerate(res):
        value = (box[2] - box[0]) * (box[3] - box[1])
        if (max_value is None or value > max_value):
            max_value = value
            max_idx = idx
    output_box = res[max_idx][0]
    kpss = res[max_idx][1]

    return output_box, kpss


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--imgpath', type=str, default='selfie.jpg', help='image path')
    # parser.add_argument('--onnxmodel', default='weights/scrfd_10g_kps.onnx', type=str,
    #                     choices=['weights/scrfd_500m_kps.onnx', 'weights/scrfd_2.5g_kps.onnx',
    #                              'weights/scrfd_10g_kps.onnx'], help='onnx model')
    # parser.add_argument('--confThreshold', default=0.5, type=float, help='class confidence')
    # parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    # args = parser.parse_args()

    # mynet = SCRFD(args.onnxmodel, confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold)
    mynet = SCRFD('weights/scrfd_10g_kps.onnx', confThreshold=0.5, nmsThreshold=0.5)
    img = cv2.imread("s_l.jpg")
    # print(type(img), img.shape)
    res = mynet.detect(img)

    max_value = None
    max_idx = None

    for idx, (box, _) in enumerate(res):
        value = (box[2] - box[0]) * (box[3] - box[1])
        if (max_value is None or value > max_value):
            max_value = value
            max_idx = idx
    output_box, kpss = res[max_idx][0]
    # kpss = res[max_idx][1]

    xmin, ymin, xamx, ymax = output_box
    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xamx), int(ymax)), (0, 0, 255), thickness=2)
    for j in range(5):
        cv2.circle(img, (int(kpss[j, 0]), int(kpss[j, 1])), 1, (0, 255, 0), thickness=-1)
    cv2.imshow("aa", img)
    cv2.waitKey()

    logger.success("success !")
