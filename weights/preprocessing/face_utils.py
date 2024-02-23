import os, sys, time
sys.path.insert(0, "retinaface")
import cv2
import numpy as np
import torch
from skimage.transform import SimilarityTransform
# from data import cfg_re50
# from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode, decode_landm
from models.retinaface import RetinaFace
ARCFACE_SRC = np.array([[
    [122.5, 141.25],
    [197.5, 141.25],
    [160.0, 178.75],
    [137.5, 225.25],
    [182.5, 225.25]
]], dtype=np.float32)



def estimate_norm(lmk,image_size,zoom_in=1):
    assert lmk.shape == (5, 2)
    tform = SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = np.inf
    src = ARCFACE_SRC*(image_size/320)
    src=src/zoom_in-image_size/2*(1/zoom_in-1)
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
    M = tform.params[0:2, :]
    results = np.dot(M, lmk_tran.T)
    results = results.T
    error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
    if error < min_error:
        min_error = error
        min_M = M
        min_index = i
    return min_M, min_index

def norm_crop(img, landmark, image_size=320,zoom_in=1):
    M, pose_index = estimate_norm(landmark,image_size,zoom_in)
    warped = cv2.warpAffine(img, M, (image_size, image_size),flags=cv2.INTER_CUBIC, borderValue=0.0)
    return warped

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class FaceDetector:
    def __init__(self, device="cuda", confidence_threshold=0.9,nms_thresh = 0.4):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.nms_thresh = nms_thresh
        self.cfg = cfg = cfg_re50
        self.variance = cfg["variance"]
        cfg["pretrain"] = False
        self.net = RetinaFace(cfg=cfg, phase="test").to(device).eval()
        self.decode_param_cache = {}
        self.off=torch.Tensor([104, 117, 123]).to(device).view(3,1,1)

    def load_checkpoint(self, path):
        checkpoint_=torch.load(path)
        checkpoint_={i[7:]:checkpoint_[i] for i in checkpoint_}
        self.net.load_state_dict(checkpoint_)

    def decode_params(self, height, width):
        cache_key = (height, width)
        try:
            result= self.decode_param_cache[cache_key]
            return tuple(i.cuda() for i in result)
        except KeyError:
            priorbox = PriorBox(self.cfg, image_size=(height, width))
            priors = priorbox.forward()
            prior_data = priors.data.to(self.device)
            scale = torch.Tensor([width, height] * 2).to(self.device)
            scale1 = torch.Tensor([width, height] * 5).to(self.device)
            result = (prior_data, scale, scale1)
            self.decode_param_cache[cache_key] = tuple(i.cpu() for i in result)
            return result


    def detect(self, imgs):
        imgs=np.float32(np.stack(imgs))
        prior_data, scale, scale1 = self.decode_params(*imgs.shape[1:3])
        imgs=np.ascontiguousarray(imgs.transpose(0,3,1,2))
        imgs=torch.from_numpy(imgs).to(self.device)
        imgs -= self.off
        loc, conf, landms_ = self.net(imgs) 
        result=[]
        for i in range(imgs.shape[0]):
            boxes = decode(loc[i], prior_data, self.variance)
            boxes = boxes * scale
            boxes = boxes.cpu().numpy()
            scores = conf[i].cpu().numpy()[:, 1]
            landms = decode_landm(landms_[i], prior_data, self.variance)
            landms = landms * scale1
            landms = landms.cpu().numpy()
            inds = scores >self.confidence_threshold
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, self.nms_thresh)
            dets = dets[keep, :]
            landms = landms[keep]
            dets = np.concatenate((dets, landms), axis=1)
            result.append(dets)
        return result