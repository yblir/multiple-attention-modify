# from face_utils import FaceDetector, norm_crop
# import os
# import sys
import itertools
import json
import os
import cv2
import pickle
import requests
import numpy as np
from PIL import Image
import multiprocessing


def crop_aligned(img, landmarks, aligned_image_size, zoom_in):
    aligned = norm_crop(img, landmarks, aligned_image_size, zoom_in)
    aligned = Image.fromarray(aligned[:, :, ::-1])
    return aligned


def crop_unalinged(img, box, unaligned_padding, unaligned_image_size):
    img_cent = ((box[:2] + box[2:])[::-1] / 2).astype(np.int)
    h_max = min(img_cent[0], img.shape[0] - img_cent[0])
    v_max = min(img_cent[1], img.shape[1] - img_cent[1])
    rr = ((box[2:] - box[:2])[::-1] / 2 * (1 + unaligned_padding)).astype(np.int)
    h_ = min(h_max, rr[0])
    v_ = min(v_max, rr[1])
    imw = img[img_cent[0] - h_:img_cent[0] + h_, img_cent[1] - v_:img_cent[1] + v_, ::-1]
    img_crop = Image.fromarray(cv2.resize(imw, unaligned_image_size))
    return img_crop


def extract_face(input_file, aligned_output_dir='', unaligned_output_dir='', aligned_image_size=320, zoom_in=1,
                 unaligned_padding=0.2, unaligned_image_size=(380, 380), gt={}):
    ##gt: input checked list of array(15)
    reader = cv2.VideoCapture(input_file)
    counter = 0
    if aligned_output_dir:
        os.makedirs(aligned_output_dir, exist_ok=True)
    if unaligned_output_dir:
        os.makedirs(unaligned_output_dir, exist_ok=True)
    for idx in itertools.count():
        success, img = reader.read()
        if not success:
            break
        if idx >= len(gt):
            break
        det = gt[idx]
        if det is None:
            continue
        box = det[:4]
        landmarks = det[5:].reshape(5, 2).astype('int')
        try:
            if aligned_output_dir:
                aligned = crop_aligned(img, landmarks, aligned_image_size, zoom_in)
                out_path = os.path.join(aligned_output_dir, "%03d.png" % idx)
                aligned.save(out_path)
            if unaligned_output_dir:
                img_crop = crop_unalinged(img, box, unaligned_padding, unaligned_image_size)
                out_path = os.path.join(unaligned_output_dir, "%03d.png" % idx)
                img_crop.save(out_path)
            counter += 1
        except Exception as e:
            print(e)
            print(input_file, idx)
    return counter


def subworker(i):
    return {'num': i['num'], 'gt': extract_face(**i['para'])}


def worker(target, threads, num_batch=100):
    pool = multiprocessing.Pool(threads)
    while True:
        r = requests.get(target, params={'num': str(num_batch)})
        data = pickle.loads(r.content)
        if not data:
            pool.close()
            return
        q = pool.map(subworker, data)
        requests.post(target, data=pickle.dumps(q))


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn',force=True)
    # host = sys.argv[1]
    # threads = int(sys.argv[2])
    host = "http://10.1.203.205:10087/"
    threads = 2
    worker(host, threads)
