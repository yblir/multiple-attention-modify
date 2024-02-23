# from datasets.data import FF_dataset, Celeb_test, deeperforensics_dataset, dfdc_dataset
# from datasets.dataset import DeepfakeDataset
import cv2

from models.MAT import MAT
import pickle
import json
import torch
import re
import os
from sklearn.metrics import roc_auc_score as AUC
import numpy as np
from copy import deepcopy


def load_model(name):
    # with open('runs/%s/config.pkl' % name, 'rb') as f:
    #     config = pickle.load(f)
    # net = MAT(**config.net_config)
    net = MAT()
    return None, net


def find_best_ckpt(name, last=False):
    if last:
        return len(os.listdir('checkpoints/%s' % name)) - 1
    with open('runs/%s/train.log' % name) as f:
        lines = f.readlines()[1::2]
    accs = [float(re.search('acc\\:(.*)\\,', a).groups()[0]) for a in lines]
    best = accs.index(max(accs))
    return best


def acc_eval(labels, preds):
    labels = np.array(labels)
    preds = np.array(preds)
    thres = 0.5
    acc = np.mean((preds >= thres) == labels)
    return thres, acc


def test_eval(net, testset):
    # test_dataset = DeepfakeDataset(phase='test', **setting)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=setting['imgs_per_video'], shuffle=False,
    #                                           pin_memory=True, num_workers=8)
    for i, (X, y) in enumerate(test_loader):
        testset[i].append([])
        if -1 in y:
            testset[i].append(0.5)
            continue
        X = X.to('cuda', non_blocking=True)
        with torch.no_grad():
            for x in torch.split(X, 20):
                logits = net(x)
                pred = torch.nn.functional.softmax(logits, dim=1)[:, 1]
                testset[i][-1] += pred.cpu().numpy().tolist()
        testset[i].append(np.mean(testset[i][-1]))


# ['ff-all', 'celeb', 'deeper']
def all_eval(name, ckpt=None, test_sets=['ff-all', 'celeb', 'deeper']):
    config, net = load_model(name)
    # setting = config.val_dataset
    # codec = setting['datalabel'].split('-')[2]
    # setting['min_frames'] = 100
    # setting['frame_interval'] = 5
    # setting['imgs_per_video'] = 20
    # setting['datalabel'] = 'ff-all-%s' % codec
    # list_of_files = os.listdir('checkpoints/%s' % name)
    # list_of_files = list(map(lambda x: int(x[5:-4]), list_of_files))
    # if ckpt is None:
    #     ckpt = find_best_ckpt(name)
    # if ckpt < 0:
    #     ckpt = max(list_of_files) + 1 + ckpt

    # state_dict = torch.load('checkpoints/%s/ckpt_%s.pth' % (name, ckpt))['state_dict']
    state_dict = torch.load("/mnt/e/DeepFakeDetection/multiple-attention-master/weights/pretrained/ff_c23.pth")
    net.load_state_dict(state_dict, strict=False)
    # os.makedirs('evaluations/%s' % name, exist_ok=True)
    net.eval()
    net.cuda()
    # img = cv2.imread(r"C:\Users\Administrator\Desktop\test_imgs\1a7ba4d4b8d144edb590d807c3878154-1.jpg")
    img = cv2.imread("/mnt/C/Users/Administrator/Desktop/test_imgs/1a7ba4d4b8d144edb590d807c3878154-1.jpg")
    img = cv2.resize(img, (224, 224))
    img = img[..., ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    input_data = torch.from_numpy(img)
    input_data.unsqueeze(0)
    res = net(input_data)
    print(res)
    # result = dict()
    # if 'ff-all' in test_sets:
    #     testset = []
    #     for i in ['Origin', 'Deepfakes', 'NeuralTextures', 'FaceSwap', 'Face2Face']:
    #         testset += FF_dataset(i, codec, 'test')
    # test_eval(net, setting, testset)


if __name__ == "__main__":
    # for name in os.listdir('checkpoints'):
    #     try:
    #         all_eval(name)
    #     except:
    #         pass
    all_eval(1)
