import os
import time
# import logging
import warnings
import numpy
import torch
import torch.nn as nn
import torch.multiprocessing as mp
# import torch.nn.functional as F
# import yaml
# from albumentations import OneOf, IAAAdditiveGaussianNoise, GaussNoise
# from torch.utils.data import DataLoader
import torch.distributed as dist
# from torchvision.transforms import transforms, GaussianBlur
# from skimage import transform as trans
from models.MAT import MAT
from datasets.dataset_change import DeepfakeDataset
from AGDA import AGDA
import cv2
from utils import dist_average, ACC, compute_metrics
from config import train_config
from omegaconf import OmegaConf
# import torch.utils.data as data
# import albumentations as alb
# from albumentations.pytorch.transforms import ToTensorV2
# sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', '..'))
from datasets.base_dataloader import create_base_dataloader
# from datasets.base_transform import create_base_transforms
# import torchvision.transforms as transforms
from datasets.cli_utils import get_params
# from PIL import Image
from sklearn.utils import shuffle
from logger_util import MyLogger

# from albumentations import *

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")
logger = MyLogger(log_level="INFO", bool_std=True, bool_file=True,
                  log_file_path='./log/multiple-attention.log').get_logger()
# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)
# GPU settings
assert torch.cuda.is_available()

args = get_params()


def get_dataloader(args, split):
    """Set dataloader.

    Args:
        args (object): Args load from get_params function.
        split (str): One of ['train', 'test']
        batch_size:
    """
    dataset_cfg = getattr(args, split).dataset
    batch_size = getattr(args, "batch_size")
    num_works = getattr(args, split).num_workers
    # num_segments = getattr(args, split).num_segments
    # num_segments: 每个视频抽帧数量, 参数在train/test模式下数据集参数中配置
    # transform = create_base_transforms(nums=dataset_cfg["params"]["num_segments"], split=split)
    dataset_params = OmegaConf.to_container(dataset_cfg.params, resolve=True)
    # dataset_params['transform'] = transform

    _dataset = eval(dataset_cfg.name)(**dataset_params)

    _dataloader = create_base_dataloader(batch_size=batch_size, dataset=_dataset, num_works=num_works, split=split)

    return _dataloader


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

# todo 当前文件用于模型调试
# def main_worker(local_rank, world_size, rank_offset, config):
def main_worker(config):
    local_rank = 0
    rank = 0
    # if rank == 0:
    #     writer = SummaryWriter("./log")
    # logging.basicConfig(
    #         filename=os.path.join('runs', config.name, 'train.log'),
    #         filemode='a',
    #         format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
    #         level=logging.INFO)
    # dist.init_process_group(backend='nccl', init_method=config.url, world_size=world_size, rank=rank)
    # dist.init_process_group(backend='nccl', init_method="env://", world_size=1, rank=local_rank)
    # if rank==0:
    #     try:
    #         os.remove('/tmp/.pytorch_distribute')
    #     except:
    #         pass
    # numpy.random.seed(1234567)
    # torch.manual_seed(1234567)
    # torch.cuda.manual_seed(1234567)
    torch.cuda.set_device(0)
    # train_dataset = DeepfakeDataset(phase='train', **config.train_dataset)
    train_loader = get_dataloader(args, 'train')
    validate_loader = get_dataloader(args, "test")
    # validate_dataset = DeepfakeDataset(phase='test', **config.val_dataset)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # validate_sampler = torch.utils.data.distributed.DistributedSampler(validate_dataset)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler,
    #                                            pin_memory=True, num_workers=config.workers)
    # validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=config.batch_size,
    #                                               sampler=validate_sampler, pin_memory=True, num_workers=config.workers)
    # logs = {}
    start_epoch = 0
    net = MAT(**config.net_config)
    # 没有使用冻结模块
    # for i in config.freeze:
    #     if 'backbone' in i:
    #         net.net.requires_grad_(False)
    #     elif 'attention' in i:
    #         net.attentions.requires_grad_(False)
    #     elif 'feature_center' in i:
    #         net.auxiliary_loss.alpha = 0
    #     elif 'texture_enhance' in i:
    #         net.texture_enhance.requires_grad_(False)
    #     elif 'fcs' in i:
    #         net.projection_local.requires_grad_(False)
    #         net.project_final.requires_grad_(False)
    #         net.ensemble_classifier_fc.requires_grad_(False)
    #     else:
    #         if 'xception' in str(type(net.net)):
    #             for j in net.net.seq:
    #                 if j[0] == i:
    #                     for t in j[1]:
    #                         t.requires_grad_(False)
    #
    #         if 'EfficientNet' in str(type(net.net)):
    #             if i == 'b0':
    #                 net.net._conv_stem.requires_grad_(False)
    #             stage_map = net.net.stage_map
    #             for c in range(len(stage_map) - 2, -1, -1):
    #                 if not stage_map[c]:
    #                     stage_map[c] = stage_map[c + 1]
    #             for c1, c2 in zip(stage_map, net.net._blocks):
    #                 if c1 == i:
    #                     c2.requires_grad_(False)

    net = nn.SyncBatchNorm.convert_sync_batchnorm(net).to(local_rank)
    # net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank,
    #                                           find_unused_parameters=True)
    AG = AGDA(**config.AGDA_config).to(local_rank)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, betas=config.adam_betas,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step,
                                                gamma=config.scheduler_gamma)
    # if config.ckpt:
    #     loc = 'cuda:{}'.format(local_rank)
    #     checkpoint = torch.load(config.ckpt, map_location=loc)
    #     logs = checkpoint['logs']
    #     start_epoch = int(logs['epoch']) + 1
    #     if load_state(net.module, checkpoint['state_dict']) and config.resume_optim:
    #         optimizer.load_state_dict(checkpoint['optimizer_state'])
    #         try:
    #             scheduler.load_state_dict(checkpoint['scheduler_state'])
    #         except:
    #             pass
    #     else:
    #         net.module.auxiliary_loss.alpha = torch.tensor(config.alpha)
    #     del checkpoint
    torch.cuda.empty_cache()
    start_epoch = 0
    for epoch in range(start_epoch, config.epochs):
        # logs['epoch'] = epoch
        # train_sampler.set_epoch(epoch)
        # train_sampler.dataset.next_epoch()
        # train_loader.sampler.set_epoch(epoch)
        train_loss_value, train_acc, train_real_acc, train_fake_acc = run(epoch, data_loader=train_loader, net=net,
                                                                          optimizer=optimizer,
                                                                          local_rank=local_rank,
                                                                          config=config,
                                                                          AG=AG, phase='train')
        val_loss_value, val_acc, val_real_acc, val_fake_acc = run(epoch, data_loader=validate_loader, net=net,
                                                                  optimizer=optimizer,
                                                                  local_rank=local_rank,
                                                                  config=config,
                                                                  AG=AG, phase='val')
        net.module.auxiliary_loss.alpha *= config.alpha_decay
        scheduler.step()
        # if local_rank == 0:
        #     writer.add_scalar("train/loss", train_loss_value, epoch)
        #     writer.add_scalar("train/train_real_acc", train_real_acc, epoch)
        #     writer.add_scalar("train/train_fake_acc", train_fake_acc, epoch)
        #
        #     checkpoints_save_name = f"Epoch-{epoch}_acc-{val_acc}_loss-{val_loss_value}.pth"
        #     # 写入tensorboard日志
        #     # writer.add_scalar("val/acc", val_acc, epoch)
        #     writer.add_scalar("val/loss", val_loss_value, epoch)
        #     writer.add_scalar("val/val_real_acc", val_real_acc, epoch)
        #     writer.add_scalar("val/val_fake_acc", val_fake_acc, epoch)
        #
        #     torch.save({
        #         # 'logs'           : logs,
        #         'state_dict'     : net.module.state_dict(),
        #         'optimizer_state': optimizer.state_dict(),
        #         'scheduler_state': scheduler.state_dict()
        #     },
        #             'checkpoints/' + checkpoints_save_name
        #     )
        #
        # dist.barrier()


def train_loss(loss_pack, config):
    if 'loss' in loss_pack:
        return loss_pack['loss']
    loss = config.ensemble_loss_weight * loss_pack['ensemble_loss'] + config.aux_loss_weight * loss_pack['aux_loss']
    if config.AGDA_loss_weight != 0:
        loss += config.AGDA_loss_weight * loss_pack['AGDA_ensemble_loss'] + config.match_loss_weight * loss_pack[
            'match_loss']
    return loss


def run(epoch, data_loader, net, optimizer, local_rank, config, AG=None, phase='train'):
    if local_rank == 0:
        print('start ', phase)
    if config.AGDA_loss_weight == 0:
        AG = None
    recorder = {}
    if config.feature_layer == 'logits':
        record_list = ['loss', 'acc']
    else:
        record_list = ['ensemble_loss', 'aux_loss', 'ensemble_acc']
        if AG is not None:
            record_list += ['AGDA_ensemble_loss', 'match_loss']
    # for i in record_list:
    #     recorder[i] = dist_average(local_rank)
    # begin training
    start_time = time.time()
    if phase == 'train':
        net.train()
    else:
        net.eval()

    data_length, val_acc, val_loss = 0, 0, 0
    temp_total_loss, temp_total_acc = 0, 0
    data_loader_length = len(data_loader)
    # total_step = data_loader_length / getattr(args, "batch_size")
    # 仅用于test
    out_list, label_list = [], []

    for i, datas in enumerate(data_loader):
        images, labels, _, _ = datas
        data_length += len(images)

        images = torch.stack(images)
        labels = torch.tensor(labels)
        images, labels = shuffle(images, labels)
        # print(images.shape, labels.shape)
        X = images.to(local_rank, non_blocking=True)
        y = labels.to(local_rank, non_blocking=True)

        with torch.set_grad_enabled(phase == 'train'):
            loss_pack = net(X, y, train_batch=True, AG=AG)

        batch_loss = train_loss(loss_pack, config)
        temp_total_loss += float(batch_loss.detach().cpu().numpy())
        # print(f"batch_loss: {batch_loss.detach().cpu().numpy()}")
        if phase == 'train':
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        out_list.append(loss_pack['ensemble_logit'])
        label_list.append(y)
        # 对于非训练模式,把模型预测和标签全部收集,然后统一处理
        if phase == "test" or phase == "val":
            continue

        with torch.no_grad():
            # config.feature_layer="b2",下面分支只能走ensemble_acc
            if config.feature_layer == 'logits':
                loss_pack['acc'] = ACC(loss_pack['logits'], y)
            else:
                # loss_pack['ensemble_acc'] = ACC(loss_pack['ensemble_logit'], y)
                # 替换原准确率计算方式,统计real,fake各自的准确率
                acc, batch_real_acc, batch_fake_acc, batch_real_cnt, batch_fake_cnt = compute_metrics(
                        loss_pack['ensemble_logit'], y)
                # temp_total_acc += float(loss_pack['ensemble_acc'].detach().cpu().numpy())
                # print(acc)
        if local_rank == 0 and i % 1 == 0 and i != 0:
            # print(y)
            logger.info(
                    f"Epoch {epoch} - ({i}/{data_loader_length}), "
                    f"acc_{round(float(acc), 4)}, "
                    f"real_acc_{round(float(batch_real_acc), 4) if batch_real_cnt != 0 else 'NAN'}, "
                    f"fake_acc_{round(float(batch_fake_acc), 4) if batch_fake_cnt != 0 else 'NAN'}, "
                    f"loss_{round(float(batch_loss.detach().cpu().numpy()), 5)}"
            )
        # for i in record_list:
        #     recorder[i].step(loss_pack[i])
    # if phase == "test" or phase == "val":
    outs = torch.stack(out_list)
    outs = outs.reshape(-1, 2)
    ys = torch.stack(label_list)
    ys = ys.reshape(-1)
    acc, real_acc, fake_acc, real_cnt, fake_cnt = compute_metrics(outs, ys)
    logger.info(
            f"Epoch {epoch}, "
            f"acc_{round(float(acc), 4)}, "
            f"real_acc_{round(float(real_acc), 4) if real_cnt != 0 else 'NAN'}, "
            f"fake_acc_{round(float(fake_acc), 4) if fake_cnt != 0 else 'NAN'}, ")
    # end of this epoch
    # batch_info = []
    # for i in record_list:
    #     mesg = recorder[i].get()
    #     logs[i] = mesg
    #     batch_info.append('{}:{:.4f}'.format(i, mesg))
    # end_time = time.time()

    # write log for this epoch
    # print("local_rank=", local_rank)
    # print(local_rank == 0 and (phase == "test" or phase == "val"))

    # if local_rank == 0 and (phase == "test" or phase == "val"):
    # logging.info('{}: {}, Time {:3.2f}'.format(phase, '  '.join(batch_info), end_time - start_time))

    val_loss = round(temp_total_loss / data_length, 5)
    val_acc = round(float(acc), 4)
    val_real_acc = round(float(real_acc), 4)
    val_fake_acc = round(float(fake_acc), 4)

    return val_loss, val_acc, val_real_acc, val_fake_acc


def distributed_train(config, world_size=0, num_gpus=0, rank_offset=0):
    if not num_gpus:
        num_gpus = torch.cuda.device_count()
    if not world_size:
        world_size = num_gpus
    # mp.spawn(main_worker, nprocs=num_gpus, args=(world_size, rank_offset, config))
    main_worker(config)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    config = train_config("mixture_dataset", ["efficientnet-b4"])
    distributed_train(config)
