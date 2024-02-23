import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class dist_average:
    def __init__(self, local_rank):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = local_rank
        self.acc = torch.zeros(1).to(local_rank)
        self.count = 0

    def step(self, input_):
        self.count += 1
        if type(input_) != torch.Tensor:
            input_ = torch.tensor(input_).to(self.local_rank, dtype=torch.float)
        else:
            input_ = input_.detach()
        self.acc += input_

    def get(self):
        dist.all_reduce(self.acc, op=dist.ReduceOp.SUM)
        self.acc /= self.world_size
        return self.acc.item() / self.count


def ACC(x, y):
    with torch.no_grad():
        a = torch.max(x, dim=1)[1]
        acc = torch.sum(a == y).float() / x.shape[0]  # 0.6667
    # print(y,a,acc)
    return acc


def compute_metrics(model_outputs, labels):
    """
    Compute the accuracy metrics.
    """
    real_probs = F.softmax(model_outputs, dim=1)[:, 0]
    bin_preds = (real_probs <= 0.5).int()
    bin_labels = (labels != 0).int()

    real_cnt = (bin_labels == 0).sum()
    fake_cnt = (bin_labels == 1).sum()

    acc = (bin_preds == bin_labels).float().mean()

    real_acc = (bin_preds == bin_labels)[torch.where(bin_labels == 0)].sum() / (real_cnt + 1e-12)
    fake_acc = (bin_preds == bin_labels)[torch.where(bin_labels == 1)].sum() / (fake_cnt + 1e-12)

    return acc.item(), real_acc.item(), fake_acc.item(), real_cnt.item(), fake_cnt.item()


def gather_tensor(inp, world_size=None, dist_=True, to_numpy=False):
    """Gather tensor in the distributed setting.

    Args:
        inp (torch.tensor):
            Input torch tensor to gather.
        world_size (int, optional):
            Dist world size. Defaults to None. If None, world_size = dist.get_world_size().
        dist_ (bool, optional):
            Whether to use all_gather method to gather all the tensors. Defaults to True.
        to_numpy (bool, optional):
            Whether to return numpy array. Defaults to False.

    Returns:
        (torch.tensor || numpy.ndarray): Returned tensor or numpy array.
    """
    inp = torch.stack(inp)
    if dist_:
        if world_size is None:
            world_size = dist.get_world_size()
        gather_inp = [torch.ones_like(inp) for _ in range(world_size)]
        dist.all_gather(gather_inp, inp)
        gather_inp = torch.cat(gather_inp)
    else:
        gather_inp = inp

    if to_numpy:
        gather_inp = gather_inp.cpu().numpy()

    return gather_inp


def cont_grad(x, rate=1):
    return rate * x + (1 - rate) * x.detach()
