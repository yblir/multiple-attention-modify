#!/bin/bash

set -e

CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.run --master_port 12345 \
--nproc_per_node=1 --nnodes=1 \
train_distributed.py -c params.yaml > train.log 2>&1 &

#CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --master_port 12345 \
#--nproc_per_node=1 --nnodes=1 \
#train_distributed.py -c params.yaml > train.log 2>&1 &
