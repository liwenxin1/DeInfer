#!/bin/bash
# the following must be performed with root privilege
sudo nvidia-smi -pm 1 #开启持久模型
sudo nvidia-smi -lgc 1750 #设置GPU的时钟频率
export CUDA_VISIBLE_DEVICES=0         # 这里以GPU0为例，其他卡类似
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS  # 让GPU0变为独享模式,这一条需要root权限，没有root权限就不打开吧,非必须
nvidia-cuda-mps-control -d            # 开启mps服务 
