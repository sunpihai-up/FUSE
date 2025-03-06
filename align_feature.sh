#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

epoch=10
bs=24
gpus=2
lr=0.000005
encoder=vitl
dataset=eventscape_align
img_size=266
event_voxel_chans=3
load_from=/data_nvme/sph/da2_checkpoints/depth_anything_v2_vitl.pth
save_path=/home/sph/event/FUSE_PUBLIC/exp/align_l1_fea_${encoder}_${dataset}_${now}

mkdir -p $save_path

python3 -m torch.distributed.launch \
    --nproc_per_node=$gpus \
    --nnodes 1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=20596 \
    align_feature.py --epoch $epoch --encoder $encoder --bs $bs --lr $lr --save-path $save_path --dataset $dataset \
    --img-size $img_size \
    --event_voxel_chans $event_voxel_chans \
    --load-from $load_from \
    --return-feature \
    --port 20596 2>&1 | tee -a $save_path/$now.log
