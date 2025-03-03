#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

epoch=10
bs=12
gpus=2
lr=0.000005
lr_lora=0.0001
encoder=vitl
dataset=mvsec_voxel # vkitti
img_size=350
min_depth=0
max_depth=1
# pretrained_from=/data/coding/upload-data/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth
# pretrained_from=/data/coding/upload-data/checkpoints/depth_anything_v2_vitl.pth
pretrained_from=/data/coding/code/da2-prompt-tuning/da2_metric_depth/exp/eventscape_voxel_nl_da2_lora_vitl_20250107_230532/latest.pth
save_path=/data/coding/code/da2-prompt-tuning/da2_metric_depth/exp/${dataset}_nl_da2_lora_${encoder}_${now}

mkdir -p $save_path

python3 -m torch.distributed.launch \
    --nproc_per_node=$gpus \
    --nnodes 1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=20596 \
    train_lora.py --epoch $epoch --encoder $encoder --bs $bs --lr $lr --save-path $save_path --dataset $dataset \
    --img-size $img_size --min-depth $min_depth --max-depth $max_depth --pretrained-from $pretrained_from \
    --normalized-depth \
    --port 20596 2>&1 | tee -a $save_path/$now.log \
    # --lr-lora $lr_lora \
