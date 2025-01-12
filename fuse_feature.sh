#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

epoch=5
bs=2
gpus=2
lr=0.000005
encoder=vitl
dataset=eventscape_fuse_cor
img_size=350
event_voxel_chans=3
load_from=/data/coding/upload-data/checkpoints/depth_anything_v2_vitl.pth
depth_anything_pretrained=$load_from
prompt_encoder_pretrained=/data/coding/code/da2-prompt-tuning/exp/align_log_l1_eventscape_align_20250108_205824/d1-0.8810270428657532-1.pth
save_path=/data/coding/code/da2-prompt-tuning/exp/fuse_log_l1_${dataset}_${now}

mkdir -p $save_path

python3 -m torch.distributed.launch \
    --nproc_per_node=$gpus \
    --nnodes 1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=20596 \
    fuse_feature.py --epoch $epoch --encoder $encoder --bs $bs --lr $lr --save-path $save_path --dataset $dataset \
    --img-size $img_size \
    --max-depth 1 \
    --event_voxel_chans $event_voxel_chans \
    --load-from $load_from \
    --prompt-encoder-pretrained $prompt_encoder_pretrained \
    --depth-anything-pretrained $depth_anything_pretrained \
    --return-feature \
    --port 20596 2>&1 | tee -a $save_path/$now.log
