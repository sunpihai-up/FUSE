#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

epoch=10
bs=24
gpus=2
lr=0.000005
encoder=vitb
dataset=eventscape_fuse_cor
img_size=266
event_voxel_chans=3
load_from=/data_nvme/sph/da2_checkpoints/depth_anything_v2_vitl.pth
depth_anything_pretrained=$load_from
prompt_encoder_pretrained=/home/sph/event/da2-prompt-tuning/exp/align_l1_fea_vitb_eventscape_align_20250227_181549/latest.pth
save_path=/home/sph/event/da2-prompt-tuning/exp/fuse_l1_fea_${encoder}_${dataset}_${now}

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
    # --half \
