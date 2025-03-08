#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

epoch=20
bs=24
gpus=2
lr=0.000005
encoder=vitl
dataset=dense
img_size=266
min_depth=0
max_depth=1000
event_voxel_chans=3
pretrained_from=/home/sph/event/da2-prompt-tuning/exp/public_checkpoints/foundation_vitl/latest.pth
save_path=/home/sph/event/fuse_public/exp/fuse_dense_${encoder}_${dataset}_${finetune_mode}_${now}

mkdir -p $save_path

python3 -m torch.distributed.launch \
    --nproc_per_node=$gpus \
    --nnodes 1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=20596 \
    train.py --epoch $epoch --encoder $encoder --bs $bs --lr $lr --save-path $save_path --dataset $dataset \
    --img-size $img_size --min-depth $min_depth --max-depth $max_depth \
    --event_voxel_chans $event_voxel_chans \
    --pretrained-from $pretrained_from \
    --port 20596 2>&1 | tee -a $save_path/$now.log
    # --normalized_depth \
    # --depth-anything-pretrained $depth_anything_pretrained \
    # --prompt-encoder-pretrained $prompt_encoder_pretrained \
    # --prompt-encoder-pretrained $prompt_encoder_pretrained \
    # --inv \