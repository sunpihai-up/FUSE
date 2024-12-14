#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

epoch=20
bs=2
gpus=1
lr=0.000005
encoder=vitl
dataset=mvsec
img_size=518
# img_size=350
event_voxel_chans=3
prompt_type=epde_deep
depth_anything_pretrained=/data/coding/upload-data/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth
finetune_mode=prompt # choices=["prompt", "decoder", "bias", "bias_and_decoder", "overall"], 
# pretrained_from=/home/sph/code/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth
save_path=/data/coding/code/da2-prompt-tuning/exp/${dataset}_${finetune_mode}_normalized_log_${now}

mkdir -p $save_path

python3 -m torch.distributed.launch \
    --nproc_per_node=$gpus \
    --nnodes 1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=20596 \
    train_rf.py --epoch $epoch --encoder $encoder --bs $bs --lr $lr --save-path $save_path --dataset $dataset \
    --img-size $img_size \
    --prompt_type $prompt_type \
    --event_voxel_chans $event_voxel_chans \
    --depth-anything-pretrained $depth_anything_pretrained \
    --finetune-mode $finetune_mode \
    --port 20596 2>&1 | tee -a $save_path/$now.log
