#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

epoch=50
bs=6
gpus=2
lr=0.000005
encoder=vitl
dataset=mvsec
img_size=350
min_depth=0
max_depth=80
event_voxel_chans=3
prompt_type=epde_deep # choices=["epde_deep", "epde_shaw", "add", "none"],
depth_anything_pretrained=/data_nvme/sph/da2_checkpoints/depth_anything_v2_vitl.pth
finetune_mode=overall # choices=["prompt", "decoder", "bias", "bias_and_decoder", "overall"], 
# pretrained_from=/home/sph/code/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth
save_path=/home/sph/event/da2-prompt-tuning/exp/ffr_${dataset}_${finetune_mode}_nl_${now}

mkdir -p $save_path

# python3 -m torch.distributed.launch \
#     --nproc_per_node=$gpus \
#     --nnodes 1 \
#     --node_rank=0 \
#     --master_addr=localhost \
#     --master_port=20596 \
#     train.py --epoch $epoch --encoder $encoder --bs $bs --lr $lr --save-path $save_path --dataset $dataset \
#     --img-size $img_size --min-depth $min_depth --max-depth $max_depth \
#     --prompt_type $prompt_type \
#     --event_voxel_chans $event_voxel_chans \
#     --depth-anything-pretrained $depth_anything_pretrained \
#     --finetune-mode $finetune_mode \
#     --normalized_depth \
#     --port 20596 2>&1 | tee -a $save_path/$now.log

python3 -m torch.distributed.launch \
    --nproc_per_node=$gpus \
    --nnodes 1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=20596 \
    train_rf.py --epoch $epoch --encoder $encoder --bs $bs --lr $lr --save-path $save_path --dataset $dataset \
    --img-size $img_size --min-depth $min_depth --max-depth $max_depth \
    --prompt_type $prompt_type \
    --event_voxel_chans $event_voxel_chans \
    --depth-anything-pretrained $depth_anything_pretrained \
    --finetune-mode $finetune_mode \
    --port 20596 2>&1 | tee -a $save_path/$now.log
    # --normalized_depth \
