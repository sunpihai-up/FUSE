#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

epoch=20
bs=8
gpus=2
lr=0.000005
encoder=vitl
dataset=mvsec_3
img_size=266
min_depth=0
max_depth=80
event_voxel_chans=3
finetune_mode=lora # choices=["lora", "feature_fusion", "decoder", "freeze", "bias_and_decoder", "overall"], 
depth_anything_pretrained=/data_nvme/sph/da2_checkpoints/depth_anything_v2_vitl.pth
# prompt_encoder_pretrained=/home/sph/event/da2-prompt-tuning/exp/align_fealoss_eventscape_align_20250211_144101/latest.pth
prompt_encoder_pretrained=/home/sph/event/da2-prompt-tuning/exp/align_l1_eventscape_align_20250210_175341/d1-0.887863278388977-9.pth
# pretrained_from=/home/sph/event/da2-prompt-tuning/exp/fuse_log_l1_eventscape_fuse_cor_20250114_110446/latest.pth
# pretrained_from=/home/sph/event/da2-prompt-tuning/exp/epde_nl_mvsec_2_decoder_20250116_190436/abs_rel-0.26532474160194397-6.pth
save_path=/home/sph/event/da2-prompt-tuning/exp/epde_dual_sigloss_${dataset}_${finetune_mode}_${now}

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
    --depth-anything-pretrained $depth_anything_pretrained \
    --prompt-encoder-pretrained $depth_anything_pretrained \
    --finetune-mode $finetune_mode \
    --port 20596 2>&1 | tee -a $save_path/$now.log
    # --prompt-encoder-pretrained $prompt_encoder_pretrained \
    # --pretrained-from $pretrained_from \
    # --normalized_depth \
    # --inv \
