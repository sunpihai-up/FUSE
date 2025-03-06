#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

epoch=20
bs=24
gpus=2
lr=0.000005
encoder=vits
dataset=dense
img_size=266
min_depth=0
max_depth=1000
event_voxel_chans=3
finetune_mode=decoder # choices=["fuse", "lora", "feature_fusion", "decoder", "freeze", "bias_and_decoder", "overall"], 
# depth_anything_pretrained=/data_nvme/sph/da2_checkpoints/depth_anything_v2_vitl.pth
# prompt_encoder_pretrained=/home/sph/event/da2-prompt-tuning/exp/align_fealoss_eventscape_align_20250211_144101/latest.pth
# prompt_encoder_pretrained=/home/sph/event/da2-prompt-tuning/exp/align_l1_eventscape_align_20250210_175341/d1-0.887863278388977-9.pth
# pretrained_from=/home/sph/event/da2-prompt-tuning/exp/fuse_log_l1_eventscape_fuse_cor_20250114_110446/latest.pth
# pretrained_from=/home/sph/event/da2-prompt-tuning/exp/epde_nl_mvsec_2_decoder_20250116_190436/abs_rel-0.26532474160194397-6.pth
# pretrained_from=/home/sph/event/da2-prompt-tuning/exp/fuse_log_l1_eventscape_fuse_cor_20250220_153402/latest.pth
# pretrained_from=/home/sph/event/da2-prompt-tuning/exp/fuse_l1_fea_vitl_eventscape_fuse_cor_20250223_155843/latest.pth
# pretrained_from=/home/sph/event/da2-prompt-tuning/exp/fuse_noalign_l1_fea_vits_eventscape_fuse_20250225_015305/latest.pth
# pretrained_from=/home/sph/event/da2-prompt-tuning/exp/fuse_l1_fea_vitb_eventscape_fuse_cor_20250228_011118/latest.pth
pretrained_from=/home/sph/event/da2-prompt-tuning/exp/fuse_l1_fea_vits_eventscape_fuse_cor_20250227_041232/latest.pth
save_path=/home/sph/event/da2-prompt-tuning/exp/lorot_sigloss_${encoder}_${dataset}_${finetune_mode}_${now}

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
    --finetune-mode $finetune_mode \
    --port 20596 2>&1 | tee -a $save_path/$now.log
    # --normalized_depth \
    # --depth-anything-pretrained $depth_anything_pretrained \
    # --prompt-encoder-pretrained $prompt_encoder_pretrained \
    # --prompt-encoder-pretrained $prompt_encoder_pretrained \
    # --inv \
