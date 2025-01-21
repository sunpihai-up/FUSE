#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

epoch=50
bs=24
gpus=2
lr=0.000005
encoder=vitl
dataset=mvsec_2
img_size=266
min_depth=0
max_depth=80
event_voxel_chans=3
finetune_mode=decoder # choices=["prompt", "decoder", "freeze", "bias_and_decoder", "overall"], 
pretrained_from=/home/sph/event/da2-prompt-tuning/exp/fuse_log_l1_eventscape_fuse_cor_20250114_110446/latest.pth
# pretrained_from=/home/sph/event/da2-prompt-tuning/exp/epde_nl_mvsec_2_decoder_20250116_190436/abs_rel-0.26532474160194397-6.pth
save_path=/home/sph/event/da2-prompt-tuning/exp/epde_metric_noclip_mixed_sigmoid_${dataset}_${finetune_mode}_${now}

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
    # --inv \
