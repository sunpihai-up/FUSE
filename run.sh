encoder=vitl
dataset=mvsec
scene=day1
img_size=266
max_depth=80
load_from=/home/sph/event/da2-prompt-tuning/exp/epde_metric_noclip_sigloss_sigmoid_mvsec_2_decoder_20250120_202107/abs_rel-0.2618878185749054-10.pth
outdir=/home/sph/event/da2-prompt-tuning/results/test/epde_metric_sigloss_${dataset}_${scene}_11
event_voxel_chans=3

python run.py \
    --encoder $encoder \
    --dataset $dataset --scene $scene \
    --input-size $img_size \
    --max-depth $max_depth --load-from $load_from \
    --outdir $outdir \
    --event-voxel-chans $event_voxel_chans \
    --save-numpy \
    # --normalized-depth
    # --return-feature \