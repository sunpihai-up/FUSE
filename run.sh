encoder=vitl
dataset=mvsec
scene=night1
img_size=350
max_depth=80
load_from=/home/sph/event/da2-prompt-tuning/exp/ffr_debug_mvsec_overall_metric_20241224_210053/abs_rel-0.3608552813529968-43.pth
outdir=/home/sph/event/da2-prompt-tuning/results/test/ffr_debug_${dataset}_metric_vitl_overall_${scene}_44_crop
event_voxel_chans=3

python run_rf.py \
    --encoder $encoder \
    --dataset $dataset --scene $scene \
    --input-size $img_size \
    --max-depth $max_depth --load-from $load_from \
    --outdir $outdir \
    --event-voxel-chans $event_voxel_chans \
    --save-numpy \
    # --normalized-depth