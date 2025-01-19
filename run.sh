encoder=vitl
dataset=mvsec
scene=night1
img_size=266
max_depth=1
load_from=/home/sph/event/da2-prompt-tuning/exp/epde_metric_mvsec_2_decoder_20250118_153930/abs_rel-0.27630615234375-33.pth
outdir=/home/sph/event/da2-prompt-tuning/results/test/epde_metric_${dataset}_${scene}_34
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
    # --return-feature \