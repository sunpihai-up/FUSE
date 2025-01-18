encoder=vitl
dataset=mvsec
scene=day1
img_size=266
max_depth=1
load_from=/home/sph/event/da2-prompt-tuning/exp/epde_nl_mvsec_2_decoder_20250117_210924/abs_rel-0.2653605043888092-17.pth
outdir=/home/sph/event/da2-prompt-tuning/results/test/epde_nl_mvsec_2_decoder_${dataset}_${scene}_18
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