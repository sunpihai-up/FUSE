encoder=vitl
dataset=mvsec
scene=night1
img_size=266
max_depth=80
load_from=/home/sph/event/da2-prompt-tuning/exp/epde_metric_80_mvsec_2_decoder_20250116_131204/abs_rel-0.2818499207496643-6.pth
outdir=/home/sph/event/da2-prompt-tuning/results/test/epde_metric_80_mvsec_2_decoder_116131204_${dataset}_${scene}_7
event_voxel_chans=3

python run_rf.py \
    --encoder $encoder \
    --dataset $dataset --scene $scene \
    --input-size $img_size \
    --max-depth $max_depth --load-from $load_from \
    --outdir $outdir \
    --event-voxel-chans $event_voxel_chans \
    --save-numpy \
    # --return-feature \
    # --normalized-depth