encoder=vitl
dataset=mvsec # vkitti
scene=night1
img_size=350
max_depth=80
load_from=/home/sph/event/da2-prompt-tuning/exp/mvsec_decoder_metric_20241218_154941/19.pth
outdir=/home/sph/event/da2-prompt-tuning/results/${dataset}_metric_disp_vitl_decoder_${scene}_20
event_voxel_chans=3

python run.py \
    --encoder $encoder \
    --dataset $dataset --scene $scene \
    --input-size $img_size \
    --max-depth $max_depth --load-from $load_from \
    --outdir $outdir \
    --event-voxel-chans $event_voxel_chans \
    --save-numpy
    # --normailzed_depth \