encoder=vitl
dataset=mvsec # vkitti
scene=night1
img_size=350
max_depth=80
load_from=/home/sph/event/da2-prompt-tuning/exp/mvsec_prompt_metric_20241220_153808/abs_rel-2.462299346923828-3.pth
outdir=/home/sph/event/da2-prompt-tuning/results/${dataset}_metric_vitl_prompt_${scene}_1
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