encoder=vitl
dataset=mvsec # vkitti
scene=night1
img_size=350
max_depth=80
load_from=/home/sph/event/da2-prompt-tuning/exp/mvsec_metric_20241216_172558/rmse-6.455599308013916-0.pth
outdir=/home/sph/event/da2-prompt-tuning/results/${dataset}_metric_vitl_${scene}_0

python run.py \
    --encoder $encoder \
    --dataset $dataset --scene $scene \
    --input-size $img_size \
    --max-depth $max_depth --load-from $load_from \
    --outdir $outdir \
    --save-numpy
    # --normailzed_depth \