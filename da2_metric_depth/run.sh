img_path=./dataset/mvsec/outdoor_day1.txt
encoder=vitl
dataset=mvsec # vkitti
img_size=350
max_depth=80
load_from=
outdir=/data/coding/code/da2-prompt-tuning/results/

python da2_metric_depth/run.py \
    --img-path $img_path --encoder $encoder \
    --dataset $dataset --input-size $img_size \
    --max-depth $max_depth --load-from $load_from \
    --outdir $outdir \
    --save-numpy \
    --normailzed_depth \