img_path=./dataset/splits/mvsec/outdoor_night1.txt
encoder=vitl
dataset=mvsec # vkitti
img_size=350
max_depth=80
load_from=/data/coding/upload-data/checkpoints/depth_anything_v2_vitl.pth
outdir=/data/coding/code/da2-prompt-tuning/results/depth_anything_v2_vitl_night1

python run.py \
    --img-path $img_path --encoder $encoder \
    --dataset $dataset --input-size $img_size \
    --max-depth $max_depth --load-from $load_from \
    --outdir $outdir \
    --save-numpy
    # --normailzed_depth \