encoder=vitl
dataset=mvsec_voxel # vkitti
scene=night1
img_size=350
max_depth=1
load_from=/data/coding/code/da2-prompt-tuning/da2_metric_depth/exp/mvsec_voxel_2_nl_da2vitl_20250107_185902/abs_rel-0.25197216868400574-5.pth
outdir=/data/coding/code/da2-prompt-tuning/da2_metric_depth/results/${dataset}_lora_2_${scene}_6

python run.py \
    --encoder $encoder \
    --dataset $dataset --scene $scene \
    --input-size $img_size \
    --max-depth $max_depth --load-from $load_from \
    --outdir $outdir \
    --save-numpy \
    --normalized-depth