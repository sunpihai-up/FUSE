encoder=vitl
dataset=mvsec # vkitti
scene=night1
img_size=350
max_depth=80
# load_from=/data_nvme/sph/da2_checkpoints/depth_anything_v2_vitl.pth
load_from=/data_nvme/sph/da2_checkpoints/depth_anything_v2_metric_vkitti_vitl.pth
outdir=/home/sph/event/da2-prompt-tuning/da2_metric_depth/results/${dataset}_${scene}_da2_vkitti

python run.py \
    --encoder $encoder \
    --dataset $dataset --scene $scene \
    --input-size $img_size \
    --max-depth $max_depth --load-from $load_from \
    --outdir $outdir \
    --save-numpy \
    # --normalized-depth