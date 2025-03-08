encoder=vitl
dataset=dense
scene=test
img_size=266
max_depth=1000

load_from=/home/sph/event/fuse_public/exp/fuse_dense_vitl_dense_decoder_20250308_165340/latest.pth
outdir=/data_nvme/sph/lorot_results/fuse_vitl_${dataset}_${scene}
event_voxel_chans=3

python run.py \
    --encoder $encoder \
    --dataset $dataset --scene $scene \
    --input-size $img_size \
    --max-depth $max_depth --load-from $load_from \
    --outdir $outdir \
    --event-voxel-chans $event_voxel_chans \
    --save-numpy