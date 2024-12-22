encoder=vitl
dataset=mvsec
scene=night1_2b
img_size=350
max_depth=1
load_from=/home/sph/event/da2-prompt-tuning/exp/eventscape_overall_nl_20241222_010732/abs_rel-0.17139266431331635-4.pth
outdir=/home/sph/event/da2-prompt-tuning/results/${dataset}_metric_vitl_overall_${scene}_5_grad_350
event_voxel_chans=3

python run.py \
    --encoder $encoder \
    --dataset $dataset --scene $scene \
    --input-size $img_size \
    --max-depth $max_depth --load-from $load_from \
    --outdir $outdir \
    --event-voxel-chans $event_voxel_chans \
    --save-numpy \
    --normalized-depth