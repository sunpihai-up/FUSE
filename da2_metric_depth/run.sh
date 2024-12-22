encoder=vitl
dataset=eventscape # vkitti
scene=test_1k
img_size=350
max_depth=1
load_from=/data/coding/code/da2-prompt-tuning/exp/eventscape_nl_disp_da2vitl_20241222_193824/abs_rel-0.12157373875379562-1.pth
outdir=/data/coding/code/da2-prompt-tuning/results/${dataset}_nl_grad_vitl_${scene}_2

python run.py \
    --encoder $encoder \
    --dataset $dataset --scene $scene \
    --input-size $img_size \
    --max-depth $max_depth --load-from $load_from \
    --outdir $outdir \
    --save-numpy \
    --normalized-depth