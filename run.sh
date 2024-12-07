python run.py \
    --split_path /data/coding/code/da2-prompt-tuning/dataset/splits/dense/test.txt \
    --dataset dense \
    --load-from /data/coding/code/da2-prompt-tuning/exp/dense_foundation_all_frozen_normalized_log_20241206_225427/abs_rel-0.6624622941017151-17.pth \
    --encoder vitl \
    --outdir /data/coding/code/da2-prompt-tuning/results/dense_all_nl_18 \
    --save-numpy
    # --img-dir
    # --event-dir 
    # --input-size 518 \