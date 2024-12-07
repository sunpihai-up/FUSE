python run.py \
    --split_path /data/coding/code/da2-prompt-tuning/dataset/splits/dense/test.txt \
    --dataset dense \
    --load-from /data/coding/code/da2-prompt-tuning/exp/dense_nothing_frozen_normalized_log_20241207_141804/abs_rel-0.11121226847171783-9.pth \
    --encoder vitl \
    --outdir /data/coding/code/da2-prompt-tuning/results/dense_no_freeze_nl_10 \
    --save-numpy
    # --img-dir
    # --event-dir 
    # --input-size 518 \