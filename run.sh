python run.py \
    --split_path /data/coding/code/da2-prompt-tuning/dataset/splits/dense/val.txt \
    --dataset dense \
    --load-from /data/coding/code/da2-prompt-tuning/exp/dense_foundation_encoders_frozen_normalized_log_20241207_080900/abs_rel-0.12848541140556335-17.pth \
    --encoder vitl \
    --outdir /data/coding/code/da2-prompt-tuning/results/dense_encoder_nl_18_val \
    --save-numpy
    # --img-dir
    # --event-dir 
    # --input-size 518 \