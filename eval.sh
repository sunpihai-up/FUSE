python evaluation.py \
    --predictions_dataset /data/coding/code/da2-prompt-tuning/results/dense_all_nl_18/npy \
    --target_dataset /data/coding/upload-data/data/DENSE/test/depth/data \
    --clip_distance 1000.0 \
    --dataset dense \
    --metric

# python evaluation.py \
#     --predictions_dataset /data/coding/code/da2-prompt-tuning/results/dense_no_freeze_nl_10/npy \
#     --target_dataset /data/coding/upload-data/data/DENSE/test/depth/data \
#     --clip_distance 1000.0 \
#     --dataset dense \
#     --inv