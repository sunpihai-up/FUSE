python evaluation.py \
    --predictions_dataset /home/sph/event/da2-prompt-tuning/results/test/epde_metric_sigloss_mvsec_day1_11/npy \
    --target_dataset /data_nvme/sph/mvsec_processed/outdoor_day1/depths \
    --clip_distance 80.0 \
    --dataset mvsec \
    --nan_mask \
    --metric \
    # --inv

# python evaluation.py \
#     --predictions_dataset /data/coding/code/da2-prompt-tuning/results/dense_no_freeze_nl_10/npy \
#     --target_dataset /data/coding/upload-data/data/DENSE/test/depth/data \
#     --clip_distance 1000.0 \
#     --dataset dense \
#     --inv