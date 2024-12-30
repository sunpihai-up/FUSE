python evaluation.py \
    --predictions_dataset /home/sph/event/da2-prompt-tuning/results/test/ffr_debug_mvsec_metric_vitl_overall_night1_44_crop/npy \
    --target_dataset /data_nvme/sph/mvsec_processed/outdoor_night1/depths \
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