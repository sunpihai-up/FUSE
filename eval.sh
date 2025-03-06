python evaluation.py \
    --predictions_dataset /data_nvme/sph/lorot_results/test/LoRoT_zeroshot_mvsec_day1_full/npy \
    --target_dataset /data_nvme/sph/mvsec_processed/outdoor_day1/depths \
    --clip_distance 80.0 \
    --dataset mvsec \
    --nan_mask \
    --metric \
    # --target_dataset /data_nvme/sph/DENSE/test/seq0/depth/data \
    # --predictions_dataset /home/sph/event/da2-prompt-tuning/results/test/LoRoT_dense_test_20_new/npy \
    # --disparity \
    # --alignment
    # --predictions_dataset /home/sph/event/da2-prompt-tuning/results/test/epde_align_sigloss_nofiltered_mvsec_night1_13/npy \

# python evaluation.py \
#     --predictions_dataset /data/coding/code/da2-prompt-tuning/results/dense_no_freeze_nl_10/npy \
#     --target_dataset /data/coding/upload-data/data/DENSE/test/depth/data \
#     --clip_distance 1000.0 \
#     --dataset dense \
#     --inv