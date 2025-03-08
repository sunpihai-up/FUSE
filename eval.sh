python evaluation.py \
    --predictions_dataset /data_nvme/sph/lorot_results/fuse_vitl_dense_test/npy \
    --target_dataset /data_nvme/sph/DENSE/test/seq0/depth/data \
    --clip_distance 1000.0 \
    --dataset dense \
    --nan_mask \
    --metric