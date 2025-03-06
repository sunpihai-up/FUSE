encoder=vitl
dataset=dense
scene=test
img_size=266
max_depth=1
# load_from=/home/sph/event/da2-prompt-tuning/exp/fuse_l1_fea_vitl_eventscape_fuse_cor_20250223_155843/latest.pth
# load_from=/home/sph/event/da2-prompt-tuning/exp/lorot_sigloss_dense_decoder_20250224_185419/latest.pth
# load_from=/home/sph/event/da2-prompt-tuning/exp/fuse_l1_fea_vitl_eventscape_fuse_cor_20250223_155843/latest.pth
# load_from=/home/sph/event/da2-prompt-tuning/exp/lorot_sigloss_dense_decoder_20250226_205325/abs_rel-0.1792384833097458-45.pth

# load_from=/home/sph/event/da2-prompt-tuning/exp/trainzero_cross_vits_mvsec_2_overall_20250227_014334/latest.pth # baseline-1
# load_from=/home/sph/event/da2-prompt-tuning/exp/trainzero_fdfim_vits_mvsec_2_overall_20250226_171443/latest.pth # baseline-2
# load_from=/home/sph/event/da2-prompt-tuning/exp/lorot_sigloss_fusenoalign_mvsec_2_decoder_20250227_152656/latest.pth # baseline-3
# load_from=/home/sph/event/da2-prompt-tuning/exp/lorot_sigloss_vits_mvsec_2_decoder_20250227_162442/latest.pth # vits
# load_from=/home/sph/event/da2-prompt-tuning/exp/lorot_sigloss_mvsec_2_decoder_20250228_170818/latest.pth # VITB
# load_from=/home/sph/event/da2-prompt-tuning/exp/lorot_sigloss_vitb_dense_decoder_20250228_185942/latest.pth
load_from=/home/sph/event/da2-prompt-tuning/exp/fuse_l1_fea_vitl_eventscape_fuse_cor_20250223_155843/latest.pth
outdir=/data_nvme/sph/lorot_results/test/lorot_vitl_zero_exp100_${dataset}_${scene}
event_voxel_chans=3

python run.py \
    --encoder $encoder \
    --dataset $dataset --scene $scene \
    --input-size $img_size \
    --max-depth $max_depth --load-from $load_from \
    --outdir $outdir \
    --event-voxel-chans $event_voxel_chans \
    --save-numpy \
    # --normalized-depth
    # --return-feature \