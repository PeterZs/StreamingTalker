export CUDA_VISIBLE_DEVICES=0
# export EGL_DEVICE_ID=1

python visualize.py \
    --cfg configs/vocaset/stage2_diffar.yaml \
    --exp diff_ar_vocaset_visualize \
    --template ./data/vocaset/templates.pkl \
    --example ./data/vocaset/wav/FaceTalk_170809_00138_TA_sentence06.wav \
    --ply ./data/vocaset/templates/FLAME_sample.ply \
    --checkpoint ./checkpoints/diffar_voca_241120.ckpt \
    --id FaceTalk_170809_00138_TA \
    --split val