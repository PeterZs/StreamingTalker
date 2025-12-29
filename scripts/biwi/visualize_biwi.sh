python visualize.py \
    --cfg configs/biwi/stage2_diffar.yaml \
    --exp diff_ar_biwi_visualize \
    --template ./data/biwi/templates.pkl \
    --example ./data/biwi/wav/F4_e36.wav \
    --ply ./data/biwi/templates/BIWI.ply \
    --checkpoint ./checkpoints/diffar_biwi_241212.ckpt \
    --id F4 \
    --split val