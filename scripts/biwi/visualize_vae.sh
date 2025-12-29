export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

python eval_vae.py \
    --cfg configs/biwi/stage1_vae.yaml \
    --exp vae_biwi_visualize \
    --ply ./data/biwi/templates/BIWI.ply \
    --vertice ./data/biwi/vertices_npy/M3_e06.npy \
    --template ./data/biwi/templates.pkl \
    --checkpoint ./checkpoints/biwi_vae.ckpt \
    --id M6