export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

python eval_vae.py \
    --cfg configs/vocaset/stage1_vae.yaml \
    --exp vae_vocaset_visualize \
    --ply ./data/vocaset/templates/FLAME_sample.ply \
    --vertice ./data/vocaset/vertices_npy/FaceTalk_170728_03272_TA_sentence02.npy \
    --template ./data/vocaset/templates.pkl \
    --checkpoint ./checkpoints/vocaset_vae.ckpt \
    --id FaceTalk_170728_03272_TA