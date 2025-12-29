export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

python train_vae.py \
    --cfg configs/biwi/stage1_vae.yaml \
    --exp biwi_vae