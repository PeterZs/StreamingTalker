export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

python train.py \
    --cfg configs/biwi/stage2_diffar.yaml \
    --exp biwi_diffar