export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

python eval.py \
    --cfg configs/biwi/stage2_diffar.yaml \
    --exp diff_ar_biwi_eval