
pretrained_bert=''
bert_cache_dir=''
result_save_dir=''
data_path=""
n_train_class=11
n_val_class=5
n_test_class=7
dataset=stego

for shot in 1 5
do
   python ../src/main.py \
    --cuda 0 \
    --way 3 \
    --shot 5 \
    --query 25 \
    --bpw \
    --dam 1 \
    --dl 0.5 \
    --mode train \
    --pretrained_bert $pretrained_bert \
    --bert_cache_dir $bert_cache_dir \
    --embedding ad_cnn \
    --classifier arts \
    --dataset=$dataset \
    --data_path=$data_path \
    --n_train_class=$n_train_class \
    --n_val_class=$n_val_class \
    --n_test_class=$n_test_class \
    --k 1 \
    --train_epochs 100 \
    --lr_g 1e-3 \
    --lr_d 1e-3 \
    --Comments "movie_close" \
    --patience 20 \
    --test_episodes 1000 \
    --train_epochs 100 \
    --result_path $result_save_dir
done
