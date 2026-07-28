#
#dataset=20newsgroup
#data_path="../data/20news.json"
#n_train_class=8
#n_val_class=5
#n_test_class=7
#
#dataset=amazon
#data_path="../data/amazon.json"
#n_train_class=10
#n_val_class=5
#n_test_class=9
#
#dataset=huffpost
#data_path="../data/huffpost.json"
#n_train_class=20
#n_val_class=5
#n_test_class=16
#
#dataset=reuters
#data_path="../data/reuters.json"
#n_train_class=15
#n_val_class=5
#n_test_class=11
#

dataset=stego
data_path="../stego_data/fewshot_movie_22way.json"
#"../stego_data/fewshot500_movie_13way.json"
#"../stego_data/fewshot500_news_13way.json"
#"../stego_data/fewshot500_twitter_13way.json"
n_train_class=11
#n_val_class=5
n_val_class=9
n_test_class=7
pretrained_bert='/home/zxs/text-meta/bert-base-uncased/'
# For P-MAML, use need to replace pretrained_bert by the output of
# https://github.com/huggingface/transformers/blob/master/examples/run_lm_finetuning.py
# Otherwise, MAML will be learned from bert_base_uncased
#
bert_cache_dir='/home/zxs/text-meta/bert-base-uncased/model.ckpt.index'
#pretrained_gpt2='/home/zxs/pre_model/gpt2'
#gpt2_cache_dir='/home/zxs/pre_model/gpt2/pytorch_model.bin'
result_save_dir='/home/zxs/MLADA-master/bin/result.txt'
#pretrained_elec='/home/zxs/pre_model/electra'
#elec_cache_dir='/home/zxs/pre_model/electra/pytorch_model.bin'
#snapshot='/home/zxs/MLADA-master/bin/tmp-runs/17070281490387788/0'
#pretrained_bart='/home/zxs/pre_model/bart'
#bart_cache_dir='/home/zxs/pre_model/bart/pytorch_model.bin'
#embedding mlada_cnn mlada_att mlada
for shot in  5
do
  python ../src/main.py \
    --cuda 0 \
    --way 3 \
    --shot $shot \
    --query 25 \
    --mode train \
    --pretrained_bert $pretrained_bert \
    --bert_cache_dir $bert_cache_dir \
    --embedding mlada_cnn\
    --bert \
    --classifier proto \
    --dataset=$dataset \
    --data_path=$data_path \
    --n_train_class=$n_train_class \
    --n_val_class=$n_val_class \
    --n_test_class=$n_test_class \
    --k 1 \
    --train_epochs 60 \
    --lr_g 1e-3 \
    --lr_d 1e-3 \
    --Comments "stego" \
    --patience 10 \
    --embedding_toD idf  \
    --test_episodes 1000 \
    --train_mode t_add_v \
    --result_path $result_save_dir
done

#scp -r /home/zxs/text-meta/wiki.en.vec /home/zxs/MLADA-master/pretrain_wordvec/wiki.en.vec
