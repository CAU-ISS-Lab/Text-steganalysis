import os
import sys
import argparse
import numpy as np
import random
import torch
from transformers import BertModel, BertTokenizer
from transformers import RobertaModel, RobertaTokenizer
from transformers import AlbertModel, AlbertTokenizer
import MyBert
import MyGPT2
import train
from DataLoader import *
from transformers import GPT2Model, GPT2Tokenizer

import tent
import norm

parser = argparse.ArgumentParser(description='MyBert Classification Test')

# data
parser.add_argument('-test-cover-dir', type=str, default='cover.txt',
                    help='the path of test cover data. [default:cover.txt]')
parser.add_argument('-test-stego-dir', type=str, default='1bpw.txt',
                    help='the path of test stego data. [default:1bpw.txt]')
parser.add_argument('-batch-size', type=int, default=64, \
                    help='batch size for training [default: 128]')
parser.add_argument('-save-dir', type=str, default='snapshot',
                    help='where to save the snapshot')
parser.add_argument('-strategy', type=str, default=None,
                    choices=['None', 'norm', 'tent', 'both'],
                    help='Adaptation strategy: norm, tent, or None both')

# device
parser.add_argument('-device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='device to use for testing [default:cuda]')

# option
parser.add_argument('-seed', type=int, default=123,
                    help='The random seed for initialization [default:123]')
parser.add_argument('-test', type=bool, default=False, \
                    help='train or test [default:False]')
parser.add_argument('-pad_size', type=int, default=32, \
                    help='pad_size [default:False]')




args = parser.parse_args()
model_path = 'roberta'
# 设置环境变量CUDA_VISIBLE_DEVICES
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 假设我们使用GPU 0

# 设置随机种子以确保可重复性
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# 加载预训练的模型和分词器
args.model = BertModel.from_pretrained('pretrained_BERT/large_uncased/')
args.tokenizer = BertTokenizer.from_pretrained('pretrained_BERT/large_uncased/')
#args.model = AlbertModel.from_pretrained('pretrained_BERT/little/')
#args.tokenizer = AlbertTokenizer.from_pretrained('pretrained_BERT/little/')
#args.model = GPT2Model.from_pretrained('GPT2/')
#args.tokenizer = GPT2Tokenizer.from_pretrained('GPT2/')
#args.model = RobertaModel.from_pretrained(model_path)
#args.tokenizer = RobertaTokenizer.from_pretrained(model_path)



# 确保模型在评估模式
args.model.eval()
args.model.to(args.device)

apply_norm = args.strategy in ['norm', 'both']
apply_tent = args.strategy in ['tent', 'both']

model = MyBert.MyBert(args, apply_norm=apply_norm,apply_tent = apply_tent )

# 加载测试数据
print('\nLoading test data...')
test_data = build_dataset(args)
test_loader = build_iterator(test_data,args)

# 测试
print('\n---------- Testing ----------')
train.data_eval(test_loader, model, args)  # 修改这里以兼容DataLoader

