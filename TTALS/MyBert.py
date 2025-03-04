import sys
import torch
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertModel, BertTokenizer
from sklearn.decomposition import PCA
from norm import Norm
from tent import Tent, collect_params, configure_model


class MyBert(nn.Module):
    def __init__(self, args, apply_tent=False, apply_norm=False):
        super(MyBert, self).__init__()
        self.apply_tent = apply_tent
        self.apply_norm = apply_norm

        # 加载预训练的 Bert 模型
        self.bert = args.model
        self.fc = nn.Linear(1024, 2)  # 假设您的任务是二分类任务

        # 根据 apply_norm 标志应用 Norm 包装
        if apply_norm:
            # 假设 Norm 类已正确定义
            self.bert = Norm(self.bert)

        # 根据 apply_tent 标志应用 Tent 逻辑
        if apply_tent:
            # 假设 collect_params 和 Tent 类已正确定义
            self.params, _ = collect_params(self.bert)  # 收集 Bert 模型中的可训练参数
            self.optimizer = torch.optim.Adam(self.params, lr=1e-5)  # 初始化优化器
            self.tent = Tent(self.bert, self.optimizer, steps=1, episodic=True)

    def forward(self, input_ids, attention_mask=None):
        if self.apply_tent:

            outputs = self.tent({'input_ids': input_ids, 'attention_mask': attention_mask})
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # 确保 outputs 是预期的格式
        pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[1]


        logits = self.fc(pooled_output)
        return logits