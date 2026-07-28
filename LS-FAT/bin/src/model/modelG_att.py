import torch
import torch.nn as nn

from embedding.rnn import RNN
import torch.nn.functional as F
from embedding.lstmatt import LSTMAtt
from embedding.avg import AVG
from embedding.idf import IDF


class ModelG_ATT(nn.Module):

    def __init__(self, ebd, args):
        super(ModelG_ATT, self).__init__()

        self.args = args

        self.ebd = ebd

        self.ebd_dim = self.ebd.embedding_dim

        #self.rnn = RNN(768, 128, 1, True, 0)
        self.lstmatt = LSTMAtt(ebd,args)
        self.lstm = nn.LSTM(input_size=300, hidden_size=128, num_layers=1, batch_first=True, dropout=0)
        if args.embedding_toD=='avg':
            self.additional=AVG(ebd,args)
        elif args.embedding_toD=='idf':
            self.additional=IDF(ebd,args)
        elif args.embedding_toD== None:
            self.additional=None

        self.seq = nn.Sequential(
            nn.Linear(256, 1),
        )

    def forward(self, data, flag=None, return_score=False):

        ebd = self.ebd(data)
        w2v = ebd

        avg_sentence_ebd = torch.mean(w2v, dim=1)

        # scale = self.compute_score(data, ebd)
        # print("\ndata.shape:", ebd.shape)  # [b, text_len, 300]

        # Generator部分
        ebd = self.lstmatt(ebd,data)

            # print('reverse_feature.shape[1]', reverse_feature.shape[1])
        # ebd, (hn, cn) = self.lstm(ebd)
        # print("\ndata.shape:", ebd.shape)  # [b, text_len, 256]
        # for i, b in enumerate(ebd):
        #ebd = self.seq(ebd).squeeze(-1)  # [b, text_len, 256] -> [b, text_len]
        # ebd = torch.max(ebd, dim=-1, keepdim=False)[0]
        # print("\ndata.shape:", ebd.shape)  # [b, text_len]
        word_weight = F.softmax(ebd, dim=-1)
       # print("word_weight",word_weight)

        #print("ebd:",ebd.shape)
        #print("word_weight.shape:", word_weight.shape)  # [b, text_len]
        #print("bert:", w2v.shape)
        #sentence_ebd = torch.sum((torch.unsqueeze(word_weight, dim=-1)) * w2v, dim=-2)
        sentence_ebd =ebd
        # print("sentence_ebd.shape:", sentence_ebd.shape)

        reverse_feature = word_weight


        if reverse_feature.shape[1] < 500:
            zero = torch.zeros((reverse_feature.shape[0], 500-reverse_feature.shape[1]))
            if self.args.cuda != -1:
               zero = zero.cuda(self.args.cuda)
            reverse_feature = torch.cat((reverse_feature, zero), dim=-1)
            # print('reverse_feature.shape[1]', reverse_feature.shape[1])
        else:
            reverse_feature = reverse_feature[:, :500]
            # print('reverse_feature.shape[1]', reverse_feature.shape[1])


        if self.args.ablation == '-IL':
            sentence_ebd = torch.cat((avg_sentence_ebd, sentence_ebd), 1)
            print("%%%%%%%%%%%%%%%%%%%%This is ablation mode: -IL%%%%%%%%%%%%%%%%%%")

        if self.args.embedding_toD== 'idf' or self.args.embedding_toD== 'avg':
          ebd_add = self.additional(ebd, data)
          word_weight_add = F.softmax(ebd_add, dim=-1)
          reverse_feature_add = word_weight_add
          if reverse_feature_add.shape[1] < 500:
            zero_add = torch.zeros((reverse_feature_add.shape[0], 500 - reverse_feature_add.shape[1]))
            if self.args.cuda != -1:
               zero_add = zero_add.cuda(self.args.cuda)
            reverse_feature_add = torch.cat((reverse_feature_add, zero_add), dim=-1)
            # print('reverse_feature.shape[1]', reverse_feature.shape[1])
          else:
            reverse_feature_add = reverse_feature_add[:, :500]

          return sentence_ebd, reverse_feature, avg_sentence_ebd,reverse_feature_add
        else:
            return sentence_ebd, reverse_feature, avg_sentence_ebd