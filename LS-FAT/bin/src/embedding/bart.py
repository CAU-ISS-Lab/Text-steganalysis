import datetime
import torch
import torch.nn as nn
from transformers import BartModel, BartConfig
from transformers import BartTokenizer

class BART(nn.Module):
    def __init__(self, pretrained_model_name_or_path=None, cache_dir=None,
                 finetune_ebd=False, return_seq=False):
        super(BART, self).__init__()

        self.finetune_ebd = finetune_ebd
        model_config = BartConfig.from_pretrained('/home/zxs/pre_model/bart')
        self.return_seq = return_seq

        print("{}, Loading pretrained BART".format(
            datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')), flush=True)

        self.model = BartModel.from_pretrained(pretrained_model_name_or_path,
                                              cache_dir=cache_dir)

        self.embedding_dim = self.model.config.d_model
        self.ebd_dim = self.model.config.d_model

    def get_bart(self, bart_id, text_len):
        len_range = torch.arange(bart_id.size()[-1], device=bart_id.device,
                                 dtype=text_len.dtype).expand(*bart_id.size())

        mask1 = (len_range < text_len.unsqueeze(-1)).long()
        mask2 = (len_range < (text_len-1).unsqueeze(-1)).float().unsqueeze(-1)

        out = self.model(input_ids=bart_id, attention_mask=mask1)

        last_layer = mask2 * out[0]#.last_hidden_state

        if self.return_seq:
            return last_layer
        else:
            return last_layer[:, 0, :]

    def forward(self, data, weights=None):
        if self.finetune_ebd:
            return self.get_bart(data['text'], data['text_len'])
        else:
            with torch.no_grad():
                return self.get_bart(data['text'], data['text_len'])