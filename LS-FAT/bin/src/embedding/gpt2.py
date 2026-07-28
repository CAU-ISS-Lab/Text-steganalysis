import datetime
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config

class GPT2(nn.Module):
    def __init__(self, pretrained_model_name_or_path=None, cache_dir=None, finetune_ebd=False, return_seq=False):
        super(GPT2, self).__init__()

        self.finetune_ebd = finetune_ebd
        model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path)
        self.return_seq = return_seq

        print("{}, Loading pretrained GPT-2".format(datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')), flush=True)

        self.model = GPT2Model.from_pretrained(pretrained_model_name_or_path, cache_dir=cache_dir)
        self.embedding_dim = self.model.config.n_embd
        self.ebd_dim = self.model.config.n_embd

    def get_gpt2(self, gpt2_id, text_len):
        len_range = torch.arange(gpt2_id.size()[-1], device=gpt2_id.device,
                dtype=text_len.dtype).expand(*gpt2_id.size())

        mask1 = (len_range < text_len.unsqueeze(-1)).long()
        mask2 = (len_range < (text_len - 1).unsqueeze(-1)).float().unsqueeze(-1)

        out = self.model(gpt2_id, attention_mask=mask1)

        last_layer = mask2 * out[0]

        if self.return_seq:
            return last_layer
        else:
            return last_layer[:, 0, :]

    def forward(self, data, weights=None):
        if self.finetune_ebd:
            return self.get_gpt2(data['text'], data['text_len'])
        else:
            with torch.no_grad():
                return self.get_gpt2(data['text'], data['text_len'])