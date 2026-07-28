import datetime
import torch
from transformers import ElectraModel, ElectraConfig
import torch.nn as nn

class ELEC(nn.Module):
    def __init__(self, pretrained_model_name_or_path=None, cache_dir=None,
                 finetune_ebd=False, return_seq=False):
        super(ELEC, self).__init__()

        self.finetune_ebd = finetune_ebd
        model_config = ElectraConfig.from_pretrained('/home/zxs/pre_model/electra')
        self.return_seq = return_seq

        print("{}, Loading pretrained electra".format(
            datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')), flush=True)

        self.model = ElectraModel.from_pretrained(pretrained_model_name_or_path,
                                                  cache_dir=cache_dir,
                                                  from_tf=False)

        self.embedding_dim = self.model.config.hidden_size
        self.ebd_dim = self.model.config.hidden_size

    def get_electra(self, electra_id, text_len):
        len_range = torch.arange(electra_id.size()[-1], device=electra_id.device,
                                 dtype=text_len.dtype).expand(*electra_id.size())

        mask1 = (len_range < text_len.unsqueeze(-1)).long()
        mask2 = (len_range < (text_len-1).unsqueeze(-1)).float().unsqueeze(-1)

        out = self.model(electra_id, attention_mask=mask1)

        last_layer = mask2 * out[0]#.last_hidden_state

        if self.return_seq:
            return last_layer
        else:
            return last_layer[:, 0, :]

    def forward(self, data, weights=None):
        if self.finetune_ebd:
            return self.get_electra(data['text'], data['text_len'])
        else:
            with torch.no_grad():
                return self.get_electra(data['text'], data['text_len'])