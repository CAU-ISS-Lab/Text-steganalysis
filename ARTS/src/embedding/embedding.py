import datetime
from embedding.wordebd import WORDEBD
from embedding.cxtebd import CXTEBD
from model.ad_cnn import ModelG_CNN
from model.modelD import ModelD
import torch

def get_embedding(vocab, args):
    print("{}, Building embedding".format(
        datetime.datetime.now()), flush=True)

    #ebd = WORDEBD(vocab, args.finetune_ebd)
    ebd = CXTEBD(args.pretrained_bert,
                 cache_dir=args.bert_cache_dir,
                 finetune_ebd=False,
                 return_seq=True)

    modelG = ModelG_CNN(ebd, args)
    modelD = ModelD(ebd, args)

    print("{}, Building embedding".format(
        datetime.datetime.now()), flush=True)
    if args.snapshot != '':
        # load pretrained models
        print("{}, Loading pretrained embedding from {}".format(
            datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
            args.snapshot + '.G'
        ))
        modelG.load_state_dict(torch.load(args.snapshot + '.G'))
    if args.cuda != -1:
        modelG = modelG.cuda(args.cuda)
        modelD = modelD.cuda(args.cuda)
        return modelG, modelD
    else:
        return modelG, modelD