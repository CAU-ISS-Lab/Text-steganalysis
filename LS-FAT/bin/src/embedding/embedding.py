import datetime
from embedding.wordebd import WORDEBD
from embedding.cxtebd import CXTEBD
from model.modelG import ModelG_RNN
from model.modelG_CNN import ModelG_CNN
from model.modelG_att import ModelG_ATT
from model.modelG_LSCNN import ModelG_LSCNN
from model.modelD import ModelD
from embedding.gpt2 import GPT2
from embedding.electra import ELEC
from embedding.bart import BART
import torch
def get_embedding(vocab, args):
    print("{}, Building embedding".format(
        datetime.datetime.now()), flush=True)

    #ebd = WORDEBD(vocab, args.finetune_ebd)
    ebd = CXTEBD(args.pretrained_bert,
                 cache_dir=args.bert_cache_dir,
                 finetune_ebd=False,
                 return_seq=True)
    #model = GPT2LMHeadModel.from_pretrained("gpt2", torchscript=True).eval()
    # tokenizer
    #tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    '''ebd=AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir,)'''
    '''ebd = GPT2(args.pretrained_gpt2,
                 cache_dir=args.gpt2_cache_dir,
                 finetune_ebd=False,
                 return_seq=True)'''
    '''ebd = ELEC(args.pretrained_elec,
               cache_dir=args.elec_cache_dir,
               finetune_ebd=False,
               return_seq=True)'''
    '''ebd = BART(args.pretrained_bart,
               cache_dir=args.bart_cache_dir,
               finetune_ebd=False,
               return_seq=True)'''

    if args.embedding=='mlada':
      modelG = ModelG_RNN(ebd, args)
    elif args.embedding=='mlada_cnn':
      modelG = ModelG_CNN(ebd, args)
    elif args.embedding == 'mlada_att':
        modelG = ModelG_ATT(ebd, args)
    elif args.embedding == 'mlada_lscnn':
        modelG = ModelG_LSCNN(ebd, args)

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