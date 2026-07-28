from dataset.utils import tprint
from model.r2d2 import R2D2
from model.proto import PROTO
from model.routing import ROUTING
import torch
def get_classifier(ebd_dim, args):
    tprint("Building classifier")
    if args.classifier=='r2d2':
        model = R2D2(ebd_dim, args)
    elif args.classifier=='proto':
        model = PROTO(ebd_dim, args)
    elif args.classifier=='routing':
        model = ROUTING(ebd_dim, args)
    if args.snapshot != '':
        # load pretrained models
        tprint("Loading pretrained classifier from {}".format(
            args.snapshot + '.clf'
        ))
        model.load_state_dict(torch.load(args.snapshot + '.clf'))
    if args.cuda != -1:
        return model.cuda(args.cuda)
    else:
        return model