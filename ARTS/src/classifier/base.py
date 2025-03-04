import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BASE(nn.Module):
    '''
        BASE model
    '''
    def __init__(self, args):
        super(BASE, self).__init__()
        self.args = args

        # cached tensor for speed
        self.I_way = nn.Parameter(torch.eye(self.args.way, dtype=torch.float),
                                  requires_grad=False)

    def _compute_l2(self, XS, XQ):
        '''
            Compute the pairwise l2 distance
            @param XS (support x): support_size x ebd_dim
            @param XQ (support x): query_size x ebd_dim

            @return dist: query_size x support_size

        '''
        diff = XS.unsqueeze(0) - XQ.unsqueeze(1)
        dist = torch.norm(diff, dim=2)

        return dist

    def _compute_cos(self, XS, XQ):
        '''
            Compute the pairwise cos distance
            @param XS (support x): support_size x ebd_dim
            @param XQ (support x): query_size x ebd_dim

            @return dist: query_size support_size

        '''
        dot = torch.matmul(
                XS.unsqueeze(0).unsqueeze(-2),
                XQ.unsqueeze(1).unsqueeze(-1)
                )
        dot = dot.squeeze(-1).squeeze(-1)

        scale = (torch.norm(XS, dim=1).unsqueeze(0) *
                 torch.norm(XQ, dim=1).unsqueeze(1))

        scale = torch.max(scale,
                          torch.ones_like(scale) * 1e-8)

        dist = 1 - dot/scale

        return dist

    def reidx_y(self, YS, YQ):
        '''
            Map the labels into 0,..., way
            @param YS: batch_size
            @param YQ: batch_size

            @return YS_new: batch_size
            @return YQ_new: batch_size
        '''
        '''
        for i in range(0,self.args.query * self.args.way):
            if YQ[i]==15 :
                YQ[i]=0
        for i in range(0, self.args.shot * self.args.way):
                if YS[i] == 15:
                    YS[i] = 0'''
            #print("1")

        unique1, inv_S = torch.unique(YS, sorted=True, return_inverse=True)
        unique2, inv_Q = torch.unique(YQ, sorted=True, return_inverse=True)
        #print("Support set classes:",unique1)
        #print("unique1",unique1)
        #print("unique2", unique2)

        if len(unique1) != len(unique2):
            raise ValueError(
                'Support set classes are different from the query set')

        if len(unique1) != self.args.way:
            print("unique1", unique1)
            raise ValueError(
                'Support set classes are different from the number of ways')

        if int(torch.sum(unique1 - unique2).item()) != 0:
            raise ValueError(
                'Support set classes are different from the query set classes')

        Y_new = torch.arange(start=0, end=self.args.way, dtype=unique1.dtype,
                device=unique1.device)

        return Y_new[inv_S], Y_new[inv_Q]

    def _init_mlp(self, in_d, hidden_ds, drop_rate):
        modules = []

        for d in hidden_ds[:-1]:
            modules.extend([
                nn.Dropout(drop_rate),
                nn.Linear(in_d, d),
                nn.ReLU()])
            in_d = d

        modules.extend([
            nn.Dropout(drop_rate),
            nn.Linear(in_d, hidden_ds[-1])])

        return nn.Sequential(*modules)

    def _label2onehot(self, Y):
        '''
            Map the labels into 0,..., way
            @param Y: batch_size

            @return Y_onehot: batch_size * ways
        '''
        Y_onehot = F.embedding(Y, self.I_way)

        return Y_onehot

    @staticmethod
    def compute_acc(pred, true, args):
        '''
            Compute the accuracy.
            @param pred: batch_size * num_classes
            @param true: batch_size
        '''
        result = torch.argmax(pred, dim=1)
        if flag == 1:
            cols = result.dim()
            temp_a = np.ones(cols)
            temp_b = np.zeros(cols)
            a = torch.tensor(temp_a, device=args.cuda)
            b = torch.tensor(temp_b, device=args.cuda)
            # print(result)
            result = torch.where(result > 0, a, b)
            true = torch.where(true > 0, a, b)
        else:
            temp_a = np.ones(args.query * args.way)
            temp_b = np.zeros(args.query * args.way)
            a = torch.tensor(temp_a, device=args.cuda)
            b = torch.tensor(temp_b, device=args.cuda)
            result = torch.where(result > 0, a, b)
            true = torch.where(true > 0, a, b)
        # print("result",result)
        # print("true", true)

        return torch.mean((result == true).float()).item()

    def compute_acc_bpw(pred, true):
        '''
            Compute the accuracy.
            @param pred: batch_size * num_classes
            @param true: batch_size
        '''
        return torch.mean((torch.argmax(pred, dim=1) == true).float()).item()

