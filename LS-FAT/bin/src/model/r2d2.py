import torch
import torch.nn as nn
import torch.nn.functional as F

from classifier.base import BASE
import numpy as np
from torch.autograd import Variable
from sklearn import metrics

class R2D2(BASE):
    '''
        META-LEARNING WITH DIFFERENTIABLE CLOSED-FORM SOLVERS
    '''
    def __init__(self, ebd_dim, args):
        super(R2D2, self).__init__(args)
        self.ebd_dim = ebd_dim

        self.args = args

        # meta parameters to learn
        self.lam = nn.Parameter(torch.tensor(-1, dtype=torch.float))
        self.alpha = nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.beta = nn.Parameter(torch.tensor(1, dtype=torch.float))
        # lambda and alpha is learned in the log space

        # cached tensor for speed
        self.I_support = nn.Parameter(
            torch.eye(self.args.shot * self.args.way, dtype=torch.float),
            requires_grad=False)
        self.I_way = nn.Parameter(torch.eye(self.args.way, dtype=torch.float),
                                  requires_grad=False)

    def _compute_w(self, XS, YS_onehot):
        '''
            Compute the W matrix of ridge regression
            @param XS: support_size x ebd_dim
            @param YS_onehot: support_size x way

            @return W: ebd_dim * way
        '''

        W = XS.t() @ torch.inverse(
                XS @ XS.t() + (10. ** self.lam) * self.I_support) @ YS_onehot

        return W

    def _label2onehot(self, Y):
        '''
            Map the labels into 0,..., way
            @param Y: batch_size

            @return Y_onehot: batch_size * ways
        '''
        Y_onehot = F.embedding(Y, self.I_way)

        return Y_onehot

    def forward(self, XS, YS, XQ, YQ, XQ_logitsD, XSource_logitsD, YQ_d, YSource_d, query_data=None):
        '''
            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size
            @param XQ (support x): query_size x ebd_dim
            @param YQ (support y): query_size

            @return acc
            @return loss
        '''

        YS, YQ = self.reidx_y(YS, YQ)

        YS_onehot = self._label2onehot(YS)

        W = self._compute_w(XS, YS_onehot)
        #print("\nXQ.shape:",XQ.shape)
        #print("XS.shape:", XS.shape)
        pred = (10.0 ** self.alpha) * XQ @ W + self.beta
        #print("YS.shape:", YS.shape)
        #print("YQ.shape:", YQ.shape)
        temp = np.ones(self.args.query * self.args.way)
        temp1 = np.zeros(self.args.query * self.args.way)
        a = torch.tensor(temp, device=self.args.cuda)
        b = torch.tensor(temp1, device=self.args.cuda)
        # print(pred)
        # YQ = YQ.float()
        result = torch.argmax(pred, dim=1)
        result_2 = torch.where(result > 0, a, b)
        true_2 = torch.where(YQ > 0, a, b)
        result_2 = Variable(result_2.float(), requires_grad=True)
        true_2 = Variable(true_2.float(), requires_grad=True)

        #loss = F.cross_entropy(pred, YQ)
        loss = F.cross_entropy(pred, YQ)

        if self.args.bpw :
            acc = BASE.compute_acc_bpw(pred, YQ)
            d_acc = (BASE.compute_acc_bpw(XQ_logitsD, YQ_d) + BASE.compute_acc_bpw(
                XSource_logitsD, YSource_d)) / 2
        else:
            acc = BASE.compute_acc(pred, YQ, self.args.way, self.args.query)
            d_acc = (BASE.compute_acc(XQ_logitsD, YQ_d, self.args.way, self.args.query) + BASE.compute_acc(
                XSource_logitsD, YSource_d, self.args.way, self.args.query)) / 2



        if self.args.mode == 'test':
            temp_a = np.ones(self.args.query * self.args.way)
            temp_b = np.zeros(self.args.query * self.args.way)
            a = torch.tensor(temp_a, device=self.args.cuda)
            b = torch.tensor(temp_b, device=self.args.cuda)
            result = torch.where(result > 0, a, b)
            true = torch.where(YQ > 0, a, b)
            labels = true.cpu().numpy()
            r = result.cpu().detach().numpy()
            precious = metrics.precision_score(labels, r, average='macro')
            recall = metrics.recall_score(labels, r, average='macro')
            # print("precious:",precious)
            # print("recall:", recall)
            if self.args.bpw :
                acc = BASE.compute_acc_bpw(pred, YQ)
            else:
                acc = BASE.compute_acc(pred, YQ, self.args.way, self.args.query)
            if query_data is not None:
                y_hat = torch.argmax(pred, dim=1)
                X_hat = query_data[y_hat != YQ]
                return acc, d_acc, loss, X_hat,precious, recall
            else:
                return acc, d_acc, loss, loss,precious, recall


        else:

            if query_data is not None:
             y_hat = torch.argmax(pred, dim=1)
             X_hat = query_data[y_hat != YQ]
             return acc, d_acc, loss, X_hat
            else:
             return acc, d_acc, loss, loss