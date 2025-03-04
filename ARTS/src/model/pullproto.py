import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from classifier.base import BASE
from torch.autograd import Variable
from sklearn import metrics
class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization.

    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
    - in_c (int): number of input channels.
    - out_c (int): number of output channels.
    - k (int or tuple): kernel size.
    - s (int or tuple): stride.
    - p (int or tuple): padding.
    """
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))

class PullProto(BASE):

    def __init__(self, ebd_dim, args):
        super(PullProto, self).__init__(args)
        self.ebd_dim = ebd_dim

        self.args = args
        self.scale_cls = 7

        self.reference = nn.Linear(self.ebd_dim, self.args.way, bias=True)
        nn.init.kaiming_normal_(self.reference.weight, a=0, mode='fan_in', nonlinearity='linear')
        nn.init.constant_(self.reference.bias, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def Shared_Matrix(self, prototype):
        C = prototype
        eps = 1e-6
        R = self.reference.weight

        power_R = ((R * R).sum(dim=1, keepdim=True)).sqrt()
        R = R / (power_R + eps)

        power_C = ((C * C).sum(dim=1, keepdim=True)).sqrt()
        C = C / (power_C + eps)

        P = torch.matmul(torch.pinverse(C), R)
        P = P.permute(1, 0)
        return P

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

    def _compute_prototype(self, XS, YS):
        '''
            Compute the prototype for each class by averaging over the ebd.

            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size

            @return prototype: way x ebd_dim
        '''
        # sort YS to make sure classes of the same labels are clustered together
        sorted_YS, indices = torch.sort(YS)
        #print("indices:",indices)
        #print("XS:", XS)
        #a = [aa.tolist() for aa in a]

        #XS_1=torch.cat(XS,dim=0)
        #indices=torch.cat(indices,dim=0)
        sorted_XS = XS[indices]
        #print("sorted_XS:",sorted_XS)

        prototype = []
        for i in range(self.args.way):
            prototype.append(torch.mean(
                sorted_XS[i*self.args.shot:(i+1)*self.args.shot], dim=0,
                keepdim=True))

        prototype = torch.cat(prototype, dim=0)

        return prototype

    def forward(self, XS, YS, XQ, YQ, XQ_logitsD, XSource_logitsD, YQ_d, YSource_d,query_data=None):
        '''
            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size
            @param XQ (support x): query_size x ebd_dim
            @param YQ (support y): query_size

            @return acc
            @return loss
        '''

        YS, YQ = self.reidx_y(YS, YQ)
        prototype = self._compute_prototype(XS, YS)
        P = self.Shared_Matrix(prototype)
        weight = P.view(P.size(0), P.size(1), 1)
        prototype_transformed = F.conv1d(prototype.squeeze(0).unsqueeze(2), weight).squeeze(2)
        XQ_transformed = F.conv1d(XQ.squeeze(0).unsqueeze(2), weight).squeeze(2)

        pred = -self._compute_cos(prototype_transformed, XQ_transformed)
        result = torch.argmax(pred, dim=1)

        discriminative_loss = 0.0

        for j in range(self.args.way):
            for k in range(self.args.way):
                if j != k:
                    sim = -self._compute_cos(prototype_transformed[j].unsqueeze(0),
                                            prototype_transformed[k].unsqueeze(0))
                    discriminative_loss = discriminative_loss + sim

        if self.args.bpw:
            acc = BASE.compute_acc_bpw(pred, YQ)
            d_acc = (BASE.compute_acc_bpw(XQ_logitsD, YQ_d) + BASE.compute_acc_bpw(
                XSource_logitsD, YSource_d)) / 2
        else:
            acc = BASE.compute_acc(pred, YQ, self.args)
            d_acc =0.4
        loss = F.cross_entropy(pred, YQ) + self.args.dl * discriminative_loss

        if self.args.mode=='test':
            if self.args.bpw:
                labels = YQ.cpu().numpy()
                result = Variable(result.float(), requires_grad=False)
                r = result.cpu().detach().numpy()
                precious = metrics.precision_score(labels, r, average='macro')
                recall = metrics.recall_score(labels, r, average='macro')


            else:
              result = Variable(result.float(), requires_grad=False)
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

            if query_data is not None:
                y_hat = torch.argmax(pred, dim=1)
                X_hat = query_data[y_hat != YQ]
                return acc, d_acc, loss, X_hat,precious, recall
            return acc, d_acc, loss, discriminative_loss,precious, recall

        else:
            if query_data is not None:
                y_hat = torch.argmax(pred, dim=1)
                X_hat = query_data[y_hat != YQ]
                return acc, d_acc, loss, X_hat
            else:
               return acc, d_acc, loss, discriminative_loss