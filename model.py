import torch
import torchvision
from torch.cuda import amp
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
import numpy as np
# import cupy
import os
import time
import argparse
from spikingjelly.clock_driven import neuron, layer, functional, surrogate

"""
class S_E(nn.Module):
    def Spiking_Encoder(C_in, C_out, tau):
        return nn.Sequential(
            nn.Conv1d(in_channels=C_in, out_channels=C_out, kernel_size=3, padding="same", bias=False),
            neuron.MultiStepLIFNode(tau=tau),
            nn.MaxPool1d(1, stride=2, ceil_mode=True)
        )
"""


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv_Block(nn.Module):
    """
    Conv--Dropout--BatchNorm--ReLU
    """

    def __init__(self, C_in, C_out, batch_size, channel, length, tau, kernel_size=3, dropout=None, pad_type='zero',
                 dropout_pos='CDBR'):
        """
        ATTENTION NEEDED TO C_in & C_out
        :param input:
        :param output:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dropout:
        :param pad_type:
        """
        super().__init__()
        """
        self.conv = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv1d(in_channels=C_in, out_channels=C_out, kernel_size=3, padding="same", bias=False)
            ),
            layer.MultiStepDropout(p=dropout),
            layer.MultiStepThresholdDependentBatchNorm1d(alpha=1, v_th=1, num_features=length, eps=1e-05,
                                                         momentum=0.1, affine=True, track_running_stats=True),
            #neuron.MultiStepLIFNode(tau=tau, surrogate_function=surrogate.ATan(), backend='cupy')
            neuron.MultiStepLIFNode(tau=tau, surrogate_function=surrogate.ATan())
        )        

        self.conv = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv1d(in_channels=C_in, out_channels=C_out, kernel_size=3, padding="same", bias=False)
            ),
            neuron.MultiStepLIFNode(tau=tau, surrogate_function=surrogate.ATan())
        )

            if pad_type == 'zero':
                self.pad = nn.ZeroPad2d(padding)
            elif pad_type == 'reflect':
                # [!] the paper used reflect padding - just for data augmentation?
                self.pad = nn.ReflectionPad2d(padding)
            else:
                raise ValueError(pad_type)
        """

        self.dropout_pos = dropout_pos

        self.conv1 = layer.SeqToANNContainer(
            nn.Conv1d(in_channels=C_in, out_channels=C_out, kernel_size=3, padding="same", bias=False))
        self.tdbn1 = layer.MultiStepThresholdDependentBatchNorm1d(alpha=100, v_th=0.5, num_features=C_out, eps=1e-05,
                                                                  momentum=0.1, affine=True, track_running_stats=True)
        self.dropout1 = layer.MultiStepDropout(p=dropout)
        self.bn1 = layer.SeqToANNContainer(nn.BatchNorm1d(C_out, affine=True))
        """
        if (dropout is not None) and (dropout>0):
            self.dropout1 = layer.Dropout(p=dropout)
        else: self.dropout1=nn.Linear(channel, batch_size, length)
        """

        """


        self.maxpool1 = layer.MultiStepContainer(nn.MaxPool1d(1))
        """


        self.dropout1 = layer.Dropout(p=dropout)
        #self.snn1 = neuron.MultiStepIFNode(surrogate_function=surrogate.ATan()),
        self.snn1 = neuron.MultiStepLIFNode(tau=tau, v_threshold=0.5, surrogate_function=surrogate.Sigmoid())
        # self.snn1 = neuron.LIFNode(tau=tau)

        self.c_b = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv1d(in_channels=C_in, out_channels=C_out, kernel_size=3, padding=1, bias=False),
                nn.MaxPool1d(1)))
        """,                nn.BatchNorm1d(C_out))
        )
                """


    def forward(self, x):
        t=[]
        x=self.c_b(x)
        x=self.dropout1(x)
        """
        print("mean")
        print(torch.mean(x,dim=1))
        print("var")
        print(torch.var_mean(x,dim=1))
        print("max")
        print(torch.max(x,dim=1))
        print("min")
        print(torch.min(x,dim=1))
        x = self.bn1(x)        
        """
        x_min=x.min(dim=3)[0].unsqueeze(3).repeat(1,1,1,x.shape[3])
        x_max=x.max(dim=3)[0].unsqueeze(3).repeat(1,1,1,x.shape[3])
        x=(x-x_min)/(x_max-x_min+1e-12)
        out = self.snn1(x)
        y=1
        """
        if self.dropout1:
            out = self.dropout1(out)

        out = x
        out = self.dropout1(out)

        out = self.snn1(out)

        out = self.conv(x)

        # out = x
        # x0=x
        # torch.save(x0.to(torch.device('cpu')), "C:/Reproduce/LIF_Fractalnet/tmp_data/input.pth")
        out = self.conv1(x)
        # x1=out
        # torch.save(x1.to(torch.device('cpu')), "C:/Reproduce/LIF_Fractalnet/tmp_data/conv1_output.pth")
        out = self.maxpool1(out)
        # x2=out
        # torch.save(x2.to(torch.device('cpu')), "C:/Reproduce/LIF_Fractalnet/tmp_data/maxpool1_output.pth")
        out = self.snn1(out)
        # x3=out
        # torch.save(x3.to(torch.device('cpu')), "C:/Reproduce/LIF_Fractalnet/tmp_data/snn1_output.pth")
        y1 = []

        """
        """
        out1 = self.tdbn1(out0)
        out2=self.bn1(out0)
        p1=sum(out1)
        p2=sum(out2)
        y2=[]
        out = self.snn1(out)
        y3=[]
        out=self.maxpool1(out)
        y4=[]
        out = self.dropout1(out)
        y5=[]        


        # out = self.tdbn1(out)
        """
        return out


class Fractal_Block(nn.Module):
    def __init__(self, num_columns, C_in, C_out, batch_size, channel, length, tau, p_ldrop, p_dropout, cb_dropout, pad_type='zero',
                 doubling=False, dropout_pos='CDBR'):
        """
        :param num_columns:
        :param C_in:
        :param C_out:
        :param p_ldrop: prob of local drop-path
        :param p_dropout: prob of drop-path
        :param pad_type: if True, doubling by 1x1 conv in front of the block
        :param doubling:
        :dropout_pos: the position of dropout:
                - CDBR (default): conv-dropout-BN-relu
                - CBRD: conv-BN-relu-dropout
                - FD: fractal_block-dropout
        """

        super().__init__()
        #self.num = 0
        self.num_columns = num_columns
        self.p_ldrop = p_ldrop
        self.dropout_pos = dropout_pos
        if dropout_pos == 'FD' and p_dropout > 0.:
            self.dropout = layer.Dropout(p=p_dropout)
            p_dropout = 0.
        else:
            self.dropout = None

        if doubling:
            # self.doubler = nn.Conv2d(C_in, C_out, 1, padding=0)
            self.doubler = Conv_Block(C_in, C_out,  channel, length, tau, kernel_size=1)
        else:
            self.doubler = None

        self.columns = nn.ModuleList([nn.ModuleList() for _ in range(num_columns)])
        self.max_depth = 2 ** (num_columns - 1)

        dist = self.max_depth
        self.count = np.zeros([self.max_depth], dtype=np.int)
        for col in self.columns:
            for i in range(self.max_depth):
                if (i + 1) % dist == 0:
                    first_block = (i + 1 == dist)  # first block in this column
                    if first_block and not doubling:
                        """                        
                        # if doubling, always input channel size is C_out.
                        """
                        cur_C_in = C_in
                    else:
                        cur_C_in = C_out

                    module = Conv_Block(cur_C_in, C_out, batch_size, channel, length, tau, dropout=p_dropout,
                                        pad_type=pad_type, dropout_pos=dropout_pos)
                    self.count[i] = self.count[i] + 1
                else:
                    module = None

                col.append(module)

            dist = dist // 2

    def drop_mask(self, B, global_columns, num_columns):
        """ Generate drop mask; [num_columns, B].
        1) generate global masks
        2) generate local masks
        3) resurrect random path in all-dead column
        4) concat global and local masks

        Args:
            - B: batch_size
            - global_columns: global columns which to alive [GB]
            - num_columns: the number of columns of mask
        """
        # global drop mask
        GB = global_columns.shape[0]
        # calc gdrop cols / samples
        gdrop_cols = global_columns - (self.num_columns - num_columns)
        gdrop_indices = np.where(gdrop_cols >= 0)[0]
        # gen gdrop mask
        gdrop_mask = np.zeros([num_columns, GB], dtype=np.float32)
        gdrop_mask[gdrop_cols[gdrop_indices], gdrop_indices] = 1.

        # local drop mask
        LB = B - GB
        ldrop_mask = np.random.binomial(1, 1. - self.p_ldrop, [num_columns, LB]).astype(np.float32)
        alive_count = ldrop_mask.sum(axis=0)
        # resurrect all-dead case
        dead_indices = np.where(alive_count == 0.)[0]
        ldrop_mask[np.random.randint(0, num_columns, size=dead_indices.shape), dead_indices] = 1.

        drop_mask = np.concatenate((gdrop_mask, ldrop_mask), axis=1)
        return torch.from_numpy(drop_mask)

    def join(self, outs, global_columns):
        """
        Args:
            - outs: the outputs to join
            - global_columns: global drop path columns
        """
        num_columns = len(outs)
        out = torch.stack(outs)  # [n_cols, B, C, L]

        if self.training:
            mask = self.drop_mask(out.size(2), global_columns, num_columns).to(out.device)  # [n_cols, B]
            # mask = mask.view(*mask.size(), 1, 1) # unsqueeze to [num_columns, B, 1, 1]
            mask = mask.view(*mask.size(), 1, 1)
            mask = mask.unsqueeze(1)
            n_alive = mask.sum(dim=0)  # [B, 1, 1, 1]
            masked_out = out * mask  # [n_cols, B, C, L]
            n_alive[n_alive == 0.] = 1.  # all-dead cases
            out = masked_out.sum(dim=0) / n_alive  # [B, C, L] / [B, 1, 1]
        else:
            out = out.mean(dim=0)  # no drop

        return out

    def forward(self, x, global_columns, deepest=False):
        """
        global_columns works only in training mode.
        """
        out = self.doubler(x) if self.doubler else x
        outs = [out] * self.num_columns

        for i in range(self.max_depth):
            st = self.num_columns - self.count[i]
            cur_outs = []  # outs of current depth
            if deepest:
                st = self.num_columns - 1  # last column only

            for c in range(st, self.num_columns):
                cur_in = outs[c]  # current input
                cur_module = self.columns[c][i]  # current module
                cur_outs.append(cur_module(cur_in))

            # join
            # print("join in depth = {}, # of in_join = {}".format(i, len(cur_out)))
            joined = self.join(cur_outs, global_columns)

            for c in range(st, self.num_columns):
                outs[c] = joined

        if self.dropout_pos == 'FD' and self.dropout:
            outs[-1] = self.dropout(outs[-1])

        return outs[-1]  # for deepest case


"""
class TEST_MODEL(nn.Module):
    def __init__(self,tau):
        super().__init__()
        self.snn1 = neuron.LIFNode(tau=tau)
        self.Fl=nn.Flatten(start_dim=-2, end_dim=-1)
        self.l=nn.Linear(channel*L,  2)

        self.snn2 = neuron.LIFNode(tau=tau)
    def forward(self, x):

        x = self.snn1(x)
        x=self.Fl(x)
        x=self.l(x)
        y=[]
        x = self.snn2(x*10000)
        return x
"""


class LIF_FractalNet(nn.Module):
    def __init__(self, T, data_shape, num_columns, init_channels, p_ldrop, dropout_probs,
                 batch_size, cb_dropout, C_L, tau,
                 gdrop_ratio, gap=0, init='xavier', pad_type='zero', doubling=False,
                 consist_gdrop=True, dropout_pos='CDBR'):
        """ LIF_FractalNet
        Args:
            - data_shape: (C, L, nb_classes)
            - num_columns: the number of columns
            - init_channels: the number of out channels in the first block
            - p_ldrop: local drop prob
            - dropout_probs: dropout probs (list)
            - gdrop_ratio: global droppath ratio
            - gap: pooling type for last block
            - init: initializer type
            - pad_type: padding type of conv
            - doubling: if True, doubling by 1x1 conv in front of the block.
            - consist_gdrop
            - dropout_pos: the position of dropout
                - CDBR (default): conv-dropout-BN-relu
                - CBRD: conv-BN-relu-dropout
                - FD: fractal_block-dropout
        """
        super().__init__()
        assert dropout_pos in ['CDBR', 'CBRD', 'FD']

        self.T = T
        self.B = len(dropout_probs)  # the number of blocks
        self.consist_gdrop = consist_gdrop
        self.gdrop_ratio = gdrop_ratio
        self.num_columns = num_columns

        L, channel, n_classes = data_shape

        size = L

        C_in = channel

        layers = nn.ModuleList()
        self.layers = nn.Sequential()
        layers_tmp = nn.Sequential()

        C_out = init_channels
        total_layers = 0
        for b, p_dropout in enumerate(dropout_probs):
            print("[block {}] Channel in = {}, Channel out = {}".format(b, C_in, C_out))
            fb = Fractal_Block(num_columns, C_in, C_out, batch_size, channel, C_L, tau, p_ldrop, p_dropout, cb_dropout,
                               pad_type=pad_type, doubling=doubling, dropout_pos=dropout_pos)
            # self.layers = nn.Sequential(self.layers, fb)
            # layers.append(fb)
            self.layers.add_module("Fractal_Block"+str(b), fb)

            if gap == 0 or b < self.B - 1:
                # Originally, every pool is max-pool in the paper (No GAP).
                # layers.append(layer.SeqToANNContainer(nn.MaxPool1d(1)))
                # self.layers = nn.Sequential(self.layers, layer.SeqToANNContainer(nn.MaxPool1d(1)))
                self.layers.add_module("MaxPool1d", layer.SeqToANNContainer(nn.MaxPool1d(1)))
            elif gap == 1:
                # last layer and gap == 1
                # layers.append(nn.AdaptiveAvgPool1d(1))  # average pooling
                # layers.append(layer.SeqToANNContainer(nn.AdaptiveAvgPool1d(1)))
                # self.layers = nn.Sequential(self.layers, layer.SeqToANNContainer(nn.AdaptiveAvgPool1d(1)))
                self.layers.add_module("AvgPool1d", layer.SeqToANNContainer(nn.AdaptiveAvgPool1d(1)))

            size = size // 2
            total_layers = total_layers + fb.max_depth
            C_in = C_out

            """
            if b < self.B - 2:
                # C_out = C_out * 2  # doubling except for last block
                C_out = int(C_out / 2)

            """
            print(self.layers)

        print("Last featuremap size = {}".format(size))
        print("Total layers = {}".format(total_layers))

        if gap == 2:
            self.layers.add_module("Conv1d", layer.SeqToANNContainer(nn.Conv1d(C_out, n_classes, 1, padding=0)))
            self.layers.add_module("AvgPool1d", layer.SeqToANNContainer(nn.AdaptiveAvgPool1d(1)))
            """
            self.layers = nn.Sequential(self.layers,
                layer.SeqToANNContainer(nn.Conv1d(C_out, n_classes, 1, padding=0)),
                layer.SeqToANNContainer(nn.AdaptiveAvgPool1d(1))
            )

            layers.append(layer.SeqToANNContainer(nn.Conv1d(C_out, n_classes, 1, padding=0)))  # 1x1 conv
            #in_channels=C_in,out_channels=C_out, kernel_size=3, padding="same", bias=False
            #layers.append(nn.Conv1d(batch_size, channel, length, bias=False))
            #layers.append(nn.AdaptiveAvgPool1d(1))  # gap
            layers.append(layer.SeqToANNContainer(nn.AdaptiveAvgPool1d(1)))
            #layers.append(Flatten())
            #layers.append(neuron.LIFNode(tau))
            """

        else:
            """
            layers.append(layer.SeqToANNContainer(nn.Flatten(start_dim=-2, end_dim=-1)))
            #layers.append(nn.Linear(C_out * size * size, n_classes))  # fc layer
            #layers.append(nn.Linear(C_out * L, n_classes))  # fc layer

            layers.append(layer.SeqToANNContainer(nn.Linear(channel*C_L,  n_classes)))  # fc layer
            #layers.append(neuron.MultiStepLIFNode(tau=tau))
            layers.append(neuron.MultiStepLIFNode(tau=tau, surrogate_function=surrogate.ATan()))
            #layers.append(neuron.MultiStepLIFNode(tau=tau,surrogate_function=surrogate.ATan(), backend='cupy'))

            self.layers = nn.Sequential(self.layers,
                layer.SeqToANNContainer(nn.Flatten(start_dim=-2, end_dim=-1),
                                        nn.Linear(channel * C_L, n_classes, bias=False)),
                neuron.MultiStepLIFNode(tau=tau, surrogate_function=surrogate.ATan())
            )
            """
            self.layers.add_module("Linear", layer.SeqToANNContainer(nn.Flatten(start_dim=-2, end_dim=-1),
                                                                     nn.Linear(C_out * C_L, n_classes, bias=False)))
            self.layers.add_module("MultiStepLIFNode1",
                                   neuron.MultiStepLIFNode(v_threshold=1., tau=tau, surrogate_function=surrogate.ATan()))


        # self.layers = layers

        # initialization
        if init != 'torch':
            initialize_ = {
                'xavier': nn.init.xavier_uniform_,
                'he': nn.init.kaiming_uniform_
            }[init]

            for n, p in self.named_parameters():
                if p.dim() > 1:  # weights only
                    initialize_(p)
                else:  # bn w/b or bias
                    if 'bn.weight' in n:
                        nn.init.ones_(p)
                    else:
                        nn.init.zeros_(p)

    def forward(self, x, deepest=False):
        if deepest:
            assert self.training is False
        GB = int(x.size(1) * self.gdrop_ratio)
        out = x
        global_columns = None
        for each_layer in self.layers:
            if not self.consist_gdrop or global_columns is None:
                global_columns = np.random.randint(0, self.num_columns, size=[GB])
            if isinstance(each_layer, Fractal_Block):
                # if not self.consist_gdrop or global_columns is None:
                # global_columns = np.random.randint(0, self.num_columns, size=[GB])
                out = each_layer(out, global_columns, deepest=deepest)
                """
                print("F_B")
                print(out)                
                """

            else:
                out = each_layer(out)
                """
                print("Linear")
                print(out)

                """

        return out
