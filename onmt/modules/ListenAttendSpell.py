# Listen, Attend and Spell model adapted from XenderLiu
# https://github.com/XenderLiu/Listen-Attend-and-Spell-Pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import onmt
from onmt.Models import EncoderBase


# BLSTM layer for pBLSTM
# Step 1. Reduce time resolution to half
# Step 2. Run through BLSTM
# Note the input should have timestep%2 == 0
class pBLSTMLayer(nn.Module):
    def __init__(self, input_feature_size, hidden_size,
                 rnn_type='LSTM', dropout=0.0):
        super(pBLSTMLayer, self).__init__()
        self.rnn_type = getattr(nn, rnn_type.upper())

        # feature dimension will be doubled since time resolution reduction
        self.BLSTM = self.rnn_type(input_feature_size * 2, hidden_size, 1,
                                   bidirectional=True,
                                   dropout=dropout,
                                   batch_first=True)

    def forward(self, input_x):
        batch_size = input_x.size(0)
        timestep = input_x.size(1)
        feature_size = input_x.size(2)
        # Reduce time resolution
        input_x = input_x.contiguous().view(batch_size,
                                            int(timestep/2),
                                            feature_size*2)
        # Bidirectional RNN
        output, hidden = self.BLSTM(input_x)
        return output, hidden


# Listener is a pBLSTM stacking 3 layers to reduce time resolution 8 times
# Input shape should be [# of sample, timestep, features]
class LasEncoder(EncoderBase):
    def __init__(self, input_feature_size, hidden_size,
                 rnn_type, dropout=0.0, num_layers=3, **kwargs):
        super(LasEncoder, self).__init__()
        modules = []
        for i in range(num_layers):
            if i == 0:
                modules.append(pBLSTMLayer(input_feature_size,
                                           hidden_size,
                                           rnn_type=rnn_type,
                                           dropout=dropout))
            else:
                modules.append(pBLSTMLayer(hidden_size * 2,
                                           hidden_size,
                                           rnn_type=rnn_type,
                                           dropout=dropout))
        self.modules = nn.ModuleList(modules)
        self.num_layers = num_layers

    def forward(self, input, lengths=None, hidden=None):
        """ See :obj:`EncoderBase.forward()`"""
        self._check_args(input, lengths, hidden)

        out = input
        for i in range(self.num_layers):
            out, hidden = self.modules[i](out)
        return hidden, out
