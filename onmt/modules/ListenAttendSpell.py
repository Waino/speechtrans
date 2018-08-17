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
        assert timestep % 2 == 0
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
        self.las_modules = nn.ModuleList(modules)
        self.num_layers = num_layers

    def forward(self, input, src_pad_mask=None, hidden=None):
        """ See :obj:`EncoderBase.forward()`"""
        self._check_args(input, None, hidden)

        out = input
        for i in range(self.num_layers):
            out, hidden = self.las_modules[i](out)
        # out is transposed to (timestep, batch, feats)
        return hidden, out.transpose(0, 1)


class E2EModel(nn.Module):
    def __init__(self, src_aud_encoder, src_txt_encoder,
                 src_txt_decoder, tgt_txt_decoder):
        super(E2EModel, self).__init__()
        self.src_aud_encoder = src_aud_encoder
        self.src_txt_encoder = src_txt_encoder
        self.src_txt_decoder = src_txt_decoder
        self.tgt_txt_decoder = tgt_txt_decoder

    def forward(self, feats, feats_mask, src, tgt, task='main'):
        if task == 'main':
            assert feats is not None
            encoder = self.src_aud_encoder
            inp = feats
            mask = feats_mask
            print('audio inp', inp.size())
            print('audio mask', mask.size())
        elif task == 'text-only':
            if feats is not None:
                print(type(feats))
            #assert feats is None
            encoder = self.src_txt_encoder
            inp = src
            padding_idx = self.src_txt_encoder.embeddings.word_padding_idx
            mask = src.data.eq(padding_idx)
            print('text inp', inp.size())
            print('text mask', mask.size())
        #elif task = 'asr':
        else:
            raise Exception('Unknown task "{}"'.format(task))

        print('src', src.size())
        print('trg', src.size())
        # exclude last timestep from inputs
        src_feed = src[:-1]
        tgt_feed = tgt[:-1]

        # encode with appropriate encoder
        enc_final, memory_bank = encoder(inp, mask)
        if task == 'main':
            # the audio feature timestep has been reduced,
            # so the padding mask is no longer valid
            print('before', mask.size())
            mask_type = type(mask.data)
            mask = Variable(mask_type(
                memory_bank.size(0), memory_bank.size(1)).zero_())
            print('after', mask.size())
        else:
            mask = mask.transpose(0, 1)

        # decode to src
        enc_state = self.src_txt_decoder.init_decoder_state(
            mask, memory_bank, enc_final)
        print('*** to src')
        print('src_feed', src_feed.size())
        print('memory_bank', memory_bank.size())
        print('enc_state.mask', enc_state.mask.size())
        print('mask', mask.size())
        # (decoder_outputs, dec_state, attns)
        src_txt_decoder_out = self.src_txt_decoder(
            src_feed, memory_bank, enc_state,
            mask=mask)

        print('*** to tgt')
        print('src_feed', src_feed.size())
        print('memory_bank', memory_bank.size())
        print('enc_state.mask', enc_state.mask.size())
        print('mask', mask.size())
        # decode to tgt
        enc_state = self.tgt_txt_decoder.init_decoder_state(
            mask, memory_bank, enc_final)
        # (decoder_outputs, dec_state, attns)
        tgt_txt_decoder_out = self.tgt_txt_decoder(
            tgt_feed, memory_bank, enc_state,
            mask=mask)

        return src_txt_decoder_out, tgt_txt_decoder_out
