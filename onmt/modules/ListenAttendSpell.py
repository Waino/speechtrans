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
        # Init encoder->decoder hidden state projection layer:
        self.num_encoder_layers = len(self.src_aud_encoder.las_modules)
        num_decoder_layers = self.src_txt_decoder.num_layers
        hidden_size = self.src_aud_encoder.las_modules[-1].BLSTM.hidden_size*2
        self.bridge_hidden_projections = [torch.nn.Linear(hidden_size, hidden_size)
                              for _ in range(num_decoder_layers)]
        self.bridge_cell_projections = [torch.nn.Linear(hidden_size, hidden_size)
                              for _ in range(num_decoder_layers)]

    def forward(self, feats, feats_mask, src, tgt, task='main'):
        if task == 'main':
            assert feats is not None
            encoder = self.src_aud_encoder
            inp = feats
            mask = feats_mask
        elif task == 'text-only':
            assert feats is None
            encoder = self.src_txt_encoder
            inp = src
            padding_idx = self.src_txt_encoder.embeddings.word_padding_idx
            mask = src.data.eq(padding_idx).squeeze(2)
        else:
            raise Exception('Unknown task "{}"'.format(task))

        # exclude last timestep from inputs
        src_feed = src[:-1]
        tgt_feed = tgt[:-1]

        # encode with appropriate encoder
        enc_final, memory_bank = encoder(inp, mask)

        # fix as done in fix onmt/Models.py->RNNDecoderBase->init_decoder_state
        # The encoder hidden is  (layers*directions) x batch x dim.
        # We need to convert it to layers x batch x (directions*dim).
        # NOTE: we are assuming directions = 2 as the LAS encoder should be bidirectional
        # NOTE: do not use with -brnn flag, as that will automatically try to fix this same thing again
        enc_hidden_final, enc_cell_final = [
            torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            for h in enc_final]

        # Project to encoder hidden state to each decoder layer hidden state init separately
        # These will be decoder_layers x batch x (directions*dim)
        projected_hidden_states = torch.cat([
            hidden_proj(enc_hidden_final) for hidden_proj in self.bridge_hidden_projections], 0)
        projected_cell_states = torch.cat([
            cell_proj(enc_hidden_final) for cell_proj in self.bridge_cell_projections], 0)
        projected_encoder_states = (projected_hidden_states, projected_cell_states)
           
        # NOTE: The mask is never recalculated!
        # TODO: Fix this bug
        #if task == 'main':
        #    # the audio feature timestep has been reduced,
        #    # so the padding mask is no longer valid
        #    mask_type = type(mask.data)
        #    mask = mask_type(
        #        memory_bank.size(0), memory_bank.size(1)).zero_()
        #mask = mask.byte()
        #memory_lengths = torch.sum(1 - mask, dim=0)
        #print(memory_lengths)

        print(mask)
        if task == 'main':
            memory_lenghts = torch.sum(1 - mask[::2**self.num_encoder_layers], dim=0)
        print(memory_lenghts)

        # decode to src
        enc_state = self.src_txt_decoder.init_decoder_state(
            mask, memory_bank, projected_encoder_states)
        # (decoder_outputs, dec_state, attns)
        src_txt_decoder_out = self.src_txt_decoder(
            src_feed, memory_bank, enc_state,
            memory_lengths = memory_lengths)

        # decode to tgt
        enc_state = self.tgt_txt_decoder.init_decoder_state(
            mask, memory_bank, projected_encoder_states)
        # (decoder_outputs, dec_state, attns)
        tgt_txt_decoder_out = self.tgt_txt_decoder(
            tgt_feed, memory_bank, enc_state,
            memory_lengths = memory_lengths)

        return src_txt_decoder_out, tgt_txt_decoder_out
