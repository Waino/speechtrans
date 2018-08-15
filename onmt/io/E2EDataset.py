# -*- coding: utf-8 -*-

from collections import Counter
from itertools import chain
import io
import codecs
import sys

import torch
import torchtext

from onmt.Utils import aeq
from onmt.io.TextDataset import TextDataset
from onmt.io.DatasetBase import (ONMTDatasetBase, UNK_WORD,
                                 PAD_WORD, BOS_WORD, EOS_WORD)


class E2EDataset(ONMTDatasetBase):
    """ Dataset for end-to-end speech translation """
    def __init__(self,
                 fields,
                 src_audio_examples_iter,
                 src_text_examples_iter,
                 tft_text_examples_iter):
        self.data_type = 'e2e'

        examples_iter = (self._join_dicts(src_audio, src_text, tgt_text)
                         for src_audio, src_text, tgt_text
                         in zip(src_audio_examples_iter,
                                src_text_examples_iter,
                                tft_text_examples_iter))

        # Peek at the first to see which fields are used.
        ex, examples_iter = self._peek(examples_iter)
        keys = ex.keys()

        out_fields = [(k, fields[k]) if k in fields else (k, None)
                      for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples_iter)

        filter_pred = lambda x: True
        super(TextDataset, self).__init__(
            out_examples, out_fields, filter_pred
        )

    def sort_key(self, ex):
        if hasattr(ex, "tgt"):
            return len(ex.src), len(ex.tgt)
        return len(ex.src)

    @staticmethod
    def make_text_examples_nfeats_tpl(path, truncate, side):
        return TextDataset.make_text_examples_nfeats_tpl(path, truncate, side)

    @staticmethod
    def make_audio_examples_nfeats_tpl(path):
        """
        Args:
            path (str): location of a src audio file.

        Returns:
            (example_dict iterator, 0) tuple.
        """
        # where example_dicts have the keys:
        # src_audio, src_audio_mask
        raise NotImplementedError() # FIXME Aku

    @staticmethod
    def get_fields():
        fields = {}

        fields["src"] = torchtext.data.Field(
            pad_token=PAD_WORD,
            include_lengths=True)

        fields["tgt"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD,
            pad_token=PAD_WORD)

        def pad_audio(data, vocab, is_train):
            # data: a sequence with one of whatever
            # was in src_audio in example_dicts
            # for each entry in the minibatch
            # returns: padded numpy float array

            # (minibatch_size, None, feat_len)
            # -> (minibatch_size, max_len, feat_len)
            raise NotImplementedError() # FIXME Aku
            

        def make_audio_mask(data, vocab, is_train):
            # data: a sequence with one of whatever
            # was in src_audio_mask in example_dicts
            # for each entry in the minibatch
            # returns: numpy float array with 0 if real value and 1 if padding

            # (minibatch_size, variable:lengths)
            # -> (minibatch_size, max_len)
            raise NotImplementedError() # FIXME Aku

        fields["src_audio"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.FloatTensor,
            postprocessing=pad_audio, sequential=False)

        fields["src_audio_mask"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.FloatTensor,
            postprocessing=make_audio_mask, sequential=False)

        fields["indices"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.LongTensor,
            sequential=False)

        return fields

class SimpleShardedCorpusIterator(object):
    """Makes shards of exactly shard_size examples"""
    def __init__(self, corpus_path, line_truncate, side, shard_size, use_chars=False):
        try:
            # The codecs module seems to have bugs with seek()/tell(),
            # so we use io.open().
            self.corpus = io.open(corpus_path, "r", encoding="utf-8")
        except IOError:
            sys.stderr.write("Failed to open corpus file: %s" % corpus_path)
            sys.exit(1)
        self.line_truncate = line_truncate
        self.side = side
        self.shard_size = shard_size
        self.use_chars = use_chars
        self.eof = False

    def __iter__(self):
        for i in range(self.shard_size):
            line = self.corpus.readline()
            if line == '':
                self.eof = True
                self.corpus.close()
                raise StopIteration
            yield self._example_dict_iter(line, i)

    def hit_end(self):
        return self.eof

    def _example_dict_iter(self, line, index):
        if self.use_chars:
            line = list(line)
        else:
            line = line.split()
        if self.line_truncate:
            line = line[:self.line_truncate]
        words, feats, n_feats = TextDataset.extract_text_features(line)
        example_dict = {self.side: words, "indices": index}
        assert not feats

        return example_dict
