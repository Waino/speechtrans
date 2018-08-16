# -*- coding: utf-8 -*-

from collections import Counter
from itertools import chain
import io
import codecs
import sys
import pathlib

import torch
import torchtext

from onmt.Utils import aeq
from onmt.io.TextDataset import TextDataset
from onmt.io.DatasetBase import (ONMTDatasetBase, UNK_WORD,
                                 PAD_WORD, BOS_WORD, EOS_WORD)
import numpy as np

class E2EDataset(ONMTDatasetBase):
    """ Dataset for end-to-end speech translation """
    def __init__(self,
                 fields,
                 src_audio_examples_iter,
                 src_text_examples_iter,
                 tgt_text_examples_iter):
        self.data_type = 'e2e'

        examples_iter = (self._join_dicts(src_audio, src_text, tgt_text)
                         for src_audio, src_text, tgt_text
                         in zip(src_audio_examples_iter,
                                src_text_examples_iter,
                                tgt_text_examples_iter))

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
        raise NotImplementedError() # FIXME Aku NOTE: Not fixed but see next

    @staticmethod
    def make_minibatch_from_shard(shard_data, indices):
        """
        shard_data is the data given by SimpleAudioShardIterator (see bottom of file)
        basically shard_data = np.load(shardpath)["mfccs"]
        indices determine what is picked to the current minibatch
        """
        example_dict = {}
        arrays = np.array([shard_data[i] for i in indices])
        lengths = np.array([arr.shape[0] for arr in arrays])
        example_dict["src_audio"] = arrays
        example_dict["src_audio_mask"] = lengths
        return example_dict

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
            max_len = max(arr.shape[0] for arr in data)
            padded = np.zeros((len(data), max_len, data[0].shape[1]))
            for i,arr in enumerate(data):
                padded[i,:arr.shape[0],:] = arr
            return padded

        def make_audio_mask(data, vocab, is_train):
            # data: a sequence with one of whatever
            # was in src_audio_mask in example_dicts
            # for each entry in the minibatch
            # returns: numpy float array with 0 if real value and 1 if padding

            # (minibatch_size)
            # -> (minibatch_size, max_len)
            max_len = max(data)
            masks = np.ones((len(data), max_len))
            for i,length in enumerate(data):
                masks[i,:length] = 0
            return masks 

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
    def __init__(self, key_corpus, corpus_path, line_truncate, side, shard_size, use_chars=False):
        assert key_corpus is None
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
        example_dict = {self.side: words, "indices": index, 'task': 'text-only'}
        assert not feats

        return example_dict


class KeyedShardedCorpusIterator(object):
    shard_template = "keys.{shardnum}.txt"
    def __init__(self, key_corpus, corpus_path, line_truncate, side, shard_size, use_chars=False):
        assert key_corpus is not None
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

        self.dirpath = pathlib.Path(key_corpus)
        self.num_shards = len(path for path in self.dirpath.iterdir()
            if path.is_file() and path.endswith('.txt'))
        self.current_shard = 0
        self.current_line = 0

    def __iter__(self):
        # slurp in entire text corpus, index by key
        lines_by_key = {}
        for line in self.corpus:
            key, line = line.rstrip().split(' ', 1)
            lines_by_key[key] = line
        self.corpus.close()
        if self.current_shard >= self.num_shards:
            self.eof = True
            raise StopIteration
        for key in self.get_shard(i):
            line = lines_by_key[key]
            yield _example_dict_iter(line, self.current_line)
            self.current_line += 1
        self.current_shard += 1

    def get_shard(self, shard_index):
        filepath = self.dirpath / shard_template.format(shardnum = shard_index)
        with open(filepath, 'r') as fobj:
            for line in fobj:
                yield line.strip()

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
        example_dict = {self.side: words, "indices": index, 'task': 'main'}
        assert not feats

        return example_dict


class SimpleAudioShardIterator(object):
    shard_template = "keys_mfccs.{shardnum}.shard.npz"
    def __init__(self, shard_dir_path):
        self.dirpath = pathlib.Path(shard_dir_path)
        self.num_shards = len(path for path in self.dirpath.iterdir()
            if path.is_file() and path.endswith('.shard.npz'))

    def __iter__(self):
        #range instead of directly iterating on the dir contents
        #so that we can go through them in correct order
        for i in range(self.num_shards):
            data = self.get_shard(i)
            yield data["mfccs"]

    def get_shard(self, shard_index):
        filepath = self.dirpath / shard_template.format(shardnum = shard_index)
        return np.load(filepath)
        