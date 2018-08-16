#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import glob
import sys

import torch

import onmt.io
import onmt.opts


def check_existing_pt_files(opt):
    # We will use glob.glob() to find sharded {train|valid}.[0-9]*.pt
    # when training, so check to avoid tampering with existing pt files
    # or mixing them up.
    for t in ['train.main', 'train.textonly', 'valid', 'vocab']:
        pattern = opt.save_data + '.' + t + '*.pt'
        if glob.glob(pattern):
            sys.stderr.write("Please backup exisiting pt file: %s, "
                             "to avoid tampering!\n" % pattern)
            sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description='preprocess_e2e.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    onmt.opts.add_md_help_argument(parser)
    onmt.opts.preprocess_e2e_opts(parser)

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    check_existing_pt_files(opt)

    return opt




def build_save_dataset(corpus_type, fields, opt):
    assert corpus_type in ['train.main', 'train.textonly', 'valid']

    if corpus_type == 'train.main':
        src_corpus = opt.train_main_src
        tgt_corpus = opt.train_main_tgt
        key_corpus = opt.train_main_key
    elif corpus_type == 'train.textonly':
        src_corpus = opt.train_textonly_src
        tgt_corpus = opt.train_textonly_tgt
        key_corpus = None
    else:
        src_corpus = opt.valid_src
        tgt_corpus = opt.valid_tgt
        key_corpus = opt.valid_key

    if opt.max_shard_size != 0:
        print(' * divide corpus into shards and build dataset separately'
              '(shard_size = %d lines).' % opt.max_shard_size)

    ret_list = []
    if corpus_type == 'train.textonly':
        iterator_type = onmt.io.E2EDataset.SimpleShardedCorpusIterator
        task = 'text-only'
    else:
        iterator_type = onmt.io.E2EDataset.KeyedShardedCorpusIterator
        task = 'main'
    src_iter = iterator_type(
        key_corpus,
        src_corpus,
        opt.src_seq_length_trunc,
        "src", task,
        opt.max_shard_size,
        use_chars=opt.use_chars)
    tgt_iter = iterator_type(
        key_corpus,
        tgt_corpus,
        opt.tgt_seq_length_trunc,
        "tgt", task,
        opt.max_shard_size,
        use_chars=opt.use_chars)

    index = 0
    while not src_iter.hit_end():
        index += 1
        dataset = onmt.io.E2EDataset.E2EDataset(
            fields, src_iter, tgt_iter)

        # We save fields in vocab.pt seperately, so make it empty.
        dataset.fields = []

        pt_file = "{:s}.{:s}.{:d}.pt".format(
            opt.save_data, corpus_type, index)
        print(" * saving %s data shard to %s." % (corpus_type, pt_file))
        torch.save(dataset, pt_file)

        ret_list.append(pt_file)

    return ret_list


def build_save_vocab(train_dataset, fields, opt):
    fields = onmt.io.build_vocab(train_dataset, fields, 'e2e',
                                 opt.share_vocab,
                                 opt.src_vocab,
                                 opt.src_vocab_size,
                                 opt.src_words_min_frequency,
                                 opt.tgt_vocab,
                                 opt.tgt_vocab_size,
                                 opt.tgt_words_min_frequency)

    # Can't save fields, so remove/reconstruct at training time.
    vocab_file = opt.save_data + '.vocab.pt'
    torch.save(onmt.io.save_fields_to_vocab(fields), vocab_file)


def main():
    opt = parse_args()

    src_nfeats = 0
    tgt_nfeats = 0

    if opt.reuse_vocab:
        print("Loading `Fields` object from %s ..." % opt.reuse_vocab)
        # Reuse previously created vocabulary,
        # to enable finetuning a model with additional data
        fields = onmt.io.load_fields_from_vocab(
            torch.load(opt.reuse_vocab, 'e2e'))
    else:
        print("Building `Fields` object...")
        fields = onmt.io.get_fields('e2e', src_nfeats, tgt_nfeats)

    train_dataset_files = []
    print("Building & saving main training data...")
    train_dataset_files.extend(build_save_dataset('train.main', fields, opt))
    print("Building & saving text-only training data...")
    train_dataset_files.extend(build_save_dataset('train.textonly', fields, opt))

    if opt.reuse_vocab:
        print("Re-saving vocabulary...")
        vocab_file = opt.save_data + '.vocab.pt'
        torch.save(onmt.io.save_fields_to_vocab(fields), vocab_file)
    else:
        print("Building & saving vocabulary...")
        build_save_vocab(train_dataset_files, fields, opt)

    print("Building & saving validation data...")
    build_save_dataset('valid', fields, opt)


if __name__ == "__main__":
    main()
