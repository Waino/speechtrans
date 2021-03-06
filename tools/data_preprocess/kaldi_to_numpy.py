#!/usr/bin/env python3
# Copyright 2018 Aalto University (author: Aku Rouhe)
# Licensed under the Apache License, Version 2.0
#
# This script concatenates Kaldi format data into numpy_arrays and saves in numpy format

import kaldi_io
import numpy as np


#These will be set again in if __name__ == "__main__" block, if run as script:
ENCODING = 'utf-8' 
COMPRESSION = 'GZIP'


#Utility functions:
def prepend_generator(generator, element):
    return ((element, *values) for values in generator)

def get_keys(example):
    return [key for name, key, feature in example]

def equal_keys(example):
    keys = get_keys(example)
    return all(key == keys[0] for key in keys)

def extract_key_as_feature(example):
    keys = get_keys(example)
    as_feature = keys[0]
    return ("key", keys[0], as_feature) 


#shard generation and writing
def make_shards(generator, shard_size):
    shard = {}
    for name, key, data in generator:
        shard.setdefault("keys", []).append(key)
        shard.setdefault("mfccs", []).append(data)
        if len(shard["keys"]) == shard_size:
            yield shard
            shard = {}
    if len(shard.get("keys", [])) > 0:
        yield shard
    raise StopIteration()

def write_shard(shard, filepath):
    np.savez_compressed(filepath, **shard)
def write_key_list(key_list, filepath):
    with open(filepath, "w") as fo:
        for key in key_list:
            print(key, file=fo)

def read_exclusion_list(path):
    with open(path) as fi:
        return set([line.strip() for line in fi])
def filter_with_exclusion_list(generator, exclude_list):
    return ((name, key, data) for name, key, data in generator if key not in exclude_list)

def filter_by_max_len(generator, max_len):
    return ((name, key, data) for name, key, data in generator if data.shape[0] <= max_len)

#Data readers (as python generators):
def feats_ark_generator(ark, name):
    generator = kaldi_io.read_mat_ark(ark)
    return prepend_generator(generator, name)

def feats_scp_generator(scp, name):
    generator = kaldi_io.read_mat_scp(scp)
    return prepend_generator(generator, name)

def text_generator(path, name):
    generator = kaldi_io.read_vec_int_ark(path)
    return prepend_generator(generator, name)

if __name__  == "__main__":
    import argparse
    import pathlib
    parser = argparse.ArgumentParser("""
    Concatenate Kaldi-style IO and write out numpy arrays
    All inputs data have to be sorted and filtered the same way.
    Use 'utils/filter_scp.pl' and 'sort' (LC_ALL=C when sorting in Kaldi)
    """)
    parser.add_argument("feats_scp", metavar="feats.scp",
            help = "Read float matrices pointed to by an scp")
    parser.add_argument("outdir",
            help = "Path to directory where the shards are placed")
    parser.add_argument("--encoding",
            default = 'utf-8',
            help = "Change text encoding. Affects at least how the key is encoded")
    parser.add_argument("--shard-size", 
            type=int,
            default = 2048,
            help = "The size of shards the set is divided into")
    parser.add_argument("--use-ark", default = False,
            help = "Read the given feats.scp as if it was an ark format.")
    parser.add_argument("--exclusion-list-file", help = "List of keys to exclude")
    parser.add_argument("--max-frame-length", type = int,
        help = "Maximum length of utterance in frames, default is no maximum", default=0)
    args = parser.parse_args()
    ENCODING = args.encoding
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.use_ark:
        generator = feats_ark_generator(args.feats_scp, "mfcc")
    else:
        generator = feats_scp_generator(args.feats_scp, "mfcc")
    if args.exclusion_list_file:
        print("Reading exclusion list from:", args.exclusion_list_file)
        exclusion_list = read_exclusion_list(args.exclusion_list_file)
        generator = filter_with_exclusion_list(generator, exclusion_list)
    if args.max_frame_length > 0:
        print("Filtering utterances longer than", args.max_frame_length, "frames")
        generator = filter_by_max_len(generator, args.max_frame_length)
    for i, shard in enumerate(make_shards(generator, args.shard_size)):
        print("Processing shard number", i)
        shard_path = outdir / "keys_mfccs.{shardnum}.shard".format(shardnum=str(i)) 
        key_list_path = outdir / "keys.{shardnum}.txt".format(shardnum=str(i))
        write_shard(shard, shard_path)
        write_key_list(shard["keys"], key_list_path)

