#!/usr/bin/env python3
import pathlib
from kaldi_to_numpy import write_shard, write_key_list
import numpy as np

def filter_shard(shardpath, max_len_frames):
    shard = np.load(shardpath)
    too_longs = [array.shape[0] > max_len_frames for array in shard["mfccs"]]
    to_pick = np.where(np.logical_not(too_longs))
    return {"keys": shard["keys"][to_pick],
        "mfccs": shard["mfccs"][to_pick]}

def get_shardfiles(dirpath):
    sharddir = pathlib.Path(dirpath) 
    return (f for f in sharddir.iterdir() if f.name.endswith(".shard.npz"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Filters utterances in shards by length")
    parser.add_argument("shard_dir", metavar="shard-dir", help = "directory where to filter")
    parser.add_argument("max_len_frames", type = int, metavar="max-len-frames", help = "number of frames to include at max")
    parser.add_argument("out_dir", metavar = "out-dir", 
        help = "directory where to put output, if same as shard_dir will overwrite")
    args = parser.parse_args()
    outdir = pathlib.Path(args.out_dir)
    outdir.mkdir(exist_ok=True, parents=True)
    for shardfile in get_shardfiles(args.shard_dir):
        shardnum = shardfile.name.split(".")[1]
        shard_filt = filter_shard(shardfile, args.max_len_frames)
        shard_path = outdir / "keys_mfccs.{shardnum}.shard".format(shardnum=shardnum) 
        key_list_path = outdir / "keys.{shardnum}.txt".format(shardnum=shardnum)
        write_shard(shard_filt, shard_path)
        write_key_list(shard_filt["keys"], key_list_path)
    
