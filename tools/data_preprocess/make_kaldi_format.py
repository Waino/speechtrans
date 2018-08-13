#!/usr/bin/env python3
import yaml

def line_iterator(filepath):
  with open(filepath, "r") as fi:
    for line in fi:
      yield line

def read_segments(filepath):
  with open(filepath) as fi:
    segments = yaml.load(fi))
  return segments

segments_template = "{uttid} {wavid} {start} {stop}"
wav_scp_template = "{wavid} /scratch/elec/puhe/p/speechtrans/data/iwslt-corpus/wav/{wavf}"
utt_spk_template = "{uttid} {spk}"
padded = "{:08}"
def parse_segments(segments):
  kaldi_segments = []
  wav_scp = []
  utt_spk = []
  uttids = []
  wavid_wav = []
  for segment in segments:
    start = segment["offset"]
    stop = start + segment["duration"]
    wavf = segment["wav"]
    wavid = wavf[:-4] #remove ".wav"
    uttid = "-".join([wavid, padded.format(start*100), padded.format(stop*100)])
    spk = segment["speaker_id"]
    kaldi_segments.append(segments_template.format(start = start,
      stop = stop, wavid = wavid, uttid = uttid))
    wav_scp.append(wav_scp_template.format(wavid = wavid, wavf = wavf))
    utt_spk.append(utt_spk_template.format(uttid = uttid, spk = spk))
    uttids.append(uttid)
  return kaldi_segments, wav_scp, utt_spk, uttids

def read_text(filepath):
  with open(filepath) as fi:
    for line in fi:
      yield line.strip()

text_template = "{uttid} {text}"
def parse_text(text, uttids):
  kaldi_text = []
  for uttid, text in zip(uttids, text):
    kaldi_text.append(text_template.format(uttid=uttid, text=text))
  return kaldi_text
  
def write_list(list_of_lines, outpath):
  with open(outpath, "w"), as fo:
    for line in list_of_lines:
      print(line, file=fo)



if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser("Create a Kaldi-style data-dir from IWSLT-corpus files")
  parser.add_argument("segments-yaml", help = "yaml file splitting the segments")
  parser.add_argument("english-text", help = "text file in the same order as the yaml")
  parser.add_argument("german-text", help = "German text file in the same order as the yaml")
  parser.add_argument("outdir", help = "dir where segments, utt2spk, text are created")
  args = parser.parse_args()

