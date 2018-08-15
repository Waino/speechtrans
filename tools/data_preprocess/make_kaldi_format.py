#!/usr/bin/env python3
import yaml

def line_iterator(filepath):
  with open(filepath, "r") as fi:
    for line in fi:
      yield line

def read_segments(filepath):
  with open(filepath) as fi:
    return yaml.load(fi)

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
    uttid = "-".join([wavid, padded.format(int(start*100)), padded.format(int(stop*100))])
    spk = segment["speaker_id"]
    kaldi_segments.append(segments_template.format(start = start,
      stop = stop, wavid = wavid, uttid = uttid))
    #Wav_scp will have many duplicates:
    wav_scp.append(wav_scp_template.format(wavid = wavid, wavf = wavf))
    utt_spk.append(utt_spk_template.format(uttid = uttid, spk = spk))
    uttids.append(uttid)
  wav_scp = list(set(wav_scp)) #Remove duplicates
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
  with open(outpath, "w") as fo:
    for line in list_of_lines:
      print(line, file=fo)



if __name__ == "__main__":
  import argparse
  import pathlib
  parser = argparse.ArgumentParser("Create a Kaldi-style data-dir from IWSLT-corpus files. "+
    "Run utils/fix_data_dir.sh on the directory after this to create spk2utt and validate.")
  parser.add_argument("segments_yaml", metavar="segments-yaml", help = "yaml file splitting the segments")
  parser.add_argument("english_text", metavar="english-text", help = "text file in the same order as the yaml")
  parser.add_argument("german_text", metavar="german-text", help = "German text file in the same order as the yaml")
  parser.add_argument("outdir", help = "dir where segments, wav.scp, utt2spk, text are created")
  args = parser.parse_args()

  segments = read_segments(args.segments_yaml)
  english = read_text(args.english_text)
  german = read_text(args.german_text)
  kaldi_segments, wav_scp, utt_spk, uttids = parse_segments(segments)
  text_en = parse_text(english, uttids)
  text_de = parse_text(german, uttids)

  outdir = pathlib.Path(args.outdir)
  outdir.mkdir(parents = True, exist_ok = True)
  write_list(kaldi_segments, outdir / "segments")
  write_list(wav_scp, outdir / "wav.scp")
  write_list(utt_spk, outdir / "utt2spk")
  write_list(text_en, outdir / "text")
  write_list(text_de, outdir / "text.de")

