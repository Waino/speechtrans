import argparse
import torch
import codecs
import os
import math

from torch.autograd import Variable
from itertools import count

import onmt.ModelConstructor
import onmt.translate.Beam
import onmt.io
import onmt.opts
import onmt.modules.Ensemble


def make_e2e_translator(opt, report_score=True, out_file=None, use_ensemble=False):
    assert not use_ensemble, 'Not implemented yet'
    if out_file is None:
        out_file = codecs.open(opt.output, 'w', 'utf-8')

    if opt.gpu > -1:
        torch.cuda.set_device(opt.gpu)

    dummy_parser = argparse.ArgumentParser(description='train.py')
    onmt.opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    if use_ensemble:
        fields, model, model_opt = \
            onmt.modules.Ensemble.load_test_model(opt, dummy_opt.__dict__)
    else:
        fields, model, model_opt = \
            onmt.ModelConstructor.load_test_model(opt, dummy_opt.__dict__)

    scorer = onmt.translate.GNMTGlobalScorer(opt.alpha,
                                             opt.beta,
                                             opt.coverage_penalty,
                                             opt.length_penalty)

    kwargs = {k: getattr(opt, k)
              for k in ["beam_size", "n_best", "max_length", "min_length",
                        "stepwise_penalty", "block_ngram_repeat",
                        "ignore_when_blocking", "dump_beam",
                        "data_type", "replace_unk", "gpu", "verbose"]}

    translator = E2ETranslator(model, fields, global_scorer=scorer,
                            out_file=out_file, report_score=report_score,
                            **kwargs)
    return translator


class E2ETranslator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
    """

    def __init__(self,
                 model,
                 fields,
                 beam_size,
                 n_best=1,
                 max_length=100,
                 global_scorer=None,
                 gpu=False,
                 dump_beam="",
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 ignore_when_blocking=[],
                 data_type="e2e",
                 report_score=True,
                 verbose=False,
                 out_file=None,
                 side='tgt'):
        self.gpu = gpu
        self.cuda = gpu > -1

        self.model = model
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.beam_size = beam_size
        self.min_length = min_length
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = set(ignore_when_blocking)
        self.data_type = data_type
        self.verbose = verbose
        self.out_file = out_file
        self.report_score = report_score
        assert side in ('src', 'tgt')
        self.side = side

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def translate(self, src_dir, src_path, tgt_path,
                  batch_size, attn_debug=False):
        # no dataset to build: just divide audio into minibatches

        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        vocab = self.fields[self.side].vocab

        all_scores = []
        for batch in data_iter:
            batch_data = self.translate_batch(batch, feats, feats_mask, vocab)
            translations = self.from_batch(batch_data, vocab)

            for trans in translations:
                pass

    def translate_batch(self, batch, feats, feats_mask, vocab):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch

        """

        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch.batch_size
        data_type = 'e2e'

        # Define a list of tokens to exclude from ngram-blocking
        # exclusion_list = ["<t>", "</t>", "."]
        exclusion_tokens = set([vocab.stoi[t]
                                for t in self.ignore_when_blocking])

        beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[onmt.io.PAD_WORD],
                                    eos=vocab.stoi[onmt.io.EOS_WORD],
                                    bos=vocab.stoi[onmt.io.BOS_WORD],
                                    min_length=self.min_length,
                                    stepwise_penalty=self.stepwise_penalty,
                                    block_ngram_repeat=self.block_ngram_repeat,
                                    exclusion_tokens=exclusion_tokens)
                for __ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a): return Variable(a, volatile=True)

        def rvar(a): return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # (1) Run the encoder on the audio.

        enc_states, memory_bank = self.model.src_aud_encoder(feats, feats_mask)

        if self.side == 'src':
            decoder = self.model.src_txt_decoder
        else:
            decoder = self.model.tgt_txt_decoder

        dec_states = decoder.init_decoder_state(
            feats_mask, memory_bank, enc_states)

        # (2) Repeat src objects `beam_size` times.
        if isinstance(memory_bank, tuple):
            memory_bank = tuple(rvar(x.data) for x in memory_bank)
        else:
            memory_bank = rvar(memory_bank.data)
        feats_mask = feats_mask.repeat(beam_size)
        dec_states.repeat_beam_size_times(beam_size)

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.get_current_state() for b in beam])
                      .t().contiguous().view(1, -1))

            # Temporary kludge solution to handle changed dim expectation
            # in the decoder
            inp = inp.unsqueeze(2)

            # Run one step.
            dec_out, dec_states, attn = decoder(
                inp, memory_bank, dec_states, mask=feats_mask)
            dec_out = dec_out.squeeze(0)
            # dec_out: beam x rnn_size

            # (b) Compute a vector of batch x beam word scores.
            out = self.model.generator.forward(dec_out).data
            out = unbottle(out)
            # beam x tgt_vocab
            beam_attn = unbottle(attn["std"])
            # (c) Advance each beam.
            for j, b in enumerate(beam):
                b.advance(out[:, j],
                          beam_attn.data[:, j, :])
                dec_states.beam_update(j, b.get_current_origin(), beam_size)

        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        ret["gold_score"] = [0] * batch_size
        ret["batch"] = batch
        return ret

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": [],
               "attention": []}
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
            ret["attention"].append(attn)
        return ret

    def from_batch(self, translation_batch, vocab):
        batch = translation_batch["batch"]
        batch_size = batch.batch_size

        preds, pred_score, attn, indices = list(zip(
            *sorted(zip(translation_batch["predictions"],
                        translation_batch["scores"],
                        translation_batch["attention"],
                        batch.indices.data),
                    key=lambda x: x[-1])))

        # Sorting
        inds, perm = torch.sort(batch.indices.data)
        src = batch.src[0].data.index_select(1, perm)
        tgt = batch.tgt.data.index_select(1, perm)

        translations = []
        for b in range(batch_size):
            #src_vocab = self.data.src_vocabs[inds[b]] \
            #    if self.data.src_vocabs else None
            #src_raw = self.data.examples[inds[b]].src
            #pred_sents = [self._build_target_tokens(
            #    src[:, b] if src is not None else None,
            #    src_vocab, src_raw,
            #    preds[b][n], attn[b][n])
            #              for n in range(self.n_best)]

            #translation = Translation(src[:, b] if src is not None else None,
            #                          src_raw, pred_sents,
            #                          attn[b], pred_score[b])
            tokens = []
            for tok in preds[b][0]:
                tokens.append(vocab.itos[tok])
                translations.append(tokens)

        return translations