from __future__ import division
"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
See train.py for a use case of this library.

Note!!! To make this a general library, we implement *only*
mechanism things here(i.e. what to do), and leave the strategy
things to users(i.e. how to do it). Also see train.py(one of the
users of this library) for the strategy things we do.
"""
import time
import sys
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

import onmt
import onmt.io
import onmt.modules

class TaskStatistics(object):
    def __init__(self):
        self.main = Statistics()
        self.tt = Statistics()
        self.asr = Statistics()
        self.ae = Statistics()

    def update(self, src_stat, trg_stat, task):
        if task == 'main':
            self.main.update(trg_stat)
            self.asr.update(src_stat)
        else:
            self.tt.update(trg_stat)
            self.ae.update(src_stat)

    def output(self, *args):
        self.main.output(*args, extra='main')
        self.tt.output(*args, extra='tt')
        self.asr.output(*args, extra='asr')
        self.ae.output(*args, extra='ae')

    def ppl(self):
        return self.main.ppl()

    def accuracy(self):
        return self.main.accuracy()

    def elapsed_time(self):
        return self.main.elapsed_time()

    @property
    def start_time(self):
        return self.main.start_time
        

class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """
    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def accuracy(self):
        if self.n_words == 0:
            return 0
        return 100 * (self.n_correct / self.n_words)

    def xent(self):
        if self.n_words == 0:
            return 0
        return self.loss / self.n_words

    def ppl(self):
        if self.n_words == 0:
            return 0
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start, extra=''):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; xent: %6.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed %s") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.ppl(),
               self.xent(),
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start,
               extra))
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)

    def log_tensorboard(self, prefix, writer, lr, step):
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "/tgtper",  self.n_words / t, step)
        writer.add_scalar(prefix + "/lr", lr, step)


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train

            train_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32, data_type='text',
                 norm_method="sents", grad_accum_count=1):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.progress_step = 0

        assert(grad_accum_count > 0)
        if grad_accum_count > 1:
            assert(self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def train(self, train_iter, epoch, report_func=None):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        total_stats = Statistics()
        report_stats = Statistics()
        idx = 0
        true_batchs = []
        accum = 0
        normalization = 0
        try:
            add_on = 0
            if len(train_iter) % self.grad_accum_count > 0:
                add_on += 1
            num_batches = len(train_iter) / self.grad_accum_count + add_on
        except NotImplementedError:
            # Dynamic batching
            num_batches = -1

        for i, batch in enumerate(train_iter):
            cur_dataset = train_iter.get_cur_dataset()
            self.train_loss.cur_dataset = cur_dataset

            true_batchs.append(batch)
            accum += 1
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:].data.view(-1) \
                    .ne(self.train_loss.padding_idx).sum()
                normalization += num_tokens
            else:
                normalization += batch.batch_size

            if accum == self.grad_accum_count:
                self._gradient_accumulation(
                        true_batchs, total_stats,
                        report_stats, normalization)

                if report_func is not None:
                    report_stats = report_func(
                            epoch, idx, num_batches,
                            self.progress_step,
                            total_stats.start_time, self.optim.lr,
                            report_stats)
                    self.progress_step += 1

                true_batchs = []
                accum = 0
                normalization = 0
                idx += 1

        if len(true_batchs) > 0:
            self._gradient_accumulation(
                    true_batchs, total_stats,
                    report_stats, normalization)
            true_batchs = []

        return total_stats

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics()

        for batch in valid_iter:
            cur_dataset = valid_iter.get_cur_dataset()
            self.valid_loss.cur_dataset = cur_dataset

            src = onmt.io.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
            else:
                src_lengths = None

            tgt = onmt.io.make_features(batch, 'tgt')

            # F-prop through the model.
            outputs, attns, _ = self.model(src, tgt, src_lengths)

            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(
                    batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def epoch_step(self, ppl, epoch):
        return self.optim.update_learning_rate(ppl, epoch)

    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': onmt.io.save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))

    def _gradient_accumulation(self, true_batchs, total_stats,
                               report_stats, normalization):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            target_size = batch.tgt.size(0)
            # Truncated BPTT
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            dec_state = None
            src = onmt.io.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
                report_stats.n_src_words += src_lengths.sum()
            else:
                src_lengths = None

            tgt_outer = onmt.io.make_features(batch, 'tgt')

            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.model.zero_grad()
                outputs, attns, dec_state = \
                    self.model(src, tgt, src_lengths, dec_state)

                # 3. Compute loss in shards for memory efficiency.
                batch_stats = self.train_loss.sharded_compute_loss(
                        batch, outputs, attns, j,
                        trunc_size, self.shard_size, normalization)

                # 4. Update the parameters and statistics.
                if self.grad_accum_count == 1:
                    self.optim.step()
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

        if self.grad_accum_count > 1:
            self.optim.step()


class E2ETrainer(Trainer):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train

            train_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
    """

    def __init__(self, model, src_train_loss, tgt_train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32, data_type='text',
                 norm_method="sents", grad_accum_count=1,
                 e2e_audio=None, las_layers=3, cuda=True, truncate_feat=None,
                 ae_weight=1.0, main_weight=1.0):
        # Basic attributes.
        self.model = model
        self.src_train_loss = src_train_loss
        self.tgt_train_loss = tgt_train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.progress_step = 0
        self.e2e_audio = e2e_audio
        self.las_layers = las_layers
        self.cuda = cuda
        self.truncate_feat = truncate_feat
        self.ae_weight = ae_weight
        self.main_weight = main_weight

        assert(self.trunc_size == 0)
        assert(grad_accum_count > 0)
        if self.truncate_feat is not None:
            time_reduction_ratio = 2 ** self.las_layers
            assert self.truncate_feat % time_reduction_ratio == 0

        # Set model in training mode.
        self.model.train()

    def train(self, train_iter, epoch, report_func=None, extra_checkpoint=None):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        total_stats = TaskStatistics()
        report_stats = TaskStatistics()
        idx = 0
        true_batchs = []
        accum = 0
        normalization = 0
        try:
            add_on = 0
            if len(train_iter) % self.grad_accum_count > 0:
                add_on += 1
            num_batches = len(train_iter) / self.grad_accum_count + add_on
        except (TypeError, NotImplementedError):
            # Dynamic batching
            num_batches = -1

        for i, batch in enumerate(train_iter):
            cur_dataset = train_iter.get_cur_dataset()
            self.src_train_loss.cur_dataset = cur_dataset
            self.tgt_train_loss.cur_dataset = cur_dataset

            true_batchs.append(batch)
            accum += 1
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:].data.view(-1) \
                    .ne(self.tgt_train_loss.padding_idx).sum()
                normalization += num_tokens
            else:
                normalization += batch.batch_size

            if accum == self.grad_accum_count:
                self._gradient_accumulation(
                        true_batchs, total_stats,
                        report_stats, normalization)

                if report_func is not None:
                    report_stats = report_func(
                            epoch, idx, num_batches,
                            self.progress_step,
                            total_stats.start_time, self.optim.lr,
                            report_stats)
                    self.progress_step += 1

                true_batchs = []
                accum = 0
                normalization = 0
                idx += 1

            if extra_checkpoint is not None and i > 0 and i % 2000 == 0:
                extra_checkpoint(i)

        if len(true_batchs) > 0:
            self._gradient_accumulation(
                    true_batchs, total_stats,
                    report_stats, normalization)
            true_batchs = []

        return total_stats

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics()

        for batch in valid_iter:
            cur_dataset = valid_iter.get_cur_dataset()
            self.valid_loss.cur_dataset = cur_dataset

            src = onmt.io.make_features(batch, 'src', self.data_type)
            _, src_lengths = batch.src
            tgt = onmt.io.make_features(batch, 'tgt')

            # absurd boilerplate because torchtext doesn't
            # allow passing through non-numerical data
            task = batch.dataset.fields['task'].vocab.itos[batch.task.data[0]]
            if task == 'text-only':
                feats, feats_mask = None, None
            else:
                shard_idx = batch.shard_idx.data[0]
                feat_idx = batch.feat_idx.data
                feats, feats_mask = self.e2e_audio.get_minibatch_features(
                    shard_idx, feat_idx, self.las_layers,
                    truncate=self.truncate_feat)
                feats = Variable(torch.FloatTensor(feats),
                                    requires_grad=False)
                feats_mask = Variable(torch.FloatTensor(feats_mask),
                                        requires_grad=False)
                if self.cuda:
                    feats = feats.cuda()
                    feats_mask = feats_mask.cuda()

            # F-prop through the model.
            # FIXME: flag to disable src decoder
            # each is a (outputs, attns, dec_state) tuple
            src_txt_decoder_out, tgt_txt_decoder_out = \
                self.model(feats, feats_mask,
                            src, tgt,
                            task=task)
            outputs, attns, dec_state = tgt_txt_decoder_out

            # Compute loss.
            _, batch_stats = self.tgt_train_loss.just_compute_loss(
                    batch, outputs)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def epoch_step(self, ppl, epoch):
        return self.optim.update_learning_rate(ppl, epoch)

    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_src_generator = (real_model.src_generator.module
                          if isinstance(real_model.src_generator, nn.DataParallel)
                          else real_model.src_generator)
        real_tgt_generator = (real_model.tgt_generator.module
                          if isinstance(real_model.tgt_generator, nn.DataParallel)
                          else real_model.tgt_generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        src_generator_state_dict = real_src_generator.state_dict()
        tgt_generator_state_dict = real_tgt_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'src_generator': src_generator_state_dict,
            'tgt_generator': tgt_generator_state_dict,
            'vocab': onmt.io.save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))

    def drop_extra_checkpoint(self, opt, epoch, fields, i):
        """ Save an additional mid-epoch checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            i (int): mb number
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_src_generator = (real_model.src_generator.module
                          if isinstance(real_model.src_generator, nn.DataParallel)
                          else real_model.src_generator)
        real_tgt_generator = (real_model.tgt_generator.module
                          if isinstance(real_model.tgt_generator, nn.DataParallel)
                          else real_model.tgt_generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        src_generator_state_dict = real_src_generator.state_dict()
        tgt_generator_state_dict = real_tgt_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'src_generator': src_generator_state_dict,
            'tgt_generator': tgt_generator_state_dict,
            'vocab': onmt.io.save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
        }
        torch.save(checkpoint,
                   '%s_extra_ep%d_mb%d.pt'
                   % (opt.save_model, epoch, i))

    def _gradient_accumulation(self, true_batchs, total_stats,
                               report_stats, normalization):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            dec_state = None
            src = onmt.io.make_features(batch, 'src', self.data_type)
            src_lengths = None
            # absurd boilerplate because torchtext doesn't
            # allow passing through non-numerical data
            task = batch.dataset.fields['task'].vocab.itos[batch.task.data[0]]
            if task == 'text-only':
                feats, feats_mask = None, None
            else:
                shard_idx = batch.shard_idx.data[0]
                feat_idx = batch.feat_idx.data
                feats, feats_mask = self.e2e_audio.get_minibatch_features(
                    shard_idx, feat_idx, self.las_layers,
                    truncate=self.truncate_feat)
                feats = Variable(torch.FloatTensor(feats),
                                    requires_grad=False)
                feats_mask = Variable(torch.FloatTensor(feats_mask),
                                        requires_grad=False)
                if self.cuda:
                    feats = feats.cuda()
                    feats_mask = feats_mask.cuda()
            assert not self.trunc_size

            src = onmt.io.make_features(batch, 'src')
            tgt = onmt.io.make_features(batch, 'tgt')

            # 2. F-prop all but generator.
            if self.grad_accum_count == 1:
                self.model.zero_grad()
            # each is a (outputs, attns, dec_state) tuple
            src_txt_decoder_out, tgt_txt_decoder_out = \
                self.model(feats, feats_mask,
                            src, tgt,
                            task=task)

            # 3. Compute loss in shards for memory efficiency.
            # src side
            outputs, attns, dec_state = src_txt_decoder_out
            src_loss, src_batch_stats = self.src_train_loss.just_compute_loss(
                    batch, outputs)
            if task == 'text-only':
                src_loss *= self.ae_weight

            # tgt side
            outputs, attns, dec_state = tgt_txt_decoder_out
            tgt_loss, tgt_batch_stats = self.tgt_train_loss.just_compute_loss(
                    batch, outputs)
            if task == 'main':
                tgt_loss *= self.main_weight

            loss = src_loss + tgt_loss
            loss.div(normalization).backward()

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                self.optim.step()
            total_stats.update(src_batch_stats, tgt_batch_stats, task)
            report_stats.update(src_batch_stats, tgt_batch_stats, task)

        if self.grad_accum_count > 1:
            self.optim.step()
