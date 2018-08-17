"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch
import torch.nn as nn
import collections

import onmt
import onmt.io
import onmt.Models
import onmt.modules
from onmt.Models import NMTModel, MeanEncoder, RNNEncoder, \
                        StdRNNDecoder, InputFeedRNNDecoder
from onmt.modules import Embeddings, ImageEncoder, CopyGenerator, \
                         TransformerEncoder, TransformerDecoder, \
                         CNNEncoder, CNNDecoder, AudioEncoder, \
                         LinkedEmbeddings, LasEncoder, E2EModel
from onmt.Utils import use_gpu
from torch.nn.init import xavier_uniform


def make_embeddings(opt, word_dict, feature_dicts, for_encoder=True):
    """
    Make an Embeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocab): words dictionary.
        feature_dicts([Vocab], optional): a list of feature dictionary.
        for_encoder(bool): make Embeddings for encoder or decoder?
    """
    try:
        if not for_encoder and opt.linked_embeddings is not None:
            return make_linked_embeddings(
                opt, word_dict, feature_dicts, for_encoder)
    except AttributeError:
        pass
    if for_encoder:
        embedding_dim = opt.src_word_vec_size
    else:
        embedding_dim = opt.tgt_word_vec_size

    word_padding_idx = word_dict.stoi[onmt.io.PAD_WORD]
    num_word_embeddings = len(word_dict)

    feats_padding_idx = [feat_dict.stoi[onmt.io.PAD_WORD]
                         for feat_dict in feature_dicts]
    num_feat_embeddings = [len(feat_dict) for feat_dict in
                           feature_dicts]

    return Embeddings(word_vec_size=embedding_dim,
                      position_encoding=opt.position_encoding,
                      feat_merge=opt.feat_merge,
                      feat_vec_exponent=opt.feat_vec_exponent,
                      feat_vec_size=opt.feat_vec_size,
                      dropout=opt.dropout,
                      word_padding_idx=word_padding_idx,
                      feat_padding_idx=feats_padding_idx,
                      word_vocab_size=num_word_embeddings,
                      feat_vocab_sizes=num_feat_embeddings,
                      sparse=opt.optim == "sparseadam")


def make_linked_embeddings(opt, word_dict, feature_dicts, for_encoder=True):
    """
    Make a LinkedEmbeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocab): words dictionary.
        feature_dicts([Vocab], optional): a list of feature dictionary.
        for_encoder(bool): make Embeddings for encoder or decoder?
    """
    if for_encoder:
        print('Warning: you probably do not want to use LinkedEmbeddings '
              'on the source side.')
        embedding_dim = opt.src_word_vec_size
    else:
        embedding_dim = opt.tgt_word_vec_size

    word_padding_idx = word_dict.stoi[onmt.io.PAD_WORD]
    num_word_embeddings = len(word_dict)

    cluster_name_to_idx = {}
    word_to_cluster_idx = collections.defaultdict(int)
    current_idx = 1
    with open(opt.linked_embeddings, 'r') as fobj:
        for line in fobj:
            word, cluster = line.strip().split('\t')
            if cluster not in cluster_name_to_idx:
                cluster_name_to_idx[cluster] = current_idx
                current_idx += 1
            word_to_cluster_idx[word] = cluster_name_to_idx[cluster]
    if opt.linked_default == 'identity':
        for word in word_dict.itos:
            if word not in word_to_cluster_idx:
                word_to_cluster_idx[word] = current_idx
                current_idx += 1
    cluster_mapping = [word_to_cluster_idx[word] for word in word_dict.itos]

    return LinkedEmbeddings(word_vec_size=embedding_dim,
                            linked_vec_size=opt.linked_vec_size,
                            word_vocab_size=num_word_embeddings,
                            word_padding_idx=word_padding_idx,
                            cluster_mapping=cluster_mapping,
                            position_encoding=opt.position_encoding,
                            dropout=opt.dropout,
                            sparse=opt.optim == "sparseadam")


def make_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    if opt.encoder_type == "transformer":
        return TransformerEncoder(opt.enc_layers, opt.rnn_size,
                                  opt.dropout, embeddings)
    elif opt.encoder_type == "cnn":
        return CNNEncoder(opt.enc_layers, opt.rnn_size,
                          opt.cnn_kernel_width,
                          opt.dropout, embeddings)
    elif opt.encoder_type == "mean":
        return MeanEncoder(opt.enc_layers, embeddings)
    else:
        # "rnn" or "brnn"
        return RNNEncoder(opt.rnn_type, opt.brnn, opt.enc_layers,
                          opt.rnn_size, opt.dropout, embeddings,
                          opt.bridge)


def make_decoder(opt, embeddings, layers=None):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    if layers is None:
        layers = opt.dec_layers
    if opt.decoder_type == "transformer":
        return TransformerDecoder(layers, opt.rnn_size,
                                  opt.global_attention, opt.copy_attn,
                                  opt.dropout, embeddings)
    elif opt.decoder_type == "cnn":
        return CNNDecoder(layers, opt.rnn_size,
                          opt.global_attention, opt.copy_attn,
                          opt.cnn_kernel_width, opt.dropout,
                          embeddings)
    elif opt.input_feed:
        return InputFeedRNNDecoder(opt.rnn_type, opt.brnn,
                                   layers, opt.rnn_size,
                                   opt.global_attention,
                                   opt.coverage_attn,
                                   opt.context_gate,
                                   opt.copy_attn,
                                   opt.dropout,
                                   embeddings,
                                   opt.reuse_copy_attn)
    else:
        return StdRNNDecoder(opt.rnn_type, opt.brnn,
                             layers, opt.rnn_size,
                             opt.global_attention,
                             opt.coverage_attn,
                             opt.context_gate,
                             opt.copy_attn,
                             opt.dropout,
                             embeddings,
                             opt.reuse_copy_attn)


def load_test_model(opt, dummy_opt, model_path=None):
    if model_path is None:
        model_path = opt.model
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)
    fields = onmt.io.load_fields_from_vocab(
        checkpoint['vocab'], data_type=opt.data_type)

    model_opt = checkpoint['opt']
    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]

    model = make_base_model(model_opt, fields,
                            use_gpu(opt), checkpoint)
    model.eval()
    model.generator.eval()
    return fields, model, model_opt


def make_base_model(model_opt, fields, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """
    assert model_opt.model_type in ["text", "img", "audio"], \
        ("Unsupported model type %s" % (model_opt.model_type))

    # Make encoder.
    if model_opt.model_type == "text":
        src_dict = fields["src"].vocab
        feature_dicts = onmt.io.collect_feature_vocabs(fields, 'src')
        src_embeddings = make_embeddings(model_opt, src_dict,
                                         feature_dicts)
        encoder = make_encoder(model_opt, src_embeddings)
    elif model_opt.model_type == "img":
        encoder = ImageEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout)
    elif model_opt.model_type == "audio":
        encoder = AudioEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout,
                               model_opt.sample_rate,
                               model_opt.window_size)

    # Make decoder.
    tgt_dict = fields["tgt"].vocab
    feature_dicts = onmt.io.collect_feature_vocabs(fields, 'tgt')
    tgt_embeddings = make_embeddings(
        model_opt, tgt_dict, feature_dicts, for_encoder=False)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        if src_dict != tgt_dict:
            raise AssertionError('The `-share_vocab` should be set during '
                                 'preprocess if you use share_embeddings!')

        tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

    decoder = make_decoder(model_opt, tgt_embeddings)

    # Make NMTModel(= encoder + decoder).
    model = NMTModel(encoder, decoder)
    model.model_type = model_opt.model_type

    # Make Generator.
    if not model_opt.copy_attn:
        generator = nn.Sequential(
            nn.Linear(model_opt.rnn_size, len(fields["tgt"].vocab)),
            nn.LogSoftmax(dim=-1))
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut_weight
    else:
        generator = CopyGenerator(model_opt.rnn_size,
                                  fields["tgt"].vocab)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform(p)
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform(p)

        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

    # Add generator to model (this registers it as parameter of model).
    model.generator = generator

    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model


def make_e2e_model(model_opt, fields, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the E2EModel.
    """

    ### Make embeddings
    src_dict = fields["src"].vocab
    feature_dicts = onmt.io.collect_feature_vocabs(fields, 'src')
    src_embeddings = make_embeddings(model_opt, src_dict,
                                     feature_dicts)
    # "for_encoder" actually means "source"
    tgt_dict = fields["tgt"].vocab
    feature_dicts = onmt.io.collect_feature_vocabs(fields, 'tgt')
    tgt_embeddings = make_embeddings(
        model_opt, tgt_dict, feature_dicts, for_encoder=False)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        if src_dict != tgt_dict:
            raise AssertionError('The `-share_vocab` should be set during '
                                 'preprocess if you use share_embeddings!')

        tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

    ### Make encoders.
    # source audio
    # LasEncoder is bidirectional, so output is twice the hidden size
    assert model_opt.rnn_size % 2 == 0
    src_aud_encoder = LasEncoder(model_opt.audio_feature_size,
                                 hidden_size=int(model_opt.rnn_size / 2),
                                 rnn_type=model_opt.rnn_type,
                                 dropout=model_opt.dropout,
                                 num_layers=model_opt.las_layers)
    # source text
    src_txt_encoder = make_encoder(model_opt, src_embeddings)

    ### Make decoders.
    # source text
    src_txt_decoder = make_decoder(model_opt, src_embeddings,
                                   layers=model_opt.src_decoder_layers)
    # target text
    tgt_txt_decoder = make_decoder(model_opt, tgt_embeddings)

    # Make end-to-end speech translation model
    model = E2EModel(src_aud_encoder,
                     src_txt_encoder,
                     src_txt_decoder,
                     tgt_txt_decoder)

    # Make Generators.
    src_generator = nn.Sequential(
        nn.Linear(model_opt.rnn_size, len(fields["src"].vocab)),
        nn.LogSoftmax(dim=-1))
    if model_opt.share_decoder_embeddings:
        src_generator[0].weight = src_txt_decoder.embeddings.word_lut.weight

    tgt_generator = nn.Sequential(
        nn.Linear(model_opt.rnn_size, len(fields["tgt"].vocab)),
        nn.LogSoftmax(dim=-1))
    if model_opt.share_decoder_embeddings:
        tgt_generator[0].weight = tgt_txt_decoder.embeddings.word_lut.weight

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])
        src_generator.load_state_dict(checkpoint['src_generator'])
        tgt_generator.load_state_dict(checkpoint['tgt_generator'])
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in src_generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in tgt_generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform(p)
            for p in src_generator.parameters():
                if p.dim() > 1:
                    xavier_uniform(p)
            for p in tgt_generator.parameters():
                if p.dim() > 1:
                    xavier_uniform(p)

        # FIXME: pretrained embeddings
        #if hasattr(model.encoder, 'embeddings'):
        #    model.encoder.embeddings.load_pretrained_vectors(
        #            model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
        #if hasattr(model.decoder, 'embeddings'):
        #    model.decoder.embeddings.load_pretrained_vectors(
        #            model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

    # Add generator to model (this registers it as parameter of model).
    model.src_generator = src_generator
    model.tgt_generator = tgt_generator

    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        print('moving model to gpu...')
        model.cuda()
        print('...done')
    else:
        model.cpu()

    return model
