import argparse
import json
import os
import random


import numpy
import torch
from allennlp.common.params import Params
from allennlp.common.util import (cleanup_global_logging, dump_metrics,
                                  prepare_global_logging, import_submodules)
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import \
    PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders.text_field_embedder import \
    TextFieldEmbedder
from allennlp.nn import RegularizerApplicator
from allennlp.training.trainer import Trainer
from allennlp.training.util import create_serialization_dir

'''
The settings follow the config file of ELMO at
https://github.com/allenai/allennlp/blob/v0.8.3/training_config/ner.jsonnet
https://github.com/allenai/allennlp/blob/v0.8.3/training_config/ner_elmo.jsonnet
The only difference is that we have only one layer of bi-lstm, while there
are two in the original ELMO paper. see line 150 and 159 of this file
'''


def setup_arguments():
    parser = argparse.ArgumentParser(
        description='Training the (skip) NER Model')
    parser.add_argument('-d', '--serialization-dir',
                        required=True,
                        type=str,
                        help='directory in which to save the model and its logs')

    parser.add_argument('--seed', default=10, type=int,
                        help='The seed for reproducing the experiments.')

    parser.add_argument('--device', type=str,  default='0',
                        help='cuda device ids')

    parser.add_argument('-e', '--is_elmo', action='store_true', required=False,
                        help='embedding type, default is glove, if set, will use elmo')

    parser.add_argument('-l', '--is_doc', action='store_true',
                        required=False, help='is document level')

    parser.add_argument('-s', '--sent_input', action='store_true',
                        required=False, help='keep input as sentence level, only avaliable when is_doc is true')

    parser.add_argument('--data_type', type=str,
                        default='conll2003', help='datatype')

    parser_args = parser.parse_args()

    return parser_args


def setup_environment(serialization_dir, seed):
    create_serialization_dir(None, serialization_dir, False, True)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataset_reader(is_elmo, data_type, is_doc):
    if is_doc and 'conll2003' in data_type:
        data_type = data_type + '-doc'
    jsonpara = {
        "type": data_type,
        "coding_scheme": "BIOUL",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": True
            },
            "token_characters": {
                "type": "characters",
                "min_padding_length": 3
            }
        }
    }
    # conll2003 need this
    if 'conll2003' in data_type:
        jsonpara["tag_label"] = "ner"
    if is_elmo:
        jsonpara['token_indexers']['elmo'] = {
            "type": "elmo_characters"
        }
    return DatasetReader.from_params(params=Params(jsonpara))


def read_data(is_elmo, data_type, is_doc, is_sent_input):
    dataset_reader = get_dataset_reader(
        is_elmo=is_elmo, data_type=data_type, is_doc=is_doc)
    if is_doc and is_sent_input:
        dataset_reader.is_sent_input = True
    else:
        dataset_reader.is_sent_input = False
    train_data_path = "CoNLL2003/eng.train" if data_type == 'conll2003' else 'DocRED/train_annotated.json'
    validation_data_path = "CoNLL2003/eng.testa" if data_type == 'conll2003' else 'DocRED/dev.json'
    test_data_path = "CoNLL2003/eng.testb" if data_type == 'conll2003' else 'DocRED/test.json'

    train_data = dataset_reader.read(train_data_path)
    validation_data = dataset_reader.read(validation_data_path)
    test_data = dataset_reader.read(test_data_path)
    return train_data, validation_data, test_data


def construct_text_field_embedder(vocab, is_elmo, is_doc):
    jsonpara = {
        "token_embedders": {
            "tokens": {
                "type": "embedding",
                "embedding_dim": 50,
                "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.50d.txt.gz",
                "trainable": True
            },
            "token_characters": {
                "type": "character_encoding",
                "embedding": {
                    "embedding_dim": 16
                },
                "encoder": {
                    "type": "cnn",
                    "embedding_dim": 16,
                    "num_filters": 128,
                    "ngram_filter_sizes": [3],
                    "conv_layer_activation": "relu"
                }
            }
        },
    }

    if is_doc:
        jsonpara['token_embedders']["el_tags"] = {
            "type": "embedding",
            "embedding_dim": 50,
            "trainable": True
        }

    if is_elmo:
        jsonpara['token_embedders']['elmo'] = {
            "type": "elmo_token_embedder",
            "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
            "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
            "do_layer_norm": False,
            "dropout": 0.0
        }
    text_field_embedder = TextFieldEmbedder.from_params(
        vocab=vocab, params=Params(jsonpara))
    return text_field_embedder


def construct_model(vocab, args):
    # is_skip = args.skip
    is_elmo = args.is_elmo
    is_doc = args.is_doc
    text_field_embedder = construct_text_field_embedder(
        vocab=vocab, is_elmo=is_elmo, is_doc=is_doc)

    input_size = (50 + 128) if not is_elmo else (50 + 128 + 1024)
    if is_doc:
        input_size += 50

    encoder = PytorchSeq2SeqWrapper(
        torch.nn.LSTM(
            input_size=input_size,
            num_layers=1,
            hidden_size=200,
            bidirectional=True,
            batch_first=True
        )
    )

    regularizer = RegularizerApplicator.from_params([[
        "scalar_parameters",
        {
            "type": "l2",
            "alpha": 0.1
        }
    ]]) if is_elmo else None

    if is_doc:
        from my_crf_tagger import MyCrfTagger as CrfTagger
    else:
        from allennlp.models.crf_tagger import CrfTagger

    model = CrfTagger(
        vocab=vocab,
        text_field_embedder=text_field_embedder,
        encoder=encoder,
        label_encoding="BIOUL",
        constrain_crf_decoding=True,
        calculate_span_f1=True,
        dropout=0.5,
        include_start_end_transitions=False,
        regularizer=regularizer
    )

    return model


def train_model(args):
    seed = args.seed
    device = args.device
    is_elmo = args.is_elmo
    is_doc = args.is_doc
    is_sent_input=args.sent_input
    data_type = args.data_type
    serialization_dir = args.serialization_dir

    setup_environment(serialization_dir, seed)
    stdout_handler = prepare_global_logging(serialization_dir, True)

    train_data, validation_data, test_data = read_data(
        is_elmo=is_elmo, data_type=data_type, is_doc=is_doc, is_sent_input = is_sent_input)
    vocab = Vocabulary.from_instances(train_data + validation_data + test_data)
    vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

    model = construct_model(vocab, args)
    batch_size = 64
    if is_doc and is_elmo and not is_sent_input:
        # cuda memory limit
        batch_size= 32
    iterator = DataIterator.from_params(params=Params({
        "type": "basic",
        "batch_size": batch_size
    }))
    iterator.index_with(model.vocab)

    max_epoch = 75
    if is_doc:
        max_epoch += 100
    trainer = Trainer.from_params(
        model=model,
        serialization_dir=serialization_dir,
        iterator=iterator,
        train_data=train_data,
        validation_data=validation_data,
        params=Params({
            "optimizer": {
                "type": "adam",
                "lr": 0.001
            },
            "validation_metric": "+f1-measure-overall",
            "num_serialized_models_to_keep": 3,
            "num_epochs": max_epoch,
            "grad_norm": 5.0,
            "patience": 25,
            "cuda_device": 0 if device != '-1' else -1
        }))

    try:
        metrics = trainer.train()
    except:
        cleanup_global_logging(stdout_handler)
        raise

    dump_metrics(os.path.join(serialization_dir,
                 "metrics.json"), metrics, log=True)
    cleanup_global_logging(stdout_handler)


def test_model(args):
    is_elmo = args.is_elmo
    data_type = args.data_type
    serialization_dir = args.serialization_dir
    is_doc = args.is_doc
    is_sent_input=args.sent_input

    _, _, test_data = read_data(
        is_elmo=is_elmo, data_type=data_type, is_doc=is_doc, is_sent_input=is_sent_input)
    vocab = Vocabulary.from_files(
        os.path.join(serialization_dir, "vocabulary"))
    model = construct_model(vocab, args)
    model_state = torch.load(os.path.join(serialization_dir, "best.th"))
    model.load_state_dict(model_state)
    model.eval()

    model.get_metrics(reset=True)
    skip_tokens = []
    for _instance in test_data:
        output = model.forward_on_instance(_instance)
    metrics = model.get_metrics()
    with open(os.path.join(serialization_dir, 'test.metric.json'), 'w', encoding='utf8') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    args = setup_arguments()
    print(args)

    if args.device != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    import_submodules("datasetreader")
    train_model(args)
    test_model(args)
