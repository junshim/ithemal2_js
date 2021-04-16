import sys
import os

from enum import Enum
import torch
from typing import Any, Callable, List, Optional, Iterator, Tuple, NamedTuple, Union

import data.data_cost as dt
import models.graph_models as md

BaseParameters = NamedTuple('BaseParameters', [
    ('data', str),
    ('embed_mode', str),
    ('embed_size', int),
    ('hidden_size', int),
    ('use_rnn', bool),
    ('rnn_type', md.RnnType),
    ('rnn_hierarchy_type', md.RnnHierarchyType),
    ('rnn_connect_tokens', bool),
    ('rnn_skip_connections', bool),
    ('rnn_learn_init', bool),
])

TrainParameters = NamedTuple('TrainParameters', [
    ('experiment_name', str),
    ('experiment_time', str),
    ('epochs', int),
    ('batch_size', int),
    ('initial_lr', float),
    ('lr_decay_rate', float),

])

PredictorDump = NamedTuple('PredictorDump', [
    ('model', md.AbstractGraphModule),
    ('dataset_params', Any),
])

def load_data(params):
    # type: (BaseParameters) -> dt.DataCost
    data = dt.load_dataset(params.data)

    return data

def load_model(params, data):
    # type: (BaseParameters, dt.DataCost) -> md.AbstractGraphModule

    if params.use_rnn:
        rnn_params = md.RnnParameters(
            embedding_size=params.embed_size,
            hidden_size=params.hidden_size,
            num_classes=1,
            connect_tokens=params.rnn_connect_tokens,
            skip_connections=params.rnn_skip_connections,
            hierarchy_type=params.rnn_hierarchy_type,
            rnn_type=params.rnn_type,
            learn_init=params.rnn_learn_init,
        )
        model = md.RNN(rnn_params)
    else:
        model = md.GraphNN(embedding_size=params.embed_size, hidden_size=params.hidden_size, num_classes=1,
                           use_residual=not params.no_residual, linear_embed=params.linear_embeddings,
                           use_dag_rnn=not params.no_dag_rnn, reduction=params.dag_reduction,
                           nonlinear_type=params.dag_nonlinearity, nonlinear_width=params.dag_nonlinearity_width,
                           nonlinear_before_max=params.dag_nonlinear_before_max,
        )

    model.set_learnable_embedding(mode=params.embed_mode, dictsize=628 or max(data.hot_idx_to_token) + 1)

    return model

def dump_model_and_data(model, data, fname):
    # type: (md.AbstractGraphMode, dt.DataCost, str) -> None
    try:
        os.makedirs(os.path.dirname(fname))
    except OSError:
        pass
    torch.save(PredictorDump(
        model=model,
        dataset_params=data.dump_dataset_params(),
    ), fname)

def load_model_and_data(fname):
    # type: (str) -> (md.AbstractGraphMode, dt.DataCost)
    dump = torch.load(fname)
    data = dt.DataInstructionEmbedding()
    data.read_meta_data()
    data.load_dataset_params(dump.dataset_params)
    return (dump.model, data)

