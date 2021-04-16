import sys
import os

import argparse
import time
import torch
import models.losses as ls
#import models.train as tr
from tqdm import tqdm
from typing import Callable, List, Optional, Iterator, Tuple, NamedTuple, Union
import random
from ithemal_utils import *
import training
import pandas as pd
#import utilities as ut

def main():
    # type: () -> None
    parser = argparse.ArgumentParser()

    # data arguments
    parser.add_argument('--data', required=True, help='The data file to load from')
    parser.add_argument('--embed-mode', help='The embedding mode to use (default: none)', default='none')
    parser.add_argument('--embed-size', help='The size of embedding to use (default: 256)', default=256, type=int)
    parser.add_argument('--hidden-size', help='The size of hidden layer to use (default: 256)', default=256, type=int)

    parser.add_argument('--use-rnn', action='store_true', default=False)
    rnn_type_group = parser.add_mutually_exclusive_group()
    rnn_type_group.add_argument('--rnn-normal', action='store_const', const=md.RnnType.RNN, dest='rnn_type')
    rnn_type_group.add_argument('--rnn-lstm', action='store_const', const=md.RnnType.LSTM, dest='rnn_type')
    rnn_type_group.add_argument('--rnn-gru', action='store_const', const=md.RnnType.GRU, dest='rnn_type')
    parser.set_defaults(rnn_type=md.RnnType.LSTM)

    rnn_hierarchy_type_group = parser.add_mutually_exclusive_group()
    rnn_hierarchy_type_group.add_argument('--rnn-token', action='store_const', const=md.RnnHierarchyType.NONE, dest='rnn_hierarchy_type')
    rnn_hierarchy_type_group.add_argument('--rnn-dense', action='store_const', const=md.RnnHierarchyType.DENSE, dest='rnn_hierarchy_type')
    rnn_hierarchy_type_group.add_argument('--rnn-multiscale', action='store_const', const=md.RnnHierarchyType.MULTISCALE, dest='rnn_hierarchy_type')
    rnn_hierarchy_type_group.add_argument('--rnn-linear-model', action='store_const', const=md.RnnHierarchyType.LINEAR_MODEL, dest='rnn_hierarchy_type')
    rnn_hierarchy_type_group.add_argument('--rnn-mop', action='store_const', const=md.RnnHierarchyType.MOP_MODEL, dest='rnn_hierarchy_type')
    parser.set_defaults(rnn_hierarchy_type=md.RnnHierarchyType.MULTISCALE)

    parser.add_argument('--rnn-skip-connections', action='store_true', default=False)
    parser.add_argument('--rnn-learn-init', action='store_true', default=False)
    parser.add_argument('--rnn-connect-tokens', action='store_true', default=False)
    

    sp = parser.add_subparsers(dest='subparser')

    train = sp.add_parser('train', help='Train an ithemal model')
    train.add_argument('--experiment-name', required=True, help='Name of the experiment to run')
    train.add_argument('--experiment-time', required=True, help='Time the experiment was started at')
    train.add_argument('--epochs', type=int, default=30, help='Number of epochs to run for')
    train.add_argument('--batch-size', type=int, default=128, help='The batch size to use in train')
    train.add_argument('--initial-lr', type=float, default=0.1, help='Initial learning rate')
    train.add_argument('--lr-decay-rate', default=1.2, help='LR division rate', type=float)

    args = parser.parse_args()

    base_params = BaseParameters(
        data=args.data,
        embed_mode=args.embed_mode,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        use_rnn=args.use_rnn,
        rnn_type=args.rnn_type,
        rnn_hierarchy_type=args.rnn_hierarchy_type,
        rnn_connect_tokens=args.rnn_connect_tokens,
        rnn_skip_connections=args.rnn_skip_connections,
        rnn_learn_init=args.rnn_learn_init,
    )

    if args.subparser == 'train':
        train_params = TrainParameters(
            experiment_name=args.experiment_name,
            experiment_time=args.experiment_time,
            epochs=args.epochs,
            batch_size=args.batch_size,
            initial_lr=args.initial_lr,
            lr_decay_rate=args.lr_decay_rate,
        )
        trainer = training.load_trainer(base_params,train_params)
        trainer.train()
    
    else:
        raise ValueError('Unknown mode "{}"'.format(args.subparser))

if __name__ == '__main__':
    main()
