import argparse
import torch

import models
import train
import optim

from utils import set_seeds, get_device
from ithemal_utils import *
from experiments.experiment import Experiment

def main():
    # type: () -> None
    parser = argparse.ArgumentParser()

    # data arguments
    parser.add_argument('--data', required=True, help='The data file to load from')
    parser.add_argument('--train_cfg', required=True, help='Configuration for train')
    parser.add_argument('--model_cfg', required=True, help='Configuration of model')
    parser.add_argument('--experiment-name', required=True, help='Name of the experiment to run')
    parser.add_argument('--experiment-time', required=True, help='Time the experiment was started at')


    args = parser.parse_args()

    data = load_data(args.data)
    
    cfg = train.Config.from_json(args.train_cfg)
    model_cfg = models.Config.from_json(args.model_cfg)
    #model_cfg.set_vocab_size(len(data.token_to_hot_idx))
    model_cfg.set_vocab_size(628)

    set_seeds(cfg.seed)

    expt = Experiment(args.experiment_name, args.experiment_time)

    model = models.Ithemal(model_cfg)
    model = model.to(get_device())
    
    dump_model_and_data(model, data, os.path.join(expt.experiment_root_path(), 'predictor.dump'))
    trainer = train.Trainer(cfg, model, data, expt, "1", get_device())
    trainer.train()


if __name__ == '__main__':
    main()
