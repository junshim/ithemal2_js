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
    #trainer = train.Trainer(cfg, model, data, expt, optim.optim4GPU(cfg, model), get_device())
    trainer = train.Trainer(cfg, model, data, expt, "1", get_device())
    trainer.train()



"""

    item = data.train[0]
    token = []
    for instr, token_inputs in zip(item.block.instrs, item.x):
        print("----------")
        print(token_inputs)
        x = torch.cuda.LongTensor(token_inputs)
        seq_len = x.size(0)
        print(seq_len)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        print(pos)


"""

"""
    print(data)
    print(data.raw_data[2])
    print(data.data[2].x)
    print(data.data[2].y)
    print(data.data[2].block)
    print(data.data[2].code_id)
    print(data.hot_idx_to_token)
"""


"""
    base_params = BaseParameters(
        data=args.data,
        embed_mode=args.embed_mode,
        use_rnn=args.use_rnn,
    )

    if args.subparser == 'train':

        train_params = TrainParameters(
            experiment_name=args.experiment_name,
            experiment_time=args.experiment_time,
            epochs=args.epochs,
        )
        trainer = training.load_trainer(base_params,train_params)
        trainer.train()
    
#    elif args.subparser == 'validate':
#        graph_model_validate(base_params, args.load_file, args.iaca_only)
    else:
        raise ValueError('Unknown mode "{}"'.format(args.subparser))
"""
if __name__ == '__main__':
    main()
