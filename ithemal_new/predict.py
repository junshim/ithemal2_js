#!/usr/bin/env python

import argparse
import data.data_cost as dt
import ithemal_utils
import os
import subprocess
import torch
from tqdm import tqdm

HOME = os.path.dirname(os.path.abspath(__file__))
_TOKENIZER = os.path.join(HOME, 'tokenizer')


def load_model_and_data(model_file, model_data_file):
    (model, data) = ithemal_utils.load_model_and_data(model_file)

    state_dict = torch.load(model_data_file)
    model_dict = model.state_dict()
    new_model_dict = {k: v for (k, v) in state_dict['model'].items() if k in model_dict}
    model_dict.update(new_model_dict)
    model.load_state_dict(model_dict)

    return (model, data)

def datum_of_code(data, raw):
    xml = subprocess.check_output([_TOKENIZER, raw, '--token'])
    xml = xml.decode()
    intel = subprocess.check_output([_TOKENIZER, raw, '--intel'])
    intel = intel.decode()

    data.raw_data = [(-1, -1, intel, xml)]
    data.data = []
    data.prepare_data(fixed=False, progress=False)
    return data.data[-1]

def predict_raw(model_arg, data_arg, raw):
    (model, data) = load_model_and_data(model_arg, data_arg)
    datum = datum_of_code(data, raw)
    print(model(datum).item())

def main():
    parser = argparse.ArgumentParser(description='Analyze a basic block')
    parser.add_argument('--model', help='Model architecture to use(dump file)', required=True)
    parser.add_argument('--model-data', help='Model data to use(mdl file)', required=True)

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--raw', help='Input raw hex')
    input_group.add_argument('--files',help='Input raw basic block file')
    input_group.add_argument('--data',help='Input pytorch data file')
    args = parser.parse_args()

    if args.raw:
        predict_raw(args.model, args.model_data, args.raw)
    elif args.files:
        (model, data) = load_model_and_data(args.model, args.model_data)
        fr = open(args.files)
        oneline = fr.readline()
        while True:
            datum = datum_of_code(data, oneline)
            print(model(datum).item())
    elif args.data:
        (model, data) = load_model_and_data(args.model, args.model_data)
        data = ithemal_utils.load_data(args.data)
        total_data = len(data.data)
        correct = 0
        tolerance = 25.
        #tolerance = 10.
        #tolerance = 5.
        print('Total data : {}'.format(total_data))
        for idx in tqdm(range(total_data)):
            datum = data.data[idx]
            output = model(datum).item()
            #output = output.cpu()
            target = datum.y
            
            percentage = abs(output-target) *100 / (target+ 1e-3)

            if percentage < tolerance:
                correct +=1
        
        accuracy = correct / total_data
        print('Accuracy : {}'.format(accuracy))


if __name__ == '__main__':
    main()

