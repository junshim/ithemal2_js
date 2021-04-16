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
    parser.add_argument('--data',help='Input pytorch data file')
    args = parser.parse_args()

    (model, data) = load_model_and_data(args.model, args.model_data)
    cuda = torch.device('cuda')
    model.to(cuda)
    data = dt.load_dataset(args.data)
    total_data = len(data.data)
    correct1 = 0
    correct2 = 0
    correct3 = 0
    tolerance1 = 5.
    tolerance2 = 10.
    tolerance3 = 25.
    print('Total data : {}'.format(total_data))
    for idx in tqdm(range(total_data)):
        datum = data.test[idx]
        output = model(datum).item()
        target = datum.y
            
        percentage = abs(output-target) *100 / (target+ 1e-3)

        if percentage < tolerance1:
            correct1 +=1
        if percentage < tolerance2: 
            correct2 +=1
        if percentage < tolerance3:
            correct3 +=1

    accuracy1 = correct1 / total_data
    accuracy2 = correct2 / total_data
    accuracy3 = correct3 / total_data
    print('Accuracy1 : {}'.format(accuracy1))
    print('Accuracy2 : {}'.format(accuracy2))
    print('Accuracy3 : {}'.format(accuracy3))


if __name__ == '__main__':
    main()

