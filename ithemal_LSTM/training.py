#!/usr/bin/env python

import sys
import os

#import models.graph_models as md
import models.losses as ls
#import models.train as tr
#import data.data_cost as dt
from experiments.experiment import Experiment
#from utils import *
from ithemal_utils import *

import atexit
import collections
from enum import Enum
import time
import torch
import torch.optim as optim
from typing import Any, Dict, List, Iterator, Tuple, Type, Union, NamedTuple, TypeVar
import zmq
from tqdm import tqdm
import subprocess
import random
import uuid
import numpy as np

# ------------------------------- LOSS REPORTING --------------------------------

class LossReporter(object):
    def __init__(self, experiment, n_datapoints):
        # type: (Experiment, int, tr.Train) -> None

        self.experiment = experiment
        self.n_datapoints = n_datapoints

        self.start_time = time.time()
        self.loss = 1.0
        self.epoch_no = 0
        self.total_processed_items = 0
        self.epoch_processed_items = 0
        self.accuracy = 0.0

        self.last_report_time = 0.0
        self.last_save_time = 0.0

        self.root_path = experiment.experiment_root_path()

        try:
            os.makedirs(self.root_path)
        except OSError:
            pass

        # line buffered
        self.loss_report_file = open(os.path.join(self.root_path, 'loss_report.log'), 'w', 1)
        self.pbar = tqdm(desc=self.format_loss(), total=self.n_datapoints)

    def format_loss(self):
        # type: () -> str

        return 'Epoch {}, Loss: {:.2}, learning rate: {:.2}'.format(
                self.epoch_no,
                self.loss,
                self.accuracy
        )

    def start_epoch(self, epoch_no):
        # type: (int, int) -> None

        self.epoch_no = epoch_no
        self.epoch_processed_items = 0
        self.accuracy = 0.0

        self.pbar.close()
        self.pbar = tqdm(desc=self.format_loss(), total=self.n_datapoints)

    def report(self, n_items, loss, t_accuracy):

        self.loss = loss
        self.accuracy = (self.accuracy * self.epoch_processed_items + t_accuracy * n_items) / (self.epoch_processed_items + n_items)
        self.epoch_processed_items += n_items
        self.total_processed_items += n_items

        desc = self.format_loss()
        self.pbar.set_description(desc)
        self.pbar.update(n_items)

    def check_point(self, model, optimizer, file_name):

        state_dict = {
            'epoch': self.epoch_no,
            'model': model.state_dict(),
            'optimizer':optimizer.state_dict(),
        }

        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError :
            pass
        
        torch.save(state_dict, file_name)

    def end_epoch(self, model, optimizer, loss):

        self.loss = loss

        t = time.time()
        message = '\t'.join(map(str, (
            self.epoch_no,
            t - self.start_time,
            self.loss,
        )))
        self.loss_report_file.write(message + '\n')

        file_name = os.path.join(self.experiment.checkpoint_file_dir(),'{}.mdl'.format(self.epoch_no))
        self.check_point(model,optimizer,file_name)

    def finish(self, model, optimizer):

        self.pbar.close()
        print("Finishing training")

        file_name = os.path.join(self.root_path, 'trained.mdl')
        self.check_point(model,optimizer,file_name)

# -------------------------------- TRAIN ---------------------------------

class OptimizerType(Enum):
    ADAM_PRIVATE = 1
    ADAM_SHARED = 2
    SGD = 3

class PredictionType(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2

def load_trainer(base_params, train_params):
    # type: (BaseParameters, TrainParameters, md.AbstractGraphModule, dt.DataCost) -> tr.Train

    expt = Experiment(train_params.experiment_name, train_params.experiment_time, base_params.data)
    
    data = load_data(base_params)
    model = load_model(base_params, data)
    #model = self.model.to('cuda')#js

    dump_model_and_data(model, data, os.path.join(expt.experiment_root_path(), 'predictor.dump'))

    return Train(model, data, expt, train_params)
        

class Train():
    
    def __init__(self,model,data,expt,train_params):

        self.expt = expt
        
        self.data = data
        self.model = model

        self.model = self.model.to('cuda')#js

        self.epochs = train_params.epochs

        #js : Class Train parameter
        self.typ = PredictionType.REGRESSION
        self.loss_fn = ls.mse_loss
        self.num_losses = 1
        self.batch_size = train_params.batch_size
        self.tolerance = 25.
        self.lr = train_params.initial_lr
        self.lr_decay_rate = train_params.lr_decay_rate
        self.momentum = 0.9
        self.nesterov=False
        self.clip = 2.
        self.opt = OptimizerType.SGD
        self.predict_log = False

        self.correct = 0

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, nesterov=self.nesterov)


        self.loss_reporter = LossReporter(expt, len(self.data.train))


    def load_checkpoint(self, filename):
        # type: (str) -> Dict[str, Any]

        state_dict = torch.load(filename)
        self.model.load_state_dict(state_dict['model'])

        try:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        except ValueError:
            print('Couldnt load optimizer!')

        return state_dict

    def print_final(self,f,x,y):
        # type: (IO[str], np.array, np.array) -> None
        if x.shape != ():
            size = x.shape[0]
            for i in range(size):
                f.write('%f,%f ' % (x[i],y[i]))
            f.write('\n')
        else:
            f.write('%f,%f\n' % (x,y))

    def validate(self,resultfile, loadfile=None):
        # type: (str, Optional[str]) -> Tuple[List[List[float]], List[List[float]]]
        
        if loadfile is not None:
            print('loaded from checkpoint for validation...')#js
            self.load_checkpoint(loadfile)

        f = open(resultfile,'w')

        self.correct = 0
        average_loss = [0] * self.num_losses
        actual = []
        predicted = []

        for j, item in enumerate(tqdm(self.data.test)): 

            #print len(item.x)
            output = self.model(item)
            target = torch.FloatTensor([item.y]).squeeze()
            
            #get the target and predicted values into a list
            if self.typ == PredictionType.CLASSIFICATION:
                actual.append((torch.argmax(target) + 1).data.numpy().tolist())
                predicted.append((torch.argmax(output) + 1).data.numpy().tolist())
            else:
                output = output.to('cpu')
                actual.append(target.data.numpy().tolist())
                predicted.append(output.data.numpy().tolist())

            self.print_final(f, output, target)
            losses = self.loss_fn(output, target)
            if self.typ == PredictionType.CLASSIFICATION:
                self.correct_classification(output, target) 
            else:
                self.correct_regression(output, target)

            #accumulate the losses
            loss = torch.zeros(1)
            for c,l in enumerate(losses):
                loss += l
                average_loss[c] = (average_loss[c] * j + l.item()) / (j + 1)

            if j % (len(self.data.test) / 100) == 0:
                p_str = str(j) + ' '
                for av in average_loss:
                    p_str += str(av) + ' '
                p_str += str(self.correct) + ' '

            #remove refs; so the gc remove unwanted tensors
            self.model.remove_refs(item)

        for loss in average_loss:
            f.write('loss - %f\n' % (loss))
        f.write('%f,%f\n' % (self.correct, len(self.data.test)))

        f.close()

        return (actual, predicted)


    def correct_classification(x,y):
        # type: (torch.tensor, torch.tensor) -> None

        x = torch.argmax(x) + 1
        y = torch.argmax(y) + 1

        percentage = torch.abs(x - y) * 100.0 / y

        if percentage < self.tolerance:
            self.correct += 1

    def correct_regression(self,x,y):
        # type: (torch.tensor, torch.tensor) -> None

        if x.shape != ():
            x = x[-1]
            y = y[-1]

        percentage = torch.abs(x - y) * 100.0 / (y + 1e-3)

        if percentage < self.tolerance:
            self.correct += 1

    def train(self):

        for epoch_no in range(self.epochs):
            
            epoch_loss_sum = 0.
            step = 0
            self.loss_reporter.start_epoch(epoch_no + 1)

            random.shuffle(self.data.train)

            #js: class train to here
            for idx in range( 0, len(self.data.train), self.batch_size):
                
                self.optimizer.zero_grad()
                loss_tensor = torch.cuda.FloatTensor([0]).squeeze()
                batch =self.data.train[idx:idx+self.batch_size]
                batch_loss_sum = 0.
                self.correct = 0

                if not batch:
                    continue
                
                for datum in batch:
                    output = self.model(datum)
                    target = torch.cuda.FloatTensor([datum.y]).squeeze()
                   
                    loss = self.loss_fn(output, target)
                    batch_loss_sum += loss[0].item()
                    loss_tensor += loss[0]

                batch_loss_avg = batch_loss_sum / len(batch)
                loss_tensor.backward()

                #clip the gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.)

                for param in self.model.parameters():
                    if param.grad is None:
                        continue

                    if torch.isnan(param.grad).any():
                        self.loss_reporter.finish(self.model, self.optimizer)
                        return

                #optimizer step to update parameters
                self.optimizer.step()

                step += 1
                epoch_loss_sum += batch_loss_avg
                self.loss_reporter.report(len(batch), batch_loss_avg, self.lr)
            
            #js : report to log
            epoch_loss_avg = epoch_loss_sum / step
            self.loss_reporter.end_epoch(self.model, self.optimizer, epoch_loss_avg)        

            # decay lr if necessary
            self.lr /= self.lr_decay_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

        self.loss_reporter.finish(self.model, self.optimizer)

        resultfile = os.path.join(self.expt.experiment_root_path(), 'validation_results.txt')
        self.validate(resultfile)

