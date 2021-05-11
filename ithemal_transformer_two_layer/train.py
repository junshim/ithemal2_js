# Copyright 2018 Dong-Hyun Lee, Kakao Brain.

""" Training Config & Helper Classes  """

import os
import json
from typing import NamedTuple
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import time
import losses as ls
import random

class Config(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    batch_size: int = 32
    lr: int = 5e-5 # learning rate
    n_epochs: int = 10 # the number of epoch
    warmup: float = 0.001

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))

class LossReporter(object):
    def __init__(self, experiment, n_data_points):
        # type: (Experiment, int, tr.Train) -> None

        self.experiment = experiment
        self.n_datapoints = n_data_points
        self.start_time = time.time()

        self.loss = 1.0
        self.epoch_no = 0
        self.total_processed_items = 0
        self.epoch_processed_items = 0
        self.accuracy = 0.0

        self.last_report_time = 0.0
        self.last_save_time = 0.0

        self.root_path = self.experiment.experiment_root_path()

        try:
            os.makedirs(self.root_path)
        except OSError:
            pass

        self.loss_report_file = open(os.path.join(self.root_path, 'loss_report.log'), 'w',1)
        self.pbar = tqdm(desc = self.format_loss(), total=self.n_datapoints)

    def format_loss(self):

        return 'Epoch {}, Loss: {:.2}, Learning Rate: {:.2}'.format(
                self.epoch_no,
                self.loss,
                self.accuracy
        )

    def start_epoch(self, epoch_no):
        
        self.epoch_no = epoch_no
        self.epoch_processed_items = 0
        self.accuracy = 0.0

        self.pbar.close()
        self.pbar = tqdm(desc=self.format_loss(), total=self.n_datapoints)

    def report(self, n_items, loss, t_accuracy):

        self.loss = loss
        #self.accuracy = (self.accuracy * self.epoch_processed_items + t_accuracy * n_items) / (self.epoch_processed_items + n_items)
        self.accuracy = t_accuracy
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
        except OSError:
            pass

        torch.save(state_dict, file_name) 

    def end_epoch(self, model, optimizer, loss):
        
        self.loss = loss

        t = time.time()
        message = '\t'.join(map(str, (
            self.epoch_no,
            t - self.start_time,
            self.loss,
            self.accuracy,
        )))
        self.loss_report_file.write(message + '\n')

        file_name = os.path.join(self.experiment.checkpoint_file_dir(),'{}.mdl'.format(self.epoch_no))
        self.check_point(model,optimizer,file_name)


    def finish(self, model, optimizer):

        self.pbar.close()
        print("Finishing training")

        file_name = os.path.join(self.root_path, 'trained.mdl')
        self.check_point(model,optimizer,file_name)


class Trainer(object):
    """ Training Helper Class """
    def __init__(self, cfg, model, data, expt, optimizer, device):
        self.cfg = cfg # config for training : see class Config
        self.model = model
        self.data = data
        self.expt = expt
        self.lr = cfg.lr
        self.optimizer = optimizer
        #self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.,nesterov=False)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.save_dir = self.expt.experiment_root_path()
        self.device = device # device name

        self.loss_fn = ls.mse_loss
        self.loss_reporter = LossReporter(expt, len(self.data.train))

        self.tolerance = 25.

    def correct_regression(self, x, y):
        
        if x.shape != ():
            x = x[-1]
            y = y[-1]

        percentage = torch.abs(x - y) * 100.0 / (y + 1e-3)

        if percentage < self.tolerance:
            self.correct += 1

    def print_final(self, f, x, y):
        
        if x.shape != ():
            size = x.shape[0]
            for i in range(size):
                f.write('%f,%f ' % (x[i],y[i]))
            f.write('\n')
        else:
            f.write('%f,%f\n' % (x,y))

    def validate(self, resultfile):
        
        f = open(resultfile,'w')

        self.correct = 0
        average_loss = 0.
        actual = []
        predicted = []

        for j , item in enumerate(tqdm(self.data.test)):

            output = self.model(item)
            target = torch.FloatTensor([item.y]).squeeze()

            output = output.to('cpu')
            actual.append(target.data.numpy().tolist())
            predicted.append(output.data.numpy().tolist())

            self.print_final(f, output, target)
            loss = self.loss_fn(output,target)
            average_loss = (average_loss * j + loss.item()) / (j+1)
            self.correct_regression(output, target)

        f.write('loss - %f\n' % (average_loss))
        f.write('%f, %f\n'%(self.correct, len(self.data.test)))

        f.close()

    def train(self):
        """ Train Loop """
        for epoch_no in range(self.cfg.n_epochs):

            epoch_loss_sum = 0.
            step = 0
            self.loss_reporter.start_epoch(epoch_no + 1) 

            random.shuffle(self.data.train)

            if epoch_no == 0:
                warm_up = int((len(self.data.train) * 0.3))

            for idx in range( 0, len(self.data.train), self.cfg.batch_size):

                if epoch_no == 0 and idx < warm_up:
                    self.lr = self.cfg.warmup
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr
                    
                elif epoch_no == 0 and idx > warm_up:
                    self.lr = self.cfg.lr
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr
              


                self.optimizer.zero_grad()
                loss_tensor = torch.cuda.FloatTensor([0]).squeeze()

                batch =self.data.train[idx:idx+self.cfg.batch_size]
                batch_loss_sum = 0.

                if not batch:
                    continue
        
                for datum in batch:
                    output = self.model(datum)
                    target = torch.cuda.FloatTensor([datum.y]).squeeze()
   
                    loss = self.loss_fn(output, target)
                    batch_loss_sum += loss.item()
                    loss_tensor += loss

                batch_loss_avg = batch_loss_sum / len(batch)
                loss_tensor.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.)
                
                for param in self.model.parameters():
                    if param.grad is None:
                        continue

                    if torch.isnan(param.grad).any():
                        self.loss_reporter.finish(self.model, self.optimizer)
                        return

                self.optimizer.step()
            
                step += 1
                epoch_loss_sum += batch_loss_avg
                self.loss_reporter.report(len(batch), batch_loss_avg, self.lr)   

            #if self.lr > 1.0e-05 :
            self.lr /= 1.2
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
            epoch_loss_avg = epoch_loss_sum / step
            self.loss_reporter.end_epoch(self.model,self.optimizer, epoch_loss_avg)
        self.loss_reporter.finish(self.model,self.optimizer)

        resultfile = os.path.join(self.expt.experiment_root_path(), 'validation_results.txt')
        self.validate(resultfile)
