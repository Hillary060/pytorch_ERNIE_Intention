import os
import sys
import numpy as np
import random
import argparse
from shutil import copyfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn import metrics

from utils import constant, helper, torch_utils
from utils import helper
from data.loader import DataLoader
from model.trainer import MyTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset')
parser.add_argument('--model_dir', type=str, default='saved_models/best_model.pt', help='Directory of the model.')
args = parser.parse_args()

print("Loading model from {}".format(args.model_dir))
opt = torch_utils.load_config(args.model_dir)
model = MyTrainer(opt)
model.load(args.model_dir)

print("Loading data from {} with batch size {}...".format(args.data_dir, opt['batch_size']))
test_batch = DataLoader(args.data_dir + '/test.tsv', opt['batch_size'], opt)

print("Evaluating...")
predictions, labels = [], []
test_loss, test_acc, test_step = 0., 0., 0
for i, batch in enumerate(test_batch):
    loss, acc, pred, label = model.predict(batch)
    test_loss += loss
    test_acc += acc
    predictions += pred
    labels += label
    test_step += 1
f1_score = metrics.f1_score(labels, predictions, average='macro')

print("test_loss: {}, test_acc: {}, f1_score: {}".format( \
                                      test_loss/test_step, \
                                      test_acc/test_step, \
                                      f1_score))
