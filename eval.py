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
from data.loader import DataLoader,read_tsv,split_test_data
from model.trainer import MyTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset/multi/其他/eval/', help='Dictionary of the test file.')
parser.add_argument('--test_filename', type=str, default='test.tsv', help='Name of the test file.')
parser.add_argument('--model_dir', type=str, default='saved_models/multi/其他/best_model.pt', help='Directory of the model.')
args = parser.parse_args()

print("Loading model from {}".format(args.model_dir))
opt = torch_utils.load_config(args.model_dir)
helper.print_config(opt)
model = MyTrainer(opt)
model.load(args.model_dir)

print("Loading data from {} with batch size {}...".format(os.path.join(args.data_dir, args.test_filename), opt['batch_size']))

# split_test_data for multi
if opt['type'] == 'multi':
    split_test_data(opt['coarse_name'])

is_multi_eval = False
if opt['type'] == 'multi':
    is_multi_eval = True

test_batch = DataLoader(os.path.join(args.data_dir, args.test_filename), opt['batch_size'], opt, is_multi_eval)

print("Evaluating...")
if opt['type'] == 'multi':
    predictions = []
    test_step = 0
    for i, batch in enumerate(test_batch):
        pred = model.predict(batch,only_pred=True)
        predictions += pred
        test_step += 1
    # save prediction
    preds_path = os.path.join(opt['res_dir'], 'preds')
    with open(preds_path, 'w') as f:
        f.write(str(predictions))
    print(opt['coarse_name']+" prediction saved to file {}".format(preds_path))
else:
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
