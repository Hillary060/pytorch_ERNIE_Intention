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
parser.add_argument('--data_dir', type=str, default='dataset', help='Dictionary of the test file.')
parser.add_argument('--test_filename', type=str, default='test.tsv', help='Name of the test file.')
parser.add_argument('--model_dir', type=str, default='saved_models/grained/best_model.pt', help='Directory of the model.')
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
    predictions,data_ids = [],[]
    test_step = 0
    for i, batch in enumerate(test_batch):
        pred,data_id = model.predict(batch,only_pred=True)
        predictions += pred
        test_step += 1
        data_ids += data_id
else:
    predictions, labels, data_ids = [], [], []
    test_loss, test_acc, test_step = 0., 0., 0
    for i, batch in enumerate(test_batch):
        loss, acc, pred, label, data_id = model.predict(batch)
        test_loss += loss
        test_acc += acc
        predictions += pred
        labels += label
        data_ids += data_id
        test_step += 1
    f1_score = metrics.f1_score(labels, predictions, average='macro')

    print("test_loss: {}, test_acc: {}, f1_score: {}".format( \
                                          test_loss/test_step, \
                                          test_acc/test_step, \
                                          f1_score))

# save preds and corresponding labels
pred_save_path = os.path.join(opt['res_dir'], 'preds')
label_save_path = os.path.join(opt['res_dir'], 'labels')

# resort the prediction and labels into origin order
preds_resort, labels_resort = {}, {}
for index, data_id in enumerate(data_ids):
    preds_resort[data_id] = predictions[index]

# resort
tmp = sorted(preds_resort.items(), key=lambda item: item[0])
preds_resort = [p[1] for j, p in enumerate((tmp))]
with open(pred_save_path, 'w') as f:
    f.write(str(preds_resort))
    print("Best prediction resorted into origin sort, and saved to file {}".format(pred_save_path))

if opt['type'] != 'multi':
    for index, data_id in enumerate(data_ids):
        labels_resort[data_id] = labels[index]
    tmp = sorted(labels_resort.items(), key=lambda item: item[0])
    labels_resort = [p[1] for j, p in enumerate((tmp))]
    with open(label_save_path, 'w') as f:
        f.write(str(labels_resort))
        print("Corresponding labels saved to file {}".format(label_save_path))