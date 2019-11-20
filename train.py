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

from utils import constant
from utils import helper
from data.loader import DataLoader
from model.trainer import MyTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset', help='Dataset directory')
parser.add_argument('--ERNIE_dir', type=str, default='pretrained_ERNIE', help='ERNIE directory')
parser.add_argument('--emb_dim', type=int, default=768, help='Word embedding dimension.')
parser.add_argument('--input_dropout', type=float, default=0.4, help='input dropout rate.')
parser.add_argument('--lr', type=float, default=5e-5, help='Applies to adam')
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam', help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=40, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=24, help='Training batch size.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
args = parser.parse_args()

# set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# make opt
opt = vars(args)
label2id = constant.LABEL_TO_ID
opt['num_class'] = len(label2id)

# print opt
helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])

# model save dir
helper.ensure_dir(opt['save_dir'], verbose=True)

# save config
helper.save_config(opt, opt['save_dir'] + '/config.json', verbose=True)
file_logger = helper.FileLogger(opt['save_dir'] + '/' + opt['log'], header="# epoch\ttrain_loss\tdev_loss\ttrain_ACC\ttest_ACC\tF1")

# load data
print("Loading data from {} with batch size {} ...".format(opt['data_dir'], opt['batch_size']))
train_batch = DataLoader(opt['data_dir'] + '/train.tsv', opt['batch_size'], opt)
test_batch = DataLoader(opt['data_dir'] + '/test.tsv', opt['batch_size'], opt)

# build model
print('Building model...')
trainer = MyTrainer(opt)

train_acc_history, train_loss_history, test_loss_history, f1_score_history = [], [], [], [0.]
test_acc_history = [0.]
for epoch in range(1, args.num_epoch+1):
    train_loss, train_acc, train_step = 0., 0., 0
    for i, batch in enumerate(train_batch):
        loss, acc = trainer.update(batch)
        train_loss += loss
        train_acc += acc
        train_step += 1
        if train_step % args.log_step == 0:
            print("train_loss: {}, train_acc: {}".format(train_loss/train_step, train_acc/train_step))

    # eval on test
    print("Evaluating on test set...")
    predictions, labels = [], []
    test_loss, test_acc, test_step = 0., 0., 0
    for i, batch in enumerate(test_batch):
        loss, acc, pred, label = trainer.predict(batch)
        test_loss += loss
        test_acc += acc
        predictions += pred
        labels += label
        test_step += 1
    # f1 score
    f1_score = metrics.f1_score(labels, predictions, average='macro')

    print("trian_loss: {}, test_loss: {}, train_acc: {}, test_acc: {}, f1_score: {}".format( \
        train_loss/train_step, test_loss/test_step, \
        train_acc/train_step, test_acc/test_step, \
        f1_score))

    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format( \
        epoch, train_loss/train_step, test_loss/test_step, \
        train_acc/train_step, test_acc/test_step, \
        f1_score))

    train_acc_history.append(train_acc/train_step)
    train_loss_history.append(train_loss/train_step)
    test_loss_history.append(test_loss/test_step)

    # save
    model_file = opt['save_dir'] + '/checkpoint_epoch_{}.pt'.format(epoch)
    trainer.save(model_file)
    # save best model
    if epoch == 1 or test_acc/test_step > max(test_acc_history):
        copyfile(model_file, opt['save_dir'] + '/best_model.pt')
        print("new best model saved.")
        file_logger.log("new best model saved at epoch {}: {:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}"\
            .format(epoch, train_loss/train_step, test_loss/test_step, \
            train_acc/train_step, test_acc/test_step, \
            f1_score))
    test_acc_history.append(test_acc/test_step)
    f1_score_history.append(f1_score)
    print("")

print("Training ended with {} epochs.".format(epoch))
bt_train_acc = max(train_acc_history)
bt_train_loss = min(train_loss_history)
bt_test_acc = max(test_acc_history)
bt_f1_score = f1_score_history[test_acc_history.index(bt_test_acc)]
bt_test_loss = min(test_loss_history)
print("best train_acc: {}, best train_loss: {}, best test_acc/f1_score: {}/{}, best test_loss: {}".format(bt_train_acc, \
                                                                                                          bt_train_loss, \
                                                                                                          bt_test_acc, \
                                                                                                          bt_f1_score, \
                                                                                                          bt_test_loss))
of = open('tmp.txt','a')
of.write(str(bt_test_acc)+","+str(bt_f1_score)+'\n')
of.close()
