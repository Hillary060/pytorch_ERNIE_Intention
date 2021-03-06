"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.ERNIE import BasicClassifier
from utils import torch_utils


class Trainer(object):
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


# 0: tokens, 1: mask_s, 2: label 3: data_id
def unpack_batch(batch, cuda):
    inputs, label, data_id = batch[0:2], batch[2], batch[3]
    if cuda:
        inputs = [Variable(i.cuda()) for i in inputs]
        label = Variable(label.cuda())
        data_id = Variable(data_id.cuda())
    else:
        inputs = [Variable(i) for i in inputs]
        label = Variable(label)
        data_id = Variable(data_id)
    return inputs, label, data_id


# 0: tokens, 1: mask_s, 2: label
class MyTrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt  # opt: batch[0:2]
        self.model = BasicClassifier(opt)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def update(self, batch):
        inputs, label, _ = unpack_batch(batch, self.opt['cuda'])
        # forward
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, label, reduction='mean')
        corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
        acc = 100.0 * np.float(corrects) / label.size()[0]
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        
        return loss.item(), acc, label.data.cpu().numpy().tolist()

    def predict(self, batch,only_pred=False):
        if self.opt['type'] == 'multi' and only_pred is True:
            # only prediction needed
            inputs, _, data_id = unpack_batch(batch, self.opt['cuda'])
            self.model.eval()
            logits = self.model(inputs)
            predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
            return predictions, data_id.data.cpu().numpy().tolist()
        else:
            inputs, label, data_id = unpack_batch(batch, self.opt['cuda'])  # inputs = batch[0:2]
            self.model.eval()
            logits = self.model(inputs)
            # loss
            loss = F.cross_entropy(logits, label, reduction='mean')
            corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
            acc = 100.0 * np.float(corrects) / label.size()[0]
            predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()

            pred_complete = logits.data.cpu().numpy()
            preds_top5 = []
            for i in range(len(pred_complete)):
                pred_temp = [[j, item] for j, item in enumerate(pred_complete[i])]  # [label,logits]
                pred_temp.sort(key=lambda x: x[1], reverse=True)
                pred_top5 = pred_temp[0:5]
                preds_top5.append(pred_top5)
            return loss.item(), acc, predictions, label.data.cpu().numpy().tolist(), data_id.data.cpu().numpy().tolist(), preds_top5
