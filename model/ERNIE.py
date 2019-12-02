import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pytorch_pretrained_bert import BertModel
from data.loader import get_current_label2id
from utils import constant, torch_utils


class BasicClassifier(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # ERNIE model
        self.ERNIE = BertModel.from_pretrained(opt['ERNIE_dir'])
        for param in self.ERNIE.parameters():
            param.requires_grad = True
        # dropout
        self.input_dropout = nn.Dropout(opt['input_dropout'])

        # label to id
        label2id = get_current_label2id(opt)

        # classifier
        self.classifier = nn.Linear(opt['emb_dim'], len(label2id))
        
    # 0: tokens, 1: mask_s
    def forward(self, inputs):
        # unpack inputs 
        tokens, mask_s = inputs
        encoder_out, pooled = self.ERNIE(tokens, attention_mask=mask_s, output_all_encoded_layers=False)
        pooled = self.input_dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits
