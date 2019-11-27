import os
import json
import random
import torch
import numpy as np
from utils import constant
from pytorch_pretrained_bert import BertTokenizer
from utils import helper


def read_tsv(filename):
    '''Load data from .tsv file'''
    tmp_list = []
    with open(filename, 'r') as of:
        for line in of.readlines():
            line = line.rstrip('\n').split('\t')
            tmp_list.append({'text_a': line[1], 'label': int(line[0])})
    return tmp_list


# filter data in current coarse class
def filter_data(data, opt, is_test=False):
    if opt['type'] == 'multi' and opt['coarse_name'] is not None:
        tmp_list = []

        if is_test:
            # select test set according to coarse predictions
            coarse_id = constant.COARSE_TO_ID[opt['coarse_name']]
            coarse_prediction = eval(open(constant.BEST_PRED_COARSE_FILE).read())
            index_rec = dict()
            for i, p in enumerate(coarse_prediction):
                if p == coarse_id:
                    tmp_list.append(data[i])
                    index_rec[i] = len(tmp_list) - 1
            index_rela_path = opt['pred_res_dir'] + '/index_relation_' + opt['coarse_name']
            with open(index_rela_path, 'w') as f:
                f.write(str(index_rec))
            print("index relation between multi test set and test.tsv saved to file {}".format(index_rela_path))
            return tmp_list

        id2label = dict([(v, k) for k, v in constant.LABEL_TO_ID.items()])
        for i, item in enumerate(data):
            item_coarse_name = constant.GRAINED_TO_COARSE[id2label[item['label']]]
            if item_coarse_name == opt['coarse_name']:
                tmp_list.append(item)
        return tmp_list

    return data


class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """

    def __init__(self, filename, batch_size, opt, is_test=False):
        self.batch_size = batch_size
        self.opt = opt
        if opt['type'] == 'coarse':
            self.label2id = constant.COARSE_TO_ID
        else:
            self.label2id = constant.LABEL_TO_ID
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(opt['ERNIE_dir'], 'vocab.txt'))
        data = read_tsv(filename)

        # filter data in multi classification
        data = filter_data(data, opt, is_test)
        self.raw_data = data
        data = self.preprocess(data, opt)
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in data:
            # tokenize
            tokens = self.tokenizer.tokenize(d['text_a'])

            # mapping to ids
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            tokens = self.tokenizer.convert_tokens_to_ids(['[CLS]']) + tokens
            l = len(tokens)

            # mask for real length
            mask_s = [1 for i in range(l)]

            processed += [(tokens, mask_s, d['label'])]
        return processed

    def __len__(self):
        return len(self.data)

    # 0: tokens, 1: mask_s, 2: label
    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 3

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, _ = sort_all(batch, lens)

        # convert to tensors
        tokens = get_long_tensor(batch[0], batch_size)
        mask_s = get_float_tensor(batch[1], batch_size)
        label = torch.LongTensor(batch[2])

        return (tokens, mask_s, label)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens


def get_float_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded FloatTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.FloatTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.FloatTensor(s)
    return tokens


def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]


def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
                else x for x in tokens]