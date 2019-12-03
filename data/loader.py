import os
import json
import random
import torch
import numpy as np
from utils import constant
from pytorch_pretrained_bert import BertTokenizer
from utils import helper
from sklearn.model_selection import train_test_split
import pandas as pd


def read_tsv(filename):
    '''Load data from .tsv file'''
    tmp_list = []
    data_id = 0
    with open(filename, 'r') as of:
        for line in of.readlines():
            line = line.rstrip('\n').split('\t')
            tmp_list.append({'text_a': line[1], 'label': int(line[0]),'data_id':data_id})
            data_id +=1
    return tmp_list


# filter data in current coarse class
def filter_data(data, opt, is_multi_eval=False):
    if opt['type'] != 'multi':
        return data

    # filter data
    id2label = dict([(v, k) for k, v in constant.LABEL_TO_ID.items()])
    grained2coarse = constant.GRAINED_TO_COARSE
    res_dir = opt['res_dir']
    tmp_list = []
    if opt['type'] == 'multi' and opt['coarse_name'] is not None:
        if is_multi_eval is True:
            for i, item in enumerate(data):
                grained_name = id2label[item['label']]
                item_coarse_name = constant.GRAINED_TO_COARSE[grained_name]
                grained_id_in_coarse = constant.GRAINED_ID_IN_COARSE[item_coarse_name][grained_name]
                # replace label id
                data[i]['label'] = grained_id_in_coarse
            return data
        else:
            # train
            for i, item in enumerate(data):
                grained_name = id2label[item['label']]
                item_coarse_name = constant.GRAINED_TO_COARSE[grained_name]
                grained_id_in_coarse = constant.GRAINED_ID_IN_COARSE[item_coarse_name][grained_name]

                # replace label id
                if item_coarse_name == opt['coarse_name']:
                    tmp_list.append({'text_a': item['text_a'], 'label': grained_id_in_coarse})
            return tmp_list

    return data


# split a data set , used in the training of second levels
def split_dataset(data_path, split_save_dir, coarse_name):
    id2label = dict([(v, k) for k, v in constant.LABEL_TO_ID.items()])
    grained2coarse = constant.GRAINED_TO_COARSE
    # data origin
    data_all = pd.read_csv(data_path, '\t', header=None)
    # select data
    data = [[],[]]
    for i in range(len(data_all)):
        if grained2coarse[id2label[data_all[0][i]]] == coarse_name:
            data[0].append(data_all[0][i])
            data[1].append(data_all[1][i])

    # x:text_a y:label id
    x, y = data[1], data[0]

    # test set prop:15 %
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=0)
    train_set = [[y_train[i], p] for i, p in enumerate(x_train)]
    test_set = [[y_test[i], p] for i, p in enumerate(x_test)]

    # save train set
    with open(os.path.join(split_save_dir, 'train.tsv'), 'w') as f:
        for i, p in enumerate(train_set):
            f.write('\t'.join([str(p[0]), p[1]]) + '\n')

    # save test set
    with open(os.path.join(split_save_dir, 'test.tsv'), 'w') as f:
        for i, p in enumerate(test_set):
            f.write('\t'.join([str(p[0]), p[1]]) + '\n')


# split_test_data for second level classification
def split_test_data(coarse_name):
    data = read_tsv('dataset/test.tsv')
    # for i,coarse_name in enumerate(constant.COARSE_INTO_MULTI):
    # save dir
    res_dir = 'result/multi/'+coarse_name
    helper.ensure_dir(res_dir)

    # select test data according to coarse predictions
    coarse_id = constant.COARSE_TO_ID[coarse_name]
    coarse_prediction = eval(open(constant.BEST_PRED_COARSE_FILE).read())
    tmp_list, index_rec, labels= [], {}, []
    for i, p in enumerate(coarse_prediction):
        if p == coarse_id:
            tmp_list.append(data[i])
            index_rec[i] = len(tmp_list) - 1  # index[0~1500] = current coarse data index
            labels.append(data[i]['label'])

    # save input data of test
    print("\nsaving data...")
    helper.ensure_dir('dataset/multi/'+coarse_name+'/eval/')
    input_path = os.path.join('dataset/multi/'+coarse_name+'/eval/', 'test.tsv')
    with open(input_path, 'w') as f:
        pass
    with open(input_path, 'a') as f:
        for i,p in enumerate(tmp_list):
            f.write('\t'.join([str(p['label']), p['text_a']]) + '\n')
    print("test input file saved to file {}".format(input_path))

    # save index relation
    index_rela_path = os.path.join(res_dir, 'index_relation')
    with open(index_rela_path, 'w') as f:
        f.write(str(index_rec))
    print("index relation between multi test set and test.tsv saved to file {}".format(index_rela_path))

    # save corresponding labels
    labels_save_path = os.path.join(res_dir, 'labels')
    with open(labels_save_path, 'w') as f:
        f.write(str(labels))
    print("corresponding labels saved to file {}".format(labels_save_path)+"\n")


# label2id
def get_current_label2id(opt):
    if opt['type'] == 'coarse':
        label2id = constant.COARSE_TO_ID
        return label2id
    if opt['type'] == 'multi':
        return constant.GRAINED_ID_IN_COARSE[opt['coarse_name']]
    return constant.LABEL_TO_ID


class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, is_multi_eval=False):
        self.batch_size = batch_size
        self.opt = opt
        self.label2id = get_current_label2id(opt)
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(opt['ERNIE_dir'], 'vocab.txt'))
        data = read_tsv(filename)

        # filter data in multi classification
        data = filter_data(data, opt, is_multi_eval)
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

            processed += [(tokens, mask_s, d['label'], d['data_id'])]
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
        if self.opt['type'] == 'grained':
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