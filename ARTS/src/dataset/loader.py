import collections
import json
from collections import defaultdict

import numpy as np
import torch
from torchtext.vocab import Vocab, Vectors

from dataset.utils import tprint
from transformers import BertTokenizer
import dataset.stats as stats

def _get_stego_classes(args):
    train_classes = [0,1,2,3,4,5,6,7,8,9,10]#[0,1,2,3,4,5,6]#[0,1,2,3,4,5,6,7,8,9,10]
    val_classes = [0,11,12,13,14]
    test_classes = [15,16,17,18,19,20,21]
    return train_classes, val_classes, test_classes

def _get_stego_draw_classes(args):
    train_classes = [0,1,2,3,4,5,6,7,8,9,10]#[0,1,2,3,4,5,6]#[0,1,2,3,4,5,6,7,8,9,10]
    val_classes = [0,11,12,13,14]
    test_classes = [16,18,21]#[15,16,17]
    return train_classes, val_classes, test_classes


def _load_json(path):
    '''
        load data file
        @param path: str, path to the data file
        @return data: list of examples
    '''
    label = {}
    text_len = []
    with open(path, 'r', errors='ignore') as f:
        data = []
        for line in f:
            row = json.loads(line)

            # count the number of examples per label
            if int(row['label']) not in label:
                label[int(row['label'])] = 1
            else:
                label[int(row['label'])] += 1

            item = {
                'label': int(row['label']),
                'text': row['text'][:500]  # truncate the text to 500 tokens
            }

            text_len.append(len(row['text']))

            keys = ['head', 'tail', 'ebd_id']
            for k in keys:
                if k in row:
                    item[k] = row[k]

            data.append(item)

        tprint('Class balance:')

        print(label)

        tprint('Avg len: {}'.format(sum(text_len) / (len(text_len))))

        return data


def _read_words(data):
    '''
        Count the occurrences of all words
        @param data: list of examples
        @return words: list of words (with duplicates)
    '''
    words = []
    for example in data:
        words += example['text']
    return words


def _meta_split(all_data, train_classes, val_classes, test_classes):
    '''
        Split the dataset according to the specified train_classes, val_classes
        and test_classes

        @param all_data: list of examples (dictionaries)
        @param train_classes: list of int
        @param val_classes: list of int
        @param test_classes: list of int

        @return train_data: list of examples
        @return val_data: list of examples
        @return test_data: list of examples
    '''
    train_data, val_data, test_data = [], [], []

    for example in all_data:
        if example['label'] in train_classes:
            train_data.append(example)
        if example['label'] in val_classes:
            val_data.append(example)
        if example['label'] in test_classes:
            test_data.append(example)

    return train_data, val_data, test_data


def _del_by_idx(array_list, idx, axis):
    '''
        Delete the specified index for each array in the array_lists

        @params: array_list: list of np arrays
        @params: idx: list of int
        @params: axis: int

        @return: res: tuple of pruned np arrays
    '''
    if type(array_list) is not list:
        array_list = [array_list]

    # modified to perform operations in place
    for i, array in enumerate(array_list):
        array_list[i] = np.delete(array, idx, axis)

    if len(array_list) == 1:
        return array_list[0]
    else:
        return array_list


def _data_to_nparray(data, vocab, args):
    '''
           Convert the data into a dictionary of np arrays for speed.
       '''
    doc_label = np.array([x['label'] for x in data], dtype=np.int64)

    raw = np.array([e['text'] for e in data], dtype=object)

    if args.bert:
        tokenizer = BertTokenizer.from_pretrained(
            args.pretrained_bert, do_lower_case=True)

        # convert to wpe
        vocab_size = 0  # record the maximum token id for computing idf
        for e in data:
            e['bert_id'] = tokenizer.convert_tokens_to_ids(['CLS'] + e['text'] + ['SEP'])
            vocab_size = max(max(e['bert_id']) + 1, vocab_size)

        text_len = np.array([len(e['bert_id']) for e in data])
        max_text_len = max(text_len)

        text = np.zeros([len(data), max_text_len], dtype=np.int64)

        del_idx = []
        # convert each token to its corresponding id
        for i in range(len(data)):
            text[i, :len(data[i]['bert_id'])] = data[i]['bert_id']

            # filter out document with only special tokens
            # unk (100), cls (101), sep (102), pad (0)
            if np.max(text[i]) < 103:
                del_idx.append(i)

        text_len = text_len

    else:
        # compute the max text length
        text_len = np.array([len(e['text']) for e in data])
        max_text_len = max(text_len)

        # initialize the big numpy array by <pad>
        text = vocab.stoi['<pad>'] * np.ones([len(data), max_text_len],
                                             dtype=np.int64)

        del_idx = []
        # convert each token to its corresponding id
        for i in range(len(data)):
            text[i, :len(data[i]['text'])] = [
                vocab.stoi[x] if x in vocab.stoi else vocab.stoi['<unk>']
                for x in data[i]['text']]

            # filter out document with only unk and pad
            if np.max(text[i]) < 2:
                del_idx.append(i)

        vocab_size = vocab.vectors.size()[0]

    text_len, text, doc_label, raw = _del_by_idx(
        [text_len, text, doc_label, raw], del_idx, 0)

    new_data = {
        'text': text,
        'text_len': text_len,
        'label': doc_label,
        'raw': raw,
        'vocab_size': vocab_size,
    }

    if 'pos' in args.auxiliary:
        # use positional information in fewrel
        head = np.vstack([e['head'] for e in data])
        tail = np.vstack([e['tail'] for e in data])

        new_data['head'], new_data['tail'] = _del_by_idx(
            [head, tail], del_idx, 0)
    return new_data


def _split_dataset(data, finetune_split):
    """
        split the data into train and val (maintain the balance between classes)
        @return data_train, data_val
    """

    # separate train and val data
    # used for fine tune
    data_train, data_val = defaultdict(list), defaultdict(list)

    # sort each matrix by ascending label order for each searching
    idx = np.argsort(data['label'], kind="stable")

    non_idx_keys = ['vocab_size', 'classes2id', 'is_train', 'n_t', 'n_d', 'avg_ebd']
    for k, v in data.items():
        if k not in non_idx_keys:
            data[k] = v[idx]

    # loop through classes in ascending order
    classes, counts = np.unique(data['label'], return_counts=True)
    start = 0
    for label, n in zip(classes, counts):
        mid = start + int(finetune_split * n)  # split between train/val
        end = start + n  # split between this/next class

        for k, v in data.items():
            if k not in non_idx_keys:
                data_train[k].append(v[start:mid])
                data_val[k].append(v[mid:end])

        start = end  # advance to next class

    # convert back to np arrays
    for k, v in data.items():
        if k not in non_idx_keys:
            data_train[k] = np.concatenate(data_train[k], axis=0)
            data_val[k] = np.concatenate(data_val[k], axis=0)

    return data_train, data_val


def load_dataset(args):

    if args.dataset == 'stego':
        train_classes, val_classes, test_classes = _get_stego_classes(args)

    elif args.dataset == 'stego_draw':
        train_classes, val_classes, test_classes = _get_stego_draw_classes(args)

    else:
        raise ValueError(
            'args.dataset should be one of'
            '[stego,stego_draw]')

    assert(len(train_classes) == args.n_train_class)
    assert(len(val_classes) == args.n_val_class)
    assert(len(test_classes) == args.n_test_class)

    if args.train_mode == 't_add_v':

        train_classes = train_classes + val_classes
        args.n_train_class = args.n_train_class + args.n_val_class

    tprint('Loading data')
    all_data = _load_json(args.data_path)

    tprint('Loading word vectors')

    vectors = Vectors(args.word_vector, cache=args.wv_path)
    vocab = Vocab(collections.Counter(_read_words(all_data)), vectors=vectors,
                  specials=['<pad>', '<unk>'], min_freq=5)
    #print("vocab:",vocab)

    # print word embedding statistics
    wv_size = vocab.vectors.size()
    tprint('Total num. of words: {}, word vector dimension: {}'.format(
        wv_size[0],
        wv_size[1]))

    num_oov = wv_size[0] - torch.nonzero(
            torch.sum(torch.abs(vocab.vectors), dim=1)).size()[0]
    tprint(('Num. of out-of-vocabulary words'
           '(they are isnitialized to zeros): {}').format( num_oov))

    # Split into meta-train, meta-val, meta-test data
    train_data, val_data, test_data = _meta_split(
            all_data, train_classes, val_classes, test_classes)
    tprint('#train {}, #val {}, #test {}'.format(
        len(train_data), len(val_data), len(test_data)))

    # Convert everything into np array for fast data loading
    train_data = _data_to_nparray(train_data, vocab, args)
    val_data = _data_to_nparray(val_data, vocab, args)
    test_data = _data_to_nparray(test_data, vocab, args)

    train_data['is_train'] = True
    # this tag is used for distinguishing train/val/test when creating source pool

    stats.precompute_stats(train_data, val_data, test_data, args)

    return train_data, val_data, test_data, vocab
