# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, unicode_literals

"""
In addition to SentEval, this script additionally requires CoVe and its requirements: https://github.com/salesforce/cove
"""

import sys, time
import torch
from torch.autograd import Variable
from exutil import dotdict
import logging
import numpy as np

# Set PATHs
GLOVE_PATH = 'glove/glove.840B.300d.txt'
COVE_PATH = '../../cove-master'
SENTEVAL_PATH = '../'
SENTEVAL_DATA_PATH = '../data/senteval_data_ptb/'

# Import CoVe
sys.path.insert(0, COVE_PATH)
from cove import MTLSTM

# Import senteval
sys.path.insert(0, SENTEVAL_PATH)
import senteval

"""
MTLSTM Encoder wrapped with same batcher as InferSent"""
class Wrapped_MTLSTM(MTLSTM):
    def __init__(self, *args):
        super(Wrapped_MTLSTM, self).__init__(*args)
        self.word_emb_dim = 300

    def set_glove_path(self, glove_path):
        self.glove_path = glove_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        if tokenize:
            from nltk.tokenize import word_tokenize
        sentences = [s.split() if not tokenize else word_tokenize(s)
                     for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict['<s>'] = ''
        word_dict['</s>'] = ''
        return word_dict

    def get_glove(self, word_dict):
        assert hasattr(self, 'glove_path'), \
               'warning : you need to set_glove_path(glove_path)'
        # create word_vec with glove vectors
        word_vec = {}
        with open(self.glove_path) as f:
            for line in f:
                word, vec = line.decode('UTF-8').split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=str(' '))
        print('Found {0}(/{1}) words with glove vectors'.format(
                    len(word_vec), len(word_dict)))
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'glove_path'), 'warning : you need \
                                             to set_glove_path(glove_path)'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_glove(word_dict)
        print('Vocab size : {0}'.format(len(self.word_vec)))

    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        if tokenize:
            from nltk.tokenize import word_tokenize
        sentences = [['<s>'] + s.split() + ['</s>'] if not tokenize else
                     ['<s>'] + word_tokenize(s) + ['</s>'] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without glove vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "{0}" (idx={1}) have glove vectors. \
                                  Replacing by "</s>"..'.format(sentences[i], i))
                s_f = ['</s>']
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : {0}/{1} ({2} %)'.format(
                n_wk, n_w, round((100.0 * n_wk) / n_w, 2)))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
            sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch = Variable(self.get_batch(
                sentences[stidx:stidx + bsize]), volatile=True)
            batch = batch.cuda()
            batch = super(Wrapped_MTLSTM, self).forward(
                (batch, lengths[stidx:stidx + bsize])).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : {0} sentences/s ({1} mode, bsize={2})'.format(
                round(len(embeddings) / (time.time() - tic), 2),
                'gpu', bsize))
        return embeddings


"""
The user has to implement two functions:
    1) "batcher" : transforms a batch of sentences into sentence embeddings.
        i) takes as input a "batch", and "params".
        ii) outputs a numpy array of sentence embeddings
        iii) Your sentence encoder should be in "params"
    2) "prepare" : sees the whole dataset, and can create a vocabulary
        i) outputs of "prepare" are stored in "params" that batcher will use.
"""

def prepare(params, samples):
    params.cove.build_vocab([' '.join(s) for s in samples], tokenize=False)
    return

def batcher(params, batch):
    sentences = [' '.join(s) for s in batch]
    embeddings = params.cove.encode(sentences, bsize=params.batch_size, tokenize=False)
    return embeddings

"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# Define transfer tasks
transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST', 'TREC', 'SICKRelatedness', 'SICKEntailment', 'MRPC', 'STS14']

# Define SentEval params
params_senteval = dotdict({'usepytorch': True, 'task_path': SENTEVAL_DATA_PATH, 'seed': 1111, 'kfold': 5})

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    params_senteval.cove = Wrapped_MTLSTM()
    params_senteval.cove.set_glove_path(GLOVE_PATH)
    params_senteval.cove.cuda(0)

    # Run SentEval
    se = senteval.SentEval(params_senteval, batcher, prepare)
    results_transfer = se.eval(transfer_tasks)

    print(results_transfer)
