# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, unicode_literals

"""
In addition to SentEval, this script additionally requires CoVe and its requirements: https://github.com/salesforce/cove
"""

import sys
import torch
from torch import nn
from torch.autograd import Variable
from exutil import dotdict
import logging
from torchtext import data
from torchtext import datasets
import numpy as np

# Set PATHs
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
The user has to implement two functions:
    1) "batcher" : transforms a batch of sentences into sentence embeddings.
        i) takes as input a "batch", and "params".
        ii) outputs a numpy array of sentence embeddings
        iii) Your sentence encoder should be in "params"
    2) "prepare" : sees the whole dataset, and can create a vocabulary
        i) outputs of "prepare" are stored in "params" that batcher will use.
"""

def prepare(params, samples):
    inputs = data.Field(lower=True, include_lengths=True, batch_first=True, tokenize="moses")
    inputs.build_vocab([' '.join(s) for s in samples])
    inputs.vocab.load_vectors('glove.840B.300d')
    params.cove.cove.embed = True
    params.cove.vectors = nn.Embedding(len(inputs.vocab), 300)
    if params.cove.vectors is not None:
        params.cove.vectors.weight.data = inputs.vocab.vectors
    return

def batcher(params, batch):
    word_vec = params.cove.vectors # TODO: See if this is correct

    def prepare_samples(sentences, bsize):
        sentences = [['<s>'] + s.split() + ['</s>'] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without glove vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "{0}" (idx={1}) have glove vectors. \
                               Replacing by "</s>"..'.format(sentences[i], i))
                s_f = ['</s>']
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def get_batch(batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), 300))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = word_vec[batch[i][j]]

        return torch.FloatTensor(embed)


    print("BATCHER")
    sentences = [' '.join(s) for s in batch]
    bsize=params.batch_size
    tokenize=False
    sentences, lengths, idx_sort = prepare_samples(sentences, bsize)

    embeddings = []
    for stidx in range(0, len(sentences), bsize):
        this_batch = Variable(get_batch(sentences[stidx:stidx + bsize]), volatile=True)
        this_batch = this_batch.cuda()
        this_batch = params.cove((this_batch, lengths[stidx:stidx + bsize])).data.cpu().numpy()
        embeddings.append(this_batch)
    embeddings = np.vstack(embeddings)

    # unsort
    idx_unsort = np.argsort(idx_sort)
    embeddings = embeddings[idx_unsort]
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
    params_senteval.cove = MTLSTM()
    params_senteval.cove.cuda(0)

    # Run SentEval
    se = senteval.SentEval(params_senteval, batcher, prepare)
    results_transfer = se.eval(transfer_tasks)

    print(results_transfer)
