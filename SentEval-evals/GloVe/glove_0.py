# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of the InferSent and SentEval
# source trees.
#
# The source code below is a modified version of source code from:
# InferSent: https://github.com/facebookresearch/InferSent
# SentEval: https://github.com/facebookresearch/SentEval
#

from __future__ import absolute_import, division, unicode_literals

import sys
import torch
from torch import nn
from torch.autograd import Variable
import logging
from torchtext import data
from torchtext import datasets
import numpy as np

"""dotdict from InferSent / SentEval"""
class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# Set PATHs
SENTEVAL_PATH = '/mnt/mmenezes/libs/SentEval/'
SENTEVAL_DATA_PATH = '/mnt/mmenezes/libs/SentEval/data/senteval_data/'

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
    params.inputs = data.Field(lower=True, include_lengths=True, batch_first=True)
    params.inputs.build_vocab(samples)
    params.inputs.vocab.load_vectors('glove.840B.300d')
    return

def batcher(params, batch):
    embeddings = []
    max_sent_len = 256 # Looks like we have to force this for SentEval to work
    for raw_sentence in batch:
        sentence = []
        for word in raw_sentence:
            sentence.append(word)
        if len(sentence) > max_sent_len:
            sentence = sentence[:max_sent_len]
        if len(sentence) == 0:
            embeddings.append(np.array([np.repeat(0.0, max_sent_len * 300)]))
            continue
        embedding = []
        for word in sentence:
            vector = params.inputs.vocab.vectors[params.inputs.vocab.stoi[word]].numpy()
            for num in vector:
                embedding.append(num)
        for pad in range((max_sent_len - len(sentence)) * 300):
            embedding.append(0.0)
        embeddings.append(np.array([embedding]))
    embeddings = np.vstack(embeddings)
    return embeddings

"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# Define transfer tasks
transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST', 'TREC', 'MRPC', 'SNLI', 'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16']

# Define SentEval params
params_senteval = dotdict({'usepytorch': True, 'task_path': SENTEVAL_DATA_PATH, 'seed': 1111, 'kfold': 5})

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # Run SentEval
    se = senteval.SentEval(params_senteval, batcher, prepare)
    results_transfer = se.eval(transfer_tasks)

    print(results_transfer)
