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
from exutil import dotdict
import logging
from torchtext import data
from torchtext import datasets

# Set PATHs
COVE_PATH = '../../cove-master'
GLOVE_PATH = 'glove/glove.840B.300d.txt'
SENTEVAL_PATH = '../'
SENTEVAL_DATA_PATH = '../data/senteval_data_moses/'

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
    # TODO
    return

def batcher(params, batch):
    # TODO
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
    # TODO

    # Run SentEval
    se = senteval.SentEval(params_senteval, batcher, prepare)
    results_transfer = se.eval(transfer_tasks)

    print(results_transfer)
