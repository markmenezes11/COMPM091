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
    #params.inputs = data.Field(lower=True, include_lengths=True, batch_first=True, tokenize="moses")
    params.inputs = data.Field(lower=True, include_lengths=True, batch_first=True) # TODO: Should this be moses tokenized?
    params.inputs.build_vocab(samples)
    params.inputs.vocab.load_vectors('glove.840B.300d')
    params.cove = MTLSTM(n_vocab=len(params.inputs.vocab), vectors=params.inputs.vocab.vectors)
    params.cove.cuda(0)
    params.cove.embed = False
    return

def batcher(params, batch):
    embeddings = []
    for raw_sentence in batch:
        sentence = [word.lower() for word in raw_sentence]
        vector_list = []
        if len(sentence) == 0:
            embeddings.append(np.array([[np.repeat(0.0, 3)]])[0]) # TODO: Do something more intuitive here if sentence is empty?
            continue
        for word in sentence:
            vector_list.append(params.inputs.vocab.vectors[params.inputs.vocab.stoi[word]])
        vector_tensor = torch.autograd.Variable(torch.stack(vector_list)).unsqueeze(1).cuda(0)
        length_tensor = (torch.LongTensor([len(vector_list)])).cuda(0)
        embeddings.append(np.array([params.cove(vector_tensor, length_tensor).cpu().data.numpy()[0][0][:3]]))
    embeddings = np.vstack(embeddings)
    return embeddings


"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# Define transfer tasks
#transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST', 'TREC', 'SICKRelatedness', 'SICKEntailment', 'MRPC', 'STS14']
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
