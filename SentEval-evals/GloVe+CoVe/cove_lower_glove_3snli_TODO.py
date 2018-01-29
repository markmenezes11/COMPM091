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
# CoVe is taken from:
# B. McCann, J. Bradbury, C. Xiong, R. Socher, Learned in Translation: Contextualized Word Vectors
# https://github.com/salesforce/cove
#

from __future__ import absolute_import, division, unicode_literals

"""
In addition to SentEval, this script additionally requires CoVe and its requirements: https://github.com/salesforce/cove
"""

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
COVE_PATH = '/mnt/mmenezes/libs/cove'
SENTEVAL_PATH = '/mnt/mmenezes/libs/SentEval/'
SENTEVAL_DATA_PATH = '/mnt/mmenezes/libs/SentEval/data/senteval_data/'

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
    #print("batch")
    embeddings = []
    max_sent_len = 256 # Looks like we have to force this for SentEval to work
    #max_sent_len = 0
    #for raw_sentence in batch:
    #    if len(raw_sentence) > max_sent_len:
    #        max_sent_len = len(raw_sentence)
    for raw_sentence in batch:
        sentence = [word.lower() for word in raw_sentence]
        if len(sentence) > max_sent_len:
            sentence = sentence[:max_sent_len]
        vector_list = []
        padded_vector_list = []
        if len(sentence) == 0:
            embeddings.append(np.zeros((1, (max_sent_len + 1) * 600), dtype=float))
            #print(str(np.zeros(((max_sent_len + 1) * 600), dtype=float).shape))
            continue
        for word in sentence:
            vector_list.append(params.inputs.vocab.vectors[params.inputs.vocab.stoi[word]])
            padded_vector_list.append(torch.cat((params.inputs.vocab.vectors[params.inputs.vocab.stoi[word]], torch.zeros(300))))
        vector_tensor = torch.autograd.Variable(torch.stack(vector_list)).unsqueeze(1).cuda(0)
        length_tensor = (torch.LongTensor([len(vector_list)])).cuda(0)
        padded_vector_tensor =  torch.autograd.Variable(torch.stack(padded_vector_list)).unsqueeze(1)
        #print("GLOVE: " + str(padded_vector_tensor.size()))
        #print("COVE: " + str(params.cove(vector_tensor, length_tensor).cpu().size()))
        #print("SENTENCE LENGTH: " + str(len(sentence)))
        padding = torch.zeros(max_sent_len - len(sentence), 1, 600)
        #print("PADDING: " + str(padding.size()))
        if (max_sent_len - len(sentence)) < 1:
            #print("NO PADDING")
            embedding_tensor = torch.cat((padded_vector_tensor, params.cove(vector_tensor, length_tensor).cpu()))
        else:
            embedding_tensor = torch.cat((padded_vector_tensor, padding, params.cove(vector_tensor, length_tensor).cpu()))
        embedding = []
        for vector in embedding_tensor.data.numpy():
            for num in vector[0]:
                embedding.append(num)
        assert(len(embedding) == (max_sent_len + 1) * 600)
        #print("EMBEDDING: " + str(np.array([embedding]).shape) + "\n")
        embeddings.append(np.array([embedding]))
    embeddings = np.vstack(embeddings)
    return embeddings


"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# Define transfer tasks
#transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST', 'TREC', 'SICKRelatedness', 'SICKEntailment', 'MRPC', 'STS14']
transfer_tasks = ['SNLI']

# Define SentEval params
params_senteval = dotdict({'usepytorch': True, 'task_path': SENTEVAL_DATA_PATH, 'seed': 1111, 'kfold': 5})

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # Run SentEval
    se = senteval.SentEval(params_senteval, batcher, prepare)
    results_transfer = se.eval(transfer_tasks)

    print(results_transfer)
