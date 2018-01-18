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
import os
import torch
import logging

"""dotdict from InferSent / SentEval"""
class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# Set PATHs
GLOVE_PATH = '/mnt/mmenezes/libs/InferSent/dataset/GloVe/glove.840B.300d.txt'
PATH_SENTEVAL = '/mnt/mmenezes/libs/SentEval/'
PATH_TO_DATA = '/mnt/mmenezes/libs/SentEval/data/senteval_data/'
MODEL_PATH = '/mnt/mmenezes/libs/InferSent/encoder/infersent.allnli.pickle'

assert os.path.isfile(MODEL_PATH) and os.path.isfile(GLOVE_PATH), \
    'Set MODEL and GloVe PATHs'

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval


def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples],
                                 tokenize=False)


def batcher(params, batch):
    # batch contains list of words
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size,
                                         tokenize=False)
    return embeddings


"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# Define transfer tasks
#transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST', 'TREC', 'SICKRelatedness', 'SICKEntailment', 'MRPC', 'STS14']
transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST', 'TREC', 'MRPC', 'SNLI', 'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16']

# define senteval params
params_senteval = dotdict({'usepytorch': True, 'task_path': PATH_TO_DATA,
                           'seed': 1111, 'kfold': 5})

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # Load model
    params_senteval.infersent = torch.load(MODEL_PATH, map_location={'cuda:1' : 'cuda:0', 'cuda:2' : 'cuda:0'})
    params_senteval.infersent.set_glove_path(GLOVE_PATH)

    se = senteval.SentEval(params_senteval, batcher, prepare)
    results_transfer = se.eval(transfer_tasks)

    print(results_transfer)
