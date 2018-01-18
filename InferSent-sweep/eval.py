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

import os
import sys
import time
import timeit
import argparse
import logging

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

"""dotdict from InferSent / SentEval"""
class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
start_time = timeit.default_timer()

parser = argparse.ArgumentParser(description='NLI evaluation')

# infersent and senteval path
parser.add_argument("--infersentpath", type=str, default='../../../InferSent-master', help="Path to InferSent repository")
parser.add_argument("--sentevalpath", type=str, default="/mnt/mmenezes/libs/SentEval", help="Path to SentEval repository")

# other paths
parser.add_argument("--inputdir", type=str, default='savedir/', help="Input directory where the model/encoder will be loaded from")
parser.add_argument("--inputmodelname", type=str, default='model.pickle')
parser.add_argument("--wordvecpath", type=str, default="../../../InferSent-master/dataset/GloVe/glove.840B.300d.txt", help="Path to word vectors txt file (e.g. GloVe)")

# gpu
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID. Set to -1 for CPU mode")

params, _ = parser.parse_known_args()

# Import InferSent
sys.path.insert(0, params.infersentpath)
from data import get_nli, get_batch, build_vocab
from mutils import get_optimizer
from models import NLINet

# Import senteval
sys.path.insert(0, params.sentevalpath)
import senteval

# cpu mode if gpu id is -1
cpu = params.gpu_id == -1

# set gpu device
if not cpu:
    torch.cuda.set_device(params.gpu_id)

"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

print("\n\n\nEvaluating model using SentEval...\n")

# Set data path
slash = "" if params.sentevalpath[-1] == '/' else "/"
PATH_TO_DATA = params.sentevalpath + slash + 'data/senteval_data'

def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)

def batcher(params, batch):
    # batch contains list of words
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)
    return embeddings

# Define transfer tasks
transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST', 'TREC', 'MRPC', 'SNLI', 'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16']

# Define senteval params
params_senteval = dotdict({'usepytorch': True, 'task_path': PATH_TO_DATA, 'seed': 1111, 'kfold': 5})

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # Model
    params_senteval.infersent = torch.load(os.path.join(params.inputdir, params.inputmodelname + '.encoder'))
    params_senteval.infersent.set_glove_path(params.wordvecpath)

    se = senteval.SentEval(params_senteval, batcher, prepare)
    results_transfer = se.eval(transfer_tasks)

    print(results_transfer)

print("Real time taken to evaluate: %s seconds" % (timeit.default_timer() - start_time))
print("All done.")
