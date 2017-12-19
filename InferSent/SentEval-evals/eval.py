# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import os
import torch
import logging
import argparse

"""dotdict from InferSent / SentEval"""
class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

parser = argparse.ArgumentParser(description='InferSent Parameter Sweep')
parser.add_argument("--modelpath", type=str, default="infersent.allnli.pickle")
parser.add_argument("--sentevalpath", type=str, default="/mnt/mmenezes/libs/SentEval", help="Path to SentEval repository")
parser.add_argument("--wordvecpath", type=str, default="/mnt/mmenezes/libs/InferSent/dataset/GloVe/glove.840B.300d.txt", help="Path to word vectors txt file")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID. Set to -1 for CPU mode")
params, _ = parser.parse_known_args()

# Set PATHs
WORD_VEC_PATH = params.wordvecpath
PATH_SENTEVAL = params.sentevalpath
slash = "" if params.sentevalpath[-1] == '/' else "/"
PATH_TO_DATA = params.sentevalpath + slash + 'data/senteval_data'
MODEL_PATH = params.modelpath

assert os.path.isfile(MODEL_PATH) and os.path.isfile(WORD_VEC_PATH), \
    'Set MODEL and GloVe PATHs'

# Import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval

def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)

def batcher(params, batch):
    # batch contains list of words
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)
    return embeddings

"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# Define transfer tasks
transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST', 'TREC', 'MRPC', 'SNLI', 'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16']

# Define senteval params
params_senteval = dotdict({'usepytorch': True, 'task_path': PATH_TO_DATA,
                           'seed': 1111, 'kfold': 5})

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # Load model
    params_senteval.infersent = torch.load(MODEL_PATH, map_location={'cuda:1' : 'cuda:' + str(params.gpu_id), 'cuda:2' : 'cuda:' + str(params.gpu_id)})
    params_senteval.infersent.set_glove_path(WORD_VEC_PATH)

    se = senteval.SentEval(params_senteval, batcher, prepare)
    results_transfer = se.eval(transfer_tasks)

    print(results_transfer)
