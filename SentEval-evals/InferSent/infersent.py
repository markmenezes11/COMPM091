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
import argparse

parser = argparse.ArgumentParser(description='SentEval Evaluation of Sentence Representations')
parser.add_argument("--transfertask", type=str, default="", help="Which SentEval transfer task to run. Leave blank to run all of them")
parser.add_argument("--sentevalpath", type=str, default="/mnt/mmenezes/libs/SentEval/", help="Path to SentEval repository")
parser.add_argument("--inputdir", type=str, default='/mnt/mmenezes/libs/InferSent/encoder/', help="Input directory where the model/encoder will be loaded from")
parser.add_argument("--inputmodelname", type=str, default='infersent.allnli.pickle')
parser.add_argument("--wordvecpath", type=str, default="/mnt/mmenezes/libs/InferSent/dataset/GloVe/glove.840B.300d.txt", help="Path to word vectors txt file (e.g. GloVe)")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID. Set to -1 for CPU mode")
params, _ = parser.parse_known_args()

# Set PATHs
MODEL_PATH = ''

# Import senteval
sys.path.insert(0, params.sentevalpath)
import senteval

# Set data path
slash = "" if params.sentevalpath[-1] == '/' else "/"
PATH_TO_DATA = params.sentevalpath + slash + 'data/senteval_data'

def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)

def batcher(params, batch):
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)
    return embeddings

"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# Define senteval params
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'seed': 1111}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # Load InferSent model
    params_senteval['infersent'] = torch.load(params.inputdir, params.inputmodelname, map_location={'cuda:1' : 'cuda:0', 'cuda:2' : 'cuda:0'})
    params_senteval['infersent'].set_glove_path(params.wordvecpath)

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SNLI',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval']
    results = se.eval(transfer_tasks)
    print(results)
