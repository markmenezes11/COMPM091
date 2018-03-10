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
import timeit

import numpy as np

start_time = timeit.default_timer()

parser = argparse.ArgumentParser(description='SentEval Evaluation of InferSent Sentence Representations')
parser.add_argument("--transfertask", type=str, default="", help="Which SentEval transfer task to run. Leave blank to run all of them")
parser.add_argument("--sentevalpath", type=str, default="/mnt/mmenezes/libs/SentEval/", help="Path to SentEval repository")
parser.add_argument("--inputdir", type=str, default='/mnt/mmenezes/libs/InferSent/encoder/', help="Input directory where the model/encoder will be loaded from")
parser.add_argument("--outputdir", type=str, default='.', help="Output directory to save results")
parser.add_argument("--inputmodelname", type=str, default='infersent.allnli.pickle')
parser.add_argument("--wordvecpath", type=str, default="/mnt/mmenezes/libs/InferSent/dataset/GloVe/glove.840B.300d.txt", help="Path to word vectors txt file (e.g. GloVe)")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID. Set to -1 for CPU mode")
params, _ = parser.parse_known_args()

# Import senteval
sys.path.insert(0, params.sentevalpath)
import senteval

# Set data path
slash = "" if params.sentevalpath[-1] == '/' else "/"
PATH_TO_DATA = params.sentevalpath + slash + 'data/senteval_data'

# Set gpu device
cpu = params.gpu_id == -1
if not cpu:
    torch.cuda.set_device(params.gpu_id)

def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)

def batcher(params, batch):
    embeddings = []
    max_sent_len = params.max_sent_len + 2 # Looks like we have to force this to allow padded sentences for SentEval to work
    for sentence in batch:
        vector_list = params.infersent.encode([' '.join(sentence)], bsize=1, tokenize=False)[0]
        assert vector_list.shape[0] <= len(sentence) + 2
        assert vector_list.shape[1] == 4096
        embedding = []
        for vector in vector_list:
            for num in vector:
                embedding.append(num)
        for pad in range((max_sent_len - vector_list.shape[0]) * 4096):
            embedding.append(0.0)
        embeddings.append(np.array([embedding]))
    embeddings = np.vstack(embeddings)
    return embeddings

"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# Define senteval params
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'seed': 1111}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}

max_sent_lens = {'STS12': 41, 'STS13': 81, 'STS14': 61, 'STS15': 57, 'STS16': 52, 'MR': 62, 'CR': 106, 'MPQA': 44,
                 'SUBJ': 122, 'SST2': 56, 'SST5': 56, 'TREC': 37, 'MRPC': 41, 'SNLI': 82,
                 'SICKEntailment': 36, 'SICKRelatedness': 36, 'STSBenchmark': 61, 'ImageCaptionRetrieval': 50}
if params.transfertask != "" and params.transfertask in max_sent_lens:
    params_senteval['max_sent_len'] = max_sent_lens[params.transfertask]
else:
    params_senteval['max_sent_len'] = 128
print("Sentences will be padded to length " + str(params_senteval['max_sent_len']) + ".")

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # Load InferSent model
    params_senteval['infersent'] = torch.load(os.path.join(params.inputdir, params.inputmodelname))
    params_senteval['infersent'].set_glove_path(params.wordvecpath)

    se = senteval.engine.SE(params_senteval, batcher, prepare)

    single_task = False
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SNLI',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval']
    if params.transfertask != "" and params.transfertask in transfer_tasks:
        single_task = True
        transfer_tasks = params.transfertask
    elif params.transfertask != "" and params.transfertask not in transfer_tasks:
        print("ERROR: Transfer task not found: " + params.transfertask)
        sys.exit()

    results = se.eval(transfer_tasks)

    print("\n\nSENTEVAL RESULTS:")
    if single_task:
        print("\nRESULTS FOR " + params.transfertask + ":\n" + str(results))
    else:
        for task in transfer_tasks:
            print("\nRESULTS FOR " + task + ":\n" + str(results[task]))

    outputslash = "" if params.outputdir[-1] == "/" else "/"
    outputtask = "_" + params.transfertask if single_task else ""
    with open(params.outputdir + outputslash + "se_results" + outputtask + ".txt", "w") as outputfile:
        outputfile.write(str(results))

    print("\n\nReal time taken to evaluate: %s seconds" % (timeit.default_timer() - start_time))
    print("All done.")
