# This file makes use of the InferSent, SentEval and CoVe libraries, and may contain adapted code from the repositories
# containing these libraries. Their licenses can be found in <this-repository>/Licenses.
#
# InferSent and SentEval:
#   Copyright (c) 2017-present, Facebook, Inc. All rights reserved.
#   InferSent repository: https://github.com/facebookresearch/InferSent
#   SentEval repository: https://github.com/facebookresearch/SentEval
#   Reference: Conneau, Alexis, Kiela, Douwe, Schwenk, Holger, Barrault, Loic, and Bordes, Antoine. Supervised learning
#              of universal sentence representations from natural language inference data. In Proceedings of the 2017
#              Conference on Empirical Methods in Natural Language Processing, pp. 670-680. Association for
#              Computational Linguistics, 2017.
#
# CoVe:
#   Copyright (c) 2017, Salesforce.com, Inc. All rights reserved.
#   Repository: https://github.com/salesforce/cove
#   Reference: McCann, Bryan, Bradbury, James, Xiong, Caiming, and Socher, Richard. Learned in translation:
#              Contextualized word vectors. In Advances in Neural Information Processing Systems 30, pp, 6297-6308.
#              Curran Associates, Inc., 2017.
#
# This code also makes use of GloVe: Pennington, Jeffrey, Socher, Richard, and Manning, Christopher D. Glove: Global
# vectors for word representation. In Empirical Methods in Natural Language Processing (EMNLP), pp. 1532-1543, 2014.
#
# This code also makes use of Keras: Chollet, Francois et al. Keras. https://github.com/keras-team/keras, 2015
#
# This code also makes use of PyTorch: Pytorch. [online]. Available at: https://github.com/pytorch/pytorch
#

from __future__ import absolute_import, division, unicode_literals

from keras.models import load_model

import sys
import os
import torch
import logging
import argparse
import timeit
import numpy as np

from torchtext import data

start_time = timeit.default_timer()

parser = argparse.ArgumentParser(description='SentEval Evaluation of CoVe Sentence Representations')
parser.add_argument("--transfertask", type=str, default="", help="Which SentEval transfer task to run. Leave blank to run all of them")
parser.add_argument("--sentevalpath", type=str, default="../../SentEval-modified/", help="Path to SentEval repository")
parser.add_argument("--modelpath", type=str, default='../../CoVe-ported/Keras_CoVe_Python2.h5', help="Path to the CoVe model")
parser.add_argument("--outputdir", type=str, default='.', help="Output directory to save results")
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
    params.inputs = data.Field(lower=True, include_lengths=True, batch_first=True)
    params.inputs.build_vocab(samples)
    params.inputs.vocab.load_vectors('glove.840B.300d')
    params.cove = load_model(params.inputdir)
    return

def batcher(params, batch):
    embeddings = []
    max_sent_len = params.max_sent_len # Looks like we have to force this to allow padded sentences for SentEval to work
    for sentence in batch:
        if len(sentence) > max_sent_len:
            print("ERROR: Sentence is longer than max_sent_len.")
            sys.exit(1)
        if len(sentence) == 0:
            embedding = np.zeros((1, max_sent_len * 600), dtype=float)
            embeddings.append(embedding)
            continue
        vector_list = []
        for word in sentence:
            if word in params.inputs.vocab.stoi and np.count_nonzero(params.inputs.vocab.vectors[params.inputs.vocab.stoi[word]].numpy()) != 0:
                vector_list.append(params.inputs.vocab.vectors[params.inputs.vocab.stoi[word]].numpy())
            else:
                vector_list.append(np.full((300), 1e-10))
        embedding = []
        context_vectors = params.cove.predict(np.array([vector_list]))
        for vector in context_vectors[0]:
            for num in vector:
                embedding.append(num)
        for pad in range((max_sent_len - len(sentence)) * 600):
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
params_senteval['inputdir'] = params.modelpath

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
    se = senteval.engine.SE(params_senteval, batcher, prepare)

    single_task = False
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SNLI',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval']
    if params.transfertask != "" and params.transfertask in transfer_tasks:
        single_task = True
        transfer_tasks = params.transfertask

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
