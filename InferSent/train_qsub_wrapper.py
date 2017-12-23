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

import argparse

from subprocess import Popen, PIPE, STDOUT

parser = argparse.ArgumentParser(description='NLI training')

# infersent path
parser.add_argument("--infersentpath", type=str, default='../../../InferSent-master', help="Path to InferSent repository")

# other paths
parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='model.pickle')
parser.add_argument("--nlipath", type=str, default='../../../InferSent-master/dataset/SNLI/', help="NLI data path (SNLI or MultiNLI)")
parser.add_argument("--wordvecpath", type=str, default="../../../InferSent-master/dataset/GloVe/glove.840B.300d.txt", help="Path to word vectors txt file (e.g. GloVe)")

# training
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--nonlinear_fc", type=float, default=0, help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

# model
parser.add_argument("--encoder_type", type=str, default='BLSTMEncoder', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")

# gpu
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID. Set to -1 for CPU mode")
parser.add_argument("--seed", type=int, default=1234, help="seed")

params, _ = parser.parse_known_args()

p = Popen("python /home/mmenezes/Dev/COMPM091/InferSent/train.py" +
              " --outputdir " + params.outputdir +
              " --outputmodelname " + params.outputmodelname +
              " --infersentpath " + params.infersentpath +
              " --gpu_id " + str(params.gpu_id) +
              " --nlipath " + params.nlipath +
              " --wordvecpath " + params.wordvecpath +
              " --n_epochs " + str(params.n_epochs) +
              " --batch_size " + str(params.batch_size) +
              " --dpout_model " + str(params.dpout_model) +
              " --dpout_fc " + str(params.dpout_fc) +
              " --nonlinear_fc " + str(params.nonlinear_fc) +
              " --optimizer " + params.optimizer +
              " --lrshrink " + str(params.lrshrink) +
              " --decay " + str(params.decay) +
              " --minlr " + str(params.minlr) +
              " --max_norm " + str(params.max_norm) +
              " --encoder_type " + params.encoder_type +
              " --enc_lstm_dim " + str(params.enc_lstm_dim) +
              " --n_enc_layers " + str(params.n_enc_layers) +
              " --fc_dim " + str(params.fc_dim) +
              " --pool_type " + params.pool_type +
              " --seed " + str(params.seed), stdout=PIPE, stderr=STDOUT, bufsize=1, shell=True)

with p.stdout, open(params.outputdir + "train_output.txt", 'ab') as file:
    for line in iter(p.stdout.readline, b''):
        print line,  # Comma to prevent duplicate newlines
        file.write(line)
p.wait()
