# CoVe is taken from:
# B. McCann, J. Bradbury, C. Xiong, R. Socher, Learned in Translation: Contextualized Word Vectors
# https://github.com/salesforce/cove
#

import sys
import argparse

parser = argparse.ArgumentParser(description='Replication of the CoVe Biattentive Classification Network (BCN)')
parser.add_argument("--task", type=str, default="SSTBinary", help="Transfer task used for training BCN and evaluating predictions (e.g. SSTBinary)")
parser.add_argument("--glovepath", type=str, default="../../Word2Vec_models/GloVe/glove.840B.300d.txt", help="Path to GloVe word embeddings. Download glove.840B.300d embeddings from https://nlp.stanford.edu/projects/glove/")
parser.add_argument("--ignoregloveheader", action="store_true", default=False, help="Set this flag if the first line of the GloVe file is a header and not a (word, embedding) pair")
parser.add_argument("--covepath", type=str, default='../CoVe-ported/Keras_CoVe_Python2.h5', help="Path to the CoVe model")
parser.add_argument("--covedim", type=int, default=600, help="Number of dimensions in CoVe embeddings (default: 600)")
parser.add_argument("--datadir", type=str, default='datasets', help="Path to the directory that contains the datasets")
parser.add_argument("--mode", type=int, default=0, help="0: Normal (train + test); 1: BCN model dry-run (just try creating the model and do nothing else); 2: Train + test dry-run (Load a smaller dataset and train + test on it)")
args, _ = parser.parse_known_args()

from data_processing import GloVeCoVeEncoder
from datasets import SSTBinaryDataset
from model import BCN

"""
HYPERPARAMETERS
"""

hyperparameters = { # TODO: Tune the following parameters using the Dev set for validation
    'n_epochs': 2, # int
    'batch_size': 64, # int

    'feedforward_weight_size': 0.1, # float
    'feedforward_bias_size': 0.1, # float
    'feedforward_activation': "ReLU6", # "ReLU or ReLU6"

    'same_bilstm_for_encoder': True, # boolean
    'bilstm_encoder_n_hidden': 200, # int
    'bilstm_encoder_forget_bias1': 1.0, # float
    'bilstm_encoder_forget_bias2': 1.0, # float - only needs to be set if same_bilstm_for_encoder is False

    'bilstm_integrate_n_hidden': 200, # int
    'bilstm_integrate_forget_bias1': 1.0, # float
    'bilstm_integrate_forget_bias2': 1.0, # float

    'self_pool_weight_size1': 0.1, # float
    'self_pool_bias_size1': 0.1, # float
    'self_pool_weight_size2': 0.1, # float
    'self_pool_bias_size2': 0.1, # float

    'bn_decay1': 0.999, # float
    'bn_epsilon1': 1e-3, # float
    'bn_decay2': 0.999, # float
    'bn_epsilon2': 1e-3, # float
    'bn_decay3': 0.999,  # float
    'bn_epsilon3': 1e-3,  # float

    'softmax_weight_size': 0.1,  # float
    'softmax_bias_size': 0.1,  # float

    'optimizer': "gradientdescent", # "adam" or "gradientdescent"
    'learning_rate': 0.001, # float
    'adam_beta1': 0.9, # float (used only if optimizer == "adam")
    'adam_beta2': 0.999, # float (used only if optimizer == "adam")
    'adam_epsilon': 1e-08 # float (used only if optimizer == "adam")
}

if args.mode == 1:
    BCN(hyperparameters, 3, 128, 900).dry_run()
    sys.exit()

"""
DATASET
"""

if args.task == "SSTBinary":
    data_encoder = GloVeCoVeEncoder(args.glovepath, args.covepath, ignore_glove_header=args.ignoregloveheader, cove_dim=args.covedim)
    embeddings_length = data_encoder.get_embeddings_length()
    dataset = SSTBinaryDataset(args.datadir, args.mode == 2)
    data = dataset.generate_embeddings(data_encoder)
    data_encoder = None # So that the GloVe embeddings and CoVe model can be garbage collected
    n_classes = dataset.get_n_classes()
    max_sent_len = dataset.get_max_sent_len()
else: # TODO: Add more tasks
    print("ERROR: Unknown transfer task. Set it correctly using the --task argument.")
    sys.exit(1)

"""
BCN MODEL
"""

bcn = BCN(hyperparameters, n_classes, max_sent_len, embeddings_length)
bcn.train(data)
bcn.test(data)
