# CoVe is taken from:
# B. McCann, J. Bradbury, C. Xiong, R. Socher, Learned in Translation: Contextualized Word Vectors
# https://github.com/salesforce/cove
#

import sys
import argparse
import timeit

start_time = timeit.default_timer()

parser = argparse.ArgumentParser(description='Replication of the CoVe Biattentive Classification Network (BCN)')

parser.add_argument("--task", type=str, default="SSTBinary", help="Transfer task used for training BCN and evaluating predictions (e.g. SSTBinary)")
parser.add_argument("--glovepath", type=str, default="../../Word2Vec_models/GloVe/glove.840B.300d.txt", help="Path to GloVe word embeddings. Download glove.840B.300d embeddings from https://nlp.stanford.edu/projects/glove/")
parser.add_argument("--ignoregloveheader", action="store_true", default=False, help="Set this flag if the first line of the GloVe file is a header and not a (word, embedding) pair")
parser.add_argument("--covepath", type=str, default='../CoVe-ported/Keras_CoVe_Python2.h5', help="Path to the CoVe model")
parser.add_argument("--covedim", type=int, default=600, help="Number of dimensions in CoVe embeddings (default: 600)")
parser.add_argument("--datadir", type=str, default='datasets', help="Path to the directory that contains the datasets")
parser.add_argument("--outputdir", type=str, default='model', help="Path to the directory where the BCN model will be saved")
parser.add_argument("--mode", type=int, default=0, help="0: Normal (train + test); 1: BCN model dry-run (just try creating the model and do nothing else); 2: Train + test dry-run (Load a smaller dataset and train + test on it)")

parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs (int)")
parser.add_argument("--batch_size", type=int, default=64, help="Number of epochs (int)")
parser.add_argument("--same_bilstm_for_encoder", action="store_true", default=False, help="Whether or not to use the same BiLSTM (when flag is set) or separate BiLSTMs (flag unset) for the encoder")
parser.add_argument("--bilstm_encoder_n_hidden", type=int, default=200, help="Number of hidden states in encoder's BiLSTM(s) (int)")
parser.add_argument("--bilstm_encoder_forget_bias1", type=float, default=1.0, help="Forget bias for encoder's first BiLSTM (float)")
parser.add_argument("--bilstm_encoder_forget_bias2", type=float, default=1.0, help="Forget bias for encoder's second BiLSTM - only needs to be set if --same_bilstm_for_encoder is not set (float)")
parser.add_argument("--bilstm_integrate_n_hidden", type=int, default=200, help="Number of hidden states in integrate's BiLSTM(s) (int)")
parser.add_argument("--bilstm_integrate_forget_bias1", type=float, default=1.0, help="Forget bias for integrate's first BiLSTM (float)")
parser.add_argument("--bilstm_integrate_forget_bias2", type=float, default=1.0, help="Forget bias for integrate's second BiLSTM (float)")
parser.add_argument("--bn_decay1", type=float, default=0.999, help="Decay for first batch normalisation (float)")
parser.add_argument("--bn_epsilon1", type=float, default=1e-3, help="Epsilon for first batch normalisation (float)")
parser.add_argument("--bn_decay2", type=float, default=0.999, help="Decay for second batch normalisation (float)")
parser.add_argument("--bn_epsilon2", type=float, default=1e-3, help="Epsilon for second batch normalisation (float)")
parser.add_argument("--bn_decay3", type=float, default=0.999, help="Decay for third batch normalisation (float)")
parser.add_argument("--bn_epsilon3", type=float, default=1e-3, help="Epsilon for third batch normalisation (float)")
parser.add_argument("--optimizer", type=str, default="gradientdescent", help="Optimizer (adam or gradientdescent)")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Leaning rate (float)")
parser.add_argument("--adam_beta1", type=float, default=0.9, help="Beta1 for adam optimiser if adam optimiser is used (float)")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="Beta2 for adam optimiser if adam optimiser is used (float)")
parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon for adam optimiser if adam optimiser is used (float)")

args, _ = parser.parse_known_args()

from data_processing import GloVeCoVeEncoder
from datasets import SSTBinaryDataset
from model import BCN

"""
HYPERPARAMETERS
"""

hyperparameters = { # TODO: Tune the following parameters using the Dev set for validation
    'n_epochs': args.n_epochs, # int
    'batch_size': args.batch_size, # int

    'same_bilstm_for_encoder': args.same_bilstm_for_encoder, # boolean
    'bilstm_encoder_n_hidden': args.bilstm_encoder_n_hidden, # int
    'bilstm_encoder_forget_bias1': args.bilstm_encoder_forget_bias1, # float
    'bilstm_encoder_forget_bias2': args.bilstm_encoder_forget_bias2, # float - only needs to be set if same_bilstm_for_encoder is False

    'bilstm_integrate_n_hidden': args.bilstm_integrate_n_hidden, # int
    'bilstm_integrate_forget_bias1': args.bilstm_integrate_forget_bias1, # float
    'bilstm_integrate_forget_bias2': args.bilstm_integrate_forget_bias2, # float

    'bn_decay1': args.bn_decay1, # float
    'bn_epsilon1': args.bn_epsilon1, # float
    'bn_decay2': args.bn_decay2, # float
    'bn_epsilon2': args.bn_epsilon2, # float
    'bn_decay3': args.bn_decay3,  # float
    'bn_epsilon3': args.bn_epsilon3,  # float

    'optimizer': args.optimizer, # "adam" or "gradientdescent"
    'learning_rate': args.learning_rate, # float
    'adam_beta1': args.adam_beta1, # float (used only if optimizer == "adam")
    'adam_beta2': args.adam_beta2, # float (used only if optimizer == "adam")
    'adam_epsilon': args.adam_epsilon # float (used only if optimizer == "adam")
}

if args.mode == 1:
    BCN(hyperparameters, 3, 128, 900, args.outputdir).dry_run()
    sys.exit()

"""
DATASET
"""

if args.task == "SSTBinary":
    encoder = GloVeCoVeEncoder(args.glovepath, args.covepath, ignore_glove_header=args.ignoregloveheader, cove_dim=args.covedim)
    embed_dim = encoder.get_embed_dim()
    dataset = SSTBinaryDataset(args.datadir, encoder, args.mode == 2)
    encoder = None
    n_classes = dataset.get_n_classes()
    max_sent_len = dataset.get_max_sent_len()
else: # TODO: Add more tasks
    print("ERROR: Unknown transfer task. Set it correctly using the --task argument.")
    sys.exit(1)

"""
BCN MODEL
"""

bcn = BCN(hyperparameters, n_classes, max_sent_len, embed_dim, args.outputdir)
bcn.train(dataset)
bcn.test(dataset)

print("\n\nReal time taken to train + test: %s seconds" % (timeit.default_timer() - start_time))
