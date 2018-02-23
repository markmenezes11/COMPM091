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
parser.add_argument("--bilstm_encoder_n_hidden", type=int, default=300, help="Number of hidden states in encoder's BiLSTM(s) (int)")
parser.add_argument("--bilstm_encoder_forget_bias1", type=float, default=1.0, help="Forget bias for encoder's first BiLSTM (float)")
parser.add_argument("--bilstm_encoder_forget_bias2", type=float, default=1.0, help="Forget bias for encoder's second BiLSTM - only needs to be set if --same_bilstm_for_encoder is not set (float)")
parser.add_argument("--bilstm_integrate_n_hidden", type=int, default=300, help="Number of hidden states in integrate's BiLSTM(s) (int)")
parser.add_argument("--bilstm_integrate_forget_bias1", type=float, default=1.0, help="Forget bias for integrate's first BiLSTM (float)")
parser.add_argument("--bilstm_integrate_forget_bias2", type=float, default=1.0, help="Forget bias for integrate's second BiLSTM (float)")
parser.add_argument("--dropout_ratio", type=float, default=0.1, help="Ratio for dropout applied before Feedforward Network and before each Batch Norm (float)")
parser.add_argument("--maxout_reduction", type=int, default=2, help="On the first and second maxout layers, the dimensionality is divided by this number (int)")
parser.add_argument("--bn_decay", type=float, default=0.999, help="Decay for each batch normalisation (float)")
parser.add_argument("--bn_epsilon", type=float, default=1e-3, help="Epsilon for each batch normalisation (float)")
parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer (adam or gradientdescent)")
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

hyperparameters = {
    'n_epochs': args.n_epochs, # int # TODO: Tune this or implement early stopping
    'batch_size': args.batch_size, # int # TODO: Tune this

    'same_bilstm_for_encoder': args.same_bilstm_for_encoder, # boolean # TODO: Tune this
    'bilstm_encoder_n_hidden': args.bilstm_encoder_n_hidden, # int. Used by McCann et al.: 300
    'bilstm_encoder_forget_bias1': args.bilstm_encoder_forget_bias1, # float # TODO: Tune this
    'bilstm_encoder_forget_bias2': args.bilstm_encoder_forget_bias2, # float - only needs to be set if same_bilstm_for_encoder is False # TODO: Tune this

    'bilstm_integrate_n_hidden': args.bilstm_integrate_n_hidden, # int. Used by McCann et al.: 300
    'bilstm_integrate_forget_bias1': args.bilstm_integrate_forget_bias1, # float # TODO: Tune this
    'bilstm_integrate_forget_bias2': args.bilstm_integrate_forget_bias2, # float # TODO: Tune this

    'dropout_ratio': args.dropout_ratio, # float. Used by McCann et al.: 0.1, 0.2 or 0.3 # TODO: Tune this as either 0.1, 0.2 or 0.3 ##############
    'maxout_reduction': args.maxout_reduction, # int. Used by McCann et al.: 2, 4 or 8 # TODO: Tune this as either 2, 4 or 8 ######################

    'bn_decay': args.bn_decay, # float # TODO: Tune this
    'bn_epsilon': args.bn_epsilon, # float # TODO: Tune this

    'optimizer': args.optimizer, # "adam" or "gradientdescent". Used by McCann et al.: "adam"
    'learning_rate': args.learning_rate, # float. Used by McCann et al.: 0.001
    'adam_beta1': args.adam_beta1, # float (used only if optimizer == "adam") # TODO: Tune this
    'adam_beta2': args.adam_beta2, # float (used only if optimizer == "adam") # TODO: Tune this
    'adam_epsilon': args.adam_epsilon # float (used only if optimizer == "adam") # TODO: Tune this
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

print("\nReal time taken to train + test: %s seconds" % (timeit.default_timer() - start_time))
