import sys, os
import argparse
import itertools

from subprocess import Popen, PIPE, STDOUT

"""
Arguments
"""

parser = argparse.ArgumentParser(description='InferSent Parameter Sweep')
parser.add_argument("--infersentpath", type=str, default="/mnt/mmenezes/libs/InferSent", help="Path to InferSent repository")
parser.add_argument("--sentevalpath", type=str, default="/mnt/mmenezes/libs/SentEval", help="Path to SentEval repository")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID. Set to -1 for CPU mode")
parser.add_argument("--outputdir", type=str, default='/mnt/mmenezes/InferSent-models/sweep', help="Output directory")
params, _ = parser.parse_known_args()

"""
Parameters to sweep
"""

# NLI data path (e.g. "[path]/AllNLI", "[path]/SNLI" or "[path]/MultiNLI") - should have 3 classes
# (entailment/neutral/contradiction). Default: "AllNLI"
nlipath      = ["/mnt/mmenezes/InferSent-datasets/SNLI_small"]

# Path to word vectors txt file (e.g. "[path]/glove.840B.300d.txt"). Default: "glove.840B.300d.txt"
wordvecpath  = ["/mnt/mmenezes/libs/InferSent/dataset/GloVe/glove.840B.300d.txt"]

# Number of epochs (int). Default: 20
n_epochs     = [20]

# Batch size (int). Default: 64
batch_size   = [64]

# Encoder dropout (float). Default: 0
dpout_model  = [0]

# Classifier dropout (float). Default: 0
dpout_fc     = [0]

# Use nonlinearity in FC (float). Default: 0
nonlinear_fc = [0]

# "adam" or "sgd,lr=0.1". Default: "sgd,lr=0.1"
optimizer    = ["sgd,lr=0.1"]

# Shrink factor for SGD (float). Default: 5
lrshrink     = [5]

# LR decay (float). Default: 0.99
decay        = [0.99]

# Minimum LR (float). Default: 1e-5
minlr        = [1e-5]

# Max norm (grad clipping) (float). Default: 5
max_norm     = [5]

# "BLSTMEncoder", "BLSTMprojEncoder", "BGRUlastEncoder", "InnerAttentionMILAEncoder", "InnerAttentionYANGEncoder",
# "InnerAttentionNAACLEncoder", "ConvNetEncoder" or "LSTMEncoder". Default: "BLSTMEncoder"
encoder_type = ["BLSTMEncoder"]

# Encoder NHID dimension (int). Default: 2048
enc_lstm_dim = [2048]

# Encoder num layers (int). Default: 1
n_enc_layers = [1]

# NHID of FC layers (int). Default: 512
fc_dim       = [512]

# "max" or "mean". Default: "max"
pool_type    = ["max"]

# Random seed (int). Default: 1234
seed         = [1234]

"""
Sweep
"""

iterations = itertools.product(nlipath, wordvecpath, n_epochs, batch_size, dpout_model, dpout_fc, nonlinear_fc,
                                   optimizer, lrshrink, decay, minlr, max_norm, encoder_type, enc_lstm_dim,
                                   n_enc_layers, fc_dim, pool_type, seed)

def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)

def batcher(params, batch):
    # batch contains list of words
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)
    return embeddings

iterationNumber = 0
for iteration in iterations:
    iterationNumber += 1
    print("\n\n\n\n\n\n\n####### Iteration " + str(iterationNumber) + " of " + str(len(list(iterations))) + "...")

    print("\n\n\nPreparing output directory...\n")

    # Get the output directory based on current params in this iteration
    slash = "" if params.outputdir[-1] == "/" else "/"
    sweepdir = ("nlipath___" + iteration[0].replace('/', '_').replace(':', '_') + "/" +
                "wordvecpath___" + iteration[1].replace('/', '_').replace(':', '_') + "/" +
                "n_epochs___" + str(iteration[2]) + "/" +
                "batch_size___" + str(iteration[3]) + "/" +
                "dpout_model___" + str(iteration[4]) + "/" +
                "dpout_fc___" + str(iteration[5]) + "/" +
                "nonlinear_fc___" + str(iteration[6]) + "/" +
                "optimizer___" + iteration[7].replace('/', '_').replace(':', '_') + "/" +
                "lrshrink___" + str(iteration[8]) + "/" +
                "decay___" + str(iteration[9]) + "/" +
                "minlr___" + str(iteration[10]) + "/" +
                "max_norm___" + str(iteration[11]) + "/" +
                "encoder_type___" + iteration[12].replace('/', '_').replace(':', '_') + "/" +
                "enc_lstm_dim___" + str(iteration[13]) + "/" +
                "n_enc_layers___" + str(iteration[14]) + "/" +
                "fc_dim___" + str(iteration[15]) + "/" +
                "pool_type___" + iteration[16].replace('/', '_').replace(':', '_') + "/" +
                "seed___" + str(iteration[17]) + "/")
    outputdir = params.outputdir + slash + sweepdir

    # Make the output directory if it doesn't exist
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    # Write the parameters to the output text file in the output directory
    formattedParams = ("  --nlipath " + iteration[0] + "\n" +
                       "  --wordvecpath " + iteration[1] + "\n" +
                       "  --n_epochs " + str(iteration[2]) + "\n" +
                       "  --batch_size " + str(iteration[3]) + "\n" +
                       "  --dpout_model " + str(iteration[4]) + "\n" +
                       "  --dpout_fc " + str(iteration[5]) + "\n" +
                       "  --nonlinear_fc " + str(iteration[6]) + "\n" +
                       "  --optimizer " + iteration[7] + "\n" +
                       "  --lrshrink " + str(iteration[8]) + "\n" +
                       "  --decay " + str(iteration[9]) + "\n" +
                       "  --minlr " + str(iteration[10]) + "\n" +
                       "  --max_norm " + str(iteration[11]) + "\n" +
                       "  --encoder_type " + iteration[12] + "\n" +
                       "  --enc_lstm_dim " + str(iteration[13]) + "\n" +
                       "  --n_enc_layers " + str(iteration[14]) + "\n" +
                       "  --fc_dim " + str(iteration[15]) + "\n" +
                       "  --pool_type " + iteration[16] + "\n" +
                       "  --seed " + str(iteration[17]) + "\n")

    print("\n\n\nParameters:\n" + formattedParams + "...\n")
    with open(outputdir + "output.txt", "a") as outputfile:
        outputfile.write("\n\n\nParameters:\n" + formattedParams + "\n")

    p = Popen("python train_and_eval.py" +
              " --outputdir " + outputdir +
              " --infersentpath " + params.infersentpath +
              " --sentevalpath " + params.sentevalpath +
              " --gpu_id " + str(params.gpu_id) +
              " --nlipath " + iteration[0] +
              " --wordvecpath " + iteration[1] +
              " --n_epochs " + str(iteration[2]) +
              " --batch_size " + str(iteration[3]) +
              " --dpout_model " + str(iteration[4]) +
              " --dpout_fc " + str(iteration[5]) +
              " --nonlinear_fc " + str(iteration[6]) +
              " --optimizer " + iteration[7] +
              " --lrshrink " + str(iteration[8]) +
              " --decay " + str(iteration[9]) +
              " --minlr " + str(iteration[10]) +
              " --max_norm " + str(iteration[11]) +
              " --encoder_type " + iteration[12] +
              " --enc_lstm_dim " + str(iteration[13]) +
              " --n_enc_layers " + str(iteration[14]) +
              " --fc_dim " + str(iteration[15]) +
              " --pool_type " + iteration[16] +
              " --seed " + str(iteration[17]), stdout=PIPE, stderr=STDOUT, bufsize=1, shell=True)

    with p.stdout, open(outputdir + "output.txt", 'ab') as file:
        for line in iter(p.stdout.readline, b''):
            print line,  # Comma to prevent duplicate newlines
            file.write(line)
    p.wait()
