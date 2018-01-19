import sys, os, time
import argparse
import itertools

from subprocess import Popen, PIPE, STDOUT

"""
Arguments
"""

parser = argparse.ArgumentParser(description='InferSent Parameter Sweep')
parser.add_argument("--mode", type=int, default=0, help="0 to run full sweep (train + eval) on local machine. 1 to run train sweep (train ONLY) using qsub for job submissions on HPC cluster. 2 to run eval sweep (eval ONLY)")
parser.add_argument("--n_jobs", type=int, default=10, help="Maximum number of qsub jobs to be running simultaneously")
parser.add_argument("--infersentpath", type=str, default="/mnt/mmenezes/libs/InferSent", help="Path to InferSent repository. If you are using Singularity, all paths must be the ones that Singularity can see (i.e. make sure to use relevant bindings)")
parser.add_argument("--sentevalpath", type=str, default="/mnt/mmenezes/libs/SentEval", help="Path to SentEval repository. If you are using Singularity, all paths must be the ones that Singularity can see (i.e. make sure to use relevant bindings)")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID. GPU is required because of SentEval. This parameter will be ignored if using qsub, as the GPU will be chosen automatically")
parser.add_argument("--outputdir", type=str, default='/cluster/project2/ishi_storage_1/mmenezes/InferSent-models/sweep', help="Output directory (where models and output will be saved). MAKE SURE IT MAPS TO THE SAME PLACE AS SINGULARITYOUTPUTDIR")
parser.add_argument("--singularitycommand", type=str, default='singularity exec --nv --bind /cluster/project2/ishi_storage_1:/mnt /cluster/project2/ishi_storage_1/mmenezes/markmenezes11-COMPM091-master.simg', help="If you are not using Singularity, make this argument blank. If you are using Singularity, it should be something along the lines of 'singularity exec --nv <extra-params> <path-to-simg-file>', e.g. 'singularity exec --nv --bind /cluster/project2/ishi_storage_1:/mnt /cluster/project2/ishi_storage_1/mmenezes/markmenezes11-COMPM091-master.simg'")
parser.add_argument("--singularityoutputdir", type=str, default='/mnt/mmenezes/InferSent-models/sweep', help="Output directory (where models and output will be saved), that Singularity can see (in case you use binding). If you are not using Singularity, make this the same as outputdir. MAKE SURE IT MAPS TO THE SAME PLACE AS OUTPUTDIR")
params, _ = parser.parse_known_args()

if params.mode == 2: # Eval sweep (eval ONLY)
    print("ERROR: Not implemented.") # TODO: Implement this (see code below)
    sys.exit()

    """
    NOTE: Wordvecpath is important here. It should be the same one used for training
    """

    # Path to word vectors txt file (e.g. "[path]/glove.840B.300d.txt"). Default: "glove.840B.300d.txt"
    wordvecpath = ["/mnt/mmenezes/libs/InferSent/dataset/GloVe/glove.840B.300d.txt"] # TODO: Make it easier to change this parameter without having to scroll through the code

    """
    Sweep
    """
    # TODO: Output dir and singularityoutputdir if needed
    """for each outputdir with specific _wordvecpath:  # TODO: From outputdir, iterate through each folder, MAKING SURE WORD2VECPATH IS THE SAME, AND ONLY COVERING WORD2VECPATHS GIVEN ABOVE
        print("\n\n\nPreparing output directory...\n")

        # Get the output directory based on current params in this iteration
        slash = "" if params.outputdir[-1] == "/" else "/"

        p = Popen("python eval.py" +
                  " --inputdir " + outputdir +
                  " --infersentpath " + params.infersentpath +
                  " --sentevalpath " + params.sentevalpath +
                  " --gpu_id " + str(params.gpu_id) +
                  " --wordvecpath " + _wordvecpath, stdout=PIPE, stderr=STDOUT, bufsize=1, shell=True)

        with p.stdout, open(outputdir + "eval_output.txt", 'ab') as file:
            for line in iter(p.stdout.readline, b''):
                print line,  # Comma to prevent duplicate newlines
                file.write(line)
        p.wait()"""

    sys.exit()

"""
Parameters to sweep. If you are using Singularity, all paths must be the ones that Singularity can see (i.e. make sure to use relevant bindings)
"""

# NLI data path (e.g. "[path]/AllNLI", "[path]/SNLI" or "[path]/MultiNLI") - should have 3 classes
# (entailment/neutral/contradiction). Default: "AllNLI"
nlipath      = ["/mnt/mmenezes/InferSent-datasets/SmallNLI"] # TODO: Make it easier to change this parameter without having to scroll through the code # TODO: Change this from SmallNLI after testing it

# Path to word vectors txt file (e.g. "[path]/glove.840B.300d.txt"). Default: "glove.840B.300d.txt"
wordvecpath  = ["/mnt/mmenezes/libs/InferSent/dataset/GloVe/glove.840B.300d.txt"] # TODO: Make it easier to change this parameter without having to scroll through the code

# Number of epochs (int). Default: 20
n_epochs     = [20] # TODO: Make it easier to change this parameter without having to scroll through the code

# Batch size (int). Default: 64
batch_size   = [32, 64, 128] # TODO: Make it easier to change this parameter without having to scroll through the code

# Encoder dropout (float). Default: 0
dpout_model  = [0] # TODO: Make it easier to change this parameter without having to scroll through the code

# Classifier dropout (float). Default: 0
dpout_fc     = [0] # TODO: Make it easier to change this parameter without having to scroll through the code

# Use nonlinearity in FC (float). Default: 0
nonlinear_fc = [0] # TODO: Make it easier to change this parameter without having to scroll through the code

# "adam" or "sgd,lr=0.1". Default: "sgd,lr=0.1"
optimizer    = ["sgd,lr=0.1", "adam"] # TODO: Make it easier to change this parameter without having to scroll through the code

# Shrink factor for SGD (float). Default: 5
lrshrink     = [5] # TODO: Make it easier to change this parameter without having to scroll through the code

# LR decay (float). Default: 0.99
decay        = [0.99] # TODO: Make it easier to change this parameter without having to scroll through the code

# Minimum LR (float). Default: 1e-5
minlr        = [1e-5] # TODO: Make it easier to change this parameter without having to scroll through the code

# Max norm (grad clipping) (float). Default: 5
max_norm     = [5] # TODO: Make it easier to change this parameter without having to scroll through the code

# "BLSTMEncoder", "BLSTMprojEncoder", "BGRUlastEncoder", "InnerAttentionMILAEncoder", "InnerAttentionYANGEncoder",
# "InnerAttentionNAACLEncoder", "ConvNetEncoder" or "LSTMEncoder". Default: "BLSTMEncoder"
encoder_type = ["BLSTMEncoder"] # TODO: Make it easier to change this parameter without having to scroll through the code

# Encoder NHID dimension (int). Default: 2048
enc_lstm_dim = [2048] # TODO: Make it easier to change this parameter without having to scroll through the code

# Encoder num layers (int). Default: 1
n_enc_layers = [1] # TODO: Make it easier to change this parameter without having to scroll through the code

# NHID of FC layers (int). Default: 512
fc_dim       = [512] # TODO: Make it easier to change this parameter without having to scroll through the code

# "max" or "mean". Default: "max"
pool_type    = ["max", "mean"] # TODO: Make it easier to change this parameter without having to scroll through the code

# Random seed (int). Default: 1234
seed         = [1234] # TODO: Make it easier to change this parameter without having to scroll through the code

"""
Sweep
"""

iterations = itertools.product(nlipath, wordvecpath, n_epochs, batch_size, dpout_model, dpout_fc, nonlinear_fc,
                                   optimizer, lrshrink, decay, minlr, max_norm, encoder_type, enc_lstm_dim,
                                   n_enc_layers, fc_dim, pool_type, seed)
iterationsToCount = itertools.product(nlipath, wordvecpath, n_epochs, batch_size, dpout_model, dpout_fc, nonlinear_fc,
                                   optimizer, lrshrink, decay, minlr, max_norm, encoder_type, enc_lstm_dim,
                                   n_enc_layers, fc_dim, pool_type, seed)
iterationNumber = 0
totalIterations = 0
for iteration in iterationsToCount:
    totalIterations += 1

for iteration in iterations:
    iterationNumber += 1
    print("\n\n\n\n\n\n\n####### Iteration " + str(iterationNumber) + " of " + str(totalIterations) + "...")

    print("\n\n\nPreparing output directory...\n")

    # Get the output directory based on current params in this iteration
    slash = "" if params.outputdir[-1] == "/" else "/"
    singularityslash = "" if params.singularityoutputdir[-1] == "/" else "/"
    sweepdir = ("nlipath___" + iteration[0].replace('/', '_').replace(':', '_') + "/" +
                "wordvecpath___" + iteration[1].replace('/', '_').replace(':', '_') + "/" +
                "n_epochs___" + str(iteration[2]) + "/" +
                "batch_size___" + str(iteration[3]) + "/" +
                "dpout_model___" + str(iteration[4]) + "/" +
                "dpout_fc___" + str(iteration[5]) + "/" +
                "nonlinear_fc___" + str(iteration[6]) + "/" +
                "optimizer___" + iteration[7].replace('/', '_').replace(':', '_').replace(',','_') + "/" +
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
    singularityoutputdir = params.singularityoutputdir + singularityslash + sweepdir

    # If the directory already exists, this iteration has already been run before
    if os.path.exists(outputdir):
        print("Path already exists with these parameters. Skipping this iteration...")
        continue

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
    with open(outputdir + "info.txt", "a") as outputfile:
        outputfile.write("\n\n\nParameters:\n" + formattedParams + "\n")

    if params.mode == 0: # Full sweep (train + eval) on local machine
        p = Popen("python train.py" +
                  " --gpu_id " + str(params.gpu_id) +
                  " --outputdir " + outputdir +
                  " --infersentpath " + params.infersentpath +
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

        with p.stdout, open(outputdir + "train_output.txt", 'ab') as file:
            for line in iter(p.stdout.readline, b''):
                print line,  # Comma to prevent duplicate newlines
                file.write(line)
        p.wait()

        p = Popen("python eval.py" +
                  " --inputdir " + outputdir +
                  " --infersentpath " + params.infersentpath +
                  " --sentevalpath " + params.sentevalpath +
                  " --gpu_id " + str(params.gpu_id) +
                  " --wordvecpath " + iteration[1], stdout=PIPE, stderr=STDOUT, bufsize=1, shell=True)

        with p.stdout, open(outputdir + "eval_output.txt", 'ab') as file:
            for line in iter(p.stdout.readline, b''):
                print line,  # Comma to prevent duplicate newlines
                file.write(line)
        p.wait()

    elif params.mode == 1: # Train sweep (train ONLY) using qsub for job submissions on HPC cluster
        # While there are >=n_jobs jobs running under current user, wait. This is so that we do not hog all of the nodes!
        p = Popen("expr $(qstat -u $USER | wc -l) - 2", stdout=PIPE, stderr=STDOUT, bufsize=1, shell=True)
        with p.stdout:
            numberOfJobs = int(p.stdout.readline(), )
        while (numberOfJobs >= params.n_jobs):
            print("Max jobs >= n_jobs. Waiting for a job space to free up...")
            time.sleep(60) # Keep polling for number of jobs, every 30 seconds, until a job space frees up
            p = Popen("expr $(qstat -u $USER | wc -l) - 2", stdout=PIPE, stderr=STDOUT, bufsize=1, shell=True)
            with p.stdout:
                numberOfJobs = int(p.stdout.readline(), )
            pass

        p = Popen("qsub -cwd -o " + outputdir + "train_output.txt -e " + outputdir + "train_error.txt train_qsub_helper.sh " + params.singularitycommand + " python train.py" +
                  " --outputdir " + singularityoutputdir +
                  " --infersentpath " + params.infersentpath +
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
                  # The --gpu_id argument is set in train_qsub_helper.sh automatically

        with p.stdout, open(outputdir + "info.txt", 'ab') as file:
            for line in iter(p.stdout.readline, b''):
                print line,  # Comma to prevent duplicate newlines
                file.write(line)
        p.wait()

        # TODO: Parse output and append it to a CSV?
    else:
        print("ERROR: Unknown mode. Set --mode argument correctly.")

