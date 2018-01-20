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

"""
Parameters to sweep. If you are using Singularity, all paths must be the ones that Singularity can see (i.e. make sure to use relevant bindings)
"""

# NLI data path (e.g. "[path]/AllNLI", "[path]/SNLI" or "[path]/MultiNLI") - should have 3 classes
# (entailment/neutral/contradiction). Default: "AllNLI"
nlipath      = ["/mnt/mmenezes/InferSent-datasets/SmallNLI"] # TODO: Change this from SmallNLI after testing it

# Path to word vectors txt file (e.g. "[path]/glove.840B.300d.txt"). Default: "glove.840B.300d.txt"
wordvecpath  = ["/mnt/mmenezes/libs/InferSent/dataset/GloVe/glove.840B.300d.txt"]

# Number of epochs (int). Default: 20
n_epochs     = [20]

# Batch size (int). Default: 64
batch_size   = [32, 64, 128]

# Encoder dropout (float). Default: 0
dpout_model  = [0]

# Classifier dropout (float). Default: 0
dpout_fc     = [0]

# Use nonlinearity in FC (float). Default: 0
nonlinear_fc = [0]

# "adam" or "sgd,lr=0.1". Default: "sgd,lr=0.1"
optimizer    = ["sgd,lr=0.1", "adam"]

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
pool_type    = ["max", "mean"]

# Random seed (int). Default: 1234
seed         = [1234]

"""
Sweep helper functions
"""

# Get every combination of the above parameters to sweep
def get_iterations():
    return itertools.product(nlipath, wordvecpath, n_epochs, batch_size, dpout_model, dpout_fc, nonlinear_fc,
                             optimizer, lrshrink, decay, minlr, max_norm, encoder_type, enc_lstm_dim,
                             n_enc_layers, fc_dim, pool_type, seed)

# Remove any characters that should not be used in folder names / paths
def replace_illegal_chars(str):
    return str.replace('/', '-').replace(':', '-').replace(',', '-').replace(' ', '-')

# Get output directories and parameters for current iteration
def get_parameter_strings(iteration):
    slash = "" if params.outputdir[-1] == "/" else "/"
    singularityslash = "" if params.singularityoutputdir[-1] == "/" else "/"
    sweepdir = ("nlipath__" + replace_illegal_chars(iteration[0]) + "/" +
                "wordvecpath__" + replace_illegal_chars(iteration[1]) + "/" +
                "n_epochs__" + replace_illegal_chars(str(iteration[2])) + "/" +
                "batch_size__" + replace_illegal_chars(str(iteration[3])) + "/" +
                "dpout_model__" + replace_illegal_chars(str(iteration[4])) + "/" +
                "dpout_fc__" + replace_illegal_chars(str(iteration[5])) + "/" +
                "nonlinear_fc__" + replace_illegal_chars(str(iteration[6])) + "/" +
                "optimizer__" + replace_illegal_chars(iteration[7]) + "/" +
                "lrshrink__" + replace_illegal_chars(str(iteration[8])) + "/" +
                "decay__" + replace_illegal_chars(str(iteration[9])) + "/" +
                "minlr__" + replace_illegal_chars(str(iteration[10])) + "/" +
                "max_norm__" + replace_illegal_chars(str(iteration[11])) + "/" +
                "encoder_type__" + replace_illegal_chars(iteration[12]) + "/" +
                "enc_lstm_dim__" + replace_illegal_chars(str(iteration[13])) + "/" +
                "n_enc_layers__" + replace_illegal_chars(str(iteration[14])) + "/" +
                "fc_dim__" + replace_illegal_chars(str(iteration[15])) + "/" +
                "pool_type__" + replace_illegal_chars(iteration[16]) + "/" +
                "seed__" + replace_illegal_chars(str(iteration[17])) + "/")
    outputdir = params.outputdir + slash + sweepdir
    singularityoutputdir = params.singularityoutputdir + singularityslash + sweepdir
    iterationParams = (" --nlipath " + iteration[0] +
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
                       " --seed " + str(iteration[17]))
    return outputdir, singularityoutputdir, iterationParams

def prepare_directory(outputdir, iterationParams):
    os.makedirs(outputdir)
    print("\n\n\nPARAMETERS: " + iterationParams + "...\n")
    with open(outputdir + "info.txt", "a") as outputfile:
        outputfile.write("\n\n\nPARAMETERS: " + iterationParams + "\n")

# Keep polling qstat until the number of submitted jobs is less than max_jobs
def wait_for_jobs(max_jobs, verbose):
    p = Popen("expr $(qstat -u $USER | wc -l) - 2", stdout=PIPE, stderr=STDOUT, bufsize=1, shell=True)
    with p.stdout:
        numberOfJobs = int(p.stdout.readline(), )
    while (numberOfJobs >= max_jobs):
        if verbose:
            print("Too many qsub jobs running. Waiting for a job space to free up...")
        time.sleep(60)
        p = Popen("expr $(qstat -u $USER | wc -l) - 2", stdout=PIPE, stderr=STDOUT, bufsize=1, shell=True)
        with p.stdout:
            numberOfJobs = int(p.stdout.readline(), )
        pass

# Run the given command and write the output to the given file (as well as printing it out)
def run_subprocess(command, output_file):
    p = Popen(command, stdout=PIPE, stderr=STDOUT, bufsize=1, shell=True)
    with p.stdout, open(output_file, 'ab') as file:
        for line in iter(p.stdout.readline, b''):
            print line,  # Comma to prevent duplicate newlines
            file.write(line)
    p.wait()

"""
Sweep
"""

iterations = get_iterations()
iterationsToCount = get_iterations()
iterationNumber = 0
totalIterations = 0
for iteration in iterationsToCount:
    totalIterations += 1

if params.mode == 0: # Full sweep (train + eval) on local machine
    for iteration in iterations:
        iterationNumber += 1
        print("\n\n\n\n\n\n\n####### Iteration " + str(iterationNumber) + " of " + str(totalIterations) + "...")
        outputdir, singularityoutputdir, iterationParams = get_parameter_strings(iteration)

        # If the directory already exists, this iteration has already been run before
        if os.path.exists(outputdir):
            print("Path already exists with these parameters. Skipping this iteration...")
            continue
        prepare_directory(outputdir, iterationParams)

        run_subprocess("python train.py" +
                       " --gpu_id " + str(params.gpu_id) +
                       " --outputdir " + outputdir +
                       " --infersentpath " + params.infersentpath +
                       iterationParams,
                       outputdir + "train_output.txt")
        run_subprocess("python eval.py" +
                       " --inputdir " + outputdir +
                       " --infersentpath " + params.infersentpath +
                       " --sentevalpath " + params.sentevalpath +
                       " --gpu_id " + str(params.gpu_id) +
                       " --wordvecpath " + iteration[1],
                       outputdir + "eval_output.txt")

elif params.mode == 1: # Train sweep (train ONLY) using qsub for job submissions on HPC cluster
    for iteration in iterations:
        iterationNumber += 1
        print("\n\n\n\n\n\n\n####### Iteration " + str(iterationNumber) + " of " + str(totalIterations) + "...")
        outputdir, singularityoutputdir, iterationParams = get_parameter_strings(iteration)

        # If the directory already exists, this iteration has already been run before
        if os.path.exists(outputdir):
            print("Path already exists with these parameters. Skipping this iteration...")
            continue
        prepare_directory(outputdir, iterationParams)

        wait_for_jobs(params.n_jobs, True)
        run_subprocess("qsub -cwd -o " + outputdir + "train_output.txt" +
                       " -e " + outputdir + "train_error.txt" +
                       " train_qsub_helper.sh " +
                       params.singularitycommand + " python train.py" +
                       " --outputdir " + singularityoutputdir +
                       " --infersentpath " + params.infersentpath +
                       iterationParams,
                       outputdir + "info.txt")
                       # The --gpu_id argument is set in train_qsub_helper.sh automatically

    print("All jobs submitted. Will now wait for them to complete, before retrying any failed jobs...")

    def retry_failed_train_jobs(current_retry):
        print("Searching for and retrying failed jobs...")
        iterations = get_iterations()
        retried = 0

        for iteration in iterations:
            print("\n\n\nPreparing output directory...\n")
            outputdir, singularityoutputdir, iterationParams = get_parameter_strings(iteration)

            if not os.path.exists(outputdir):
                print("ERROR: Could not retry. Output directory does not exist: " + outputdir)

            if not os.path.exists(outputdir + "model.pickle"):
                print("\n\n\nPARAMETERS: " + iterationParams + "...\n")
                retried += 1
                wait_for_jobs(params.n_jobs, True)
                run_subprocess("qsub -cwd -o " + outputdir + "train_output.txt" + str(current_retry) +
                               " -e " + outputdir + "train_error.txt" + str(current_retry) +
                               " train_qsub_helper.sh " +
                               params.singularitycommand + " python train.py" +
                               " --outputdir " + singularityoutputdir +
                               " --infersentpath " + params.infersentpath +
                               iterationParams,
                               outputdir + "info.txt")
                # The --gpu_id argument is set in train_qsub_helper.sh automatically
        return retried

    retried = 1
    max_retries = 10
    current_retry = 0
    while (retried > 0):
        current_retry += 1
        print("Retry " + str(current_retry) + " of " + str(max_retries) + "...")
        if current_retry > max_retries:
            break
        wait_for_jobs(1, False)
        retried = retry_failed_train_jobs(current_retry)

elif params.mode == 2: # Eval sweep (eval ONLY)
    print("ERROR: Not implemented.") # TODO: Implement this (see code below) - local machine or qsub?
    """
    NOTE: Wordvecpath is important here. It should be the same one used for training
    """
    # TODO: Output dir and singularityoutputdir if needed
    """for each outputdir with specific wordvecpath:  # TODO: From outputdir, iterate through each folder, MAKING SURE WORD2VECPATH IS THE SAME, AND ONLY COVERING WORD2VECPATHS GIVEN ABOVE
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

else:
    print("ERROR: Unknown mode. Set --mode argument correctly.")