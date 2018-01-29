import sys, os, time
import argparse
import itertools

from subprocess import Popen, PIPE, STDOUT

"""
Arguments
"""

parser = argparse.ArgumentParser(description='InferSent Parameter Sweep Using Grid Search and SentEval')
parser.add_argument("--mode", type=int, default=0, help="0 to run the train+eval sweep on local machine. 1 to run train+eval sweep using qsub for job submissions on HPC cluster")
parser.add_argument("--n_jobs", type=int, default=10, help="Maximum number of qsub jobs to be running simultaneously")
parser.add_argument("--infersentpath", type=str, default="/mnt/mmenezes/libs/InferSent", help="Path to InferSent repository. If you are using Singularity, all paths must be the ones that Singularity can see (i.e. make sure to use relevant bindings)")
parser.add_argument("--sentevalpath", type=str, default="/mnt/mmenezes/libs/SentEval", help="Path to SentEval repository. If you are using Singularity, all paths must be the ones that Singularity can see (i.e. make sure to use relevant bindings)")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use when running InferSent/SentEval code. This parameter will be ignored if using qsub, as the GPU will be chosen automatically")
parser.add_argument("--outputdir", type=str, default='/cluster/project2/ishi_storage_1/mmenezes/InferSent-models/sweep', help="Output directory (where models and output will be saved). MAKE SURE IT MAPS TO THE SAME PLACE AS SINGULARITYOUTPUTDIR")
parser.add_argument("--singularitycommand", type=str, default='singularity exec --nv --bind /cluster/project2/ishi_storage_1:/mnt /cluster/project2/ishi_storage_1/mmenezes/markmenezes11-COMPM091-master.simg', help="If you are not using Singularity, make this argument blank. If you are using Singularity, it should be something along the lines of 'singularity exec --nv <extra-params> <path-to-simg-file>', e.g. 'singularity exec --nv --bind /cluster/project2/ishi_storage_1:/mnt /cluster/project2/ishi_storage_1/mmenezes/markmenezes11-COMPM091-master.simg'")
parser.add_argument("--singularityoutputdir", type=str, default='/mnt/mmenezes/InferSent-models/sweep', help="Output directory (where models and output will be saved), that Singularity can see (in case you use binding). If you are not using Singularity, make this the same as outputdir. MAKE SURE IT MAPS TO THE SAME PLACE AS OUTPUTDIR")
params, _ = parser.parse_known_args()

"""
Parameters to sweep. If you are using Singularity, all paths must be the ones that Singularity can see (i.e. make sure to use relevant bindings)
"""

# NLI data path (e.g. "[path]/AllNLI", "[path]/SNLI" or "[path]/MultiNLI") - should have 3 classes
# (entailment/neutral/contradiction). Default: "AllNLI"
nlipath      = ["/mnt/mmenezes/InferSent-datasets/AllNLI"]

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
optimizer    = ["sgd,lr=0.1", "adam"]

# Shrink factor for SGD (float). Default: 5
lrshrink     = [5]

# LR decay (float). Default: 0.99
decay        = [0.9, 0.99]

# Minimum LR (float). Default: 1e-5
minlr        = [1e-5]

# Max norm (grad clipping) (float). Default: 5
max_norm     = [3, 5, 7]

# "BLSTMEncoder", "BLSTMprojEncoder", "BGRUlastEncoder", "InnerAttentionMILAEncoder", "InnerAttentionYANGEncoder",
# "InnerAttentionNAACLEncoder", "ConvNetEncoder" or "LSTMEncoder". Default: "BLSTMEncoder"
encoder_type = ["BLSTMEncoder"]

# Encoder NHID dimension (int). Default: 2048
enc_lstm_dim = [2048]

# Encoder num layers (int). Default: 1
n_enc_layers = [1, 2]

# NHID of FC layers (int). Default: 512
fc_dim       = [512]

# "max" or "mean". Default: "max"
pool_type    = ["max", "mean"]

# Random seed (int). Default: 1234
seed         = [1234]

"""
Transfer tasks to be used for evaluation
"""

# Possible transfer tasks:
#                ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC',
#                 'MRPC', 'SNLI', 'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', ImageCaptionRetrieval']
transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC',
                  'MRPC', 'SICKEntailment', 'SICKRelatedness', 'STSBenchmark']

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
    print("\nPARAMETERS: " + iterationParams + "\n")
    print("\nOUTPUT DIRECTORY: " + outputdir + "\n")
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
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

iterationsToCount = get_iterations()
totalIterations = 0
for iteration in iterationsToCount:
    totalIterations += 1

if params.mode == 0: # Full sweep (train + eval) on local machine
    print("\n\n\n\n########## TRAIN ##########\n\n")
    iterations = get_iterations()
    iterationNumber = 0
    for iteration in iterations:
        iterationNumber += 1
        print("\n\n\n####### Iteration " + str(iterationNumber) + " of " + str(totalIterations) + "...")
        outputdir, singularityoutputdir, iterationParams = get_parameter_strings(iteration)

        # If the model/encoder already exists, this iteration does not need to be rerun
        if os.path.exists(outputdir + "model.pickle") and os.path.exists(outputdir + "model.pickle.encoder"):
            print("\nModel/encoder already exists with these parameters. Skipping this iteration...")
            continue

        prepare_directory(outputdir, iterationParams)

        run_subprocess("python train.py" +
                       " --gpu_id " + str(params.gpu_id) +
                       " --outputdir " + outputdir +
                       " --infersentpath " + params.infersentpath +
                       " --outputmodelname " + "model.pickle" +
                       iterationParams,
                       outputdir + "train_output.txt")

    print("\n\n\n\n########## EVAL ##########\n\n")
    iterations = get_iterations()
    iterationNumber = 0
    for iteration in iterations:
        iterationNumber += 1
        print("\n\n\n####### Iteration " + str(iterationNumber) + " of " + str(totalIterations) + "...")
        outputdir, singularityoutputdir, iterationParams = get_parameter_strings(iteration)

        # If the model/encoder does not exist, this iteration cannot be rerun
        if not os.path.exists(outputdir + "model.pickle") or not os.path.exists(outputdir + "model.pickle.encoder"):
            print("\nModel/encoder does not exist with these parameters. Skipping this iteration...")
            continue

        prepare_directory(outputdir, iterationParams)

        for transfer_task in transfer_tasks:
            # If the eval results already exists, this iteration does not need to be rerun
            if os.path.exists(outputdir + "se_results_" + transfer_task + ".txt"):
                print("\nEval results already exists with these parameters. Skipping this iteration...")
                continue

            run_subprocess("python ../SentEval-evals/InferSent/eval.py" +
                           " --inputdir " + outputdir +
                           " --outputdir " + outputdir +
                           " --sentevalpath " + params.sentevalpath +
                           " --gpu_id " + str(params.gpu_id) +
                           " --wordvecpath " + iteration[1] +
                           " --inputmodelname " + "model.pickle.encoder" +
                           " --transfertask " + transfer_task,
                           outputdir + "se_output_" + transfer_task + ".txt")

elif params.mode == 1: # Train sweep (train ONLY) using qsub for job submissions on HPC cluster
    print("\n\n\n\n########## TRAIN ##########\n\n")
    iterations = get_iterations()
    iterationNumber = 0
    for iteration in iterations:
        iterationNumber += 1
        print("\n\n\n####### Iteration " + str(iterationNumber) + " of " + str(totalIterations) + "...")
        outputdir, singularityoutputdir, iterationParams = get_parameter_strings(iteration)

        # If the model/encoder already exists, this iteration does not need to be rerun
        if os.path.exists(outputdir + "model.pickle") and os.path.exists(outputdir + "model.pickle.encoder"):
            print("\nModel/encoder already exists with these parameters. Skipping this iteration...")
            continue

        prepare_directory(outputdir, iterationParams)

        wait_for_jobs(params.n_jobs, True)
        run_subprocess("qsub -cwd -o " + outputdir + "train_output.txt" +
                       " -e " + outputdir + "train_error.txt" +
                       " train_qsub_helper.sh " +
                       params.singularitycommand + " python train.py" +
                       " --outputdir " + singularityoutputdir +
                       " --infersentpath " + params.infersentpath +
                       " --outputmodelname " + "model.pickle" +
                       iterationParams,
                       outputdir + "info.txt")
        # The --gpu_id argument is set in the qsub script automatically

    print("\n\n\n####### All train jobs submitted. Will now wait for them to complete, before retrying any failed jobs...")

    def retry_failed_train_jobs(current_retry):
        iterations = get_iterations()
        retried = 0
        for iteration in iterations:
            outputdir, singularityoutputdir, iterationParams = get_parameter_strings(iteration)

            if not os.path.exists(outputdir):
                print("\n\n\nERROR: Could not retry. Output directory does not exist: " + outputdir)

            if not os.path.exists(outputdir + "model.pickle") or not os.path.exists(outputdir + "model.pickle.encoder"):
                print("\n\n\nPARAMETERS: " + iterationParams + "\n")
                print("\nOUTPUT DIRECTORY: " + outputdir + "\n")
                retried += 1
                wait_for_jobs(params.n_jobs, True)
                run_subprocess("qsub -cwd -o " + outputdir + "train_output" + str(current_retry) + ".txt" +
                               " -e " + outputdir + "train_error" + str(current_retry) + ".txt" +
                               " train_qsub_helper.sh " +
                               params.singularitycommand + " python train.py" +
                               " --outputdir " + singularityoutputdir +
                               " --infersentpath " + params.infersentpath +
                               " --outputmodelname " + "model.pickle" +
                               iterationParams,
                               outputdir + "info.txt")
                # The --gpu_id argument is set in the qsub script automatically
        return retried

    retried = 1
    max_retries = 10
    current_retry = 0
    while (retried > 0):
        current_retry += 1
        if current_retry > max_retries:
            break
        wait_for_jobs(1, False)
        print("\n\n\n####### Retry " + str(current_retry) + " of " + str(max_retries) + "...")
        retried = retry_failed_train_jobs(current_retry)
    print("Nothing left to retry.")

    print("\n\n\n\n########## EVAL ##########\n\n")
    iterations = get_iterations()
    iterationNumber = 0
    for iteration in iterations:
        iterationNumber += 1
        print("\n\n\n####### Iteration " + str(iterationNumber) + " of " + str(totalIterations) + "...")
        outputdir, singularityoutputdir, iterationParams = get_parameter_strings(iteration)

        # If the model/encoder does not exist, this iteration cannot be rerun
        if not os.path.exists(outputdir + "model.pickle") or not os.path.exists(outputdir + "model.pickle.encoder"):
            print("\nModel/encoder does not exist with these parameters. Skipping this iteration...")
            continue

        prepare_directory(outputdir, iterationParams)

        for transfer_task in transfer_tasks:
            # If the eval results already exists, this iteration does not need to be rerun
            if os.path.exists(outputdir + "se_results_" + transfer_task + ".txt"):
                print("\nEval results already exists with these parameters. Skipping this iteration...")
                continue

            wait_for_jobs(params.n_jobs, True)
            if transfer_task == "SNLI":
                qsub_script = "snli_eval_qsub_helper.sh"
            elif transfer_task == "ImageCaptionRetrieval":
                qsub_script = "icr_eval_qsub_helper.sh"
            else:
                qsub_script = "eval_qsub_helper.sh"
            run_subprocess("qsub -cwd -o " + outputdir + "se_output_" + transfer_task + ".txt" +
                           " -e " + outputdir + "se_error_" + transfer_task + ".txt" +
                           " " + qsub_script + " " +
                           params.singularitycommand + " python ../SentEval-evals/InferSent/eval.py" +
                           " --inputdir " + singularityoutputdir +
                           " --outputdir " + singularityoutputdir +
                           " --sentevalpath " + params.sentevalpath +
                           " --wordvecpath " + iteration[1] +
                           " --inputmodelname " + "model.pickle.encoder" +
                           " --transfertask " + transfer_task,
                           outputdir + "info.txt")
            # The --gpu_id argument is set in the qsub script automatically

    print("\n\n\n####### All eval jobs submitted. Will now wait for them to complete, before retrying any failed jobs...")

    def retry_failed_eval_jobs(current_retry):
        iterations = get_iterations()
        retried = 0
        for iteration in iterations:
            outputdir, singularityoutputdir, iterationParams = get_parameter_strings(iteration)

            if not os.path.exists(outputdir):
                print("\n\n\nERROR: Could not retry. Output directory does not exist: " + outputdir)

            # If the model/encoder does not exist, this iteration cannot be rerun
            if not os.path.exists(outputdir + "model.pickle") or not os.path.exists(outputdir + "model.pickle.encoder"):
                print("\n\n\nERROR: Could not retry. Model/encoder does not exist with these parameters: " + outputdir)
                continue

            for transfer_task in transfer_tasks:
                # If the eval results already exists, this iteration does not need to be rerun
                if not os.path.exists(outputdir + "se_results_" + transfer_task + ".txt"):
                    print("\n\n\nPARAMETERS: " + iterationParams + "\n")
                    print("\nOUTPUT DIRECTORY: " + outputdir + "\n")
                    print("\nTRANSFER TASK: " + transfer_task + "\n")
                    retried += 1
                    wait_for_jobs(params.n_jobs, True)
                    if transfer_task == "SNLI":
                        qsub_script = "snli_eval_qsub_helper.sh"
                    elif transfer_task == "ImageCaptionRetrieval":
                        qsub_script = "icr_eval_qsub_helper.sh"
                    else:
                        qsub_script = "eval_qsub_helper.sh"
                    run_subprocess("qsub -cwd -o " + outputdir + "se_output_" + transfer_task + str(current_retry) + ".txt" +
                                   " -e " + outputdir + "se_error_" + transfer_task + str(current_retry) + ".txt" +
                                   " " + qsub_script + " " +
                                   params.singularitycommand + " python ../SentEval-evals/InferSent/eval.py" +
                                   " --inputdir " + singularityoutputdir +
                                   " --outputdir " + singularityoutputdir +
                                   " --sentevalpath " + params.sentevalpath +
                                   " --wordvecpath " + iteration[1] +
                                   " --inputmodelname " + "model.pickle.encoder" +
                                   " --transfertask " + transfer_task,
                                   outputdir + "info.txt")
                    # The --gpu_id argument is set in the qsub script automatically
        return retried

    retried = 1
    max_retries = 10
    current_retry = 0
    while (retried > 0):
        current_retry += 1
        if current_retry > max_retries:
            break
        wait_for_jobs(1, False)
        print("\n\n\n####### Retry " + str(current_retry) + " of " + str(max_retries) + "...")
        retried = retry_failed_eval_jobs(current_retry)
    print("Nothing left to retry.")

else:
    print("ERROR: Unknown mode. Set --mode argument correctly.")
