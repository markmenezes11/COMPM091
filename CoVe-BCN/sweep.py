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

import os, time
import argparse
import itertools

from subprocess import Popen, PIPE, STDOUT

"""
Arguments
"""

parser = argparse.ArgumentParser(description='Parameter sweep script for the CoVe Biattentive Classification Network (BCN). Jobs are submitted on HPC using qsub.')

parser.add_argument("--n_jobs", type=int, default=20, help="Maximum number of qsub jobs to be running simultaneously")
parser.add_argument("--n_retries", type=int, default=7, help="Maximum number of retries for failed qsub jobs before giving up")

parser.add_argument("--glovepath", type=str, default="/mnt/mmenezes/libs/InferSent/dataset/GloVe/glove.840B.300d.txt", help="Path to GloVe word embeddings. Download glove.840B.300d embeddings from https://nlp.stanford.edu/projects/glove/")
parser.add_argument("--ignoregloveheader", action="store_true", default=False, help="Set this flag if the first line of the GloVe file is a header and not a (word, embedding) pair")
parser.add_argument("--covepath", type=str, default='../CoVe-ported/Keras_CoVe_Python2.h5', help="Path to the CoVe model")
parser.add_argument("--covedim", type=int, default=600, help="Number of dimensions in CoVe embeddings (default: 600)")
parser.add_argument("--infersentpath", type=str, default="/mnt/mmenezes/libs/InferSent/encoder/infersent.allnli.pickle", help="Path to InferSent repository. If you are using Singularity, all paths must be the ones that Singularity can see (i.e. make sure to use relevant bindings)")
parser.add_argument("--infersentdim", type=int, default=4096, help="Number of dimensions in InferSent embeddings (default: 4096)")
parser.add_argument("--datadir", type=str, default='datasets', help="Path to the directory that contains the datasets")
parser.add_argument("--outputdir", type=str, default='/cluster/project2/ishi_storage_1/mmenezes/BCN-models/sweep', help="Output directory (where models and output will be saved). MAKE SURE IT MAPS TO THE SAME PLACE AS SINGULARITYOUTPUTDIR")

parser.add_argument("--singularitycommand", type=str, default='singularity exec --nv --bind /cluster/project2/ishi_storage_1:/mnt /cluster/project2/ishi_storage_1/mmenezes/markmenezes11-COMPM091-master.simg', help="If you are not using Singularity, make this argument blank. If you are using Singularity, it should be something along the lines of 'singularity exec --nv <extra-params> <path-to-simg-file>', e.g. 'singularity exec --nv --bind /cluster/project2/ishi_storage_1:/mnt /cluster/project2/ishi_storage_1/mmenezes/markmenezes11-COMPM091-master.simg'")
parser.add_argument("--singularityoutputdir", type=str, default='/mnt/mmenezes/BCN-models/sweep', help="Output directory (where models and output will be saved), that Singularity can see (in case you use binding). If you are not using Singularity, make this the same as outputdir. MAKE SURE IT MAPS TO THE SAME PLACE AS OUTPUTDIR")

params, _ = parser.parse_known_args()

"""
Parameters to sweep. If you are using Singularity, all paths must be the ones that Singularity can see (i.e. make sure to use relevant bindings)
"""

# Number of epochs (int). After 5 epochs of worse dev accuracy, training will early stopped and the best epoch will be saved (based on dev accuracy).
n_epochs = [20]

# Batch size (int)
batch_size = [32, 64, 128] # TODO: Tune this if needed

# Whether or not to use the same BiLSTM (when flag is set) or separate BiLSTMs (flag unset) for the encoder
same_bilstm_for_encoder = [True, False] # TODO: Tune this as True or False as it is unclear in CoVe paper ##############

# Number of hidden states in encoder's BiLSTM(s) (int)
bilstm_encoder_n_hidden = [300]

# Forget bias for encoder's BiLSTM(s) (float)
bilstm_encoder_forget_bias = [1.0]

# Number of hidden states in integrate's BiLSTMs (int)
bilstm_integrate_n_hidden = [300]

# Forget bias for integrate's BiLSTMs (float)
bilstm_integrate_forget_bias = [1.0]

# Ratio for dropout applied before Feedforward Network and before each Batch Norm (float)
dropout_ratio = [0.1, 0.2, 0.3] # TODO: Tune this as 0.1, 0.2 or 0.3  as done in CoVe paper ############################

# On the first and second maxout layers, the dimensionality is divided by this number (int)
maxout_reduction = [2, 4, 8] # TODO: Tune this as 2, 4 or 8 as done in CoVe paper ######################################

# Decay for each batch normalisation layer (float)
bn_decay = [0.999, 0.99, 0.9] # TODO: Tune this if needed

# Epsilon for each batch normalisation layer (float)
bn_epsilon = [1e-3, 1e-5]

# Optimizer (adam or gradientdescent)
optimizer = ["adam"]

# Leaning rate (float)
learning_rate = [0.001]

# Beta1 for adam optimiser if adam optimiser is used (float)
adam_beta1 = [0.9]

# Beta2 for adam optimiser if adam optimiser is used (float)
adam_beta2 = [0.999]

# Epsilon for adam optimiser if adam optimiser is used (float)
adam_epsilon = [1e-8]

"""
Model types (str: GloVe, InferSent, CoVe, CoVe_without_GloVe)
"""

types = ["CoVe", "InferSent", "GloVe", "CoVe_without_GloVe", "GloVe+InferSent"]

"""
Transfer tasks to be used for training BCN and evaluating predictions (str: "SSTBinary", "SSTFine", "SSTBinary_lower", "SSTFine_lower", "TREC6", "TREC50", "TREC6_lower", "TREC50_lower")
"""

transfer_tasks = ["SSTBinary", "SSTFine", "SSTBinary_lower", "SSTFine_lower", "TREC6", "TREC50", "TREC6_lower", "TREC50_lower"]

"""
Sweep helper functions
"""

# Get every combination of the above parameters to sweep
def get_iterations():
    return itertools.product(n_epochs, batch_size, same_bilstm_for_encoder, bilstm_encoder_n_hidden,
                             bilstm_encoder_forget_bias, bilstm_integrate_n_hidden, bilstm_integrate_forget_bias,
                             dropout_ratio, maxout_reduction, bn_decay, bn_epsilon, optimizer, learning_rate,
                             adam_beta1, adam_beta2, adam_epsilon)

# Remove any characters that should not be used in folder names / paths
def replace_illegal_chars(string):
    return string.replace('/', '-').replace(':', '-').replace(',', '-').replace(' ', '-')

# Get output directories and parameters for current iteration
def get_parameter_strings(iteration_, type_, transfer_task_):
    slash = "" if params.outputdir[-1] == "/" else "/"
    singularityslash = "" if params.singularityoutputdir[-1] == "/" else "/"
    sweepdir = ("n_epochs__" + replace_illegal_chars(str(iteration_[0])) + "/" +
                "batch_size__" + replace_illegal_chars(str(iteration_[1])) + "/" +
                "same_bilstm_for_encoder__" + replace_illegal_chars(str(iteration_[2])) + "/" +
                "bilstm_encoder_n_hidden__" + replace_illegal_chars(str(iteration_[3])) + "/" +
                "bilstm_encoder_forget_bias__" + replace_illegal_chars(str(iteration_[4])) + "/" +
                "bilstm_integrate_n_hidden__" + replace_illegal_chars(str(iteration_[5])) + "/" +
                "bilstm_integrate_forget_bias__" + replace_illegal_chars(str(iteration_[6])) + "/" +
                "dropout_ratio__" + replace_illegal_chars(str(iteration_[7])) + "/" +
                "maxout_reduction__" + replace_illegal_chars(str(iteration_[8])) + "/" +
                "bn_decay__" + replace_illegal_chars(str(iteration_[9])) + "/" +
                "bn_epsilon__" + replace_illegal_chars(str(iteration_[10])) + "/" +
                "optimizer__" + replace_illegal_chars(str(iteration_[11])) + "/" +
                "learning_rate__" + replace_illegal_chars(str(iteration_[12])) + "/" +
                "adam_beta1__" + replace_illegal_chars(str(iteration_[13])) + "/" +
                "adam_beta2__" + replace_illegal_chars(str(iteration_[14])) + "/" +
                "adam_epsilon__" + replace_illegal_chars(str(iteration_[15])) + "/")
    outputdir_ = (params.outputdir + slash + replace_illegal_chars(type_) + "/" + replace_illegal_chars(transfer_task_)
                 + "/" + sweepdir)
    singularityoutputdir_ = (params.singularityoutputdir + singularityslash + replace_illegal_chars(type_)
                            + "/" + replace_illegal_chars(transfer_task_) + "/" + sweepdir)
    iteration_params_ = (" --type " + type_ +
                         " --transfer_task " + transfer_task_ +
                         " --n_epochs " + str(iteration_[0]) +
                         " --batch_size " + str(iteration_[1]) +
                         " --same_bilstm_for_encoder " + str(iteration_[2]) +
                         " --bilstm_encoder_n_hidden " + str(iteration_[3]) +
                         " --bilstm_encoder_forget_bias " + str(iteration_[4]) +
                         " --bilstm_integrate_n_hidden " + str(iteration_[5]) +
                         " --bilstm_integrate_forget_bias " + str(iteration_[6]) +
                         " --dropout_ratio " + str(iteration_[7]) +
                         " --maxout_reduction " + str(iteration_[8]) +
                         " --bn_decay " + str(iteration_[9]) +
                         " --bn_epsilon " + str(iteration_[10]) +
                         " --optimizer " + str(iteration_[11]) +
                         " --learning_rate " + str(iteration_[12]) +
                         " --adam_beta1 " + str(iteration_[13]) +
                         " --adam_beta2 " + str(iteration_[14]) +
                         " --adam_epsilon " + str(iteration_[15]))
    return outputdir_, singularityoutputdir_, iteration_params_

def prepare_directory(outputdir_, iteration_params_, type_, transfer_task_):
    print("\nTYPE: " + type_ + "\n")
    print("\nTRANSFER TASK: " + transfer_task_ + "\n")
    print("\nPARAMETERS: " + iteration_params_ + "\n")
    print("\nOUTPUT DIRECTORY: " + outputdir_ + "\n")
    if not os.path.exists(outputdir_):
        os.makedirs(outputdir_)

# Keep polling qstat until the number of submitted jobs is less than max_jobs
def wait_for_jobs(max_jobs, verbose):
    p = Popen("expr $(qstat -u $USER | wc -l) - 2", stdout=PIPE, stderr=STDOUT, bufsize=1, shell=True)
    with p.stdout:
        numberOfJobs = int(p.stdout.readline(), )
    while numberOfJobs >= max_jobs:
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
    with p.stdout, open(output_file, 'ab') as log_file:
        for line in iter(p.stdout.readline, b''):
            print line,  # Comma to prevent duplicate newlines
            log_file.write(line)
    p.wait()

"""
Sweep
"""

iterationsToCount = get_iterations()
totalIterations = 0
for iteration in iterationsToCount:
    totalIterations += 1
totalIterations *= (len(types) * len(transfer_tasks))
iterationNumber = 0

for transfer_task in transfer_tasks:
    for model_type in types:
        iterations = get_iterations()
        for iteration in iterations:
            iterationNumber += 1
            print("\n\n\n####### Iteration " + str(iterationNumber) + " of " + str(totalIterations) + "...")
            outputdir, singularityoutputdir, iterationParams = get_parameter_strings(iteration, model_type, transfer_task)

            # If the model/encoder already exists, this iteration does not need to be rerun
            if os.path.exists(outputdir + "accuracy.txt"):
                print("\nThis iteration has already been run. Skipping...")
                continue

            prepare_directory(outputdir, iterationParams, model_type, transfer_task)

            wait_for_jobs(params.n_jobs, True)
            qsub_helper = "qsub_helper.sh" # Can also use qsub_helper_long.sh for longer tasks but this is not needed yet
            run_subprocess("qsub -cwd -o " + outputdir + "output.txt" +
                           " -e " + outputdir + "error.txt" +
                           " " + qsub_helper + " " +
                           params.singularitycommand + " python eval.py" +
                           " --glovepath " + params.glovepath +
                           " --ignoregloveheader " + str(params.ignoregloveheader) +
                           " --covepath " + params.covepath +
                           " --covedim " + str(params.covedim) +
                           " --infersentpath " + params.infersentpath +
                           " --infersentdim " + str(params.infersentdim) +
                           " --datadir " + params.datadir +
                           " --outputdir " + singularityoutputdir +
                           iterationParams,
                           outputdir + "log.txt")

print("\n\n\n####### All train jobs submitted. Will now wait for them to complete, before retrying any failed jobs...")

def retry_failed_jobs(current_retry_):
    retried_ = 0
    for transfer_task in transfer_tasks:
        for model_type in types:
            iterations = get_iterations()
            for iteration in iterations:
                outputdir, singularityoutputdir, iterationParams = get_parameter_strings(iteration, model_type, transfer_task)

                if not os.path.exists(outputdir):
                    print("\n\n\nERROR: Could not retry. Output directory does not exist: " + outputdir)

                if not os.path.exists(outputdir + "accuracy.txt"):
                    print("\n\n\nTYPE: " + model_type + "\n")
                    print("\nTRANSFER TASK: " + transfer_task + "\n")
                    print("\nPARAMETERS: " + iterationParams + "\n")
                    print("\nOUTPUT DIRECTORY: " + outputdir + "\n")
                    retried_ += 1
                    wait_for_jobs(params.n_jobs, True)
                    qsub_helper = "qsub_helper.sh" # Can also use qsub_helper_long.sh for longer tasks but this is not needed yet
                    run_subprocess("qsub -cwd -o " + outputdir + "output" + str(current_retry_) + ".txt" +
                                   " -e " + outputdir + "error" + str(current_retry_) + ".txt" +
                                   " " + qsub_helper + " " +
                                   params.singularitycommand + " python eval.py" +
                                   " --glovepath " + params.glovepath +
                                   " --ignoregloveheader " + str(params.ignoregloveheader) +
                                   " --covepath " + params.covepath +
                                   " --covedim " + str(params.covedim) +
                                   " --infersentpath " + params.infersentpath +
                                   " --infersentdim " + str(params.infersentdim) +
                                   " --datadir " + params.datadir +
                                   " --outputdir " + singularityoutputdir +
                                   iterationParams,
                                   outputdir + "log.txt")
    return retried_

retried = 1
max_retries = params.n_retries
current_retry = 0
while retried > 0:
    current_retry += 1
    if current_retry > max_retries:
        break
    wait_for_jobs(1, False)
    print("\n\n\n####### Retry " + str(current_retry) + " of " + str(max_retries) + "...")
    retried = retry_failed_jobs(current_retry)
if retried == 0:
    print("Nothing left to retry.")
else:
    print("ERROR: Reached max number of retries. Giving up.")
