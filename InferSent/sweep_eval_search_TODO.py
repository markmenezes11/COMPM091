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
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID. GPU is required because of SentEval")
parser.add_argument("--outputdir", type=str, default='/mnt/mmenezes/InferSent-models/sweep', help="Root directory to run the eval sweep recursively on each subfolder")
params, _ = parser.parse_known_args()

"""
NOTE: Wordvecpath is important here. It should be the same one used for training
"""

# Path to word vectors txt file (e.g. "[path]/glove.840B.300d.txt"). Default: "glove.840B.300d.txt"
wordvecpath  = ["/mnt/mmenezes/libs/InferSent/dataset/GloVe/glove.840B.300d.txt"]

"""
Sweep
"""
# TODO: Output dir and singularityoutputdir if needed
for each outputdir with specific _wordvecpath: # TODO: From outputdir, iterate through each folder, MAKING SURE WORD2VECPATH IS THE SAME, AND ONLY COVERING WORD2VECPATHS GIVEN ABOVE
    print("\n\n\nPreparing output directory...\n")

    # Get the output directory based on current params in this iteration
    slash = "" if params.outputdir[-1] == "/" else "/"

    p = Popen("python /home/mmenezes/Dev/COMPM091/InferSent/eval.py" +
              " --inputdir " + outputdir +
              " --infersentpath " + params.infersentpath +
              " --sentevalpath " + params.sentevalpath +
              " --gpu_id " + str(params.gpu_id) +
              " --wordvecpath " + _wordvecpath, stdout=PIPE, stderr=STDOUT, bufsize=1, shell=True)

    with p.stdout, open(outputdir + "eval_output.txt", 'ab') as file:
        for line in iter(p.stdout.readline, b''):
            print line,  # Comma to prevent duplicate newlines
            file.write(line)
    p.wait()
