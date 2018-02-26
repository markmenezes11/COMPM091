import os
import re
import argparse

parser = argparse.ArgumentParser(description='Results parser for CoVe-BCN')
parser.add_argument("rootdir", type=str, help="Root directory where depth-first search will start")
params, _ = parser.parse_known_args()

def parse_results(dir):
    outputslash = "" if dir[-1] == "/" else "/"
    infopath = dir + outputslash + "info.txt"
    accuracypath = dir + outputslash + "accurac.txt"
    if os.path.exists(infopath) and os.path.exists(accuracypath):
        print("\nResults for: " + dir)
        with open(infopath, "r") as outputfile:
            lines = outputfile.readlines()
            print("Sentence embeddings: " + lines[0])
            print("Transfer task: " + lines[1])
            #print("Hyperparameters: " + lines[2])
        with open(accuracypath, "r") as outputfile:
            lines = outputfile.readlines()
            print("Accuracy: " + lines[0])

for subdir, dirs, files in os.walk(params.rootdir):
    parse_results(subdir)