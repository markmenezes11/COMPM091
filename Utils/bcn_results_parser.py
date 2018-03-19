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

import os
import argparse
import math

parser = argparse.ArgumentParser(description='Results parser for CoVe-BCN')
parser.add_argument("rootdir", type=str, help="Root directory where depth-first search will start")
params, _ = parser.parse_known_args()

class Result(object):
    def __init__(self, hyperparameters, accuracy):
        self.hyperparameters = hyperparameters
        self.test_accuracy = accuracy['test']
        self.dev_accuracy = accuracy['dev']

    def get_hyperparameters(self):
        return self.hyperparameters

    def get_test_accuracy(self):
        return self.test_accuracy

    def get_dev_accuracy(self):
        return self.dev_accuracy

def remove_newlines(input):
    return input.replace('\n', '').replace('\r', '')

def find_and_parse_results(root):
    results = {}
    def parse_results(dir):
        infopath = os.path.join(dir, "info.txt")
        accuracypath = os.path.join(dir, "accuracy.txt")
        if os.path.exists(infopath) and os.path.exists(accuracypath):
            with open(infopath, "r") as outputfile:
                lines = outputfile.readlines()
                embeddings_type = remove_newlines(lines[0])
                transfer_task = remove_newlines(lines[1])
                hyperparameters = eval(remove_newlines(lines[2]))
            with open(accuracypath, "r") as outputfile:
                lines = outputfile.readlines()
                accuracy = eval(remove_newlines(lines[0]).replace("nan", "float('nan')"))

            if transfer_task not in results:
                results[transfer_task] = {}
            if embeddings_type not in results[transfer_task]:
                results[transfer_task][embeddings_type] = []
            results[transfer_task][embeddings_type].append(Result(hyperparameters, accuracy))
    for subdir, dirs, files in os.walk(root):
        parse_results(subdir)
    return results

results = find_and_parse_results(params.rootdir)

print("\n\n##############################################################")
print("########################## SUMMARY: ##########################")
print("##############################################################")

for transfer_task, embeddings_types in results.iteritems():
    print("\nTRANSFER TASK: " + transfer_task)
    for embeddings_type, results_ in embeddings_types.iteritems():
        print("\n   EMBEDDINGS TYPE: " + embeddings_type)
        result_with_best_test_accuracy = Result(None, {'dev': -1.0, 'test': -1.0})
        result_with_best_dev_accuracy = Result(None, {'dev': -1.0, 'test': -1.0})
        for result in results_:
            if not math.isnan(result.get_dev_accuracy()) and not math.isnan(result.get_test_accuracy()):
                if result.get_test_accuracy() > result_with_best_test_accuracy.get_test_accuracy():
                    result_with_best_test_accuracy = result
                if result.get_dev_accuracy() > result_with_best_dev_accuracy.get_dev_accuracy():
                    result_with_best_dev_accuracy = result

        print("\n      BEST TEST ACCURACY (IGNORING DEV ACCURACY - ***NOT A GOOD METRIC FOR FINAL RESULTS***):\n"
              + "         Hyperparameters: " + str(result_with_best_test_accuracy.get_hyperparameters()) + "\n"
              + "         Dev Accuracy: " + str(result_with_best_test_accuracy.get_dev_accuracy()) + "\n"
              + "         Test Accuracy: " + str(result_with_best_test_accuracy.get_test_accuracy()))

        print("\n      BEST DEV ACCURACY (AND ITS CORRESPONDING TEST ACCURACY):\n"
              + "         Hyperparameters: " + str(result_with_best_dev_accuracy.get_hyperparameters()) + "\n"
              + "         Dev Accuracy: " + str(result_with_best_dev_accuracy.get_dev_accuracy()) + "\n"
              + "         Test Accuracy: " + str(result_with_best_dev_accuracy.get_test_accuracy()))

print("\n\n##############################################################")
print("########################### LATEX: ###########################")
print("##############################################################")

for transfer_task, embeddings_types in results.iteritems():
    print("\nTRANSFER TASK: " + transfer_task)
    print("\\hline Representation & Best Test Accuracy & Best Dev Accuracy & Actual Test Accuracy \\\\")
    print("\\hline")

    for embeddings_type, results_ in embeddings_types.iteritems():
        table_row = [embeddings_type.replace("CoVe", "GloVe+CoVe\\textsubscript{full}").replace("InferSent", "InferSent\\textsubscript{full}")]

        result_with_best_test_accuracy = Result(None, {'dev': -1.0, 'test': -1.0})
        result_with_best_dev_accuracy = Result(None, {'dev': -1.0, 'test': -1.0})
        for result in results_:
            if not math.isnan(result.get_dev_accuracy()) and not math.isnan(result.get_test_accuracy()):
                if result.get_test_accuracy() > result_with_best_test_accuracy.get_test_accuracy():
                    result_with_best_test_accuracy = result
                if result.get_dev_accuracy() > result_with_best_dev_accuracy.get_dev_accuracy():
                    result_with_best_dev_accuracy = result

        table_row.append(str(result_with_best_test_accuracy.get_test_accuracy()))
        table_row.append(str(result_with_best_dev_accuracy.get_dev_accuracy()))
        table_row.append("\\textbf{" + str(result_with_best_dev_accuracy.get_test_accuracy()) + "}")
        print("\\hline " + " & ".join(table_row) + " \\\\")
    print("\\hline")

print("\n\n##############################################################")
print("######################## RAW RESULTS: ########################")
print("##############################################################")
for transfer_task, embeddings_types in results.iteritems():
    print("\nTRANSFER TASK: " + transfer_task)
    for embeddings_type, results in embeddings_types.iteritems():
        print("\n   EMBEDDINGS TYPE: " + embeddings_type)
        for result in results:
            if not math.isnan(result.get_dev_accuracy()) and not math.isnan(result.get_test_accuracy()):
                print("\n      Hyperparameters: " + str(result.get_hyperparameters()))
                print("      Dev Accuracy: " + str(result.get_dev_accuracy()))
                print("      Test Accuracy: " + str(result.get_test_accuracy()))
