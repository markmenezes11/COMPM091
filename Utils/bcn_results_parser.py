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
                accuracy = eval(remove_newlines(lines[0]))

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
    for embeddings_type, results in embeddings_types.iteritems():
        print("\n   EMBEDDINGS TYPE: " + embeddings_type)
        result_with_best_test_accuracy = Result(None, {'dev': -1.0, 'test': -1.0})
        result_with_best_dev_accuracy = Result(None, {'dev': -1.0, 'test': -1.0})
        for result in results:
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