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

import sys
import os
import re
import argparse

parser = argparse.ArgumentParser(description='Results parser for SentEval-evals')
parser.add_argument("rootdir", type=str, help="Root directory where depth-first search will start")
params, _ = parser.parse_known_args()

transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark',
                  'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SNLI',
                  'SICKEntailment', 'SICKRelatedness', 'ImageCaptionRetrieval']

def parse_results1(results):
    match = re.compile("u'all':\s*.*?}}").findall(results)
    results = [eval("{" + match[0] + "}")["all"]["pearson"]["wmean"]]
    assert len(results) == 1
    return results

def parse_results2(results):
    results = [eval(results)["acc"]]
    assert len(results) == 1
    return results

def parse_results3(results):
    match = re.compile("u'pearson':\s*([\-0-9.]+)").findall(results)
    results = [float(match[0])]
    assert len(results) == 1
    return results

def parse_results4(results):
    match = re.compile("\[\(([\-0-9.]+),\s*([\-0-9.]+),\s*([\-0-9.]+),\s*[\-0-9.]+\),\s*\(([\-0-9.]+),\s*([\-0-9.]+),\s*([\-0-9.]+)").findall(results) # TODO
    results = [float(match[0][0]), float(match[0][1]), float(match[0][2]), # Cap (K=1), Cap (K=5), Cap (K=10)
               float(match[0][3]), float(match[0][4]), float(match[0][5])] # Img (K=1), Img (K=5), Img (K=10)
    assert len(results) == 6
    return results

def find_and_parse_results(root):
    # Results are presented differently depending on the transfer task, so different parsers are needed
    parsers = {"STS12": parse_results1, "STS13": parse_results1, "STS14": parse_results1, "STS15": parse_results1, "STS16": parse_results1,
               "MR": parse_results2, "CR": parse_results2, "MPQA": parse_results2, "SUBJ": parse_results2,
               "SST2": parse_results2, "SST5": parse_results2, "TREC": parse_results2, "MRPC": parse_results2,
               "SNLI": parse_results2, "SICKEntailment": parse_results2,
               "SICKRelatedness": parse_results3, "STSBenchmark": parse_results3,
               "ImageCaptionRetrieval": parse_results4}
    results = {}
    def parse_results(dir):
        for transfer_task in transfer_tasks:
            path = os.path.join(dir, "se_results" + "_" + transfer_task + ".txt")
            if os.path.exists(path):
                with open(path, "r") as outputfile:
                    lines = outputfile.readlines()
                    parsed_results = parsers[transfer_task]("".join(lines).replace('\n', ' ').replace('\r', ''))
                if dir not in results:
                    results[dir] = {}
                results[dir][transfer_task] = parsed_results
    for subdir, dirs, files in os.walk(root):
        parse_results(subdir)
    return results

results = find_and_parse_results(params.rootdir)

print("\n\n##############################################################")
print("########################## SUMMARY: ##########################")
print("##############################################################")

for transfer_task in transfer_tasks:
    if transfer_task != "ImageCaptionRetrieval" and transfer_task != "SNLI":
        print("\n" + transfer_task + ":")
        results_for_transfer_task = {}
        for dir in sorted(results):
            if transfer_task in results[dir]:
                results_for_transfer_task[dir] = results[dir][transfer_task][0]
        for key, val in sorted(results_for_transfer_task.items(), key=lambda x: x[1])[::-1]:
            print("   " + str(val) + "  " + key)

print("\n\n##############################################################")
print("########################### LATEX: ###########################")
print("##############################################################")

def make_table(lower, sts_and_sst):
    print("\nLaTeX Table code for lower=" + str(lower) + ", sts=" + str(sts_and_sst) + ":")
    table_rows = []
    included_transfer_tasks = [x for x in transfer_tasks if x != "ImageCaptionRetrieval" and x != "SNLI"
                               and ((sts_and_sst and ("STS" in x or "SST" in x)) or (not sts_and_sst and ("STS" not in x and "SST" not in x)))]
    table_rows.append(["Representation"] + [x.replace("SICKEntailment", "SICKE").replace("SICKRelatedness", "SICKR")
                                            for x in included_transfer_tasks])
    for dir in sorted(results):
        if ((lower and "_lower" in dir) or (not lower and "_lower" not in dir)):
            table_row = []
            table_row.append(dir.replace("./", "").replace("_lower", "").replace("_full", "\\textsubscript{full}").replace("_max", "\\textsubscript{max}").replace("_mean", "\\textsubscript{mean}"))
            for transfer_task in included_transfer_tasks:
                if transfer_task in results[dir]:
                    table_row.append(str(round(results[dir][transfer_task][0], 2)))
                else:
                    table_row.append("-")
            table_rows.append(table_row)
    print("\\hline " + " & ".join(table_rows[0]) + " \\\\")
    print("\\hline")
    for table_row in table_rows[1:]:
        print("\\hline " + " & ".join(table_row) + " \\\\")
    print("\\hline")

make_table(lower=False, sts_and_sst=True)
make_table(lower=False, sts_and_sst=False)
make_table(lower=True, sts_and_sst=True)
make_table(lower=True, sts_and_sst=False)

print("\n\n##############################################################")
print("######################## RAW RESULTS: ########################")
print("##############################################################")

for dir, transfer_tasks_ in results.iteritems():
    print("\nResults for: " + dir)
    for transfer_task in transfer_tasks:
        if transfer_task in transfer_tasks_:
            print("   " + transfer_task + ":" + str(transfer_tasks_[transfer_task]))