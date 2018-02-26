import sys
import os
import re
import argparse

parser = argparse.ArgumentParser(description='Results parser for SentEval-evals')
parser.add_argument("rootdir", type=str, help="Root directory where depth-first search will start")
params, _ = parser.parse_known_args()

transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                  'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SNLI',
                  'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval']

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
    match = re.compile("u'pearson':\s*([0-9.]+)").findall(results)
    results = [float(match[0])]
    assert len(results) == 1
    return results

def parse_results4(results):
    match = re.compile("\[\(([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*[0-9.]+\),\s*\(([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)").findall(results) # TODO
    results = [float(match[0][0]), float(match[0][1]), float(match[0][2]), # Cap (K=1), Cap (K=5), Cap (K=10)
               float(match[0][3]), float(match[0][4]), float(match[0][5])] # Img (K=1), Img (K=5), Img (K=10)
    assert len(results) == 6
    return results

def parse_results(dir, parsers):
    printed_header = False
    for transfer_task in transfer_tasks:
        outputslash = "" if dir[-1] == "/" else "/"
        path = dir + outputslash + "se_results" + "_" + transfer_task + ".txt"
        if os.path.exists(path):
            if not printed_header:
                print("\nResults for: " + dir)
                printed_header = True
            with open(path, "r") as outputfile:
                lines = outputfile.readlines()
                results = "".join(lines).replace('\n', ' ').replace('\r', '')
                parsed_results = parsers[transfer_task](results)
                print(transfer_task + ": " + str(parsed_results))
                # TODO: Do something with the parsed results (e.g. LaTeX table, LaTeX graph)

# Results are presented differently depending on the transfer task, so different parsers are needed
parsers = {"STS12": parse_results1,
           "STS13": parse_results1,
           "STS14": parse_results1,
           "STS15": parse_results1,
           "STS16": parse_results1,
           "MR": parse_results2,
           "CR": parse_results2,
           "MPQA": parse_results2,
           "SUBJ": parse_results2,
           "SST2": parse_results2,
           "SST5": parse_results2,
           "TREC": parse_results2,
           "MRPC": parse_results2,
           "SNLI": parse_results2,
           "SICKEntailment": parse_results2,
           "SICKRelatedness": parse_results3,
           "STSBenchmark": parse_results3,
           "ImageCaptionRetrieval": parse_results4}

for subdir, dirs, files in os.walk(params.rootdir):
    parse_results(subdir, parsers)