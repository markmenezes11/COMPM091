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

parser = argparse.ArgumentParser(description='Results parser for SentEval-evals. Note: LaTeX code will not be produced for SNLI or ImageCaptionRetrieval results')
parser.add_argument("rootdir", type=str, help="Root directory where depth-first search will start")
params, _ = parser.parse_known_args()

transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness', 'SICKEntailment',
                  'MRPC', 'SST2', 'SST5', 'MR', 'MPQA', 'CR', 'TREC', 'SUBJ', 'SNLI', 'ImageCaptionRetrieval']

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

print("\n############## FULL TABLE ##############")

def make_full_table(lower, include, representations):
    print("\nLaTeX Table code for lower=" + str(lower) + ", tasks=" + str(include) + ":")
    table_rows = []
    included_transfer_tasks = [x for x in transfer_tasks if
                               x != "ImageCaptionRetrieval" and x != "SNLI" and x in include]
    table_rows.append(["\\textbf{Representation}"] + ["\\textbf{" + x.replace("SICKEntailment", "SICKE").replace("SICKRelatedness", "SICKR").replace("STSBenchmark", "STSB") + "}" for x in included_transfer_tasks])

    raw_table_rows = []
    for representation in representations:
        for dir in sorted(results):
            if ((lower and "_lower" in dir) or (not lower and "_lower" not in dir)) and dir.replace("./", "").replace(
                    "_lower", "") == representation:
                raw_table_row = []
                raw_table_row.append(
                    dir.replace("./", "").replace("_lower", "").replace("_full", "\\textsubscript{full}").replace(
                        "_max", "\\textsubscript{max}").replace("_mean", "\\textsubscript{mean}"))
                for transfer_task in included_transfer_tasks:
                    if transfer_task in results[dir]:
                        raw_table_row.append(format(results[dir][transfer_task][0], ".2f"))
                    else:
                        raw_table_row.append("-")
                raw_table_rows.append(raw_table_row)

    if len(raw_table_rows) > 0:
        for raw_table_row in raw_table_rows:
            table_row = []
            for i in range(len(raw_table_rows[0])):
                if i == 0:
                    table_row.append(raw_table_row[i])
                else:
                    ranked_set = set()
                    ranked_set.add(-100000)
                    ranked_set.add(-100000)  # Yeah this was rushed
                    for ranked_row in raw_table_rows:
                        if ranked_row[i] != "-":
                            ranked_set.add(int(ranked_row[i].replace(".", "")))
                    ranked = sorted(list(ranked_set))
                    if raw_table_row[i] == "-":
                        table_row.append(raw_table_row[i])
                    elif int(raw_table_row[i].replace(".", "")) == ranked[-1]:
                        table_row.append("\\textbf{\\underline{" + raw_table_row[i] + "}}")
                    elif int(raw_table_row[i].replace(".", "")) == ranked[-2]:
                        table_row.append("\\textbf{" + raw_table_row[i] + "}")
                    else:
                        table_row.append(raw_table_row[i])
            table_rows.append(table_row)

    print("\\hline " + " & ".join(table_rows[0]) + " \\\\")
    print("\\hline")
    for table_row in table_rows[1:]:
        print(" & ".join(table_row) + " \\\\")
    print("\\hline")

make_full_table(lower=False, include=['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness'], representations = ["GloVe_full", "GloVe_max", "GloVe_mean", "CoVe_full", "CoVe_max", "CoVe_mean", "GloVe+CoVe_full", "GloVe+CoVe_max", "GloVe+CoVe_mean", "InferSent_full", "InferSent_max", "InferSent_mean"])
make_full_table(lower=False, include=['SICKEntailment', 'MRPC', 'SST2', 'SST5', 'MR', 'MPQA', 'CR', 'TREC', 'SUBJ'], representations = ["GloVe_full", "GloVe_max", "GloVe_mean", "CoVe_full", "CoVe_max", "CoVe_mean", "GloVe+CoVe_full", "GloVe+CoVe_max", "GloVe+CoVe_mean", "InferSent_full", "InferSent_max", "InferSent_mean"])
make_full_table(lower=True, include=['SICKRelatedness', 'SICKEntailment', 'MRPC', 'MR', 'TREC', 'SUBJ'], representations = ["GloVe_full", "GloVe_max", "GloVe_mean", "CoVe_full", "CoVe_max", "CoVe_mean", "GloVe+CoVe_full", "GloVe+CoVe_max", "GloVe+CoVe_mean", "InferSent_full", "InferSent_max", "InferSent_mean"])

print("\n############## SPLIT TABLES ##############")

def make_table(lower, include, representations):
    print("\nLaTeX Table code for lower=" + str(lower) + ", tasks=" + str(include) + ":")
    table_rows = []
    included_transfer_tasks = [x for x in transfer_tasks if x != "ImageCaptionRetrieval" and x != "SNLI" and x in include]
    table_rows.append(["\\textbf{Representation}"] + ["\\textbf{" + x.replace("SICKEntailment", "SICKE").replace("SICKRelatedness", "SICKR").replace("STSBenchmark", "STSB") + "}" for x in included_transfer_tasks])

    raw_table_rows = []
    for representation in representations:
        for dir in sorted(results):
            if ((lower and "_lower" in dir) or (not lower and "_lower" not in dir)) and dir.replace("./", "").replace("_lower", "") == representation:
                raw_table_row = []
                raw_table_row.append(dir.replace("./", "").replace("_lower", "").replace("_full", "\\textsubscript{full}").replace("_max", "\\textsubscript{max}").replace("_mean", "\\textsubscript{mean}"))
                for transfer_task in included_transfer_tasks:
                    if transfer_task in results[dir]:
                        raw_table_row.append(format(results[dir][transfer_task][0], ".2f"))
                    else:
                        raw_table_row.append("-")
                raw_table_rows.append(raw_table_row)

    if len(raw_table_rows) > 0:
        for raw_table_row in raw_table_rows:
            table_row = []
            for i in range(len(raw_table_rows[0])):
                if i == 0:
                    table_row.append(raw_table_row[i])
                else:
                    ranked_set = set()
                    ranked_set.add(-100000)
                    ranked_set.add(-100000) # Yeah this was rushed
                    for ranked_row in raw_table_rows:
                        if ranked_row[i] != "-":
                            ranked_set.add(int(ranked_row[i].replace(".", "")))
                    ranked = sorted(list(ranked_set))
                    if raw_table_row[i] == "-":
                        table_row.append(raw_table_row[i])
                    elif int(raw_table_row[i].replace(".", "")) == ranked[-1]:
                        table_row.append("\\textbf{\\underline{" + raw_table_row[i] + "}}")
                    #elif int(raw_table_row[i].replace(".", "")) == ranked[-2]:
                    #    table_row.append("\\textbf{" + raw_table_row[i] + "}")
                    else:
                        table_row.append(raw_table_row[i])
            table_rows.append(table_row)

    print("\\hline " + " & ".join(table_rows[0]) + " \\\\")
    print("\\hline")
    for table_row in table_rows[1:]:
        print(" & ".join(table_row) + " \\\\")
    print("\\hline")

make_table(lower=False, include=['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness'], representations = ["GloVe_full", "CoVe_full", "GloVe+CoVe_full", "InferSent_full"])
make_table(lower=False, include=['SICKEntailment', 'MRPC', 'SST2', 'SST5', 'MR', 'MPQA', 'CR', 'TREC', 'SUBJ'], representations = ["GloVe_full", "CoVe_full", "GloVe+CoVe_full", "InferSent_full"])
make_table(lower=True, include=['SICKRelatedness', 'SICKEntailment', 'MRPC', 'MR', 'TREC', 'SUBJ'], representations = ["GloVe_full", "CoVe_full", "GloVe+CoVe_full", "InferSent_full"])

make_table(lower=False, include=['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness'], representations = ["GloVe_max", "CoVe_max", "GloVe+CoVe_max", "InferSent_max"])
make_table(lower=False, include=['SICKEntailment', 'MRPC', 'SST2', 'SST5', 'MR', 'MPQA', 'CR', 'TREC', 'SUBJ'], representations = ["GloVe_max", "CoVe_max", "GloVe+CoVe_max", "InferSent_max"])
make_table(lower=True, include=['SICKRelatedness', 'SICKEntailment', 'MRPC', 'MR', 'TREC', 'SUBJ'], representations = ["GloVe_max", "CoVe_max", "GloVe+CoVe_max", "InferSent_max"])

make_table(lower=False, include=['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness'], representations = ["GloVe_mean", "CoVe_mean", "GloVe+CoVe_mean", "InferSent_mean"])
make_table(lower=False, include=['SICKEntailment', 'MRPC', 'SST2', 'SST5', 'MR', 'MPQA', 'CR', 'TREC', 'SUBJ'], representations = ["GloVe_mean", "CoVe_mean", "GloVe+CoVe_mean", "InferSent_mean"])
make_table(lower=True, include=['SICKRelatedness', 'SICKEntailment', 'MRPC', 'MR', 'TREC', 'SUBJ'], representations = ["GloVe_mean", "CoVe_mean", "GloVe+CoVe_mean", "InferSent_mean"])

print("\n############## MERGED TABLE ##############\n")

def make_merged_table(lower, include, representations, header=False):
    representations_to_rank = ["GloVe_full", "CoVe_full", "GloVe+CoVe_full", "InferSent_full", "GloVe_max", "CoVe_max",
                               "GloVe+CoVe_max", "InferSent_max", "GloVe_mean", "CoVe_mean", "GloVe+CoVe_mean", "InferSent_mean"]

    table_rows = []
    included_transfer_tasks = [x for x in transfer_tasks if
                               x != "ImageCaptionRetrieval" and x != "SNLI" and x in include]
    if header:
        table_rows.append(["\\textbf{Representation}"] + ["\\textbf{" + x.replace("SICKEntailment", "SICKE").replace("SICKRelatedness", "SICKR").replace("STSBenchmark", "STSB") + "}" for x in included_transfer_tasks])
    all_table_rows = []
    for representation in representations_to_rank:
        for dir in sorted(results):
            if ((lower and "_lower" in dir) or (not lower and "_lower" not in dir)) and dir.replace("./", "").replace(
                    "_lower", "") == representation:
                all_table_row = []
                all_table_row.append(
                    dir.replace("./", "").replace("_lower", "").replace("_full", "\\textsubscript{full}").replace(
                        "_max", "\\textsubscript{max}").replace("_mean", "\\textsubscript{mean}"))
                for transfer_task in included_transfer_tasks:
                    if transfer_task in results[dir]:
                        all_table_row.append(format(results[dir][transfer_task][0], ".2f"))
                    else:
                        all_table_row.append("-")
                all_table_rows.append(all_table_row)

    raw_table_rows = []
    for representation in representations:
        for dir in sorted(results):
            if ((lower and "_lower" in dir) or (not lower and "_lower" not in dir)) and dir.replace("./", "").replace("_lower", "") == representation:
                raw_table_row = []
                raw_table_row.append(dir.replace("./", "").replace("_lower", "").replace("_full", "\\textsubscript{full}").replace("_max", "\\textsubscript{max}").replace("_mean", "\\textsubscript{mean}"))
                for transfer_task in included_transfer_tasks:
                    if transfer_task in results[dir]:
                        raw_table_row.append(format(results[dir][transfer_task][0], ".2f"))
                    else:
                        raw_table_row.append("-")
                raw_table_rows.append(raw_table_row)

    if len(raw_table_rows) > 0:
        for raw_table_row in raw_table_rows:
            table_row = []
            for i in range(len(raw_table_rows[0])):
                if i == 0:
                    table_row.append(raw_table_row[i])
                else:
                    # Bold the best accuracy for that variant (full/max/mean)
                    bold_ranked_set = set()
                    bold_ranked_set.add(-100000)
                    bold_ranked_set.add(-100000) # Yeah this was rushed
                    for ranked_row in raw_table_rows:
                        if ranked_row[i] != "-":
                            bold_ranked_set.add(int(ranked_row[i].replace(".", "")))
                    bold_ranked = sorted(list(bold_ranked_set))

                    # Underline the best accuracy for that representation
                    underlined_ranked_set = set()
                    underlined_ranked_set.add(-100000)
                    underlined_ranked_set.add(-100000)  # Yeah this was rushed
                    for all_table_row in all_table_rows:
                        if all_table_row[i] != "-" and table_row[0].split("\\")[0] == all_table_row[0].split('\\')[0]:
                            underlined_ranked_set.add(int(all_table_row[i].replace(".", "")))
                    underlined_ranked = sorted(list(underlined_ranked_set))

                    # Colour the best accuracy overall
                    coloured_ranked_set = set()
                    coloured_ranked_set.add(-100000)
                    coloured_ranked_set.add(-100000)  # Yeah this was rushed
                    for all_table_row in all_table_rows:
                        if all_table_row[i] != "-":
                            coloured_ranked_set.add(int(all_table_row[i].replace(".", "")))
                    coloured_ranked = sorted(list(coloured_ranked_set))

                    if raw_table_row[i] == "-":
                        table_row.append(raw_table_row[i])
                    else:
                        bold = False
                        underline = False
                        coloured = False
                        if int(raw_table_row[i].replace(".", "")) == bold_ranked[-1]:
                            bold = True
                        if int(raw_table_row[i].replace(".", "")) == underlined_ranked[-1]:
                            underline = True
                        if int(raw_table_row[i].replace(".", "")) == coloured_ranked[-1]:
                            coloured = True

                        cell = ""
                        if coloured:
                            cell = cell + "\cellcolor{yellow!18}"
                        if bold:
                            cell = cell + "\\textbf{"
                        if underline:
                            cell = cell + "\\underline{"
                        cell = cell + raw_table_row[i]
                        if bold:
                            cell = cell + "}"
                        if underline:
                            cell = cell + "}"
                        table_row.append(cell)
            table_rows.append(table_row)

    if header:
        print("\\hline")
    print(" & ".join(table_rows[0]) + " \\\\")
    if header:
        print("\\hline")
        print("\\hline")
    for table_row in table_rows[1:]:
        print(" & ".join(table_row) + " \\\\")
    print("\\hline")

make_merged_table(lower=False, include=['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness'], representations = ["GloVe_full", "CoVe_full", "GloVe+CoVe_full", "InferSent_full"], header=True)
make_merged_table(lower=False, include=['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness'], representations = ["GloVe_max", "CoVe_max", "GloVe+CoVe_max", "InferSent_max"])
make_merged_table(lower=False, include=['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness'], representations = ["GloVe_mean", "CoVe_mean", "GloVe+CoVe_mean", "InferSent_mean"])

print("\n")

make_merged_table(lower=False, include=['SICKEntailment', 'MRPC', 'SST2', 'SST5', 'MR', 'MPQA', 'CR', 'TREC', 'SUBJ'], representations = ["GloVe_full", "CoVe_full", "GloVe+CoVe_full", "InferSent_full"], header=True)
make_merged_table(lower=False, include=['SICKEntailment', 'MRPC', 'SST2', 'SST5', 'MR', 'MPQA', 'CR', 'TREC', 'SUBJ'], representations = ["GloVe_max", "CoVe_max", "GloVe+CoVe_max", "InferSent_max"])
make_merged_table(lower=False, include=['SICKEntailment', 'MRPC', 'SST2', 'SST5', 'MR', 'MPQA', 'CR', 'TREC', 'SUBJ'], representations = ["GloVe_mean", "CoVe_mean", "GloVe+CoVe_mean", "InferSent_mean"])

print("\n")

make_merged_table(lower=True, include=['SICKRelatedness', 'SICKEntailment', 'MRPC', 'MR', 'TREC', 'SUBJ'], representations = ["GloVe_full", "CoVe_full", "GloVe+CoVe_full", "InferSent_full"], header=True)
make_merged_table(lower=True, include=['SICKRelatedness', 'SICKEntailment', 'MRPC', 'MR', 'TREC', 'SUBJ'], representations = ["GloVe_max", "CoVe_max", "GloVe+CoVe_max", "InferSent_max"])
make_merged_table(lower=True, include=['SICKRelatedness', 'SICKEntailment', 'MRPC', 'MR', 'TREC', 'SUBJ'], representations = ["GloVe_mean", "CoVe_mean", "GloVe+CoVe_mean", "InferSent_mean"])

print("\n\n##############################################################")
print("######################## RAW RESULTS: ########################")
print("##############################################################")

for dir, transfer_tasks_ in results.iteritems():
    print("\nResults for: " + dir)
    for transfer_task in transfer_tasks:
        if transfer_task in transfer_tasks_:
            print("   " + transfer_task + ":" + str(transfer_tasks_[transfer_task]))