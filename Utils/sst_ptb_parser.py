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
# This code also makes use of the Stanford SST dataset: Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang,
# Christopher Manning, Andrew Ng and Christopher Potts. 2013. Recursive Deep Models for Semantic Compositionality Over a
# Sentiment Treebank. In Conference on Empirical Methods in Natural Language Processing (EMNLP 2013).
#

import sys
import re
import io
import argparse

parser = argparse.ArgumentParser(description='Parses SST dataset files from PTB format to line-by-line format. These SST files can be found at: https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip')
parser.add_argument("infile", type=str, help="File in PTB format to parse")
parser.add_argument("outfile", type=str, help="Filename of output")
parser.add_argument("--binary", action='store_true', help="Whether to ignore neutrals and have single positive/negative classes")
parser.add_argument("--lower", action='store_true', help="Whether or not all words/sentences should be lowercased")
parser.add_argument("--ignore_subtrees", action='store_true', help="Only look at the outer-most sentence on each line")
params, _ = parser.parse_known_args()

# Convert PTB tree format into list of lists
# Source: https://stackoverflow.com/questions/27612254/split-nested-string-by-parentheses-into-nested-list
def make_tree(data):
    items = re.findall(r"\(|\)|[^\(^\)^\s]+", data)

    def req(index):
        result = []
        item = items[index]
        while item != ")":
            if item == "(":
                subtree, index = req(index + 1)
                result.append(subtree)
            else:
                result.append(item)
            index += 1
            item = items[index]
        return result, index

    return req(1)[0]

# We don't want brackets ( and ) to be in the tree otherwise the parser got confused
def check_for_illegals(node):
    if isinstance(node, (str, unicode)):
        if "(" in node or ")" in node:
            return True
        return False
    verdicts = []
    for item in node:
        verdicts.append(check_for_illegals(item))
    return any(verdicts)

parsed_lines = []
with io.open(params.infile, 'r', encoding='utf-8') as f:
    for line in f:
        parsed_tree = make_tree(line)
        assert(check_for_illegals(parsed_tree) == False)
        parsed_lines.append(parsed_tree)

def get_samples_from_parsed_lines(nodes, lower=False, ignore_subtrees=False):
    sentences_by_tags = [[], [], [], [], []]
    def dfs(node, ignore_subtrees, outer_node):
        if isinstance(node, (str, unicode)):
            return [node]
        assert isinstance(node, list)

        if len(node) == 1:
            return dfs(node[0], ignore_subtrees, False)

        tag = node[0]
        sentence = node[1:]

        parsed_sentence = []
        for item in sentence:
            parsed_sentence = parsed_sentence + dfs(item, ignore_subtrees, False)

        #print(str(tag) + " " + str(parsed_sentence))
        if not ignore_subtrees or outer_node:
            if lower:
                sentences_by_tags[int(tag)].append((' '.join(parsed_sentence)).lower())
            else:
                sentences_by_tags[int(tag)].append((' '.join(parsed_sentence)))
        return parsed_sentence
    for node in nodes:
        dfs(node, ignore_subtrees, True)
    return sentences_by_tags

sentences_by_tags = get_samples_from_parsed_lines(parsed_lines, lower=params.lower, ignore_subtrees=params.ignore_subtrees)
with open(params.outfile, "w") as outputfile:
    if not params.binary:
        for i in range(len(sentences_by_tags)):
            for item in sentences_by_tags[i]:
                outputfile.write(str(i) + " " + item.encode('utf-8') + "\n")
    else:
        for item in sentences_by_tags[0]:
            outputfile.write("0" + " " + item.encode('utf-8') + "\n")
        for item in sentences_by_tags[1]:
            outputfile.write("0" + " " + item.encode('utf-8') + "\n")
        for item in sentences_by_tags[3]:
            outputfile.write("1" + " " + item.encode('utf-8') + "\n")
        for item in sentences_by_tags[4]:
            outputfile.write("1" + " " + item.encode('utf-8') + "\n")
