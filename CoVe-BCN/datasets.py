# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of the SentEval source tree.
# The license can also be found in the datasets folder.
#
# The source code below is a modified version of source code from:
# SentEval: https://github.com/facebookresearch/SentEval

import os
import io
import numpy as np

class SSTBinaryDataset():
    def __init__(self, data_dir):
        print("\nLoading SST Binary dataset...")
        self.n_classes = 2
        train = self.load_file(os.path.join(data_dir, 'SSTBinary/sentiment-train'))
        dev = self.load_file(os.path.join(data_dir, 'SSTBinary/sentiment-dev'))
        test = self.load_file(os.path.join(data_dir, 'SSTBinary/sentiment-test'))
        self.textual_data = {'train': train, 'dev': dev, 'test': test}
        self.max_sent_len = -1
        self.total_sentences = 0
        for key in self.textual_data:
            for tokenized_sentence in self.textual_data[key]['X']:
                self.total_sentences += 1
                if len(tokenized_sentence) > self.max_sent_len:
                    self.max_sent_len = len(tokenized_sentence)
        print("Successfully loaded dataset (classes: " + str(self.n_classes)
              + ", max sentence length: " + str(self.max_sent_len) + ").")

    def load_file(self, fpath):
        textual_data = {'X': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if self.n_classes == 2:
                    sample = line.strip().split('\t')
                    textual_data['y'].append(int(sample[1]))
                    textual_data['X'].append(sample[0].split())
                elif self.n_classes == 5:
                    sample = line.strip().split(' ', 1)
                    textual_data['y'].append(int(sample[0]))
                    textual_data['X'].append(sample[1].split())
        assert max(textual_data['y']) == self.n_classes - 1
        return textual_data

    def generate_embeddings(self, encoder):
        print("\nGenerating sentence embeddings,,,")
        embeddings = {'train': {}, 'dev': {}, 'test': {}}
        done = 0
        milestones = {int(self.total_sentences * 0.1): "10%", int(self.total_sentences * 0.2): "20%",
                      int(self.total_sentences * 0.3): "30%", int(self.total_sentences * 0.4): "40%",
                      int(self.total_sentences * 0.5): "50%", int(self.total_sentences * 0.6): "60%",
                      int(self.total_sentences * 0.7): "70%", int(self.total_sentences * 0.8): "80%",
                      int(self.total_sentences * 0.9): "90%", self.total_sentences: "100%"}
        for key in self.textual_data:
            embeddings[key]['X1'] = []
            for tokenized_sentence in self.textual_data[key]['X']:
                embeddings[key]['X1'].append([encoder.encode_sentence(tokenized_sentence, self.max_sent_len)])
                done += 1
                if done in milestones:
                    print("  " + milestones[done])
            embeddings[key]['X1'] = np.vstack(embeddings[key]['X1'])
            embeddings[key]['X2'] = embeddings[key]['X1']
            embeddings[key]['y'] = np.array(self.textual_data[key]['y'])
        print("Successfully generated sentence embeddings,")
        return embeddings

    def get_n_classes(self):
        return self.n_classes

    def get_max_sent_len(self):
        return self.max_sent_len