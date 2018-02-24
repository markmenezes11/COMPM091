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
import random
import numpy as np

class SSTDataset(object):
    def __init__(self, n_classes, folder, data_dir, encoder, dry_run=False):
        self.n_classes = n_classes
        if dry_run:
            train = self.load_file(os.path.join(data_dir, folder, 'sentiment-dev'))
        else:
            train = self.load_file(os.path.join(data_dir, folder, 'sentiment-train'))
        dev = self.load_file(os.path.join(data_dir, folder, 'sentiment-dev'))
        test = self.load_file(os.path.join(data_dir, folder, 'sentiment-test'))
        train_cut_indexes = random.sample(range(len(train['y'])), len(dev['y']))
        train_cut = {'X': [train['X'][i] for i in train_cut_indexes], 'y': [train['y'][i] for i in train_cut_indexes]}
        textual_data = {'train': train, 'dev': dev, 'test': test, 'train_cut': train_cut}
        self.max_sent_len = -1
        self.total_sentences = 0
        for key in textual_data:
            for tokenized_sentence in textual_data[key]['X']:
                self.total_sentences += 1
                if len(tokenized_sentence) > self.max_sent_len:
                    self.max_sent_len = len(tokenized_sentence)
        print("Successfully loaded dataset (classes: " + str(self.n_classes)
              + ", max sentence length: " + str(self.max_sent_len) + ").")
        self.data = self.generate_embeddings(textual_data, encoder)
        self.embed_dim = encoder.get_embed_dim()

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

    def generate_embeddings(self, textual_data, encoder):
        print("\nGenerating sentence embeddings,,,")
        data = dict()
        done = 0
        milestones = {int(self.total_sentences * 0.1): "10%", int(self.total_sentences * 0.2): "20%",
                      int(self.total_sentences * 0.3): "30%", int(self.total_sentences * 0.4): "40%",
                      int(self.total_sentences * 0.5): "50%", int(self.total_sentences * 0.6): "60%",
                      int(self.total_sentences * 0.7): "70%", int(self.total_sentences * 0.8): "80%",
                      int(self.total_sentences * 0.9): "90%", self.total_sentences: "100%"}
        for key in textual_data:
            data[key] = {}
            data[key]['X1'] = []
            for tokenized_sentence in textual_data[key]['X']:
                data[key]['X1'].append(encoder.encode_sentence(tokenized_sentence))
                done += 1
                if done in milestones:
                    print("  " + milestones[done])
            data[key]['X1'] = np.array(data[key]['X1'])
            data[key]['X2'] = data[key]['X1'] # Only one input sentence is needed for SSTBinary so X1 is duplicated
            data[key]['y'] = np.array(textual_data[key]['y'])
        print("Successfully generated sentence embeddings,")
        return data

    def get_total_samples(self, key):
        return len(self.data[key]['y'])

    def pad(self, embeddings):
        padded_embeddings = []
        for embedding in embeddings:
            for pad in range(self.max_sent_len - len(embedding)):
                embedding = np.append(embedding, np.full((1, self.embed_dim), 0.0), axis=0)
            assert embedding.shape == (len(embedding), self.embed_dim)
            padded_embeddings.append([embedding])
        return np.vstack(padded_embeddings)

    def get_batch(self, key, indexes=None):
        if indexes is None:
            return self.pad(self.data[key]['X1']), self.pad(self.data[key]['X2']), self.data[key]['y']
        X1 = self.pad(np.take(self.data[key]['X1'], indexes, axis=0))
        X2 = self.pad(np.take(self.data[key]['X2'], indexes, axis=0))
        y = np.take(self.data[key]['y'], indexes, axis=0)
        return X1, X2, y

    def get_n_classes(self):
        return self.n_classes

    def get_max_sent_len(self):
        return self.max_sent_len

    def get_embed_dim(self):
        return self.embed_dim

class SSTBinaryDataset(SSTDataset):
    def __init__(self, data_dir, encoder, dry_run=False):
        print("\nLoading SST Binary dataset...")
        super(SSTBinaryDataset, self).__init__(2, "SSTBinary", data_dir, encoder, dry_run=dry_run)

class SSTFineDataset(SSTDataset):
    def __init__(self, data_dir, encoder, dry_run=False):
        print("\nLoading SST Fine dataset...")
        super(SSTFineDataset, self).__init__(5, "SSTFine", data_dir, encoder, dry_run=dry_run)
