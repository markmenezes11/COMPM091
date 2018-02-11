# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Pytorch Classifier class in the style of scikit-learn
Classifiers include Logistic Regression and MLP
"""

from __future__ import absolute_import, division, unicode_literals

import numpy as np
import copy
from senteval import utils

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import pickle

class PyTorchClassifier(object):
    def __init__(self, inputdim, nclasses, l2reg=0., batch_size=64, seed=1111,
                 cudaEfficient=False):
        # fix seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.inputdim = inputdim
        self.nclasses = nclasses
        self.l2reg = l2reg
        self.batch_size = batch_size
        self.cudaEfficient = cudaEfficient

    def prepare_split(self, X, y, validation_data=None, validation_split=None):
        # Preparing validation data
        assert validation_split or validation_data
        if validation_data is not None:
            trainX, trainy = X, y
            devX, devy = validation_data
        else:
            permutation = np.random.permutation(len(X))
            trainidx = permutation[int(validation_split*len(X)):]
            devidx = permutation[0:int(validation_split*len(X))]
            trainX, trainy = X[trainidx], y[trainidx]
            devX, devy = X[devidx], y[devidx]
        return trainX, trainy, devX, devy

    def cast_to_float_tensor(self, arr):
        if not self.cudaEfficient:
            return torch.FloatTensor(arr).cuda()
        else:
            return torch.FloatTensor(arr)

    def cast_to_long_tensor(self, arr):
        if not self.cudaEfficient:
            return torch.LongTensor(arr).cuda()
        else:
            return torch.LongTensor(arr)

    def fit(self, X, y, validation_data=None, validation_split=None,
            early_stop=True):
        self.nepoch = 0
        bestaccuracy = -1
        stop_train = False
        early_stop_count = 0

        # Preparing validation data
        trainX, trainy, devX, devy = self.prepare_split(X, y, validation_data, validation_split)
        trainy = self.cast_to_long_tensor(trainy)
        devy = self.cast_to_long_tensor(devy)

        # Training
        while not stop_train and self.nepoch <= self.max_epoch:
            self.trainepoch(trainX, trainy, epoch_size=self.epoch_size)
            accuracy = self.score(devX, devy)
            if accuracy > bestaccuracy:
                bestaccuracy = accuracy
                bestmodel = copy.deepcopy(self.model)
            elif early_stop:
                if early_stop_count >= self.tenacity:
                    stop_train = True
                early_stop_count += 1
        self.model = bestmodel
        return bestaccuracy

    def trainepoch(self, X, y, epoch_size=1):
        self.model.train()
        for _ in range(self.nepoch, self.nepoch + epoch_size):
            permutation = np.random.permutation(len(X))
            all_costs = []
            for i in range(0, len(X), self.batch_size):
                # forward
                idx = permutation[i:i + self.batch_size]

                # EDITED: Load embeddings from temp pickle file using the given filenames and array indexes
                Xbatch_embeddings = []
                files = dict()
                for j in idx:
                    if j < len(X):
                        filename = X[j][0]
                        index = int(X[j][1])
                        if filename not in files:
                            with open(filename) as f:
                                files[filename] = pickle.load(f)
                        Xbatch_embeddings.append(np.array([files[filename][index]]))
                Xbatch = np.vstack(Xbatch_embeddings)
                Xbatch = self.cast_to_float_tensor(Xbatch)
                y_idx = torch.LongTensor(idx)
                if isinstance(y, torch.cuda.LongTensor):
                    y_idx = y_idx.cuda()
                Xbatch = Variable(Xbatch)
                ybatch = Variable(y.index_select(0, y_idx))
                ###############################

                if self.cudaEfficient:
                    Xbatch = Xbatch.cuda()
                    ybatch = ybatch.cuda()
                output = self.model(Xbatch)
                # loss
                loss = self.loss_fn(output, ybatch)
                all_costs.append(loss.data[0])
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                # Update parameters
                self.optimizer.step()
        self.nepoch += epoch_size

    def score(self, devX, devy):
        self.model.eval()
        correct = 0
        for i in range(0, len(devX), self.batch_size):
            # EDITED: Load embeddings from temp pickle file using the given filenames and array indexes
            devX_embeddings = []
            files = dict()
            for j in range(i, i + self.batch_size):
                if j < len(devX):
                    filename = devX[j][0]
                    index = int(devX[j][1])
                    if filename not in files:
                        with open(filename) as f:
                            files[filename] = pickle.load(f)
                    devX_embeddings.append(np.array([files[filename][index]]))
            devX_embeddings = np.vstack(devX_embeddings)
            devX_embeddings = self.cast_to_float_tensor(devX_embeddings)
            ###############################

            if not isinstance(devX_embeddings, torch.cuda.FloatTensor) or self.cudaEfficient:
                devX_embeddings = torch.FloatTensor(devX_embeddings).cuda()
                devy = torch.LongTensor(devy).cuda()
            Xbatch = Variable(devX_embeddings, volatile=True)
            ybatch = Variable(devy[i:i + self.batch_size], volatile=True)
            if self.cudaEfficient:
                Xbatch = Xbatch.cuda()
                ybatch = ybatch.cuda()
            output = self.model(Xbatch)
            pred = output.data.max(1)[1]
            correct += pred.long().eq(ybatch.data.long()).sum()
        accuracy = 1.0*correct / len(devX)
        return accuracy

    def predict(self, devX):
        self.model.eval()
        yhat = np.array([])
        for i in range(0, len(devX), self.batch_size):
            # EDITED: Load embeddings from temp pickle file using the given filenames and array indexes
            devX_embeddings = []
            files = dict()
            for j in range(i, i + self.batch_size):
                if j < len(devX):
                    filename = devX[j][0]
                    index = int(devX[j][1])
                    if filename not in files:
                        with open(filename) as f:
                            files[filename] = pickle.load(f)
                    devX_embeddings.append(np.array([files[filename][index]]))
            devX_embeddings = np.vstack(devX_embeddings)
            ###############################

            if not isinstance(devX_embeddings, torch.cuda.FloatTensor):
                devX_embeddings = torch.FloatTensor(devX_embeddings).cuda()
            Xbatch = Variable(devX_embeddings, volatile=True)
            output = self.model(Xbatch)
            yhat = np.append(yhat,
                             output.data.max(1)[1].cpu().numpy())
        yhat = np.vstack(yhat)
        return yhat

    def predict_proba(self, devX):
        self.model.eval()
        probas = []
        for i in range(0, len(devX), self.batch_size):
            # EDITED: Load embeddings from temp pickle file using the given filenames and array indexes
            devX_embeddings = []
            files = dict()
            for j in range(i, i + self.batch_size):
                if j < len(devX):
                    filename = devX[j][0]
                    index = int(devX[j][1])
                    if filename not in files:
                        with open(filename) as f:
                            files[filename] = pickle.load(f)
                    devX_embeddings.append(np.array([files[filename][index]]))
            devX_embeddings = np.vstack(devX_embeddings)
            ###############################

            Xbatch = Variable(devX_embeddings, volatile=True)
            vals = F.softmax(self.model(Xbatch).data.cpu().numpy())
            if not probas:
                probas = vals
            else:
                probas = np.concatenate(probas, vals, axis=0)
        return probas


"""
MLP with Pytorch (nhid=0 --> Logistic Regression)
"""

class MLP(PyTorchClassifier):
    def __init__(self, params, inputdim, nclasses, l2reg=0., batch_size=64,
                 seed=1111, cudaEfficient=False):
        super(self.__class__, self).__init__(inputdim, nclasses, l2reg,
                                             batch_size, seed, cudaEfficient)
        """
        PARAMETERS:
        -nhid:       number of hidden units (0: Logistic Regression)
        -optim:      optimizer ("sgd,lr=0.1", "adam", "rmsprop" ..)
        -tenacity:   how many times dev acc does not increase before stopping
        -epoch_size: each epoch corresponds to epoch_size pass on the train set
        -max_epoch:  max number of epoches
        -dropout:    dropout for MLP
        """

        self.nhid = 0 if "nhid" not in params else params["nhid"]
        self.optim = "adam" if "optim" not in params else params["optim"]
        self.tenacity = 5 if "tenacity" not in params else params["tenacity"]
        self.epoch_size = 4 if "epoch_size" not in params else params["epoch_size"]
        self.max_epoch = 200 if "max_epoch" not in params else params["max_epoch"]
        self.dropout = 0. if "dropout" not in params else params["dropout"]
        self.batch_size = 64 if "batch_size" not in params else params["batch_size"]

        if params["nhid"] == 0:
            self.model = nn.Sequential(
                nn.Linear(self.inputdim, self.nclasses),
                ).cuda()
        else:
            self.model = nn.Sequential(
                nn.Linear(self.inputdim, params["nhid"]),
                nn.Dropout(p=self.dropout),
                nn.Sigmoid(),
                nn.Linear(params["nhid"], self.nclasses),
                ).cuda()

        self.loss_fn = nn.CrossEntropyLoss().cuda()
        self.loss_fn.size_average = False

        optim_fn, optim_params = utils.get_optimizer(self.optim)
        self.optimizer = optim_fn(self.model.parameters(), **optim_params)
        self.optimizer.param_groups[0]['weight_decay'] = self.l2reg
