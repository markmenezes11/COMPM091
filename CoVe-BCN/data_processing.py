# CoVe is taken from:
# B. McCann, J. Bradbury, C. Xiong, R. Socher, Learned in Translation: Contextualized Word Vectors
# https://github.com/salesforce/cove
#

import sys

import numpy as np

from keras.models import load_model

class GloVeCoVeEncoder:
    def __init__(self, glove_path, cove_path, ignore_glove_header=False, cove_dim=900):
        print("\nLoading GloVe embeddings...")
        f = open(glove_path)
        self.glove_embeddings_dict = dict()
        self.glove_dim = -1
        first_line = True
        for line in f:
            if first_line and ignore_glove_header:
                first_line = False
                continue
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            if self.glove_dim == -1:
                self.glove_dim = len(embedding)
            assert self.glove_dim == len(embedding)
            self.glove_embeddings_dict[word] = embedding
        f.close()
        if len(self.glove_embeddings_dict) == 0 or self.glove_dim == -1:
            print("ERROR: Failed to load GloVe embeddings.")
            sys.exit(1)
        print("Successfully loaded GloVe embeddings (vocab size: " + str(
            len(self.glove_embeddings_dict)) + ", dimensions: " + str(self.glove_dim) + ").")

        # Load CoVe model. Ported version from https://github.com/rgsachin/CoVe
        # - Input: GloVe vectors of dimension - (<batch_size>, <sentence_len>, <self.glove_dim>)
        # - Output: CoVe vectors of dimension - (<batch_size>, <sentence_len>, <self.cove_dim>)
        # - Example: self.cove_model.predict(np.random.rand(1, 10, self.glove_dime))
        # - For unknown words, use a dummy value different to the one used for padding - small non-zero value e.g. 1e-10
        print("\nLoading CoVe model...")
        self.cove_dim = cove_dim
        self.glove_cove_dim = self.glove_dim + self.cove_dim
        self.cove_model = load_model(cove_path)
        test = self.cove_model.predict(np.random.rand(1, 10, 300))
        assert test.shape == (1, 10, self.cove_dim)
        print("Successfully loaded CoVe model.")

    # Input sequence (tokenized sentence) w is converted to sequence of vectors: w' = [GloVe(w); CoVe(w)]
    def encode_sentence(self, tokenized_sentence):
        glove_embeddings = []
        for word in tokenized_sentence:
            try:
                glove_embedding = np.array(self.glove_embeddings_dict[word])
            except KeyError:
                # 1e-10 for unknown words, as recommended on https://github.com/rgsachin/CoVe
                glove_embedding = np.full(self.glove_dim, 1e-10)
            assert glove_embedding.shape == (self.glove_dim,)
            glove_embeddings.append(glove_embedding)
        glove = np.array([glove_embeddings])
        assert glove.shape == (1, len(tokenized_sentence), self.glove_dim)
        cove = self.cove_model.predict(glove)
        assert cove.shape == (1, len(tokenized_sentence), self.cove_dim)
        glove_cove = np.concatenate([glove[0], cove[0]], axis=1)
        assert glove_cove.shape == (len(tokenized_sentence), self.glove_dim + self.cove_dim)
        return glove_cove

    def get_embed_dim(self):
        return self.glove_cove_dim
