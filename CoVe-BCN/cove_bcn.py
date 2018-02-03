# CoVe is taken from:
# B. McCann, J. Bradbury, C. Xiong, R. Socher, Learned in Translation: Contextualized Word Vectors
# https://github.com/salesforce/cove
#

import argparse
import gensim
import numpy as np
import tensorflow as tf

from keras.models import load_model

parser = argparse.ArgumentParser(description='Replication of the CoVe Biattentive Classification Network (BCN)')
parser.add_argument("--glovepath", type=str, default="../../Word2Vec_models/GloVe/glove.840B.300d.txt", help="Path to GloVe word embeddings. Download glove.840B.300d embeddings from https://nlp.stanford.edu/projects/glove/")
parser.add_argument("--glovebinary", action="store_true", default=False, help="Whether or not the GloVe model is a binary (bin) file")
parser.add_argument("--glovedimensions", type=int, default=300, help="Number of dimensions in GloVe embeddings (default: 300)")
parser.add_argument("--covepath", type=str, default='../CoVe-ported/Keras_CoVe_Python2.h5', help="Path to the CoVe model")
parser.add_argument("--covedimensions", type=int, default=600, help="Number of dimensions in CoVe embeddings (default: 600)")
params, _ = parser.parse_known_args()

"""
DATASET
"""

# TODO: Get dataset, split it into train/test, and adjust max_sent_len and n_classes accordingly
max_sent_len = 256
n_classes = 3

"""
EMBEDDINGS
"""

"""glove_model = gensim.models.KeyedVectors.load_word2vec_format(params.glovepath, binary = params.glovebinary, unicode_errors = 'ignore')
print("Successfully loaded GloVe model.")

# Load CoVe model. Ported version from https://github.com/rgsachin/CoVe
# - Input: GloVe vectors of dimension - (<batch_size>, <sentence_len>, <glove_dimensions>)
# - Output: CoVe vectors of dimension - (<batch_size>, <sentence_len>, <cove_dimensions>)
# - Example: cove_model.predict(np.random.rand(1, 10, params.glovedimensions))
# - For unknown words, use a dummy value different to the dummy value used for padding - small non-zero value e.g. 1e-10
cove_model = load_model(params.covepath)
print("Successfully loaded CoVe model.")

# Input sequence (sentences) w is converted to sequence of vectors: w' = [GloVe(w); CoVe(w)]
def sentence_to_glove_cove(sentence):
    glove_embeddings = []
    for word in sentence:
        try:
            glove_embedding = np.array(glove_model[word])
            assert glove_embedding.shape == (params.glovedimensions,)
            glove_embeddings.append(glove_embedding)
        except KeyError:
            glove_embedding = np.full(params.glovedimensions, 1e-10) # 1e-10 for unknown words, as recommended on https://github.com/rgsachin/CoVe
            assert glove_embedding.shape == (params.glovedimensions,)
            glove_embeddings.append(glove_embedding)
    glove = np.array([glove_embeddings])
    assert glove.shape == (1, len(sentence), params.glovedimensions)
    cove = cove_model.predict(glove)
    assert cove.shape == (1, len(sentence), params.covedimensions)
    glove_cove = np.concatenate([glove[0], cove[0]], axis=1)
    for pad in range(max_sent_len - len(sentence)):
        glove_cove = np.append(glove_cove, np.full((1, params.glovedimensions + params.covedimensions), 0.0), axis=0)
    assert glove_cove.shape == (max_sent_len, params.glovedimensions + params.covedimensions)
    return glove_cove"""

"""
HYPERPARAMETERS
"""

# TODO: Tune the following parameters
n_epochs = 100 # int
batch_size = 60 # int

feedforward_weight_size = 0.1 # float
feedforward_bias_size = 0.1 # float
feedforward_activation = tf.nn.relu6 # tf.nn.relu or tf.nn.relu6

same_bilstm_for_encoder = True # boolean
bilstm_encoder_n_hidden1 = 200 # int
bilstm_encoder_forget_bias1 = 1.0 # float
bilstm_encoder_n_hidden2 = 200 # int - only needs to be set if same_bilstm_for_encoder is False
bilstm_encoder_forget_bias2 = 1.0 # float - only needs to be set if same_bilstm_for_encoder is False

bilstm_integrate_n_hidden1 = 200 # int
bilstm_integrate_forget_bias1 = 1.0 # float
bilstm_integrate_n_hidden2 = 200 # int
bilstm_integrate_forget_bias2 = 1.0 # float

self_pool_weight_size1 = 0.1
self_pool_bias_size1 = 0.1
self_pool_weight_size2 = 0.1
self_pool_bias_size2 = 0.1

"""
MODEL
"""

# Takes 2 input sequences, each of the form [GloVe(w); CoVe(w)] (duplicated if only one input sequence is needed)
inputs1 = tf.placeholder(tf.float32, shape=[batch_size, max_sent_len, 900]) # (n_sentences, max_sent_len, 900)
inputs2 = tf.placeholder(tf.float32, shape=[batch_size, max_sent_len, 900]) # (n_sentences, max_sent_len, 900)
labels = tf.placeholder(tf.float32, [None, n_classes])  # (n_sentences, n_classes)
assert inputs1.shape == (batch_size, max_sent_len, 900)
assert inputs2.shape == (batch_size, max_sent_len, 900)

# Feedforward network with ReLU activation, applied to each word embedding (word) in the sequence (sentence)
feedforward_weight = tf.get_variable("feedforward_weight", shape=[900, 900], initializer=tf.random_uniform_initializer(-feedforward_weight_size, feedforward_weight_size))
feedforward_bias = tf.get_variable("feedforward_bias", shape=[900], initializer=tf.constant_initializer(feedforward_bias_size))
def feedforward(feedforward_inputs):
    feedforward_inputs_unstacked = tf.unstack(feedforward_inputs) # Split on axis 0 into (max_sent_len, 900)
    feedforward_outputs = []
    for feedforward_input in feedforward_inputs_unstacked: # Loop through each sentence
        feedforward_outputs.append(feedforward_activation(tf.matmul(feedforward_input, feedforward_weight) + feedforward_bias)) # TODO: Should this be a a fully_connected_layer or something? For reuse of ReLU?
    return tf.stack(feedforward_outputs)
feedforward_outputs1 = feedforward(inputs1)
feedforward_outputs2 = feedforward(inputs2)
assert feedforward_outputs1.shape == (batch_size, max_sent_len, 900)
assert feedforward_outputs2.shape == (batch_size, max_sent_len, 900)

# BiLSTM processes the resulting sequences
# The BCN is symmetrical - not sure whether to use the same BiLSTM or or two separate BiLSTMs (one for each side).
# Therefore implemented both, e.g. For SNLI you might want to use different models for inputs1 and inputs2, whereas
# for sentiment analysis you mightwant to use the same model
if same_bilstm_for_encoder:
    with tf.variable_scope("bilstm_encoder_scope") as bilstm_encoder_scope:
        bilstm_encoder_fw_cell = tf.contrib.rnn.LSTMCell(bilstm_encoder_n_hidden1, forget_bias=bilstm_encoder_forget_bias1)
        bilstm_encoder_bw_cell = tf.contrib.rnn.LSTMCell(bilstm_encoder_n_hidden1, forget_bias=bilstm_encoder_forget_bias1)
        bilstm_encoder_inputs = tf.concat((feedforward_outputs1, feedforward_outputs2), 0)
        bilstm_encoder_raw_outputs, _ = tf.nn.bidirectional_dynamic_rnn(bilstm_encoder_fw_cell, bilstm_encoder_bw_cell, bilstm_encoder_inputs, dtype=tf.float32)
        bilstm_encoder_outputs1, bilstm_encoder_outputs2 = tf.split(tf.concat([bilstm_encoder_raw_outputs[0], bilstm_encoder_raw_outputs[-1]], 2), 2, axis=0)
        bilstm_encoder_n_hidden2 = bilstm_encoder_n_hidden1 # For the assert below to work
else:
    with tf.variable_scope("bilstm_encoder_scope1") as bilstm_encoder_scope1:
        bilstm_encoder_fw_cell1 = tf.contrib.rnn.LSTMCell(bilstm_encoder_n_hidden1, forget_bias=bilstm_encoder_forget_bias1)
        bilstm_encoder_bw_cell1 = tf.contrib.rnn.LSTMCell(bilstm_encoder_n_hidden1, forget_bias=bilstm_encoder_forget_bias1)
        bilstm_encoder_raw_outputs1, _ = tf.nn.bidirectional_dynamic_rnn(bilstm_encoder_fw_cell1, bilstm_encoder_bw_cell1, feedforward_outputs1, dtype=tf.float32)
        bilstm_encoder_outputs1 = tf.concat([bilstm_encoder_raw_outputs1[0], bilstm_encoder_raw_outputs1[-1]], 2)
    with tf.variable_scope("bilstm_encoder_scope2") as bilstm_encoder_scope2:
        bilstm_encoder_fw_cell2 = tf.contrib.rnn.LSTMCell(bilstm_encoder_n_hidden2, forget_bias=bilstm_encoder_forget_bias2)
        bilstm_encoder_bw_cell2 = tf.contrib.rnn.LSTMCell(bilstm_encoder_n_hidden2, forget_bias=bilstm_encoder_forget_bias2)
        bilstm_encoder_raw_outputs2, _ = tf.nn.bidirectional_dynamic_rnn(bilstm_encoder_fw_cell2, bilstm_encoder_bw_cell2, feedforward_outputs2, dtype=tf.float32)
        bilstm_encoder_outputs2 = tf.concat([bilstm_encoder_raw_outputs2[0], bilstm_encoder_raw_outputs2[-1]], 2)
assert bilstm_encoder_outputs1.shape == (batch_size, max_sent_len, bilstm_encoder_n_hidden1*2)
assert bilstm_encoder_outputs2.shape == (batch_size, max_sent_len, bilstm_encoder_n_hidden2*2)

# Biattention mechanism [Seo et al., 2017, Xiong et al., 2017]
Xs = tf.unstack(bilstm_encoder_outputs1)
Ys = tf.unstack(bilstm_encoder_outputs2)
biattention_outputs1 = []
biattention_outputs2 = []
for X, Y in zip(Xs, Ys):
    # Affinity matrix A=XY^T
    A = tf.matmul(X, Y, transpose_b=True)
    assert A.shape == (max_sent_len, max_sent_len)

    # Column-wise normalisation to extract attention weights
    Ax = tf.nn.softmax(A)
    Ay = tf.nn.softmax(tf.transpose(A))
    assert Ax.shape == (max_sent_len, max_sent_len)
    assert Ay.shape == (max_sent_len, max_sent_len)

    # Context summaries
    Cx = tf.matmul(Ax, X, transpose_a=True)
    Cy = tf.matmul(Ay, X, transpose_a=True)
    assert Cx.shape == (max_sent_len, bilstm_encoder_n_hidden1*2)
    assert Cy.shape == (max_sent_len, bilstm_encoder_n_hidden2*2)

    biattention_outputs1.append(tf.concat([X, X - Cy, tf.multiply(X, Cy)], 1))
    biattention_outputs2.append(tf.concat([Y, Y - Cx, tf.multiply(Y, Cx)], 1))
biattention_outputs1 = tf.stack(biattention_outputs1)
biattention_outputs2 = tf.stack(biattention_outputs2)
assert biattention_outputs1.shape == (batch_size, max_sent_len, bilstm_encoder_n_hidden1*2*3)
assert biattention_outputs2.shape == (batch_size, max_sent_len, bilstm_encoder_n_hidden2*2*3)

# Integrate with two separate one-layer BiLSTMs
with tf.variable_scope("bilstm_integrate_scope1") as bilstm_integrate_scope1:
    bilstm_integrate_fw_cell1 = tf.contrib.rnn.LSTMCell(bilstm_integrate_n_hidden1, forget_bias=bilstm_integrate_forget_bias1)
    bilstm_integrate_bw_cell1 = tf.contrib.rnn.LSTMCell(bilstm_integrate_n_hidden1, forget_bias=bilstm_integrate_forget_bias1)
    bilstm_integrate_raw_outputs1, _ = tf.nn.bidirectional_dynamic_rnn(bilstm_integrate_fw_cell1, bilstm_integrate_bw_cell1, biattention_outputs1, dtype=tf.float32)
    bilstm_integrate_outputs1 = tf.concat([bilstm_integrate_raw_outputs1[0], bilstm_integrate_raw_outputs1[-1]], 2)
with tf.variable_scope("bilstm_integrate_scope2") as bilstm_integrate_scope2:
    bilstm_integrate_fw_cell2 = tf.contrib.rnn.LSTMCell(bilstm_integrate_n_hidden2, forget_bias=bilstm_integrate_forget_bias2)
    bilstm_integrate_bw_cell2 = tf.contrib.rnn.LSTMCell(bilstm_integrate_n_hidden2, forget_bias=bilstm_integrate_forget_bias2)
    bilstm_integrate_raw_outputs2, _ = tf.nn.bidirectional_dynamic_rnn(bilstm_integrate_fw_cell2, bilstm_integrate_bw_cell2, biattention_outputs2, dtype=tf.float32)
    bilstm_integrate_outputs2 = tf.concat([bilstm_integrate_raw_outputs2[0], bilstm_integrate_raw_outputs2[-1]], 2)
assert bilstm_integrate_outputs1.shape == (batch_size, max_sent_len, bilstm_integrate_n_hidden1*2)
assert bilstm_integrate_outputs2.shape == (batch_size, max_sent_len, bilstm_integrate_n_hidden2*2)

# Max, mean, min and self-attentive pooling
Xys = tf.unstack(bilstm_integrate_outputs1)
Yxs = tf.unstack(bilstm_integrate_outputs2)
pool_outputs1 = []
pool_outputs2 = []
self_pool_weight1 = tf.get_variable("self_pool_weight1", shape=[bilstm_integrate_n_hidden1*2, 1], initializer=tf.random_uniform_initializer(-self_pool_weight_size1, self_pool_weight_size1))
self_pool_bias1 = tf.get_variable("self_pool_bias1", shape=[1], initializer=tf.constant_initializer(self_pool_bias_size1))
self_pool_weight2 = tf.get_variable("self_pool_weight2", shape=[bilstm_integrate_n_hidden2*2, 1], initializer=tf.random_uniform_initializer(-self_pool_weight_size2, self_pool_weight_size2))
self_pool_bias2 = tf.get_variable("self_pool_bias2", shape=[1], initializer=tf.constant_initializer(self_pool_bias_size2))
for Xy, Yx in zip(Xys, Yxs):
    assert Xy.shape == (max_sent_len, bilstm_integrate_n_hidden1*2)
    assert Xy.shape == (max_sent_len, bilstm_integrate_n_hidden2*2)

    # Max pooling - just take the 256 "columns" in the matrix and get the max in each of them.
    max_Xy = tf.reduce_max(Xy, axis=0)
    max_Yx = tf.reduce_max(Yx, axis=0)
    assert max_Xy.shape == (bilstm_integrate_n_hidden1*2)
    assert max_Yx.shape == (bilstm_integrate_n_hidden2*2)

    # Mean pooling - just take the 256 "columns" in the matrix and get the mean in each of them.
    mean_Xy = tf.reduce_mean(Xy, axis=0)
    mean_Yx = tf.reduce_mean(Yx, axis=0)
    assert mean_Xy.shape == (bilstm_integrate_n_hidden1*2)
    assert mean_Yx.shape == (bilstm_integrate_n_hidden2*2)

    # Min pooling - just take the 256 "columns" in the matrix and get the min in each of them.
    min_Xy = tf.reduce_min(Xy, axis=0)
    min_Yx = tf.reduce_min(Yx, axis=0)
    assert min_Xy.shape == (bilstm_integrate_n_hidden1*2)
    assert min_Yx.shape == (bilstm_integrate_n_hidden2*2)

    # Self-attentive pooling
    Bx = tf.nn.softmax((tf.matmul(Xy, self_pool_weight1)) + self_pool_bias1)
    By = tf.nn.softmax((tf.matmul(Yx, self_pool_weight2)) + self_pool_bias2)
    x_self = tf.squeeze(tf.matmul(Xy, Bx, transpose_a=True)) # TODO: Output of this hsould be 256 by 400? Or 400? Think 400 is correct
    y_self = tf.squeeze(tf.matmul(Yx, By, transpose_a=True)) # TODO: Output of this hsould be 256 by 400? Or 400? Think 400 is correct
    assert x_self.shape == (bilstm_integrate_n_hidden1*2)
    assert y_self.shape == (bilstm_integrate_n_hidden2*2)

    # Combine pooled representations
    pool_outputs1.append(tf.concat([max_Xy, mean_Xy, min_Xy, x_self], 0))
    pool_outputs2.append(tf.concat([max_Yx, mean_Yx, min_Yx, y_self], 0))
pool_outputs1 = tf.stack(pool_outputs1)
pool_outputs2 = tf.stack(pool_outputs2)
assert pool_outputs1.shape == (batch_size, bilstm_encoder_n_hidden1*2*4)
assert pool_outputs2.shape == (batch_size, bilstm_encoder_n_hidden2*2*4)

# Max-out network (3 layers, batch normalised)
# TODO: Implement this ( https://arxiv.org/pdf/1502.03167.pdf ) ( https://arxiv.org/pdf/1302.4389.pdf )

# TODO: train_step and predict variable for the code below to use
# TODO: Uncomment the code below

print("Successfully created BCN model.")

"""
TRAIN
"""

"""train_X1 = [["This", "is", "a", "train", "sentence", "."]] # TODO: Get train data
train_X2 = [["This", "is", "a", "train", "sentence", "."]] # TODO: Get train data
train_y = [0] # TODO: Get train data

# TODO: Shuffle data

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(n_epochs):
        print("Epoch " + str(epoch+1) + " of " + str(n_epochs))
        for i in range(len(train_y) // batch_size):
            batch_X1 = train_X1[i * batch_size: (i+1) * batch_size]
            batch_X2 = train_X2[i * batch_size: (i + 1) * batch_size]
            batch_y = train_y[i * batch_size: (i + 1) * batch_size]
            sess.run(train_step, feed_dict={inputs1: [sentence_to_glove_cove(sentence) for sentence in batch_X1],
                                            inputs2: [sentence_to_glove_cove(sentence) for sentence in batch_X2],
                                            labels : batch_y})"""

"""
TEST
"""

"""test_X1 = [["This", "is", "a", "test", "sentence", "."]] # TODO: Get test data
test_X2 = [["This", "is", "a", "test", "sentence", "."]] # TODO: Get test data
test_y = [0] # TODO: Get test data
outputs1 = []
outputs2 = []
with tf.Session() as sess:
    outputs = sess.run(predict, feed_dict={inputs1: [sentence_to_glove_cove(sentence) for sentence in test_X1],
                                           inputs2: [sentence_to_glove_cove(sentence) for sentence in test_X2],
                                           labels: test_y})"""
