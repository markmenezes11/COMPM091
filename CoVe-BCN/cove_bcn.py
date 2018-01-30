# CoVe is taken from:
# B. McCann, J. Bradbury, C. Xiong, R. Socher, Learned in Translation: Contextualized Word Vectors
# https://github.com/salesforce/cove
#

import numpy as np
import tensorflow as tf

from keras.models import load_model

"""
DATASET
"""

# TODO: Get dataset, split it into train/test, and adjust max_sent_len and n_classes accordingly
max_sent_len = 256
n_classes = 3

"""
EMBEDDINGS
"""

# TODO: Load GloVe 840B 300d embeddings, e.g. with PyTorch or https://github.com/iamalbert/pytorch-wordemb or gensim

# Load CoVe model. Ported version from https://github.com/rgsachin/CoVe
# - Iinput: GloVe vectors of dimension - (<batch_size>, <sentence_len>, 300)
# - Output: CoVe vectors of dimension - (<batch_size>, <sentence_len>, 600)
# - Example: cove_model.predict(np.random.rand(1,10,300))
# - For unknown words, use a dummy value different to the dummy value used for padding - small non-zero value e.g. 1e-10 # TODO: Make sure this is implemented - 1e-10 for unknown words
cove_model = load_model('../CoVe-ported/Keras_CoVe_Python2.h5')
print("Successfully loaded CoVe model.")

# Input sequence (sentences) w is converted to sequence of vectors: w' = [GloVe(w); CoVe(w)]
def sentence_to_glove_cove(sentence):
    glove = []
    for word in sentence:
        glove.append(np.random.rand(1, 300)) # TODO: Get actual GloVe(w) embedding (or 1e-10 if word not found)
    glove = np.array(glove) # (len(sentence), 300)
    cove = cove_model.predict(glove) # (len(sentence), 600)
    glove_cove = np.concatenate([glove, cove], axis=1) # (len(sentence), 900)
    for pad in range(max_sent_len - len(sentence)):
        np.append(glove_cove, np.full((1, 900), 0.0), axis=0)
    return glove_cove # (max_sent_len, 900)

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
        feedforward_outputs.append(feedforward_activation(tf.matmul(feedforward_input, feedforward_weight) + feedforward_bias)) # TODO: Have I done this right? Or do I have to use a fully_connected_layer or something so I can reuse the same relu? ##################################################################################################################################
    return tf.stack(feedforward_outputs)
feedforward_outputs1 = feedforward(inputs1)
feedforward_outputs2 = feedforward(inputs2)
assert feedforward_outputs1.shape == (batch_size, max_sent_len, 900)
assert feedforward_outputs2.shape == (batch_size, max_sent_len, 900)

#  BiLSTM processes the resulting sequences
if same_bilstm_for_encoder:
    with tf.variable_scope("bilstm_encoder_scope") as bilstm_encoder_scope:
        bilstm_encoder_fw_cell = tf.contrib.rnn.LSTMCell(bilstm_encoder_n_hidden1, forget_bias=bilstm_encoder_forget_bias1)
        bilstm_encoder_bw_cell = tf.contrib.rnn.LSTMCell(bilstm_encoder_n_hidden1, forget_bias=bilstm_encoder_forget_bias1)
        bilstm_encoder_inputs = tf.concat((feedforward_outputs1, feedforward_outputs2), 0)
        bilstm_encoder_raw_outputs, _ = tf.nn.bidirectional_dynamic_rnn(bilstm_encoder_fw_cell, bilstm_encoder_bw_cell, bilstm_encoder_inputs, dtype=tf.float32)
        bilstm_encoder_outputs1, bilstm_encoder_outputs2 = tf.split(tf.concat([bilstm_encoder_raw_outputs[0], bilstm_encoder_raw_outputs[-1]], 2), 2, axis=0)
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
assert bilstm_encoder_outputs1.shape == (batch_size, max_sent_len, bilstm_encoder_n_hidden1 * 2)
assert bilstm_encoder_outputs2.shape == (batch_size, max_sent_len, bilstm_encoder_n_hidden2 * 2)

# Biattention mechanism [Seo et al., 2017, Xiong et al., 2017]
Xs = tf.unstack(bilstm_encoder_outputs1)
Ys = tf.unstack(bilstm_encoder_outputs2)
biattention_outputs1 = []
biattention_outputs2 = []
for X, Y in zip(Xs, Ys):
    # Affinity matrix A=XY^T
    A = tf.matmul(X, Y, adjoint_b=True)
    assert A.shape == (max_sent_len, max_sent_len)

    # Column-wise normalisation to extract attention weights
    Ax = tf.nn.softmax(A)
    Ay = tf.nn.softmax(tf.transpose(A))
    assert Ax.shape == (max_sent_len, max_sent_len)
    assert Ay.shape == (max_sent_len, max_sent_len)

    # Context summaries
    Cx = tf.matmul(Ax, X, adjoint_a=True)
    Cy = tf.matmul(Ay, X, adjoint_a=True)
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

Xys = tf.unstack(bilstm_integrate_outputs1)
Yxs = tf.unstack(bilstm_integrate_outputs2)
self_pool_outputs1 = []
self_pool_outputs2 = []
self_pool_weight1 = tf.get_variable("self_pool_weight1", shape=[UNKNOWN, UNKNOWN], initializer=tf.random_uniform_initializer(-self_pool_weight_size1, self_pool_weight_size1))
self_pool_bias1 = tf.get_variable("self_pool_bias1", shape=[UNKNOWN], initializer=tf.constant_initializer(self_pool_bias_size1))
self_pool_weight2 = tf.get_variable("self_pool_weight2", shape=[UNKNOWN, UNKNOWN], initializer=tf.random_uniform_initializer(-self_pool_weight_size2, self_pool_weight_size2))
self_pool_bias2 = tf.get_variable("self_pool_bias2", shape=[UNKNOWN], initializer=tf.constant_initializer(self_pool_bias_size2))
for Xy, Yx in zip(Xys, Yxs):
    # Self-attentive pooling
    Bx = tf.nn.softmax((tf.matmul(Xy, self_pool_weight1)) + self_pool_bias1) # TODO: Are these weights and biases? Is this how we do it? ########################################################################################################################################################################################################################
    By = tf.nn.softmax((tf.matmul(Yx, self_pool_weight2)) + self_pool_bias2) # TODO: Are these weights and biases? Is this how we do it? ########################################################################################################################################################################################################################
    x_self = tf.matmul(Xy, Bx, adjoint_a=True)
    y_self = tf.matmul(Yx, By, adjoint_a=True)

    # TODO: Combine pooled representations
    max_Xy = UNKNOWN
    mean_Xy = UNKNOWN
    min_Xy = UNKNOWN
    max_Yx = UNKNOWN
    mean_Yx = UNKNOWN
    min_Yx = UNKNOWN

    self_pool_outputs1.append(tf.concat([max_Xy, mean_Xy, min_Xy, x_self], 1))
    self_pool_outputs2.append(tf.concat([max_Yx, mean_Yx, min_Yx, y_self], 1))
self_pool_outputs1 = tf.stack(self_pool_outputs1)
self_pool_outputs2 = tf.stack(self_pool_outputs2)
assert biattention_outputs1.shape == (batch_size, max_sent_len, bilstm_encoder_n_hidden1*2*3)
assert biattention_outputs2.shape == (batch_size, max_sent_len, bilstm_encoder_n_hidden2*2*3)

# TODO: Max-out network (3 layers, batch normalised)

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
