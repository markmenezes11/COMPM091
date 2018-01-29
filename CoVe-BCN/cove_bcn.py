# CoVe is taken from:
# B. McCann, J. Bradbury, C. Xiong, R. Socher, Learned in Translation: Contextualized Word Vectors
# https://github.com/salesforce/cove
#

import numpy as np
import tensorflow as tf

from keras.models import load_model

"""
SATASET
"""

# TODO: Get dataset, split it into train/test, and use it below

"""
EMBEDDINGS
"""

# TODO: Load GloVe 840B 300d embeddings, e.g. with PyTorch or https://github.com/iamalbert/pytorch-wordemb

# Load CoVe model. Ported version from https://github.com/rgsachin/CoVe
# - Iinput: GloVe vectors of dimension - (<batch_size>, <sentence_len>, 300)
# - Output: CoVe vectors of dimension - (<batch_size>, <sentence_len>, 600)
# - Example: cove_model.predict(np.random.rand(1,10,300))
# - For unknown words, use a dummy value different to the dummy value used for padding - small non-zero value e.g. 1e-10 # TODO: Make sure this is implemented - 1e-10 for unknown words
cove_model = load_model('../CoVe-ported/Keras_CoVe_Python2.h5')
print("Loaded CoVe model.\n")

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

debug = True
max_sent_len = 100 # TODO: Get max sentence length depending on dataset
n_classes = 3 # TODO: Get n_classes depending on dataset
n_epochs = 100  # TODO: Choose n_epochs
batch_size = 60  # TODO: Choose batch_size
bilstm_n_hidden = 256
bilstm_forget_bias = 1.0

"""
MODEL
"""

# Takes 2 input sequences, each of the form [GloVe(w); CoVe(w)] (duplicated if only one input sequence is needed)
inputs1 = tf.placeholder(tf.float32, shape=[batch_size, max_sent_len, 900]) # (n_sentences, max_sent_len, 900)
inputs2 = tf.placeholder(tf.float32, shape=[batch_size, max_sent_len, 900]) # (n_sentences, max_sent_len, 900)
labels = tf.placeholder(tf.float32, [None, n_classes])  # (n_sentences, n_classes)
if debug: print("inputs1 shape: " + str(inputs1.shape))

# Feedforward network with ReLU activation applied to each word embedding in the sequence
weight = tf.get_variable("weight", shape=[900, 900], initializer=tf.random_uniform_initializer(-0.1, 0.1))  # TODO: Choose weight
bias = tf.get_variable("bias", shape=[900], initializer=tf.constant_initializer(0.1))  # TODO: Choose bias
def feedforward(feedforward_inputs): # TODO: Same or different feedforward network for inputs1 and inputs2?
    feedforward_inputs_unstacked = tf.unstack(feedforward_inputs) # Split on axis 0 into (max_sent_len, 900)
    feedforward_outputs = []
    for feedforward_input in feedforward_inputs_unstacked: # Loop through each sentence
        feedforward_outputs.append(tf.nn.relu6(tf.matmul(feedforward_input, weight) + bias)) # TODO: Choose ReLU
    return tf.stack(feedforward_outputs)
feedforward_outputs1 = feedforward(inputs1)
feedforward_outputs2 = feedforward(inputs2)
if debug: print("feedforward_outputs1 shape: " + str(feedforward_outputs1.shape))

#  Bidirectional LSTM processes the resulting sequences to obtain task-specific representations
fw_cell = tf.contrib.rnn.LSTMCell(bilstm_n_hidden, forget_bias=bilstm_forget_bias)
bw_cell = tf.contrib.rnn.LSTMCell(bilstm_n_hidden, forget_bias=bilstm_forget_bias)
def bidirectional_lstm(bidirectional_lstm_inputs): # TODO: Same or different bilstm network for inputs1 and inputs2?
    bilstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, bidirectional_lstm_inputs, dtype=tf.float32)
    return tf.concat([bilstm_outputs[0], bilstm_outputs[-1]], 2) # TODO: Do we concatenate the outputs of each bilstm direction like this?
bilstm_outputs1 = bidirectional_lstm(feedforward_outputs1)
bilstm_outputs2 = bidirectional_lstm(feedforward_outputs2)
if debug: print("bilstm_outputs1 shape: " + str(bilstm_outputs1.shape))

# Biattention mechanism [Seo et al., 2017, Xiong et al., 2017]
def biattention(Xs, Ys):
    Xs_unstacked = tf.unstack(Xs)
    Ys_unstacked = tf.unstack(Ys)
    biattention_outputs1 = []
    biattention_outputs2 = []
    for X, Y in zip(Xs_unstacked, Ys_unstacked):
        # TODO: Do mathsy stuff and append results to biattention_outputs1 (X result) and biattention_outputs2 (Y result)
    return tf.stack(biattention_outputs1), tf.stack(biattention_outputs2)
biattention_outputs1, biattention_outputs2 = biattention(bilstm_outputs1, bilstm_outputs2)
if debug: print("biattention_outputs1 shape: " + str(biattention_outputs1.shape))

# TODO: The rest of the BCN
# TODO: train_step and predict variable for the code below to use
# TODO: Uncomment the code below

"""
TRAIN
"""

"""train_X1 = [["This", "is", "a", "train", "sentence", "."]] # TODO: Get train data
train_X2 = [["This", "is", "a", "train", "sentence", "."]] # TODO: Get train data
train_y = [0] # TODO: Get train data

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
