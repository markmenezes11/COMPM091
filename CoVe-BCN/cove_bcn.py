# CoVe is taken from:
# B. McCann, J. Bradbury, C. Xiong, R. Socher, Learned in Translation: Contextualized Word Vectors
# https://github.com/salesforce/cove
#

import sys
import argparse
import random

parser = argparse.ArgumentParser(description='Replication of the CoVe Biattentive Classification Network (BCN)')
parser.add_argument("--glovepath", type=str, default="../../Word2Vec_models/GloVe/glove.840B.300d.txt", help="Path to GloVe word embeddings. Download glove.840B.300d embeddings from https://nlp.stanford.edu/projects/glove/")
parser.add_argument("--ignoregloveheader", action="store_true", default=False, help="Set this flag if the first line of the GloVe file is a header and not a (word, embedding) pair")
parser.add_argument("--covepath", type=str, default='../CoVe-ported/Keras_CoVe_Python2.h5', help="Path to the CoVe model")
parser.add_argument("--covedimensions", type=int, default=600, help="Number of dimensions in CoVe embeddings (default: 600)")
args, _ = parser.parse_known_args()

import numpy as np
import tensorflow as tf

from maxout import maxout_with_batch_norm
from keras.models import load_model

"""
EMBEDDINGS
"""

print("\nLoading GloVe model...")
f = open(args.glovepath)
glove_embeddings_dict = dict()
glove_dimensions = -1
first_line = True
for line in f:
    if first_line and args.ignoregloveheader:
        first_line = False
        continue
    values = line.split()
    word = values[0]
    embedding = np.asarray(values[1:], dtype='float32')
    if glove_dimensions == -1:
        glove_dimensions = len(embedding)
    assert glove_dimensions == len(embedding)
    glove_embeddings_dict[word] = embedding
f.close()
if len(glove_embeddings_dict) == 0 or glove_dimensions == -1:
    print("ERROR: Failed to load GloVe embeddings.")
    sys.exit(1)
print("Successfully loaded GloVe embeddings (vocab size: " + str(len(glove_embeddings_dict)) + ", dimensions: " + str(glove_dimensions) + ").")

# Load CoVe model. Ported version from https://github.com/rgsachin/CoVe
# - Input: GloVe vectors of dimension - (<batch_size>, <sentence_len>, <glove_dimensions>)
# - Output: CoVe vectors of dimension - (<batch_size>, <sentence_len>, <cove_dimensions>)
# - Example: cove_model.predict(np.random.rand(1, 10, glove_dimensions))
# - For unknown words, use a dummy value different to the dummy value used for padding - small non-zero value e.g. 1e-10
print("\nLoading CoVe model...")
cove_dimensions = args.covedimensions
glove_cove_dimensions = glove_dimensions + cove_dimensions
cove_model = load_model(args.covepath)
test = cove_model.predict(np.random.rand(1,10,300))
assert test.shape == (1, 10, cove_dimensions)
print("Successfully loaded CoVe model.")

# Input sequence (tokenized sentence) w is converted to sequence of vectors: w' = [GloVe(w); CoVe(w)]
def sentence_to_glove_cove(tokenized_sentence, max_sent_len, glove_dimensions, cove_dimensions):
    glove_embeddings = []
    for word in tokenized_sentence:
        try:
            glove_embedding = np.array(glove_embeddings_dict[word])
            assert glove_embedding.shape == (glove_dimensions,)
            glove_embeddings.append(glove_embedding)
        except KeyError:
            glove_embedding = np.full(glove_dimensions, 1e-10) # 1e-10 for unknown words, as recommended on https://github.com/rgsachin/CoVe
            assert glove_embedding.shape == (glove_dimensions,)
            glove_embeddings.append(glove_embedding)
    glove = np.array([glove_embeddings])
    assert glove.shape == (1, len(sentence), glove_dimensions)
    cove = cove_model.predict(glove)
    assert cove.shape == (1, len(sentence), cove_dimensions)
    glove_cove = np.concatenate([glove[0], cove[0]], axis=1)
    for pad in range(max_sent_len - len(sentence)):
        glove_cove = np.append(glove_cove, np.full((1, glove_dimensions + cove_dimensions), 0.0), axis=0)
    assert glove_cove.shape == (max_sent_len, glove_dimensions + cove_dimensions)
    return glove_cove

"""
DATASET
"""

# TODO: Get actual dataset, split it into train/test, and adjust max_sent_len and n_classes accordingly
max_sent_len = 256
n_classes = 4

dummy_dataset_X1 = [["This", "is", "the", "0th", "dummy", "sentence", "."],
                    ["This", "is", "the", "1st", "dummy", "sentence", "."],
                    ["This", "is", "the", "2nd", "dummy", "sentence", "."],
                    ["This", "is", "the", "3rd", "dummy", "sentence", "."],
                    ["This", "is", "the", "4th", "dummy", "sentence", "."],
                    ["This", "is", "the", "5th", "dummy", "sentence", "."],
                    ["This", "is", "the", "6th", "dummy", "sentence", "."],
                    ["This", "is", "the", "7th", "dummy", "sentence", "."],
                    ["This", "is", "the", "8th", "dummy", "sentence", "."],
                    ["This", "is", "the", "9th", "dummy", "sentence", "."]]
dummy_dataset_X2 = [["This", "is", "the", "0th", "dummy", "sentence", "."],
                    ["This", "is", "the", "1st", "dummy", "sentence", "."],
                    ["This", "is", "the", "2nd", "dummy", "sentence", "."],
                    ["This", "is", "the", "3rd", "dummy", "sentence", "."],
                    ["This", "is", "the", "4th", "dummy", "sentence", "."],
                    ["This", "is", "the", "5th", "dummy", "sentence", "."],
                    ["This", "is", "the", "6th", "dummy", "sentence", "."],
                    ["This", "is", "the", "7th", "dummy", "sentence", "."],
                    ["This", "is", "the", "8th", "dummy", "sentence", "."],
                    ["This", "is", "the", "9th", "dummy", "sentence", "."]]
dummy_dataset_y = [0, 2, 2, 1, 3, 0, 2, 1, 0, 1]

split = int(len(dummy_dataset_y) * 0.8)

train_X1 = [sentence_to_glove_cove(sentence, max_sent_len, glove_dimensions, cove_dimensions) for sentence in dummy_dataset_X1[:split]]
train_X2 = [sentence_to_glove_cove(sentence, max_sent_len, glove_dimensions, cove_dimensions) for sentence in dummy_dataset_X2[:split]]
train_y = dummy_dataset_y[:split]

test_X1 = [sentence_to_glove_cove(sentence, max_sent_len, glove_dimensions, cove_dimensions) for sentence in dummy_dataset_X1[split:]]
test_X2 = [sentence_to_glove_cove(sentence, max_sent_len, glove_dimensions, cove_dimensions) for sentence in dummy_dataset_X2[split:]]
test_y = dummy_dataset_y[split:]

"""
HYPERPARAMETERS
"""

# TODO: Tune the following parameters
hyperparameters = {
    'n_epochs': 10, # int
    'batch_size': 60, # int

    'feedforward_weight_size': 0.1, # float
    'feedforward_bias_size': 0.1, # float
    'feedforward_activation': tf.nn.relu6, # tf.nn.relu or tf.nn.relu6

    'same_bilstm_for_encoder': True, # boolean
    'bilstm_encoder_n_hidden1': 200, # int
    'bilstm_encoder_forget_bias1': 1.0, # float
    'bilstm_encoder_n_hidden2': 200, # int - only needs to be set if same_bilstm_for_encoder is False
    'bilstm_encoder_forget_bias2': 1.0, # float - only needs to be set if same_bilstm_for_encoder is False

    'bilstm_integrate_n_hidden1': 200, # int
    'bilstm_integrate_forget_bias1': 1.0, # float
    'bilstm_integrate_n_hidden2': 200, # int
    'bilstm_integrate_forget_bias2': 1.0, # float

    'self_pool_weight_size1': 0.1, # float
    'self_pool_bias_size1': 0.1, # float
    'self_pool_weight_size2': 0.1, # float
    'self_pool_bias_size2': 0.1, # float

    'bn_weight_size1': 0.1, # float
    'bn_bias_size1': 0.1, # float
    'bn_decay1': 0.999, # float
    'bn_epsilon1': 1e-3, # float
    'bn_weight_size2': 0.1, # float
    'bn_bias_size2': 0.1, # float
    'bn_decay2': 0.999, # float
    'bn_epsilon2': 1e-3, # float
    'bn_weight_size3': 0.1,  # float
    'bn_bias_size3': 0.1,  # float
    'bn_decay3': 0.999,  # float
    'bn_epsilon3': 1e-3,  # float

    'maxout_n_units1': 3200, # int
    'maxout_n_units2': 3200, # int
    'maxout_n_units3': 3200,  # int

    'optimizer': "gradientdescent", # "adam" or "gradientdescent"
    'learning_rate': 0.001, # float
    'adam_beta1': 0.9, # float (used only if optimizer == "adam")
    'adam_beta2': 0.999, # float (used only if optimizer == "adam")
    'adam_epsilon': 1e-08 # float (used only if optimizer == "adam")
}

"""
MODEL
"""

# Helper function for asserting that dimensions are correct and allowing for "None" dimensions
def dimensions_equal(dim1, dim2):
    return all([d1 == d2 or (d1 == d2) is None for d1, d2 in zip(dim1, dim2)])

def BCN(params, is_training, max_sent_len, glove_cove_dimensions):
    print("\nCreating BCN model...")

    # Takes 2 input sequences, each of the form [GloVe(w); CoVe(w)] (duplicated if only one input sequence is needed)
    inputs1 = tf.placeholder(tf.float32, shape=[None, max_sent_len, glove_cove_dimensions])
    inputs2 = tf.placeholder(tf.float32, shape=[None, max_sent_len, glove_cove_dimensions])
    labels = tf.placeholder(tf.int32, [None])  # (n_sentences, n_classes)
    assert dimensions_equal(inputs1.shape, (params['batch_size'], max_sent_len, glove_cove_dimensions))
    assert dimensions_equal(inputs2.shape, (params['batch_size'], max_sent_len, glove_cove_dimensions))

    # Feedforward network with ReLU activation, applied to each word embedding (word) in the sequence (sentence)
    with tf.variable_scope("feedforward"):
        feedforward_weight = tf.get_variable("feedforward_weight", shape=[glove_cove_dimensions, glove_cove_dimensions], initializer=tf.random_uniform_initializer(-params['feedforward_weight_size'], params['feedforward_weight_size']))
        feedforward_bias = tf.get_variable("feedforward_bias", shape=[glove_cove_dimensions], initializer=tf.constant_initializer(params['feedforward_bias_size']))
        def feedforward(feedforward_input):
            return params['feedforward_activation'](tf.matmul(feedforward_input, feedforward_weight) + feedforward_bias)
        feedforward_outputs1 = tf.map_fn(feedforward, inputs1)
        feedforward_outputs2 = tf.map_fn(feedforward, inputs2)
        assert dimensions_equal(feedforward_outputs1.shape, (params['batch_size'], max_sent_len, glove_cove_dimensions))
        assert dimensions_equal(feedforward_outputs2.shape, (params['batch_size'], max_sent_len, glove_cove_dimensions))

    # BiLSTM processes the resulting sequences
    # The BCN is symmetrical - not sure whether to use the same BiLSTM or or two separate BiLSTMs (one for each side).
    # Therefore implemented both, e.g. For SNLI you might want to use different models for inputs1 and inputs2, whereas
    # for sentiment analysis you mightwant to use the same model
    if params['same_bilstm_for_encoder']:
        with tf.variable_scope("bilstm_encoder_scope"):
            bilstm_encoder_fw_cell = tf.contrib.rnn.LSTMCell(params['bilstm_encoder_n_hidden1'], forget_bias=params['bilstm_encoder_forget_bias1'])
            bilstm_encoder_bw_cell = tf.contrib.rnn.LSTMCell(params['bilstm_encoder_n_hidden1'], forget_bias=params['bilstm_encoder_forget_bias1'])
            bilstm_encoder_inputs = tf.concat((feedforward_outputs1, feedforward_outputs2), 0)
            bilstm_encoder_raw_outputs, _ = tf.nn.bidirectional_dynamic_rnn(bilstm_encoder_fw_cell, bilstm_encoder_bw_cell, bilstm_encoder_inputs, dtype=tf.float32)
            bilstm_encoder_outputs1, bilstm_encoder_outputs2 = tf.split(tf.concat([bilstm_encoder_raw_outputs[0], bilstm_encoder_raw_outputs[-1]], 2), 2, axis=0)
            params['bilstm_encoder_n_hidden2'] = params['bilstm_encoder_n_hidden1'] # For the assert below to work
    else:
        with tf.variable_scope("bilstm_encoder_scope1"):
            bilstm_encoder_fw_cell1 = tf.contrib.rnn.LSTMCell(params['bilstm_encoder_n_hidden1'], forget_bias=params['bilstm_encoder_forget_bias1'])
            bilstm_encoder_bw_cell1 = tf.contrib.rnn.LSTMCell(params['bilstm_encoder_n_hidden1'], forget_bias=params['bilstm_encoder_forget_bias1'])
            bilstm_encoder_raw_outputs1, _ = tf.nn.bidirectional_dynamic_rnn(bilstm_encoder_fw_cell1, bilstm_encoder_bw_cell1, feedforward_outputs1, dtype=tf.float32)
            bilstm_encoder_outputs1 = tf.concat([bilstm_encoder_raw_outputs1[0], bilstm_encoder_raw_outputs1[-1]], 2)
        with tf.variable_scope("bilstm_encoder_scope2"):
            bilstm_encoder_fw_cell2 = tf.contrib.rnn.LSTMCell(params['bilstm_encoder_n_hidden2'], forget_bias=params['bilstm_encoder_forget_bias2'])
            bilstm_encoder_bw_cell2 = tf.contrib.rnn.LSTMCell(params['bilstm_encoder_n_hidden2'], forget_bias=params['bilstm_encoder_forget_bias2'])
            bilstm_encoder_raw_outputs2, _ = tf.nn.bidirectional_dynamic_rnn(bilstm_encoder_fw_cell2, bilstm_encoder_bw_cell2, params['feedforward_outputs2'], dtype=tf.float32)
            bilstm_encoder_outputs2 = tf.concat([bilstm_encoder_raw_outputs2[0], bilstm_encoder_raw_outputs2[-1]], 2)
    assert dimensions_equal(bilstm_encoder_outputs1.shape, (params['batch_size'], max_sent_len, params['bilstm_encoder_n_hidden1']*2))
    assert dimensions_equal(bilstm_encoder_outputs2.shape, (params['batch_size'], max_sent_len, params['bilstm_encoder_n_hidden2']*2))

    # Biattention mechanism [Seo et al., 2017, Xiong et al., 2017]
    def biattention(biattention_input):
        X = biattention_input[0]
        Y = biattention_input[1]

        # Affinity matrix A=XY^T
        A = tf.matmul(X, Y, transpose_b=True)
        assert dimensions_equal(A.shape, (max_sent_len, max_sent_len))

        # Column-wise normalisation to extract attention weights
        Ax = tf.nn.softmax(A)
        Ay = tf.nn.softmax(tf.transpose(A))
        assert dimensions_equal(Ax.shape, (max_sent_len, max_sent_len))
        assert dimensions_equal(Ay.shape, (max_sent_len, max_sent_len))

        # Context summaries
        Cx = tf.matmul(Ax, X, transpose_a=True)
        Cy = tf.matmul(Ay, X, transpose_a=True)
        assert dimensions_equal(Cx.shape, (max_sent_len, params['bilstm_encoder_n_hidden1']*2))
        assert dimensions_equal(Cy.shape, (max_sent_len, params['bilstm_encoder_n_hidden2']*2))

        biattention_output1 = tf.concat([X, X - Cy, tf.multiply(X, Cy)], 1)
        biattention_output2 = tf.concat([Y, Y - Cx, tf.multiply(Y, Cx)], 1)
        return biattention_output1, biattention_output2
    biattention_outputs1, biattention_outputs2 = tf.map_fn(biattention, (bilstm_encoder_outputs1, bilstm_encoder_outputs2))
    assert dimensions_equal(biattention_outputs1.shape, (params['batch_size'], max_sent_len, params['bilstm_encoder_n_hidden1']*2*3))
    assert dimensions_equal(biattention_outputs2.shape, (params['batch_size'], max_sent_len, params['bilstm_encoder_n_hidden2']*2*3))

    # Integrate with two separate one-layer BiLSTMs
    with tf.variable_scope("bilstm_integrate_scope1"):
        bilstm_integrate_fw_cell1 = tf.contrib.rnn.LSTMCell(params['bilstm_integrate_n_hidden1'], forget_bias=params['bilstm_integrate_forget_bias1'])
        bilstm_integrate_bw_cell1 = tf.contrib.rnn.LSTMCell(params['bilstm_integrate_n_hidden1'], forget_bias=params['bilstm_integrate_forget_bias1'])
        bilstm_integrate_raw_outputs1, _ = tf.nn.bidirectional_dynamic_rnn(bilstm_integrate_fw_cell1, bilstm_integrate_bw_cell1, biattention_outputs1, dtype=tf.float32)
        bilstm_integrate_outputs1 = tf.concat([bilstm_integrate_raw_outputs1[0], bilstm_integrate_raw_outputs1[-1]], 2)
    with tf.variable_scope("bilstm_integrate_scope2"):
        bilstm_integrate_fw_cell2 = tf.contrib.rnn.LSTMCell(params['bilstm_integrate_n_hidden2'], forget_bias=params['bilstm_integrate_forget_bias2'])
        bilstm_integrate_bw_cell2 = tf.contrib.rnn.LSTMCell(params['bilstm_integrate_n_hidden2'], forget_bias=params['bilstm_integrate_forget_bias2'])
        bilstm_integrate_raw_outputs2, _ = tf.nn.bidirectional_dynamic_rnn(bilstm_integrate_fw_cell2, bilstm_integrate_bw_cell2, biattention_outputs2, dtype=tf.float32)
        bilstm_integrate_outputs2 = tf.concat([bilstm_integrate_raw_outputs2[0], bilstm_integrate_raw_outputs2[-1]], 2)
    assert dimensions_equal(bilstm_integrate_outputs1.shape, (params['batch_size'], max_sent_len, params['bilstm_integrate_n_hidden1']*2))
    assert dimensions_equal(bilstm_integrate_outputs2.shape, (params['batch_size'], max_sent_len, params['bilstm_integrate_n_hidden2']*2))

    # Max, mean, min and self-attentive pooling
    with tf.variable_scope("pool"):
        self_pool_weight1 = tf.get_variable("self_pool_weight1", shape=[params['bilstm_integrate_n_hidden1']*2, 1], initializer=tf.random_uniform_initializer(-params['self_pool_weight_size1'], params['self_pool_weight_size1']))
        self_pool_bias1 = tf.get_variable("self_pool_bias1", shape=[1], initializer=tf.constant_initializer(params['self_pool_bias_size1']))
        self_pool_weight2 = tf.get_variable("self_pool_weight2", shape=[params['bilstm_integrate_n_hidden2']*2, 1], initializer=tf.random_uniform_initializer(-params['self_pool_weight_size2'], params['self_pool_weight_size2']))
        self_pool_bias2 = tf.get_variable("self_pool_bias2", shape=[1], initializer=tf.constant_initializer(params['self_pool_bias_size2']))
        def pool(pool_input):
            Xy = pool_input[0]
            Yx = pool_input[1]
            assert dimensions_equal(Xy.shape, (max_sent_len, params['bilstm_integrate_n_hidden1']*2,))
            assert dimensions_equal(Xy.shape, (max_sent_len, params['bilstm_integrate_n_hidden2']*2,))

            # Max pooling - just take the 256 "columns" in the matrix and get the max in each of them.
            max_Xy = tf.reduce_max(Xy, axis=0)
            max_Yx = tf.reduce_max(Yx, axis=0)
            assert dimensions_equal(max_Xy.shape, (params['bilstm_integrate_n_hidden1']*2,))
            assert dimensions_equal(max_Yx.shape, (params['bilstm_integrate_n_hidden2']*2,))

            # Mean pooling - just take the 256 "columns" in the matrix and get the mean in each of them.
            mean_Xy = tf.reduce_mean(Xy, axis=0)
            mean_Yx = tf.reduce_mean(Yx, axis=0)
            assert dimensions_equal(mean_Xy.shape, (params['bilstm_integrate_n_hidden1']*2,))
            assert dimensions_equal(mean_Yx.shape, (params['bilstm_integrate_n_hidden2']*2,))

            # Min pooling - just take the 256 "columns" in the matrix and get the min in each of them.
            min_Xy = tf.reduce_min(Xy, axis=0)
            min_Yx = tf.reduce_min(Yx, axis=0)
            assert dimensions_equal(min_Xy.shape, (params['bilstm_integrate_n_hidden1']*2,))
            assert dimensions_equal(min_Yx.shape, (params['bilstm_integrate_n_hidden2']*2,))

            # Self-attentive pooling
            Bx = tf.nn.softmax((tf.matmul(Xy, self_pool_weight1)) + self_pool_bias1)
            By = tf.nn.softmax((tf.matmul(Yx, self_pool_weight2)) + self_pool_bias2)
            x_self = tf.squeeze(tf.matmul(Xy, Bx, transpose_a=True))
            y_self = tf.squeeze(tf.matmul(Yx, By, transpose_a=True))
            assert dimensions_equal(x_self.shape, (params['bilstm_integrate_n_hidden1']*2,))
            assert dimensions_equal(y_self.shape, (params['bilstm_integrate_n_hidden2']*2,))

            # Combine pooled representations
            pool_output1 = tf.concat([max_Xy, mean_Xy, min_Xy, x_self], 0)
            pool_output2 = tf.concat([max_Yx, mean_Yx, min_Yx, y_self], 0)
            return pool_output1, pool_output2
        pool_outputs1, pool_outputs2 = tf.map_fn(pool, (bilstm_integrate_outputs1, bilstm_integrate_outputs2))
        assert dimensions_equal(pool_outputs1.shape, (params['batch_size'], params['bilstm_encoder_n_hidden1']*2*4))
        assert dimensions_equal(pool_outputs2.shape, (params['batch_size'], params['bilstm_encoder_n_hidden2']*2*4))

    # Max-out network (3 layers, batch normalised)
    with tf.variable_scope("maxout"):
        joined_representation = tf.concat([pool_outputs1, pool_outputs2], 1)

        maxout_weight1 = tf.get_variable("maxout_weight1", shape=[params['bilstm_encoder_n_hidden1']*2*4*2, params['bilstm_encoder_n_hidden1']*2*4*2], initializer=tf.random_uniform_initializer(-params['bn_weight_size1'], params['bn_weight_size1']))
        maxout_bias1 = tf.get_variable("maxout_bias1", shape=[params['bilstm_encoder_n_hidden1']*2*4*2], initializer=tf.constant_initializer(params['bn_bias_size1']))
        z1 = tf.matmul(joined_representation, maxout_weight1) + maxout_bias1
        maxout_outputs1 = maxout_with_batch_norm(z1, params['maxout_n_units1'], params['bn_decay1'], params['bn_epsilon1'], is_training)

        maxout_weight2 = tf.get_variable("maxout_weight2", shape=[params['bilstm_encoder_n_hidden1']*2*4*2, params['bilstm_encoder_n_hidden1']*2*4*2], initializer=tf.random_uniform_initializer(-params['bn_weight_size2'], params['bn_weight_size2']))
        maxout_bias2 = tf.get_variable("maxout_bias2", shape=[params['bilstm_encoder_n_hidden1']*2*4*2], initializer=tf.constant_initializer(params['bn_bias_size2']))
        z2 = tf.matmul(maxout_outputs1, maxout_weight2) + maxout_bias2
        maxout_ouputs2 = maxout_with_batch_norm(z2, params['maxout_n_units2'], params['bn_decay2'], params['bn_epsilon2'], is_training)

        maxout_weight3 = tf.get_variable("maxout_weight3", shape=[params['bilstm_encoder_n_hidden1']*2*4*2, n_classes], initializer=tf.random_uniform_initializer(-params['bn_weight_size3'], params['bn_weight_size3']))
        maxout_bias3 = tf.get_variable("maxout_bias3", shape=[n_classes], initializer=tf.constant_initializer(params['bn_bias_size3']))
        z3 = tf.matmul(maxout_ouputs2, maxout_weight3) + maxout_bias3
        maxout_outputs3 = maxout_with_batch_norm(z3, params['maxout_n_units3'], params['bn_decay3'], params['bn_epsilon3'], is_training)

        loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(maxout_outputs3), reduction_indices=1))

        if params['optimizer'] == "adam":
            train_step = tf.train.AdamOptimizer(params['learning_rate'], beta1=params['adam_beta1'], beta2=params['adam_beta2'], epsilon=params['adam_epsilon']).minimize(loss)
        elif params['optimizer'] == "gradientdescent":
            train_step = tf.train.GradientDescentOptimizer(params['learning_rate']).minimize(loss)
        else:
            print("ERROR: Invalid optimizer: \"" + params['optimizer'] + "\"")
            sys.exit(1)

        predict = tf.argmax(maxout_outputs3, axis=1)

    print("Successfully created BCN model.")
    return inputs1, inputs2, labels, predict, loss, train_step

"""
TRAIN
"""

# Shuffle the training data
zipped_train_data = zip(train_X1, train_X2, train_y)
random.Random(0).shuffle(list(zipped_train_data))
train_X1, train_X2, train_y = zip(*zipped_train_data)

tf.reset_default_graph()
with tf.Graph().as_default() as graph:
    inputs1, inputs2, labels, _, loss, train_step = BCN(hyperparameters, True, max_sent_len, glove_cove_dimensions)
with tf.Session(graph=graph) as sess:
    print("\nTraining model...")
    tf.global_variables_initializer().run()
    for epoch in range(hyperparameters['n_epochs']):
        print("Epoch " + str(epoch+1) + " of " + str(hyperparameters['n_epochs']))
        for i in range(len(train_y) // hyperparameters['batch_size']):
            batch_X1 = train_X1[i * hyperparameters['batch_size']: (i+1) * hyperparameters['batch_size']]
            batch_X2 = train_X2[i * hyperparameters['batch_size']: (i + 1) * hyperparameters['batch_size']]
            batch_y = train_y[i * hyperparameters['batch_size']: (i + 1) * hyperparameters['batch_size']]
            _, loss = sess.run([train_step, loss], feed_dict={inputs1: batch_X1, inputs2: batch_X2, labels : batch_y})
    saved_model = tf.train.Saver().save(sess, 'model/model')

"""
TEST
"""

tf.reset_default_graph()
with tf.Graph().as_default() as graph:
    inputs1, inputs2, labels, predict, _, _ = BCN(hyperparameters, False, max_sent_len, glove_cove_dimensions)
with tf.Session(graph=graph) as sess:
    print("\nTesting model...")
    tf.global_variables_initializer().run()
    saver = tf.train.import_meta_graph('model/model.meta')
    saver.restore(sess, 'model/model')
    predicted = list(sess.run(predict, feed_dict={inputs1: test_X1, inputs2: test_X2, labels: test_y}))
    accuracy = sum([p == a for p, a in zip(predicted, test_y)]) / float(len(test_y))
    print("Predictions: " + str(predicted))
    print("Actual:      " + str(test_y))
    print("Accuracy:    " + str(accuracy))
