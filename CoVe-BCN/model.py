# CoVe is taken from:
# B. McCann, J. Bradbury, C. Xiong, R. Socher, Learned in Translation: Contextualized Word Vectors
# https://github.com/salesforce/cove
#

import sys

import numpy as np
import tensorflow as tf

# Helper function for asserting that dimensions are correct and allowing for "None" dimensions
def dimensions_equal(dim1, dim2):
    return all([d1 == d2 or (d1 == d2) is None for d1, d2 in zip(dim1, dim2)])

class BCN:
    def __init__(self, params, n_classes, max_sent_len, glove_cove_dimensions):
        self.params = params
        self.n_classes = n_classes
        self.max_sent_len = max_sent_len
        self.glove_cove_dimensions = glove_cove_dimensions

    def create_model(self, is_training):
        print("\nCreating BCN model...")

        # Takes 2 input sequences, each of the form [GloVe(w); CoVe(w)] (duplicated if only one input sequence is needed)
        inputs1 = tf.placeholder(tf.float32, shape=[None, self.max_sent_len, self.glove_cove_dimensions])
        inputs2 = tf.placeholder(tf.float32, shape=[None, self.max_sent_len, self.glove_cove_dimensions])
        labels = tf.placeholder(tf.int32, [None])  # (n_sentences, n_classes)
        assert dimensions_equal(inputs1.shape, (self.params['batch_size'], self.max_sent_len, self.glove_cove_dimensions))
        assert dimensions_equal(inputs2.shape, (self.params['batch_size'], self.max_sent_len, self.glove_cove_dimensions))

        # Feedforward network with ReLU activation, applied to each word embedding (word) in the sequence (sentence)
        feedforward_weight = tf.get_variable("feedforward_weight", shape=[self.glove_cove_dimensions, self.glove_cove_dimensions], initializer=tf.random_uniform_initializer(-self.params['feedforward_weight_size'], self.params['feedforward_weight_size']))
        feedforward_bias = tf.get_variable("feedforward_bias", shape=[self.glove_cove_dimensions], initializer=tf.constant_initializer(self.params['feedforward_bias_size']))
        with tf.variable_scope("feedforward"):
            def feedforward(feedforward_input):
                if self.params['feedforward_activation'] == "ReLU":
                    return tf.nn.relu(tf.matmul(feedforward_input, feedforward_weight) + feedforward_bias)
                elif self.params['feedforward_activation'] == "ReLU6":
                    return tf.nn.relu6(tf.matmul(feedforward_input, feedforward_weight) + feedforward_bias)
                else:
                    print("ERROR: Invalid feedforward activation: \"" + self.params['feedforward_activation'] + "\"")
                    sys.exit(1)
            feedforward_outputs1 = tf.map_fn(feedforward, inputs1)
            feedforward_outputs2 = tf.map_fn(feedforward, inputs2)
            assert dimensions_equal(feedforward_outputs1.shape, (self.params['batch_size'], self.max_sent_len, self.glove_cove_dimensions))
            assert dimensions_equal(feedforward_outputs2.shape, (self.params['batch_size'], self.max_sent_len, self.glove_cove_dimensions))

        # BiLSTM processes the resulting sequences
        # The BCN is symmetrical - not sure whether to use the same BiLSTM or or two separate BiLSTMs (one for each side).
        # Therefore implemented both, e.g. For SNLI you might want to use different models for inputs1 and inputs2, whereas
        # for sentiment analysis you mightwant to use the same model
        if self.params['same_bilstm_for_encoder']:
            with tf.variable_scope("bilstm_encoder_scope"):
                bilstm_encoder_fw_cell = tf.contrib.rnn.LSTMCell(self.params['bilstm_encoder_n_hidden'], forget_bias=self.params['bilstm_encoder_forget_bias1'])
                bilstm_encoder_bw_cell = tf.contrib.rnn.LSTMCell(self.params['bilstm_encoder_n_hidden'], forget_bias=self.params['bilstm_encoder_forget_bias1'])
                bilstm_encoder_inputs = tf.concat((feedforward_outputs1, feedforward_outputs2), 0)
                bilstm_encoder_raw_outputs, _ = tf.nn.bidirectional_dynamic_rnn(bilstm_encoder_fw_cell, bilstm_encoder_bw_cell, bilstm_encoder_inputs, dtype=tf.float32)
                bilstm_encoder_outputs1, bilstm_encoder_outputs2 = tf.split(tf.concat([bilstm_encoder_raw_outputs[0], bilstm_encoder_raw_outputs[-1]], 2), 2, axis=0)
        else:
            with tf.variable_scope("bilstm_encoder_scope1"):
                bilstm_encoder_fw_cell1 = tf.contrib.rnn.LSTMCell(self.params['bilstm_encoder_n_hidden'], forget_bias=self.params['bilstm_encoder_forget_bias1'])
                bilstm_encoder_bw_cell1 = tf.contrib.rnn.LSTMCell(self.params['bilstm_encoder_n_hidden'], forget_bias=self.params['bilstm_encoder_forget_bias1'])
                bilstm_encoder_raw_outputs1, _ = tf.nn.bidirectional_dynamic_rnn(bilstm_encoder_fw_cell1, bilstm_encoder_bw_cell1, feedforward_outputs1, dtype=tf.float32)
                bilstm_encoder_outputs1 = tf.concat([bilstm_encoder_raw_outputs1[0], bilstm_encoder_raw_outputs1[-1]], 2)
            with tf.variable_scope("bilstm_encoder_scope2"):
                bilstm_encoder_fw_cell2 = tf.contrib.rnn.LSTMCell(self.params['bilstm_encoder_n_hidden'], forget_bias=self.params['bilstm_encoder_forget_bias2'])
                bilstm_encoder_bw_cell2 = tf.contrib.rnn.LSTMCell(self.params['bilstm_encoder_n_hidden'], forget_bias=self.params['bilstm_encoder_forget_bias2'])
                bilstm_encoder_raw_outputs2, _ = tf.nn.bidirectional_dynamic_rnn(bilstm_encoder_fw_cell2, bilstm_encoder_bw_cell2, feedforward_outputs2, dtype=tf.float32)
                bilstm_encoder_outputs2 = tf.concat([bilstm_encoder_raw_outputs2[0], bilstm_encoder_raw_outputs2[-1]], 2)
        assert dimensions_equal(bilstm_encoder_outputs1.shape, (self.params['batch_size'], self.max_sent_len, self.params['bilstm_encoder_n_hidden']*2))
        assert dimensions_equal(bilstm_encoder_outputs2.shape, (self.params['batch_size'], self.max_sent_len, self.params['bilstm_encoder_n_hidden']*2))

        # Biattention mechanism [Seo et al., 2017, Xiong et al., 2017]
        def biattention(biattention_input):
            X = biattention_input[0]
            Y = biattention_input[1]

            # Affinity matrix A=XY^T
            A = tf.matmul(X, Y, transpose_b=True)
            assert dimensions_equal(A.shape, (self.max_sent_len, self.max_sent_len))

            # Column-wise normalisation to extract attention weights
            Ax = tf.nn.softmax(A)
            Ay = tf.nn.softmax(tf.transpose(A))
            assert dimensions_equal(Ax.shape, (self.max_sent_len, self.max_sent_len))
            assert dimensions_equal(Ay.shape, (self.max_sent_len, self.max_sent_len))

            # Context summaries
            Cx = tf.matmul(Ax, X, transpose_a=True)
            Cy = tf.matmul(Ay, X, transpose_a=True)
            assert dimensions_equal(Cx.shape, (self.max_sent_len, self.params['bilstm_encoder_n_hidden']*2))
            assert dimensions_equal(Cy.shape, (self.max_sent_len, self.params['bilstm_encoder_n_hidden']*2))

            biattention_output1 = tf.concat([X, X - Cy, tf.multiply(X, Cy)], 1)
            biattention_output2 = tf.concat([Y, Y - Cx, tf.multiply(Y, Cx)], 1)
            return biattention_output1, biattention_output2
        biattention_outputs1, biattention_outputs2 = tf.map_fn(biattention, (bilstm_encoder_outputs1, bilstm_encoder_outputs2))
        assert dimensions_equal(biattention_outputs1.shape, (self.params['batch_size'], self.max_sent_len, self.params['bilstm_encoder_n_hidden']*2*3))
        assert dimensions_equal(biattention_outputs2.shape, (self.params['batch_size'], self.max_sent_len, self.params['bilstm_encoder_n_hidden']*2*3))

        # Integrate with two separate one-layer BiLSTMs
        with tf.variable_scope("bilstm_integrate_scope1"):
            bilstm_integrate_fw_cell1 = tf.contrib.rnn.LSTMCell(self.params['bilstm_integrate_n_hidden'], forget_bias=self.params['bilstm_integrate_forget_bias1'])
            bilstm_integrate_bw_cell1 = tf.contrib.rnn.LSTMCell(self.params['bilstm_integrate_n_hidden'], forget_bias=self.params['bilstm_integrate_forget_bias1'])
            bilstm_integrate_raw_outputs1, _ = tf.nn.bidirectional_dynamic_rnn(bilstm_integrate_fw_cell1, bilstm_integrate_bw_cell1, biattention_outputs1, dtype=tf.float32)
            bilstm_integrate_outputs1 = tf.concat([bilstm_integrate_raw_outputs1[0], bilstm_integrate_raw_outputs1[-1]], 2)
        with tf.variable_scope("bilstm_integrate_scope2"):
            bilstm_integrate_fw_cell2 = tf.contrib.rnn.LSTMCell(self.params['bilstm_integrate_n_hidden'], forget_bias=self.params['bilstm_integrate_forget_bias2'])
            bilstm_integrate_bw_cell2 = tf.contrib.rnn.LSTMCell(self.params['bilstm_integrate_n_hidden'], forget_bias=self.params['bilstm_integrate_forget_bias2'])
            bilstm_integrate_raw_outputs2, _ = tf.nn.bidirectional_dynamic_rnn(bilstm_integrate_fw_cell2, bilstm_integrate_bw_cell2, biattention_outputs2, dtype=tf.float32)
            bilstm_integrate_outputs2 = tf.concat([bilstm_integrate_raw_outputs2[0], bilstm_integrate_raw_outputs2[-1]], 2)
        assert dimensions_equal(bilstm_integrate_outputs1.shape, (self.params['batch_size'], self.max_sent_len, self.params['bilstm_integrate_n_hidden']*2))
        assert dimensions_equal(bilstm_integrate_outputs2.shape, (self.params['batch_size'], self.max_sent_len, self.params['bilstm_integrate_n_hidden']*2))

        # Max, mean, min and self-attentive pooling
        with tf.variable_scope("pool"):
            self_pool_weight1 = tf.get_variable("self_pool_weight1", shape=[self.params['bilstm_integrate_n_hidden']*2, 1], initializer=tf.random_uniform_initializer(-self.params['self_pool_weight_size1'], self.params['self_pool_weight_size1']))
            self_pool_bias1 = tf.get_variable("self_pool_bias1", shape=[1], initializer=tf.constant_initializer(self.params['self_pool_bias_size1']))
            self_pool_weight2 = tf.get_variable("self_pool_weight2", shape=[self.params['bilstm_integrate_n_hidden']*2, 1], initializer=tf.random_uniform_initializer(-self.params['self_pool_weight_size2'], self.params['self_pool_weight_size2']))
            self_pool_bias2 = tf.get_variable("self_pool_bias2", shape=[1], initializer=tf.constant_initializer(self.params['self_pool_bias_size2']))
            def pool(pool_input):
                Xy = pool_input[0]
                Yx = pool_input[1]
                assert dimensions_equal(Xy.shape, (self.max_sent_len, self.params['bilstm_integrate_n_hidden']*2,))
                assert dimensions_equal(Xy.shape, (self.max_sent_len, self.params['bilstm_integrate_n_hidden']*2,))

                # Max pooling - just take the (max_sent_len) "columns" in the matrix and get the max in each of them.
                max_Xy = tf.reduce_max(Xy, axis=0)
                max_Yx = tf.reduce_max(Yx, axis=0)
                assert dimensions_equal(max_Xy.shape, (self.params['bilstm_integrate_n_hidden']*2,))
                assert dimensions_equal(max_Yx.shape, (self.params['bilstm_integrate_n_hidden']*2,))

                # Mean pooling - just take the (max_sent_len) "columns" in the matrix and get the mean in each of them.
                mean_Xy = tf.reduce_mean(Xy, axis=0)
                mean_Yx = tf.reduce_mean(Yx, axis=0)
                assert dimensions_equal(mean_Xy.shape, (self.params['bilstm_integrate_n_hidden']*2,))
                assert dimensions_equal(mean_Yx.shape, (self.params['bilstm_integrate_n_hidden']*2,))

                # Min pooling - just take the (max_sent_len) "columns" in the matrix and get the min in each of them.
                min_Xy = tf.reduce_min(Xy, axis=0)
                min_Yx = tf.reduce_min(Yx, axis=0)
                assert dimensions_equal(min_Xy.shape, (self.params['bilstm_integrate_n_hidden']*2,))
                assert dimensions_equal(min_Yx.shape, (self.params['bilstm_integrate_n_hidden']*2,))

                # Self-attentive pooling
                Bx = tf.nn.softmax((tf.matmul(Xy, self_pool_weight1)) + self_pool_bias1)
                By = tf.nn.softmax((tf.matmul(Yx, self_pool_weight2)) + self_pool_bias2)
                assert dimensions_equal(Bx.shape, (self.max_sent_len, 1))
                assert dimensions_equal(By.shape, (self.max_sent_len, 1))
                x_self = tf.squeeze(tf.matmul(Xy, Bx, transpose_a=True))
                y_self = tf.squeeze(tf.matmul(Yx, By, transpose_a=True))
                assert dimensions_equal(x_self.shape, (self.params['bilstm_integrate_n_hidden']*2,))
                assert dimensions_equal(y_self.shape, (self.params['bilstm_integrate_n_hidden']*2,))

                # Combine pooled representations
                pool_output1 = tf.concat([max_Xy, mean_Xy, min_Xy, x_self], 0)
                pool_output2 = tf.concat([max_Yx, mean_Yx, min_Yx, y_self], 0)
                return pool_output1, pool_output2
            pool_outputs1, pool_outputs2 = tf.map_fn(pool, (bilstm_integrate_outputs1, bilstm_integrate_outputs2))
            assert dimensions_equal(pool_outputs1.shape, (self.params['batch_size'], self.params['bilstm_integrate_n_hidden']*2*4))
            assert dimensions_equal(pool_outputs2.shape, (self.params['batch_size'], self.params['bilstm_integrate_n_hidden']*2*4))

        # Max-out network (3 D->D batch-normalised maxout layers, followed by a softmax)
        with tf.variable_scope("maxout"):
            joined_representation = tf.concat([pool_outputs1, pool_outputs2], 1)
            assert dimensions_equal(joined_representation.shape, (self.params['batch_size'], self.params['bilstm_integrate_n_hidden']*2*4*2))

            # tf.contrib.layers.axout wrongly outputs a tensor of unknown shape. I have wrapped it in this function to fix that.
            # Source: https://github.com/tensorflow/tensorflow/issues/16225 https://github.com/tensorflow/tensorflow/pull/16114/files
            def maxout_patched(inputs, num_units, axis=-1, name=None):
                outputs = tf.contrib.layers.maxout(inputs, num_units, axis, name)
                shape = inputs.get_shape().as_list()
                shape[axis] = num_units
                outputs.set_shape(shape)
                return outputs

            bn1 = tf.layers.batch_normalization(joined_representation, momentum=self.params['bn_decay1'], epsilon=self.params['bn_epsilon1'], center=True, scale=True, training=is_training, name='bn1')
            assert dimensions_equal(bn1.shape, (self.params['batch_size'], self.params['bilstm_integrate_n_hidden']*2*4*2))
            maxout_outputs1 = maxout_patched(bn1, self.params['bilstm_integrate_n_hidden']*2*4*2) # n_units must be a factor of number of features (bilstm_integrate_n_hidden*2*4*2)
            assert dimensions_equal(maxout_outputs1.shape, (self.params['batch_size'], self.params['bilstm_integrate_n_hidden']*2*4*2))

            bn2 = tf.layers.batch_normalization(maxout_outputs1, momentum=self.params['bn_decay2'], epsilon=self.params['bn_epsilon2'], center=True, scale=True, training=is_training, name='bn2')
            assert dimensions_equal(bn2.shape, (self.params['batch_size'], self.params['bilstm_integrate_n_hidden']*2*4*2))
            maxout_outputs2 = maxout_patched(bn2, self.params['bilstm_integrate_n_hidden']*2*4*2) # n_units must be a factor of number of features (bilstm_integrate_n_hidden*2*4*2)
            assert dimensions_equal(maxout_outputs2.shape, (self.params['batch_size'], self.params['bilstm_integrate_n_hidden']*2*4*2))

            bn3 = tf.layers.batch_normalization(maxout_outputs2, momentum=self.params['bn_decay3'], epsilon=self.params['bn_epsilon3'], center=True, scale=True, training=is_training, name='bn3')
            assert dimensions_equal(bn3.shape, (self.params['batch_size'], self.params['bilstm_integrate_n_hidden']*2*4*2))
            maxout_outputs3 = maxout_patched(bn3, self.params['bilstm_integrate_n_hidden']*2*4*2)
            assert dimensions_equal(maxout_outputs3.shape, (self.params['batch_size'], self.params['bilstm_integrate_n_hidden']*2*4*2))

            softmax_weight = tf.get_variable("softmax_weight", shape=[self.params['bilstm_integrate_n_hidden']*2*4*2, self.n_classes], initializer=tf.random_uniform_initializer(-self.params['softmax_weight_size'], self.params['softmax_weight_size']))
            softmax_bias = tf.get_variable("softmax_bias", shape=[self.n_classes], initializer=tf.constant_initializer(self.params['softmax_bias_size']))
            logits = tf.nn.softmax(tf.matmul(maxout_outputs3, softmax_weight) + softmax_bias)
            assert dimensions_equal(logits.shape, (self.params['batch_size'], self.n_classes))

            one_hot_labels = tf.one_hot(labels, self.n_classes, dtype=tf.float32)
            assert dimensions_equal(one_hot_labels.shape, (self.params['batch_size'], self.n_classes))
            cost = tf.reduce_mean(-tf.reduce_sum(tf.cast(one_hot_labels, tf.float32) * tf.log(logits), reduction_indices=1))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops): # Must attach update_ops to train_step for batch normalization to work properly
                if self.params['optimizer'] == "adam":
                    train_step = tf.train.AdamOptimizer(self.params['learning_rate'], beta1=self.params['adam_beta1'], beta2=self.params['adam_beta2'], epsilon=self.params['adam_epsilon']).minimize(loss)
                elif self.params['optimizer'] == "gradientdescent":
                    train_step = tf.train.GradientDescentOptimizer(self.params['learning_rate']).minimize(cost)
                else:
                    print("ERROR: Invalid optimizer: \"" + self.params['optimizer'] + "\".")
                    sys.exit(1)

            predict = tf.argmax(logits, axis=1)

        print("Successfully created BCN model.")
        return inputs1, inputs2, labels, predict, cost, train_step

    def dry_run(self):
        tf.reset_default_graph()
        with tf.Graph().as_default() as graph:
            return self.create_model(is_training=True)

    def train(self, data):
        tf.reset_default_graph()
        with tf.Graph().as_default() as graph:
            inputs1, inputs2, labels, _, cost, train_step = self.create_model(is_training=True)
        with tf.Session(graph=graph) as sess:
            print("\nTraining model...")
            tf.global_variables_initializer().run()
            train_data_len = len(data['train']['y'])
            indexes = np.random.permutation(train_data_len)
            total_batches = train_data_len // self.params['batch_size']
            milestones = {int(total_batches * 0.1): "10%", int(total_batches * 0.2): "20%",
                          int(total_batches * 0.3): "30%", int(total_batches * 0.4): "40%",
                          int(total_batches * 0.5): "50%", int(total_batches * 0.6): "60%",
                          int(total_batches * 0.7): "70%", int(total_batches * 0.8): "80%",
                          int(total_batches * 0.9): "90%", total_batches: "100%"}
            for epoch in range(self.params['n_epochs']):
                print("  Epoch " + str(epoch + 1) + " of " + str(self.params['n_epochs']))
                done = 0
                for i in range(total_batches):
                    batch_indexes = indexes[i * self.params['batch_size']: (i + 1) * self.params['batch_size']]
                    batch_X1 = np.take(data['train']['X1'], batch_indexes, axis=0)
                    batch_X2 = np.take(data['train']['X2'], batch_indexes, axis=0)
                    batch_y = np.take(data['train']['y'], batch_indexes, axis=0)
                    _, loss = sess.run([train_step, cost], feed_dict={inputs1: batch_X1, inputs2: batch_X2, labels: batch_y})
                    done += 1
                    if done in milestones:
                        print("    " + milestones[done])
            saved_model = tf.train.Saver().save(sess, 'model/model')
            print("Finished training model. Model is saved in the model folder.")

    def test(self, data):
        tf.reset_default_graph()
        with tf.Graph().as_default() as graph:
            inputs1, inputs2, labels, predict, _, _ = self.create_model(is_training=False)
        with tf.Session(graph=graph) as sess:
            print("\nTesting model...")
            tf.global_variables_initializer().run()
            saver = tf.train.import_meta_graph('model/model.meta')
            saver.restore(sess, 'model/model')
            predicted = list(sess.run(predict, feed_dict={inputs1: data['test']['X1'], inputs2: data['test']['X2'],
                                                          labels: data['test']['y']}))
            accuracy = sum([p == a for p, a in zip(predicted, data['test']['y'])]) / float(len(data['test']['y']))
            #print("Predictions: " + str(predicted))
            #print("Actual:      " + str(data['test']['y']))
            print("Accuracy:    " + str(accuracy))
