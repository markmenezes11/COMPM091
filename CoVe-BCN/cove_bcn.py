import numpy as np
import tensorflow as tf

from keras.models import load_model

# Load CoVe model:
# - Iinput: GloVe vectors of dimension - (<batch_size>, <sentence_len>, 300)
# - Output: CoVe vectors of dimension - (<batch_size>, <sentence_len>, 600)
# - Example: cove_model.predict(np.random.rand(1,10,300))
# - For unknown words, use a dummy value different to the dummy value used for padding, - a small non-zero value e.g. 1e-10
cove_model = load_model('../CoVe-ported/Keras_CoVe_Python2.h5')

# TODO: Get actual training data and test data and use it below

# TODO: Use pytorch (or otherwise) to load GloVe embeddings

max_sent_len = 100 # TODO: Get actual max sentence length and implement padding below

"""
MODEL
"""

# Input sequences (sentences) w are converted to sequences of vectors: w' = [GloVe(w); CoVe(w)]
# Takes 2 input sequences but they are duplicated if only one input sequence is needed
print("\n##### CONVERT TO [GLOVE(W); COVE(W)] #####")
# TODO: Change glove1 and glove2 below to actual GloVe(w) embeddings and handle padding correctly (see above)
glove1 = np.random.rand(10000, max_sent_len, 300) # (n_sentences, max_sent_len, 300)
glove2 = np.random.rand(10000, max_sent_len, 300) # (n_sentences, max_sent_len, 300)
cove1 = cove_model.predict(glove1) # (n_sentences, max_sent_len, 600)
cove2 = cove_model.predict(glove2) # (n_sentences, max_sent_len, 600)
w1 = np.concatenate([glove1, cove1], axis=2) # (n_sentences, max_sent_len, 900)
w2 = np.concatenate([glove2, cove2], axis=2) # (n_sentences, max_sent_len, 900)

# Feedforward network with ReLU activation applied to each word embedding in the sequence
print("\n##### FEEDFORWARD NETWORK #####")
inputs = tf.placeholder(tf.float32, shape=[None, 900])
weight = tf.get_variable("weight", shape=[900, 900], initializer=tf.random_uniform_initializer(-0.1, 0.1)) # TODO: Choose weight
bias = tf.get_variable("bias", shape=[max_sent_len * 900], initializer=tf.constant_initializer(0.1)) # TODO: Choose bias
logits = tf.nn.relu6(tf.matmul(inputs, weight) + bias) # TODO: Choose relu





#labels = tf.placeholder(tf.float32, [None, 900])
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
#train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy) # TODO: Choose optimiser / params
#predict = tf.argmax(logits, axis=1)








"""
TRAIN
"""

train_data = []  # TODO: Get train data - ???
n_epochs = 100  # TODO: Choose n_epochs
batch_size = 100  # TODO: Choose batch_size
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()
    for epoch in range(n_epochs):
        print("Epoch " + str(epoch+1) + " of " + str(n_epochs))
        for i in range(len(train_data) // batch_size):
            batch_inputs = train_data[i * batch_size: (i+1) * batch_size]
            sess.run(train_step, feed_dict={inputs: batch_inputs, labels : batch_inputs})

"""
TEST
"""

# Input sequences (sentences) w are converted to sequences of vectors: w' = [GloVe(w); CoVe(w)]
# Takes 2 input sequences but they are duplicated if only one input sequence is needed
# TODO: Change glove1 and glove2 below to actual GloVe(w) embeddings and handle padding correctly (see above)
glove1 = np.random.rand(1, max_sent_len, 300) # (n_sentences, max_sent_len, 300)
glove2 = np.random.rand(1, max_sent_len, 300) # (n_sentences, max_sent_len, 300)
cove1 = cove_model.predict(glove1) # (n_sentences, max_sent_len, 600)
cove2 = cove_model.predict(glove2) # (n_sentences, max_sent_len, 600)
w1 = np.concatenate([glove1, cove1], axis=2) # (n_sentences, max_sent_len, 900)
w2 = np.concatenate([glove2, cove2], axis=2) # (n_sentences, max_sent_len, 900)

# Feedforward network with ReLU activation applied to each word embedding in the sequence
outputs1 = []
outputs2 = []
with tf.Session() as sess:
    for word_vector in w1:
        outputs.append(sess.run(predict, feed_dict={inputs: word_vector, labels: word_vector}))

    for word_vector in w2:
        outputs.append(sess.run(predict, feed_dict={inputs: word_vector, labels: word_vector}))
outputs1 = np.array(outputs1)
outputs2 = np.array(outputs2)

#  Bidirectional LSTM processes the resulting sequences to obtain task-specific representations
