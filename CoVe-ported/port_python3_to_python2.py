from keras.models import Sequential, Model, Input
from keras.layers import Dense, Activation, Bidirectional
from keras.layers import LSTM, Multiply, Lambda
from keras.layers.core import Masking
from keras import backend as K

#Building Keras MTLSTM model without short-cut fix for keras masking + Bidirectional issue
keras_model = Sequential()
keras_model.add(Masking(mask_value=0.,input_shape=(None,300)))
keras_model.add(Bidirectional(LSTM(300, return_sequences=True, recurrent_activation='sigmoid', name='lstm1'),name='bidir_1'))
keras_model.add(Bidirectional(LSTM(300, return_sequences=True, recurrent_activation='sigmoid', name='lstm2'),name='bidir_2'))

#Building Keras MTLSTM model with short-cut fix for keras masking + Bidirectional issue
x = Input(shape=(None,300))
y = Masking(mask_value=0.,input_shape=(None,300))(x)
y = Bidirectional(LSTM(300, return_sequences=True, recurrent_activation='sigmoid', name='lstm1'),name='bidir_1')(y)
y = Bidirectional(LSTM(300, return_sequences=True, recurrent_activation='sigmoid', name='lstm2'),name='bidir_2')(y)

# These 2 layer are short-cut fix for the issue -
y_rev_mask_fix = Lambda(lambda x: K.cast(K.any(K.not_equal(x, 0.), axis=-1, keepdims=True), K.floatx()))(x)
y = Multiply()([y,y_rev_mask_fix])

keras_model = Model(inputs=x,outputs=y)

# Load the Python3 port of the model - MAKE SURE THIS FILE EXISTS BEFORE RUNNING THE SCRIPT - GET IT FROM https://github.com/rgsachin/CoVe
keras_model.load_weights('Keras_CoVe.h5')

# Save a new Python2 port of the model
keras_model.save('Keras_CoVe_Python2.h5')

print("Done")