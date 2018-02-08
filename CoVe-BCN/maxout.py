# This is a modified version of the tf.contrib.layers.maxout class:
#
#   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/layers/maxout.py
#
# It has been modified to contain batch normalization.
# =============================================================================
#  Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# pylint: disable=unused-import,g-bad-import-order
"""Contains the maxout layer
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_array_ops

import tensorflow as tf

from tensorflow.python.layers import base

# This batch_norm_wrapper function was taken from:
#   https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
# This is a simpler version of Tensorflow's 'official' version. See:
#   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py#L102
def batch_norm_wrapper(inputs, decay, epsilon, is_training):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs, [0])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
    else:
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

def maxout_with_batch_norm(inputs, num_units, decay, epsilon, is_training, axis=-1, name=None):
  """Adds a maxout op from https://arxiv.org/abs/1302.4389
  "Maxout Networks" Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron
  Courville,
   Yoshua Bengio
   Usually the operation is performed in the filter/channel dimension. This can
   also be
   used after fully-connected layers to reduce number of features.
   Arguments:
   inputs: Tensor input
   num_units: Specifies how many features will remain after maxout in the `axis`
     dimension
         (usually channel). This must be multiple of number of `axis`.
   axis: The dimension where max pooling will be performed. Default is the
   last dimension.
   name: Optional scope for name_scope.
   Returns:
    A `Tensor` representing the results of the pooling operation.
   Raises:
    ValueError: if num_units is not multiple of number of features.
  """
  return MaxOut(num_units=num_units, axis=axis, name=name)(inputs, decay, epsilon, is_training)


class MaxOut(base.Layer):
  """Adds a maxout op from https://arxiv.org/abs/1302.4389
  "Maxout Networks" Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron
  Courville, Yoshua
  Bengio
  Usually the operation is performed in the filter/channel dimension. This can
  also be
  used after fully-connected layers to reduce number of features.
  Arguments:
    inputs: Tensor input
    num_units: Specifies how many features will remain after maxout in the
      `axis` dimension
         (usually channel).
    This must be multiple of number of `axis`.
    axis: The dimension where max pooling will be performed. Default is the
    last dimension.
    name: Optional scope for name_scope.
  Returns:
    A `Tensor` representing the results of the pooling operation.
  Raises:
    ValueError: if num_units is not multiple of number of features.
  """

  def __init__(self, num_units, axis=-1, name=None, **kwargs):
    super(MaxOut, self).__init__(name=name, trainable=False, **kwargs)
    self.axis = axis
    self.num_units = num_units

  def call(self, inputs, decay, epsilon, is_training):
    inputs = ops.convert_to_tensor(inputs)
    shape = inputs.get_shape().as_list()
    num_channels = shape[self.axis]
    if num_channels % self.num_units:
      raise ValueError('number of features({}) is not '
                       'a multiple of num_units({})'.format(
                           num_channels, self.num_units))
    shape[self.axis] = -1
    shape += [num_channels // self.num_units]

    # Dealing with batches with arbitrary sizes
    for i in range(len(shape)):
      if shape[i] is None:
        shape[i] = gen_array_ops.shape(inputs)[i]

    inputs = batch_norm_wrapper(inputs, decay, epsilon, is_training)

    outputs = math_ops.reduce_max(gen_array_ops.reshape(inputs, shape), -1, keep_dims=False)

    return outputs