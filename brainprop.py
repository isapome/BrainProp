# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import losses, metrics, layers, initializers
from tensorflow.python.keras.utils import tf_utils, losses_utils
from tensorflow.python.ops import array_ops


class BrainPropLayer(layers.Layer):

    def __init__(self, output_dim, activation=None, use_bias=None, kernel_regularizer=None, kernel_initializer=None, **kwargs):
        super(BrainPropLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.regularizer = kernel_regularizer
        self.initializer = kernel_initializer #initializers.TruncatedNormal(stddev=0.1)
        if isinstance(self.initializer, initializers.GlorotUniform) or isinstance(self.initializer, initializers.GlorotNormal):
          self.initializer = tf.random_normal_initializer(mean=0., stddev=0.05)
        self.epsilon = 0.02

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=self.initializer,#'uniform',
                                      regularizer=self.regularizer,
                                      trainable=True)
    #    self.bias = self.add_weight(name='bias',
    #                                  shape=(self.output_dim),
    #                                  initializer=initializers.Constant(0.),
    #                                  trainable=False)
        super(BrainPropLayer, self).build(input_shape)

        
    def call(self, x, training=None):
        if training is None:
            training = K.learning_phase()
        def exploration():
            batch_size, n_classes = tf.shape(x)[0], self.output_dim
            argmax_vector = tf.math.argmax(activations, axis=-1)
            random_vector = tf.random.uniform([batch_size], minval=0, maxval=1)
            multinomial_vector = tf.reshape(tf.random.categorical(activations, 1), [-1])
            selected_classes = tf.where(tf.greater(random_vector, self.epsilon), argmax_vector, multinomial_vector)
            selected_classes_onehot = tf.one_hot(selected_classes, n_classes)
            y_pred = tf.multiply(selected_classes_onehot, activations)
            return y_pred
        
        activations = K.dot(x, self.kernel)
        output = tf_utils.smart_cond(training, #if training==1 do exploration, else do lambda
                                     exploration,
                                     lambda: tf.one_hot(tf.math.argmax(activations, axis=-1), self.output_dim))
        return output
        
        
    def get_config(self):
        config = super(BrainPropLayer, self).get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'regularizer': self.regularizer,
            'initializer': self.initializer,
        })

        return config


class BrainPropLoss(losses.Loss):
    def __init__(self, batch_size, n_classes, replicas, reduction=losses.Reduction.SUM, name='loss', **kwargs):
        super(BrainPropLoss, self).__init__(reduction=reduction, name=name, **kwargs)
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.replicas = replicas

    def call(self, y_true, y_pred):
        y_true = K.switch(tf.shape(y_true)[-1] == self.n_classes, y_true, tf.squeeze(tf.one_hot(tf.cast(y_true, tf.int32), self.n_classes)))
        selected_classes = tf.where(y_pred!=0, y_pred*(1/y_pred), y_pred)
        labels = tf.where(selected_classes==0, y_pred, y_true)
        loss = 0.5 * K.square(labels - y_pred) #* tf.cast(self.n_classes, dtype=tf.float32)
        return loss * (1/self.batch_size) * (1/self.replicas)

