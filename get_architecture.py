# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks, initializers, optimizers
import numpy as np

class fetch_model():
    def __init__(self, dataset, model):
        """Set the architecture and the hyperparameters for training.
        
        
        Keyword arguments:
        dataset -- which dataset is gonna be used.
        model -- which architecture the model is gonna be trained on.
        """
        self.batch_dim = 128
        self.learning_rate = 1e-1
        self.bias = False
        self.pad = 'same'
        self.initializer = tf.random_normal_initializer(mean=0., stddev=0.005)
        self.regularizer = None

        if dataset == 'MNIST' and model == 'dense':
            self.learning_rate = 1e-1
            self.initializer = tf.random_normal_initializer(mean=0., stddev=0.01)
            self.get_model = self.get_model_dense
        elif dataset != 'TinyImageNet' and model == 'conv':
            self.get_model = self.get_model_conv
        elif dataset != 'TinyImageNet' and model == 'loccon':
            self.pad = 'valid'
            self.batch_dim = 32
            self.get_model = self.get_model_loccon
        elif dataset != 'MNIST' and model == 'deep':
            self.regularizer = regularizers.l2(5e-4)
            self.get_model = self.get_model_deep
        else:
            raise Exception("Combination dataset-model not included. Check the allowed combinations in the README file.")
            
    def get_hyperparams(self):
        """Return the batch size and the learning rate, needed to train the model.
        """
        return self.batch_dim, self.learning_rate


    def get_model_deep(self, input_shape, output_layer, output_activation_function, n_classes):
        """Deep architecture, used on CIFAR10, CIFAR100 and TinyImageNet. Made of 8 hidden layers (7 convolutional each followed by batch normalization and 1 dense), 
        all with ReLU activation function. After the second, the fourth and the seventh layers a max pooling layer is added,
        followed by dropout. For more details, see the paper.


        Keyword arguments:
        input_shape -- shape of the input.
        output_layer -- type of output layer.
        output_activation_function -- activation function for the output layer.
        n_classes -- number of classes the inputs can belong to.
        """
        
        bias = self.bias
        initializer = self.initializer
        regularizer = self.regularizer
        pad = self.pad
        model = models.Sequential()
        model.add(layers.Conv2D(64, (3,3), strides=(1,1), input_shape=input_shape, use_bias=bias, kernel_initializer=initializer, kernel_regularizer=regularizer, padding=pad))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(64, (3,3), strides=(1,1), use_bias=bias, kernel_initializer=initializer, kernel_regularizer=regularizer, padding=pad))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Dropout(0.3))
        model.add(layers.Conv2D(128, (3,3), strides=(1,1), use_bias=bias, kernel_initializer=initializer, kernel_regularizer=regularizer, padding=pad))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(128, (3,3), strides=(1,1), use_bias=bias, kernel_initializer=initializer, kernel_regularizer=regularizer, padding=pad))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Dropout(0.3))
        model.add(layers.Conv2D(128, (3,3), strides=(1,1), use_bias=bias, kernel_initializer=initializer, kernel_regularizer=regularizer, padding=pad))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(128, (3,3), strides=(1,1), use_bias=bias, kernel_initializer=initializer, kernel_regularizer=regularizer, padding=pad))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(128, (3,3), strides=(1,1), use_bias=bias, kernel_initializer=initializer, kernel_regularizer=regularizer, padding=pad))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Dropout(0.8))
        model.add(layers.Flatten())
        model.add(layers.Dense(500, use_bias=bias, kernel_initializer=initializer, kernel_regularizer=regularizer))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.3))
        last_layer = output_layer(n_classes, activation=output_activation_function, use_bias=bias, kernel_regularizer=regularizer, kernel_initializer=initializer)
        model.add(last_layer)
        return model


    def get_model_dense(self, input_shape, output_layer, output_activation_function, n_classes):
        """Dense architecture, used on MNIST. Made of 3 dense hidden layers. For more details, see the paper.


        Keyword arguments:
        input_shape -- shape of the input.
        output_layer -- type of output layer.
        output_activation_function -- activation function for the output layer.
        n_classes -- number of classes the inputs can belong to.
        """
        
        bias = self.bias
        initializer = self.initializer
        regularizer = self.regularizer
        pad = self.pad
        model = models.Sequential()
        model.add(layers.Flatten(input_shape=input_shape))
        model.add(layers.Dense(1500, use_bias=bias, kernel_initializer=initializer, kernel_regularizer=regularizer))
        model.add(layers.Activation('relu'))
        model.add(layers.Dense(1000, use_bias=bias, kernel_initializer=initializer, kernel_regularizer=regularizer))
        model.add(layers.Activation('relu'))
        model.add(layers.Dense(500, use_bias=bias, kernel_initializer=initializer, kernel_regularizer=regularizer))
        model.add(layers.Activation('relu'))
        last_layer = output_layer(n_classes, activation=output_activation_function, use_bias=bias, kernel_regularizer=regularizer, kernel_initializer=initializer)
        model.add(last_layer)
        return model




    def get_model_loccon(self, input_shape, output_layer, output_activation_function, n_classes):
        """Loccon architecture, used on MNIST, CIFAR10 and CIFAR100. Made of 3 hidden layers (2 locally conencted each followed by batch normalization and 1 dense), 
        all with ReLU activation function. It requires the usage of batch size 32 because computationally very expensive and valid padding because same is not implemented yet.
        For more details, see the paper.


        Keyword arguments:
        input_shape -- shape of the input.
        output_layer -- type of output layer.
        output_activation_function -- activation function for the output layer.
        n_classes -- number of classes the inputs can belong to.
        """
        
        bias = self.bias
        initializer = self.initializer
        regularizer = self.regularizer
        pad = self.pad
        model = models.Sequential()
        model.add(layers.LocallyConnected2D(32, (3, 3), strides=(1,1), input_shape=input_shape, use_bias=bias, kernel_initializer=initializer, kernel_regularizer=regularizer, padding=pad))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.LocallyConnected2D(32, (3, 3), strides=(2,2), use_bias=bias, kernel_initializer=initializer, kernel_regularizer=regularizer, padding=pad))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        model.add(layers.Flatten())
        model.add(layers.Dense(500, use_bias=bias, kernel_initializer=initializer, kernel_regularizer=regularizer))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.3))
        last_layer = output_layer(n_classes, activation=output_activation_function, use_bias=bias, kernel_regularizer=regularizer, kernel_initializer=initializer)
        model.add(last_layer)
        return model



    def get_model_conv(self, input_shape, output_layer, output_activation_function, n_classes):
        """Conv architecture, used on MNIST, CIFAR10 and CIFAR100. Made of 3 hidden layers (2 convolutional each followed by batch normalization and 1 dense), 
        all with ReLU activation function. For more details, see the paper.


        Keyword arguments:
        input_shape -- shape of the input.
        output_layer -- type of output layer.
        output_activation_function -- activation function for the output layer.
        n_classes -- number of classes the inputs can belong to.
        """
        
        bias = self.bias
        initializer = self.initializer
        regularizer = self.regularizer
        pad = self.pad
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), strides=(1,1), input_shape=input_shape, use_bias=bias, kernel_initializer=initializer, kernel_regularizer=regularizer, padding=pad))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(32, (3, 3), strides=(2,2), use_bias=bias, kernel_initializer=initializer, kernel_regularizer=regularizer, padding=pad))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        model.add(layers.Flatten())
        model.add(layers.Dense(500, use_bias=bias, kernel_initializer=initializer, kernel_regularizer=regularizer))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.3))
        last_layer = output_layer(n_classes, activation=output_activation_function, use_bias=bias, kernel_regularizer=regularizer, kernel_initializer=initializer)
        model.add(last_layer)
        return model
        
        
        
