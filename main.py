# -*- coding: utf-8 -*-
""" BrainProp implementation.
Usage:
python main.py <dataset> <architecture> <algorithm>
Use the optional argument -s to save training outputs or -l to load weights (specify then the file name)
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import sys, os, importlib
import tensorflow as tf
from tensorflow.keras import datasets, layers, losses, metrics, optimizers, callbacks
import matplotlib.pyplot as plt
import numpy as np
import pickle
import datetime


#initialization
zeropath = ''

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("dataset", help="specify the dataset you want to train the model on.", choices=['MNIST', 'CIFAR10', 'CIFAR100', 'TinyImageNet'])
parser.add_argument("architecture", help="specify the architecture you want to train the model on.", choices=['dense', 'conv', 'loccon', 'deep'])
parser.add_argument("learning_algorithm", help="specify which learning rule should be used.", choices=['BrainProp', 'EBP'])
parser.add_argument("-l", "--load", help="load pre-trained model and evaluate it")
parser.add_argument("-s", "--save", action="store_true", help="save accuracy plot, history and weights")

args = parser.parse_args()

if args.load:
  train_or_eval = 'evaluating'
  str = 'a model trained '
else:
  train_or_eval = 'training'
  str = ''

tic_all = datetime.datetime.now()


def import_from_path(module_name, file_path=None):
    """Import the other python files as modules
    
    Keyword arguments:
    module_name -- the name of the python file (with extension, if file_path is None)
    file_path -- path to the file if not in the current directory (default: None)
    """
    if not file_path:
        if module_name.endswith('.py'):
            file_path = module_name
        else:
            file_path = module_name + '.py'
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return importlib.import_module(module_name)


learning_algorithm = args.learning_algorithm
dataset = args.dataset
architecture = args.architecture

save_plots = False
save_history = False
save_weights = False

if args.save:
    save_plots = True
    save_history = True
    save_weights = True

architecture_selection = import_from_path('get_architecture', file_path=zeropath+"get_architecture.py")
architecture_selected = architecture_selection.fetch_model(dataset, architecture)
batch_dim, learning_rate = architecture_selected.get_hyperparams()

print("Experiment begins, {} on {} {}with {}".format(train_or_eval, dataset, str, learning_algorithm))

seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)

#initialization ends, next: loading and preprocessing the dataset

tic_preprocessing = datetime.datetime.now()

if dataset == 'TinyImageNet':
  tinyimagenet = import_from_path('tinyimagenet', file_path=zeropath+"tinyimagenet.py")
  train_images, train_labels, test_images, test_labels = tinyimagenet.prepare_tinyimagenet(num_classes=None, path='')
elif dataset == 'MNIST':
  (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
  if len(np.shape(train_images)) < 4:
      train_images = tf.expand_dims(train_images, -1).numpy()
      test_images = tf.expand_dims(test_images, -1).numpy()
elif dataset == 'CIFAR10':
  (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
elif dataset == 'CIFAR100':
  (train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data(label_mode='fine')
else:
  raise Exception("Unknown dataset. Choose either \'MNIST\', \'CIFAR10\', \'CIFAR100\' or \'TinyImageNet\'")

if tf.reduce_max(train_images) >1:
  train_images = train_images / 255.0
if tf.reduce_max(test_images) >1:
  test_images = test_images / 255.0
  
image_shape = np.shape(train_images)[1:]
n_classes = tf.cast(tf.reduce_max(train_labels)+1, dtype=tf.int32)
n_batches = len(train_images)//batch_dim

train_labels = tf.keras.utils.to_categorical(train_labels, n_classes, dtype='float32')
test_labels = tf.keras.utils.to_categorical(test_labels, n_classes, dtype='float32')
    
toc_preprocessing = datetime.datetime.now()

print("Preprocessing, elapsed: {} seconds.".format((toc_preprocessing - tic_preprocessing).seconds))

#preparing architecture and optimizer depending on the selected learning algorithm
if learning_algorithm == 'EBP':
    output_activation_function = 'softmax'
    loss = 'categorical_crossentropy'
    metric = 'accuracy'
    output_layer = layers.Dense
elif learning_algorithm == 'BrainProp':
    output_activation_function = 'linear'
    metric = 'accuracy'
    brainprop = import_from_path('brainprop', file_path=zeropath+"brainprop.py")
    loss = brainprop.BrainPropLoss(batch_size=batch_dim, n_classes=n_classes, replicas=1)
    output_layer = brainprop.BrainPropLayer
else:
    raise Exception("Unknown learning algorithm. Choose between \'EBP\' and \'BrainProp\' ")
      
optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.)
model = architecture_selected.get_model(image_shape, output_layer, output_activation_function, n_classes)
model.summary()

#evaluation (if the flag -l was used)/training
if args.load:
    saved_weights = args.load
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    print("Loading weights {}".format(saved_weights))
    model.load_weights(saved_weights)
    
    history = model.evaluate(test_images, test_labels, batch_size=batch_dim, verbose=2)
    
else:
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    
    epochs = 500
    
    lr_schedule = callbacks.LearningRateScheduler(lambda epoch: learning_rate * (0.5 ** (epoch // 100)), verbose=0)
    terminate_on_NaN = callbacks.TerminateOnNaN()
    earlystopping = callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=45, verbose=1, mode='max', baseline=None, restore_best_weights=False)
    callbacks_list = list(filter(None, [lr_schedule, terminate_on_NaN, earlystopping]))
    
    tic_training = datetime.datetime.now()
    history = model.fit(train_images, train_labels, batch_size=batch_dim, epochs=epochs, validation_data=(test_images, test_labels), shuffle=True, verbose=2, callbacks=callbacks_list)

    toc_training = datetime.datetime.now()
    print("Training, elapsed: {} minutes.".format((toc_training - tic_training).seconds//60))


    def get_filename(type):
        """Computes the filename for the outputs of the training 
        (checks whether the file already exists, in that case adds a number to the filename 
        to avoid overriding it)
        
        Keyword arguments:
        type -- string, the type of output (accuracy.pdf, history.pkl or weights.h5)
        """
        filename = "{}_{}_{}_{}".format(dataset, architecture, learning_algorithm, type)
        num = 0
        while os.path.isfile(filename):
            filename="{}_{}_{}_{}_{}".format(dataset, architecture, learning_algorithm, num, type)
            num += 1
        return filename

    if save_plots == True: #save a plot of the accuracy as a function of the epochs
        filename_plot = get_filename('accuracy.pdf')

        n_epochs = len(history.history['accuracy'])

        plt.figure()
        plt.title("{} - {}".format(learning_algorithm, dataset) , fontsize=16)
        plt.plot(history.history['accuracy'], label='accuracy', linewidth=2)
        plt.plot(history.history['val_accuracy'], label = 'validation accuracy', linewidth=2)
        maximum_val_accuracy = np.max(history.history['val_accuracy'])
        argmax_val_accuracy = np.argmax(history.history['val_accuracy'])
        plt.plot([argmax_val_accuracy,argmax_val_accuracy], [-0.4,maximum_val_accuracy], '--', color='green', linewidth=1)
        plt.plot(argmax_val_accuracy,maximum_val_accuracy,'ks', markersize = 7, label='maximum = {:.5}'.format(maximum_val_accuracy))
        plt.xticks(list(plt.xticks()[0]) + [argmax_val_accuracy])
        plt.gca().get_xticklabels()[-1].set_color("white")
        plt.gca().get_xticklabels()[-1].set_fontweight('bold')
        plt.gca().get_xticklabels()[-1].set_bbox(dict(facecolor='green', edgecolor='white', alpha=0.8))
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.xlim([-0.4, (n_epochs-.5)])
        plt.ylim([0.0, 1.05])
        plt.legend(loc='lower right', fontsize=12)
        print("Saving the accuracy plot as \'{}\'".format(filename_plot))
        plt.savefig(zeropath+filename_plot, dpi=300, bbox_inches='tight')

    if save_history == True:  #save the history file of the training (contains accuracy, validation accuracy, epochs, loss)
        filename_history = get_filename('history.pkl')
        print("Saving the history as \'{}\'".format(filename_history))
        with open(zeropath+filename_history, 'wb') as file:
            pickle.dump([dataset, learning_algorithm, history.history], file)
        
        
    if save_weights == True: #save the weights of the trained model (the value they had at the epoch of the best validtion accuracy)
        filename_w = get_filename('weights.h5')
        print("Saving the weights as \'{}\'".format(filename_w))
        model.save_weights(zeropath+filename_w)
        

    toc_all = datetime.datetime.now()
    print("End of the experiment. Elapsed: {} seconds ({} minutes).".format((toc_all - tic_all).seconds, (toc_all - tic_all).seconds//60))
