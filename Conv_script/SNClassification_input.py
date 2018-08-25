from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from six.moves import xrange  
import tensorflow as tf
import os
import random

def pad(matrix,flux_size):
    s = tf.shape(matrix)
    paddings = [[0, m-s[i]] for (i,m) in enumerate([4,flux_size])]
    return tf.pad(matrix, paddings, 'CONSTANT', constant_values=0)
  

def decode_test(serialized_example):    
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'dense_matrix': tf.VarLenFeature(tf.float32),
            'name': tf.FixedLenFeature([], tf.string),
            'flux_size': tf.FixedLenFeature([], tf.int64),
            'parameters': tf.VarLenFeature(tf.float32),
        })

    label = tf.cast(features['label'], tf.int32)
    name = tf.cast(features['name'], tf.string)
    flux_size = tf.cast(features['flux_size'], tf.int32)

    matrix = features['dense_matrix'].values
    parameters = features['parameters'].values
    matrix = tf.reshape(matrix, [4, flux_size])
    matrix = tf.cond(flux_size <= 32,lambda: pad(matrix,32),
                    lambda: matrix)
    matrix = tf.reshape(matrix, [4, tf.shape(matrix)[1],1])


    return matrix,parameters, label, name


def decode_train(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'dense_matrix': tf.VarLenFeature(tf.float32),
            'name': tf.FixedLenFeature([], tf.string),
            'flux_size': tf.FixedLenFeature([], tf.int64),
            'parameters': tf.VarLenFeature(tf.float32),
        })

    label = tf.cast(features['label'], tf.int32)
    name = tf.cast(features['name'], tf.string)
    flux_size = tf.cast(features['flux_size'], tf.int32)
    parameters = features['parameters'].values

    matrix = features['dense_matrix'].values
    

    
    matrix = tf.reshape(matrix, [4, flux_size])

    flux_size = tf.cond(tf.greater(tf.random_uniform((), minval=0., maxval=1., dtype=tf.float32),0.8), 
                        lambda: tf.cast(tf.cast(flux_size,tf.float32)*tf.random_uniform((), minval=0.4, maxval=0.8, dtype=tf.float32),tf.int32),lambda: flux_size)    
    matrix = tf.cond(flux_size <= 32,lambda: pad(tf.random_crop(matrix,[4,flux_size]),32),
                    lambda: (tf.random_crop(matrix,[4,flux_size])))

    matrix = tf.reshape(matrix, [4, tf.shape(matrix)[1],1])

    return matrix, parameters, label, name


def generate_test_input(datadir, filename, batch_size, num_epochs):
    if not num_epochs:
        num_epochs = None
    #filenames = [os.path.join(datadir, filename[0]),os.path.join(datadir, filename[1])]
    filename = os.path.join(datadir, filename)

    with tf.name_scope('input'):
        dataset = tf.data.TFRecordDataset(filename)

        dataset = dataset.map(decode_test)
        dataset = dataset.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None,None,1]),tf.TensorShape([None]),tf.TensorShape([]),tf.TensorShape([])))
        dataset = dataset.prefetch(buffer_size=10000)

        iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

def generate_predict_input(datadir, filename, batch_size):

    filename = os.path.join(datadir, filename)

    with tf.name_scope('input'):
        dataset = tf.data.TFRecordDataset(filename)

        dataset = dataset.shuffle(buffer_size=210000)
        dataset = dataset.map(decode_test)
        dataset = dataset.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None,None,1]),tf.TensorShape([None]),tf.TensorShape([]),tf.TensorShape([])))
        dataset = dataset.prefetch(buffer_size=10000)

        iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

def generate_train_input(datadir, filename, batch_size, num_epochs):
    if not num_epochs:
        num_epochs = None
    filename = os.path.join(datadir, filename)
    #filenames = [os.path.join(datadir, filename[0]),os.path.join(datadir, filename[1])]

    with tf.name_scope('input'):
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=210000, count = num_epochs))
        dataset = dataset.map(decode_train)
        dataset = dataset.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None,None,1]),tf.TensorShape([None]),tf.TensorShape([]),tf.TensorShape([])))
        dataset = dataset.prefetch(buffer_size=10000)
        iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

def load_dataset(datadir,filename, dataset_size):
    #25% split
    filename = os.path.join(datadir, filename)
    dataset = tf.data.TFRecordDataset(filename)
    dataset_1 = dataset.take(int(dataset_size*0.25))
    dataset_2 = dataset.skip(int(dataset_size*0.25)).take(int(dataset_size*0.25))
    dataset_3 = dataset.skip(int(dataset_size*0.50)).take(int(dataset_size*0.25))
    dataset_4 = dataset.skip(int(dataset_size*0.75)).take(int(dataset_size*0.25))

    return [dataset_1,dataset_2,dataset_3,dataset_4]


def kfold_train_input_fn(list_dataset, idx_test, batch_size, num_epochs):

    dataset = None
    for i in range(len(list_dataset)):
        if i != idx_test:
            if dataset == None:
                dataset = list_dataset[i]
            else:
                dataset = dataset.concatenate(list_dataset[i])

    with tf.name_scope('input'):
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=25000, count = num_epochs))
        dataset = dataset.map(decode_train)
        dataset = dataset.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None,None,1]),tf.TensorShape([None]),tf.TensorShape([]),tf.TensorShape([])))
        dataset = dataset.prefetch(buffer_size=10000)
        iterator = dataset.make_one_shot_iterator()
    images_tensor, parameters, labels_tensor, name =  iterator.get_next()
    return {"x": images_tensor,"parameters": parameters}, labels_tensor


def kfold_predict_input_fn(list_dataset, idx_test, batch_size, num_epochs):
    dataset = list_dataset[idx_test]
    with tf.name_scope('input'):

        dataset = dataset.map(decode_test)
        dataset = dataset.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None,None,1]),tf.TensorShape([None]),tf.TensorShape([]),tf.TensorShape([])))
        dataset = dataset.prefetch(buffer_size=10000)

        iterator = dataset.make_one_shot_iterator()
    images_tensor,parameters, labels_tensor, name =  iterator.get_next()
    return {"x": images_tensor,"parameters": parameters,"label": labels_tensor}, labels_tensor

def kfold_test_input_fn(list_dataset, idx_test, batch_size, num_epochs):
    dataset = list_dataset[idx_test]
    with tf.name_scope('input'):

        dataset = dataset.map(decode_test)
        dataset = dataset.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None,None,1]),tf.TensorShape([None]),tf.TensorShape([]),tf.TensorShape([])))
        dataset = dataset.prefetch(buffer_size=10000)

        iterator = dataset.make_one_shot_iterator()
    images_tensor,parameters, labels_tensor, name =  iterator.get_next()
    return {"x": images_tensor,"parameters": parameters}, labels_tensor



def load_challenge_dataset(datadir,filename, dataset_train_size):

    filename = os.path.join(datadir, filename)
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.shuffle(25000)
    dataset_1 = dataset.take(int(dataset_train_size))
    dataset_2 = dataset.skip(int(dataset_train_size))

    return [dataset_1,dataset_2]

def challenge_train_input_fn(dataset, batch_size, num_epochs):


    with tf.name_scope('input'):
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=25000, count = num_epochs))
        dataset = dataset.map(decode_train)
        dataset = dataset.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None,None,1]),tf.TensorShape([None]),tf.TensorShape([]),tf.TensorShape([])))
        dataset = dataset.prefetch(buffer_size=10000)
        iterator = dataset.make_one_shot_iterator()
    images_tensor, parameters, labels_tensor, name =  iterator.get_next()
    return {"x": images_tensor,"parameters": parameters}, labels_tensor

def challenge_test_input_fn(dataset, batch_size, num_epochs):


    with tf.name_scope('input'):
        dataset = dataset.map(decode_test)
        dataset = dataset.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None,None,1]),tf.TensorShape([None]),tf.TensorShape([]),tf.TensorShape([])))
        dataset = dataset.prefetch(buffer_size=10000)

        iterator = dataset.make_one_shot_iterator()
    images_tensor, parameters, labels_tensor, name =  iterator.get_next()
    return {"x": images_tensor,"parameters": parameters,"label": labels_tensor}, labels_tensor



def get_train_input_data(datadir, filename, batch_size, num_epochs):
    images_tensor,parameters, labels_tensor,name = generate_train_input(datadir, filename, batch_size, num_epochs)
    return {"x": images_tensor,"parameters": parameters}, labels_tensor


def get_test_input_data(datadir, filename, batch_size, num_epochs):
    images_tensor,parameters, labels_tensor,name = generate_test_input(datadir, filename, batch_size, num_epochs)
    return {"x": images_tensor,"parameters": parameters}, labels_tensor


def get_predict_input_data(datadir, filename, batch_size):
    images_tensor,parameters, labels_tensor,name = generate_predict_input(datadir, filename, batch_size)
    return {"x": images_tensor,"parameters": parameters,"label": labels_tensor}, labels_tensor, name




