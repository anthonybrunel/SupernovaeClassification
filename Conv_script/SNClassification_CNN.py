from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf


NUM_CLASSES = 2


def _inceptionV1(input, num_filters, stride, mode):
    """
    Inception module adapted for light curves
    """
	
    if stride == 1 :
        strides = [1,1]
    else :
        strides = [1,2]

    conv11_1 = tf.layers.conv2d(
        inputs=input,
        filters=num_filters[0],
        kernel_size=[1, 1],
        padding="same"
        )
    conv11_1 = tf.nn.relu((conv11_1))
    
    conv3 = tf.layers.conv2d(
        inputs=conv11_1,
        filters=num_filters[1],
        kernel_size=[1, 3],
        strides=strides,
        padding="same"
        )
    conv3 = tf.nn.relu((conv3))

    conv11_2 = tf.layers.conv2d(
        inputs=input,
        filters=num_filters[2],
        kernel_size=[1, 1],
        padding="same"
        )
    conv5 = tf.nn.relu((conv11_2))

    conv5 = tf.layers.conv2d(
        inputs=conv11_2,
        filters=num_filters[3],
        kernel_size=[1, 5],
        strides=strides,
        padding="same"
        )
    conv5 = tf.nn.relu((conv5))

    conv11_3 = tf.layers.conv2d(
        inputs=input,
        filters=num_filters[4],
        kernel_size=[1, 1],
        strides=strides,
        padding="same"
        )
    conv11_3 = tf.nn.relu((conv11_3))

    max_pool = tf.layers.max_pooling2d(inputs=input, pool_size=[1, 3], strides=strides, padding="same")
    
    conv11_4 = tf.layers.conv2d(
        inputs=max_pool,
        filters=num_filters[5],
        kernel_size=[1, 1],
        padding="same"
        )

    conv11_4 = tf.nn.relu((conv11_4))

    return tf.concat([conv3,conv5,conv11_3,conv11_4],3)



def layer_1(input, mode):

    conv1_3 = tf.layers.conv2d(
        inputs=input,
        filters=64,
        kernel_size=[1, 3],
        padding="same",
        name='conv1_3')

    conv1_3_bn = tf.nn.relu((conv1_3))
    conv1_5 = tf.layers.conv2d(
        inputs=input,
        filters=32,
        kernel_size=[1, 5],
        padding="same",
        name='conv1_5')

    conv1_5_bn = tf.nn.relu((conv1_5))
    

    conv1_7 = tf.layers.conv2d(
        inputs=input,
        filters=16,
        kernel_size=[1, 7],
        padding="same",
        name='conv1_7')

    conv1_7_bn = tf.nn.relu((conv1_7))

    concat1 = tf.concat([conv1_3_bn, conv1_5_bn, conv1_7_bn], 3)


    return concat1


def conv_color(input,filters, mode):
    conv11 = tf.layers.conv2d(
        inputs=input,
        filters=filters[0],
        kernel_size=[1, 1],
        padding="same")
    conv11 = tf.nn.relu((conv11))

    conv_color = tf.layers.conv2d(
        inputs=conv11,
        filters=filters[1],
        kernel_size=[4,1],
        padding="valid")
    conv_color = tf.nn.relu((conv_color))


    return conv_color


def layer_11(input, mode):

    conv11_11_1 = tf.layers.conv2d(
        inputs=input,
        filters=192,
        kernel_size=[1, 1],
        padding="same"
	)
    conv11_11_1_bn = tf.nn.relu((conv11_11_1))

    conv11_3_1 = tf.layers.conv2d(
        inputs=conv11_11_1_bn,
        filters=384,
        kernel_size=[1, 3],
        strides=[1, 2],
        padding="same")
    conv11_3_1_bn = tf.nn.relu((conv11_3_1))

    max_pool = tf.layers.max_pooling2d(inputs=input, pool_size=[1, 2], strides=[1, 2],padding="same")

    conv11_11_2 = tf.layers.conv2d(
        inputs=max_pool,
        filters=320,
        kernel_size=[1, 1],
        padding="same")
    conv11_11_2_bn = tf.nn.relu((conv11_11_2))

    conv11_11_3 = tf.layers.conv2d(
        inputs=input,
        filters=384,
        kernel_size=[1, 1],
        strides=(1,2),
        padding="same")
    conv11_11_3_bn = tf.nn.relu((conv11_11_3))


    # [1,12,196+224+248]
    return tf.concat([conv11_3_1_bn,  conv11_11_2_bn, conv11_11_3_bn],3)


def cnn(input_data,parameters, mode):
    """
    CNN architecture
    """
    with tf.variable_scope("layer_1"):
        out1 = layer_1(input_data,mode)

    with tf.variable_scope("layer_2"):
        out2 = _inceptionV1(input = out1, num_filters = [48,64,16,32,64,64],stride = 2, mode = mode)

    with tf.variable_scope("layer_3"):
        out3 = _inceptionV1(input = out2, num_filters = [64,96,32,64,96,96],stride = 1, mode = mode)

    with tf.variable_scope("layer_4"):
        out4 = _inceptionV1(input = out3, num_filters = [96,128,64,96,128,96],stride = 1, mode = mode)
    
    with tf.variable_scope("layer_5"):
        out5 = _inceptionV1(input = out4, num_filters = [128,160,64,96,160,128],stride = 2, mode = mode)

    with tf.variable_scope("layer_6"):
        out6 = _inceptionV1(input = out5, num_filters = [144,196,96,128,196,164],stride = 1, mode = mode)

    with tf.variable_scope("layer_7"):
        out7 = conv_color(out6,[256,320], mode = mode) 
    
    with tf.variable_scope("layer_8"):
        out8 = _inceptionV1(input = out7, num_filters = [144,196,96,128,256,196],stride = 1, mode = mode)
    
    with tf.variable_scope("layer_9"):
        out9 = _inceptionV1(input = out8, num_filters = [160,256,128,144,224,256],stride = 2, mode = mode)

    with tf.variable_scope("layer_x1"):
        out10 = _inceptionV1(input = out9, num_filters = [160,320,128,144,288,256],stride = 2, mode = mode)

    with tf.variable_scope("layer_x2"):
        out11 = layer_11(out10, mode = mode)


    global_max_pooling = tf.reduce_max(out11, [1,2])


    #parameters = tf.reshape(parameters,[tf.shape(global_max_pooling)[0],1])

    #features_vector = tf.concat([global_max_pooling, parameters],1)
    

    with tf.variable_scope("fully_connected"):
        
        dense =  tf.layers.dense(inputs=global_max_pooling, units=1024)
        
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
        
        result = tf.layers.dense(inputs=dropout, units=2)

	

    return result,global_max_pooling







