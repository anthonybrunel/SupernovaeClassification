from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import SNClassification_CNN
import SNClassification_input
import numpy as np


LEARNING_RATE = 0.01 
DECAY_STEPS = 9000 #MAX_STEP*2
FLAGS = tf.app.flags.FLAGS
tf.logging.set_verbosity(tf.logging.DEBUG)
#os.environ["CUDA_VISIBLE_DEVICES"]= "0"    



def eval_confusion_matrix(labels, predictions):
    """
	Compute the confusion matrix
    """
    with tf.variable_scope("eval_confusion_matrix"):
        con_matrix = tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=2)

        con_matrix_sum = tf.Variable(tf.zeros(shape=(2,2), dtype=tf.int32),
                                            trainable=False,
                                            name="confusion_matrix_result",
                                            collections=[tf.GraphKeys.LOCAL_VARIABLES])

        update_op = tf.assign_add(con_matrix_sum, con_matrix)

        return tf.convert_to_tensor(con_matrix_sum), update_op


def compute_loss(logits, labels):
    """
	Computes softmax cross entropy between `logits` and `labels
    """

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean

def train_op_fn(lr, loss):
    """
	train function used by tf.estimator.EstimatorSpec
    """
    optimizer = tf.train.AdamOptimizer(learning_rate=lr,epsilon=0.1)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        return optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())


 
def model_fn(features, labels, mode):
    """
	model fn used in tf.estimator.Estimator it contain cnn tensor, predict mode, train mode, eval model
    """
    input_layer = features["x"]
    with tf.device('/device:GPU:1'):
        cnn_logits,embeddings = SNClassification_CNN.cnn(input_layer,features['parameters'],mode)

        if mode == tf.estimator.ModeKeys.PREDICT:
            pred = {'embeddings': tf.nn.softmax(cnn_logits) ,'label':features['label']}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=pred)
        
        predictions = {
            "cnn_classes": tf.argmax(input=cnn_logits, axis=1),
        }

        loss = compute_loss(cnn_logits,labels)

        if mode == tf.estimator.ModeKeys.TRAIN:
            with tf.name_scope("train"):
                with tf.name_scope("learning_rate"):
                    learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE,
                                                               global_step=tf.train.get_global_step(),
                                                               decay_steps=int(DECAY_STEPS),
                                                               decay_rate=0.1)
                    tf.summary.scalar('learning_rate', learning_rate)

                    train_op = train_op_fn(learning_rate,loss)

                    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


        with tf.name_scope("test"):
            threshold=np.linspace(0., 1., 100)
            threshold = np.float32(threshold)
            eval_metric_ops = { 
                "cnn_accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["cnn_classes"]),
                "cnn_conf_matrix": eval_confusion_matrix(labels, predictions["cnn_classes"]),
                "AUC-PR": tf.metrics.auc(tf.one_hot(labels, depth=2), tf.nn.softmax(cnn_logits),curve='PR'),
                "AUC-ROC": tf.metrics.auc(tf.one_hot(labels, depth=2), tf.nn.softmax(cnn_logits)),
            }

            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def kfold_run():
    """
	Train k model with 3750 light curves and test on the other 1750 light curves (kfold 25% split)
	Number of iteration 4500
	Batch size 128
	Model are save in Model directory
    """
    DATA_DIR = "BASE1/"
    DATA_FILE = 'sn_base_5000_1.tfrecords'

    list_dataset = SNClassification_input.load_dataset(DATA_DIR, DATA_FILE, 5000)
    eval_results = []
    for i in range(4):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        PATH = os.path.join(ROOT_DIR, '../Model/model_conv_kfold_5000_'+str(i)+'_1')
        run_config = tf.contrib.learn.RunConfig()
        run_config = run_config.replace(save_summary_steps=100)
     

        sn_classifier = tf.estimator.Estimator(
            model_fn=model_fn, model_dir=PATH,config=run_config)
        tensors_to_log = {}


        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100)

        train_input_fn = lambda : SNClassification_input.kfold_train_input_fn(list_dataset, i, 128, None)
        test_input_fn = lambda : SNClassification_input.kfold_test_input_fn(list_dataset, i, 128, 1)

        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,hooks=[logging_hook],max_steps = 4500)
        eval_spec = tf.estimator.EvalSpec(input_fn=test_input_fn,steps=None,throttle_secs=120)
        

        tf.estimator.train_and_evaluate(sn_classifier, train_spec, eval_spec)
        

        result = (sn_classifier.evaluate(input_fn=test_input_fn))
        print(result)
        eval_results.append(result)
    print(eval_results)

def challenge_run():
    """
	Function used to compare our network with https://arxiv.org/pdf/1606.07442.pdf result
	Train five model on five different train base and eval on the other part of the database
	Model are save in Model directory
	Database used come from https://arxiv.org/pdf/1606.07442.pdf
	Number of data for the training base is 5330
	Number of iteration 4500
	Batch size 128
    """
    eval_results = []
    for i in range(5):
        DATA_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_FILE = '../BASE2/sn_base_train_challenge_'+str(i)+'.tfrecords'

        filename = os.path.join(DATA_DIR, DATA_FILE)
        train_dataset = tf.data.TFRecordDataset(filename)

        DATA_FILE = '../BASE2/sn_base_test_challenge_'+str(i)+'.tfrecords'
        filename = os.path.join(DATA_DIR, DATA_FILE)
        test_dataset = tf.data.TFRecordDataset(filename)


        DECAY_STEPS = 9000

        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        PATH = os.path.join(ROOT_DIR, '../Model/model_conv_challenge_'+str(i))
        run_config = tf.contrib.learn.RunConfig()
        run_config = run_config.replace(save_summary_steps=100)


        sn_classifier = tf.estimator.Estimator(
            model_fn=model_fn, model_dir=PATH,config=run_config)
        tensors_to_log = {}


        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100)

        train_input_fn = lambda : SNClassification_input.challenge_train_input_fn(train_dataset, 128, None)
        test_input_fn = lambda : SNClassification_input.challenge_test_input_fn(test_dataset, 128, 1)

        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,hooks=[logging_hook],max_steps = 4500)
        eval_spec = tf.estimator.EvalSpec(input_fn=test_input_fn,steps=None,throttle_secs=120)
        

        tf.estimator.train_and_evaluate(sn_classifier, train_spec, eval_spec)
        

        result = (sn_classifier.evaluate(input_fn=test_input_fn))
        print(result)
        eval_results.append(result)
    print(eval_results)



def main(argv=None):
    #kfold_run()
    challenge_run()


if __name__ == '__main__':
    tf.app.run()
