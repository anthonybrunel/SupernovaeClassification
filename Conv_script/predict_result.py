"""Train the model"""

import argparse
import os
import pathlib
import shutil
import csv

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from SNClassification_train import model_fn
import SNClassification_input
from scipy.integrate import trapz
from sklearn import metrics

os.environ["CUDA_VISIBLE_DEVICES"]="0"    


"""
if __name__ == '__main__':
    DATA_DIR = "../../../SN_DATA"
    DATA_FILE = 'sn_base_5000_1.tfrecords'

    list_dataset = SNClassification_input.load_dataset(DATA_DIR, DATA_FILE, 5000)
    size_predict = 5000
    embeddings = np.zeros((size_predict, 2))
    label = np.zeros((size_predict))
    auc_list = np.zeros((4))
    acc_list = np.zeros((4))
    for i in range(4):
        tf.reset_default_graph()
        tf.logging.set_verbosity(tf.logging.INFO)

        tf.logging.info("Creating the datasets...")
        
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        PATH = os.path.join(ROOT_DIR, '../model_conv_kfold_5000_noredshift_'+str(i)+'_2')
        
        tf.logging.info("Creating the model...")
        config = tf.estimator.RunConfig(tf_random_seed=230,
                                        model_dir=PATH,
                                        save_summary_steps=500)
        estimator = tf.estimator.Estimator(model_fn, config=config)
        predict_input = lambda : SNClassification_input.kfold_predict_input_fn(list_dataset, i, 128, 1)

        tf.logging.info("Predicting")

        predictions = estimator.predict(predict_input)
        label_per_classifier = np.zeros((1250))
        pred_per_classifieur = np.zeros((1250,2))
        for j, p in enumerate(predictions):
            embeddings[j+int((size_predict/4)*i)] = p['embeddings']
            label[j+int((size_predict/4)*i)]= p['label']
            label_per_classifier[j] =  p['label']
            pred_per_classifieur[j] =  p['embeddings']
        threshold=np.linspace(0., 1., 500)
        tpr=[0]*len(threshold)
        fpr=[0]*len(threshold)
        bestAccuracy = -1.0
        bestThresh = -1.0
        bestThreshIndex = -1
        for k in range(len(threshold)):
            preds = np.ones((1250))
            for j in range(len(pred_per_classifieur)):
                if pred_per_classifieur[j][0] >= threshold[k]:
                    preds[j] =  0
            TP=sum((preds==0) & (label_per_classifier==0))
            FP=sum((preds==0) &(label_per_classifier==1))
            TN=sum((preds==1) & (label_per_classifier==1))
            FN=sum((preds==1) & (label_per_classifier==0))
            if TP==0:
                tpr[k]=0
            else:
                tpr[k]=TP/(float)(TP+FN)
            fpr[k]=FP/(float)(FP+TN)
            current_acc = (TP+TN) / (TP+FP+FN+TN)

            if bestAccuracy < current_acc:
                bestAccuracy = current_acc
                bestThresh = threshold[i]
                bestThreshIndex = i

        fpr=np.array(fpr)[::-1]
        tpr=np.array(tpr)[::-1]
        
        auc=trapz(tpr, fpr)
        print(i,auc)
        auc_list[i] = auc
        acc_list[i] = bestAccuracy
    print("list auc:",auc_list)    
    print("list acc:",acc_list)    
    threshold=np.linspace(0., 1., 500)
    tpr=[0]*len(threshold)
    fpr=[0]*len(threshold)
    tp = np.zeros((500))
    fp = np.zeros((500))
    tn = np.zeros((500))
    fn = np.zeros((500))
    bestAccuracy = -1.0
    bestThresh = -1.0
    bestThreshIndex = -1
    for i in range(len(threshold)):
        preds = np.ones((size_predict))
        for j in range(len(embeddings)):
            if embeddings[j][0] >= threshold[i]:
                preds[j] =  0
        TP=sum((preds==0) & (label==0))
        FP=sum((preds==0) &(label==1))
        TN=sum((preds==1) & (label==1))
        FN=sum((preds==1) & (label==0))
        tp[i] = TP
        fp[i] = FP
        tn[i] = TN
        fn[i] = FN
        if TP==0:
            tpr[i]=0
        else:
            tpr[i]=TP/(float)(TP+FN)
            
        fpr[i]=FP/(float)(FP+TN)
        current_acc = (TP+TN) / (TP+FP+FN+TN)

        if bestAccuracy < current_acc:
            bestAccuracy = current_acc
            bestThresh = threshold[i]
            bestThreshIndex = i
    print("tp:",{'test':tp})
    print("fp:",{'test':fp})
    print("tn:",{'test':tn})
    print("fn:",{'test':fn})
    fpr=np.array(fpr)[::-1]
    tpr=np.array(tpr)[::-1]
    
    auc=trapz(tpr, fpr)
    print("AUC",auc)
    print("Best Accuracy",bestAccuracy)
    print("Best Thresh",bestThresh)
    print("ConfMatrix",[[tp[bestThreshIndex],fp[bestThreshIndex]],[tn[bestThreshIndex],fn[bestThreshIndex]]])
"""

if __name__ == '__main__':

    size_predict = 15989*5
    embeddings = np.zeros((size_predict, 2))
    label = np.zeros((size_predict))
    auc_list = np.zeros((5))
    acc_list = np.zeros((5))
    for i in range(5):
        DATA_DIR = "../../../SN_DATA/"
        DATA_FILE = 'sn_base_test_challenge_'+str(i)+'.tfrecords'
        filename = os.path.join(DATA_DIR, DATA_FILE)
        test_dataset = tf.data.TFRecordDataset(filename)


        tf.reset_default_graph()
        tf.logging.set_verbosity(tf.logging.INFO)

        tf.logging.info("Creating the datasets...")
        
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        PATH = os.path.join(ROOT_DIR, '../model_conv_challenge_noredshift_'+str(i))
        
        tf.logging.info("Creating the model...")
        config = tf.estimator.RunConfig(tf_random_seed=230,
                                        model_dir=PATH,
                                        save_summary_steps=500)
        estimator = tf.estimator.Estimator(model_fn, config=config)
        predict_input = lambda : SNClassification_input.challenge_test_input_fn(test_dataset, 128, 1)

        tf.logging.info("Predicting")
        predictions = estimator.predict(predict_input)
        label_per_classifier = np.zeros(15989)
        pred_per_classifieur = np.zeros((15989,2))
        for j, p in enumerate(predictions):
            embeddings[j+int((size_predict/5)*i)] = p['embeddings']
            label[j+int((size_predict/5)*i)]= p['label']
            label_per_classifier[j] =  p['label']
            pred_per_classifieur[j] =  p['embeddings']
        threshold=np.linspace(0., 1., 500)
        tpr=[0]*len(threshold)
        fpr=[0]*len(threshold)
        bestAccuracy = -1.0
        bestThresh = -1.0
        bestThreshIndex = -1
        for k in range(len(threshold)):
            preds = np.ones((15989))
            for j in range(len(pred_per_classifieur)):
                if pred_per_classifieur[j][0] >= threshold[k]:
                    preds[j] =  0
            TP=sum((preds==0) & (label_per_classifier==0))
            FP=sum((preds==0) &(label_per_classifier==1))
            TN=sum((preds==1) & (label_per_classifier==1))
            FN=sum((preds==1) & (label_per_classifier==0))
            if TP==0:
                tpr[k]=0
            else:
                tpr[k]=TP/(float)(TP+FN)
            fpr[k]=FP/(float)(FP+TN)
            current_acc = (TP+TN) / (TP+FP+FN+TN)

            if bestAccuracy < current_acc:
                bestAccuracy = current_acc
                bestThresh = threshold[i]
                bestThreshIndex = i

        fpr=np.array(fpr)[::-1]
        tpr=np.array(tpr)[::-1]
        
        auc=trapz(tpr, fpr)
        print(i,auc)
        auc_list[i] = auc
        acc_list[i] = bestAccuracy
    print("list auc:",auc_list)    
    print("list acc:",acc_list)    
    threshold=np.linspace(0., 1., 500)
    tpr=[0]*len(threshold)
    fpr=[0]*len(threshold)
    tp = np.zeros((500))
    fp = np.zeros((500))
    tn = np.zeros((500))
    fn = np.zeros((500))
    bestAccuracy = -1.0
    bestThresh = -1.0
    bestThreshIndex = -1
    for i in range(len(threshold)):
        preds = np.ones((size_predict))
        for j in range(len(embeddings)):
            if embeddings[j][0] >= threshold[i]:
                preds[j] =  0
        TP=sum((preds==0) & (label==0))
        FP=sum((preds==0) &(label==1))
        TN=sum((preds==1) & (label==1))
        FN=sum((preds==1) & (label==0))
        tp[i] = TP
        fp[i] = FP
        tn[i] = TN
        fn[i] = FN
        if TP==0:
            tpr[i]=0
        else:
            tpr[i]=TP/(float)(TP+FN)
            
        fpr[i]=FP/(float)(FP+TN)
        current_acc = (TP+TN) / (TP+FP+FN+TN)

        if bestAccuracy < current_acc:
            bestAccuracy = current_acc
            bestThresh = threshold[i]
            bestThreshIndex = i
    print("tp:",{'test':tp})
    print("fp:",{'test':fp})
    print("tn:",{'test':tn})
    print("fn:",{'test':fn})
    fpr=np.array(fpr)[::-1]
    tpr=np.array(tpr)[::-1]
    
    print("fpr:",{'test':fpr})
    print("tpr:",{'test':tpr})

    auc=trapz(tpr, fpr)
    print("AUC",auc)
    print("Best Accuracy",bestAccuracy)
    print("Best Thresh",bestThresh)
    print("ConfMatrix",[[tp[bestThreshIndex],fp[bestThreshIndex]],[tn[bestThreshIndex],fn[bestThreshIndex]]])
