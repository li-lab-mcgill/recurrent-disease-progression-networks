import os
import pickle
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve, roc_curve
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K
import argparse

def load_data(path):    
    with open(path + '/cms.pickle', 'rb') as f:
        cms = pickle.load(f)

    with open(path + '/hf_labels.pickle', 'rb') as f:
        labels = pickle.load(f)

    with open(path + '/train_ids.pickle', 'rb') as f:
        train_id = pickle.load(f)

    with open(path + '/test_ids.pickle', 'rb') as f:
        test_id = pickle.load(f)

    with open(path + '/remaining_patient_ids.pickle', 'rb') as f:
        all_patient_ids = pickle.load(f)

    return cms, labels, train_id, test_id, all_patient_ids

def train_test_preprocess(cms, labels, train_id, test_id, all_patient_ids):    
    train_labels = []
    train_cms = []

    test_labels = []
    test_cms = []
    ordered_test_ids = []

    for patient_id, cm, target in tqdm(zip(all_patient_ids, cms, labels)):
        if patient_id in train_id:
            train_labels.append(target)
            train_cms.append(cm)
        else:
            test_labels.append(target)
            test_cms.append(cm)
            ordered_test_ids.append(patient_id)

    return train_labels, train_cms, test_labels, test_cms, ordered_test_ids

def get_baseline_model(model_path):
    model = joblib.load(model_path)
    return model

def test_flat(test_labels, test_cms):
    test_flat_labels = []
    test_flat_cms = []

    for label, cm, in zip(test_labels, test_cms):
        test_flat_labels.extend(label)
        test_flat_cms.extend(cm)

    test_flat_cms = np.vstack(test_flat_cms)
    test_flat_labels = np.array(test_flat_labels)
    
    return test_flat_cms, test_flat_labels

def pr_roc_lr_svm(model, mode, X_test, y_test):
    if(mode == 'lr'):
        y_proba = model.predict_proba(X_test)[:, 1]
    if(mode == 'svm'):
        y_proba = model.decision_function(X_test)
    
    precision, recall, thresh_pr = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)   
    fpr, tpr, thresh_roc = roc_curve(y_test, y_proba)
    roc = roc_auc_score(y_test, y_proba)

    
    return precision, recall, thresh_pr, ap, fpr, tpr, thresh_roc, roc

def balanced_crossentropy(y_true, y_pred, alpha=0.5):
    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1. - y_pred)
    p_t = K.clip(p_t, K.epsilon(), 1. - K.epsilon())

    a_t = tf.where(tf.equal(y_true, 1), alpha, 1. - alpha)

    return K.mean(-a_t * K.log(p_t), axis=-1)

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.):
    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1. - y_pred)
    p_t = K.clip(p_t, K.epsilon(), 1. - K.epsilon())

    a_t = tf.where(tf.equal(y_true, 1), alpha, 1. - alpha)

    return K.mean(-a_t * K.pow(1. - p_t, gamma) * K.log(p_t), axis=-1)

def compute_flat_pred(model, X_test, test_labels, is_dhtm=False):
    with tf.device('/cpu:0'):
        y_pred = model.predict(X_test)
    
    if is_dhtm:
        y_pred = y_pred[0]
    
    y_pred = np.squeeze(y_pred).tolist()
    
    true_test = []
    pred_test = []

    for pred, true in zip(y_pred, test_labels):
        pred_test.extend(pred[-len(true):])
        true_test.extend(true)

    true_test = np.array(true_test)
    pred_test = np.array(pred_test)
    
    return true_test, pred_test

def get_model(model_path):
    with tf.device('/cpu:0'):
        model = load_model(
            model_path,
            custom_objects={'focal_loss': focal_loss, 'balanced_crossentropy': balanced_crossentropy}
        )
    return model

def pr_roc(model, mode, test_cms, test_labels):
    X_test = pad_sequences(test_cms, value=-1).astype(float)
    if(mode == 'dhtmc'):
        y_test_flat, y_pred = compute_flat_pred(model, X_test, test_labels, is_dhtm=True)
    else:
        y_test_flat, y_pred = compute_flat_pred(model, X_test, test_labels, is_dhtm=False)
    precision, recall, thresh_pr = precision_recall_curve(y_test_flat, y_pred)
    ap = average_precision_score(y_test_flat, y_pred)
    fpr, tpr, thresh_roc = roc_curve(y_test_flat, y_pred)
    roc = roc_auc_score(y_test_flat, y_pred)
    return precision, recall, thresh_pr, ap, fpr, tpr, thresh_roc, roc

def plot_pr_zoomed(mode, precision, recall, ap):
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'{mode} (AUC={ap:.4f})')
    plt.xlim(0, 0.2)
    plt.savefig(f'plots/pr_curve_{mode}_zoomed.pdf')

def plot_pr(mode, precision, recall, ap):
    plt.figure(figsize=(10, 5))
    plt.plot(recall, precision, label=f'{mode} (AUC={ap:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(f'plots/pr_curve_{mode}.pdf')

def plot_roc_zoomed(mode, fpr, tpr, roc):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'{mode} (AUC={roc:.4f})')
    plt.xlim(0, 0.2)
    plt.savefig(f'plots/roc_curve_{mode}_zoomed.pdf')

def plot_roc(mode, fpr, tpr, roc):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'{mode} (AUC={roc:.4f})')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate(TPR)')
    plt.legend()
    plt.savefig(f'plots/roc_curve_{mode}.pdf')

def parse_arguments(parser):
    parser.add_argument('path', type=str, metavar='<path>', help='Enter the relative path to the folder containing the patient data')
    parser.add_argument('mode', type=str, metavar='<mode>', choices = ['lr', 'svm', 'standard', 'dhtm', 'dhtmc'], help='Choose a mode among: lr, svm, standard, dhtm, and dhtmc')
    parser.add_argument('model_path', type=str, metavar='<model_path>', help='Enter the relative path to the model file. E.g. \'./model/dhtmc.h5\'')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parse_arguments(parser)
    args = parser.parse_args()
    
    path = args.path
    mode = args.mode
    model_path = args.model_path
    
    cms, labels, train_id, test_id, all_patient_ids = load_data(path)
    train_labels, train_cms, test_labels, test_cms, ordered_test_ids = train_test_preprocess(cms, labels, train_id, test_id, all_patient_ids)
    test_flat_cms, test_flat_labels = test_flat(test_labels, test_cms)
    if (mode == 'svm' or mode == 'lr'):
        model = get_baseline_model(model_path)
        precision, recall, thresh_pr, ap, fpr, tpr, thresh_roc, roc = pr_roc_lr_svm(model, mode, test_flat_cms, test_flat_labels)
    else:
        model = get_model(model_path)
        precision, recall, thresh_pr, ap, fpr, tpr, thresh_roc, roc = pr_roc(model, mode, test_cms, test_labels)
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plot_pr_zoomed(mode, precision, recall, ap)
    plot_pr(mode, precision, recall, ap)
    plot_roc_zoomed(mode, fpr, tpr, roc)
    plot_roc(mode, fpr, tpr, roc)



