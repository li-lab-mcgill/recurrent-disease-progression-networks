import os
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, recall_score, precision_score
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import Model, Sequential, load_model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, GRU, Dense, Input, Masking, concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tqdm import tqdm
import argparse

def load_data_dhtmc(path):
    with open(path + '/cms.pickle', 'rb') as f:
        cms = pickle.load(f)

    with open(path + '/hf_labels.pickle', 'rb') as f:
        hf_labels = pickle.load(f)

    with open(path + '/train_ids.pickle', 'rb') as f:
        train_id = pickle.load(f)

    with open(path + '/test_ids.pickle', 'rb') as f:
        test_id = pickle.load(f)

    with open(path + '/remaining_patient_ids.pickle', 'rb') as f:
        all_patient_ids = pickle.load(f)
        
    with open(path + '/cm_labels.pickle', 'rb') as f:
        cm_labels = pickle.load(f)
        
    return cms, hf_labels, train_id, test_id, all_patient_ids, cm_labels

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

def update_X_regular(X, y_pred_hf, hf_thresh):
    new_hfs = y_pred_hf.squeeze()[:, -1]
    new_hfs = (new_hfs > hf_thresh).astype(np.float64)
    
    last_timestep = X[:, -1, :]
    last_age = last_timestep[:, -1]
    
    new_timestep = last_timestep.copy()
    new_timestep[:, -1] = last_age + 0.5
    new_timestep[:, -2] = new_hfs
    new_timestep[:, -5:-2] = 0
    new_timestep = np.expand_dims(new_timestep, axis=1)
    
    X_new = np.concatenate([X, new_timestep], axis=1)
    
    return X_new

def update_X_dhtm(X, y_pred_hf, y_pred_cm, hf_thresh, cm_thresh):
    new_hfs = y_pred_hf.squeeze()[:, -1]
    new_cms = y_pred_cm[:, -2, :]
    
    # thresholding
    new_cms = (new_cms > cm_thresh).astype(np.float64)
    new_hfs = (new_hfs > hf_thresh).astype(np.float64)

    last_timestep = X[:, -1, :]
    last_age = last_timestep[:, -1]
    
    new_timestep = last_timestep.copy()
    new_timestep[:, -1] = last_age + 0.5
    new_timestep[:, -2] = new_hfs
    new_timestep[:, -5:-2] = new_cms
    new_timestep = np.expand_dims(new_timestep, axis=1)
    
    X_new = np.concatenate([X, new_timestep], axis=1)
    
    return X_new

def get_model(mode, model_path):
    with tf.device('/cpu:0'):
        if(mode == 'dhtmc' or 'dhtm'):
            model = load_model(
                model_path,
                custom_objects={'focal_loss': focal_loss, 'balanced_crossentropy': balanced_crossentropy}
            )
        else:
            model = load_model(model_path)
    return model

def compute_threshold_regular(model, X_test, y_test):
    with tf.device('/cpu:0'):
        y_pred_hf = model.predict(X_test)
    y_pred_hf = y_pred_hf.squeeze()

    hf_thresh = np.extract(y_test, y_pred_hf).mean()
    
    return hf_thresh

def compute_threshold_dhtm(model, X_test, y_test, X_test_cm_labels):
    y_test_cm = X_test_cm_labels

    with tf.device('/cpu:0'):
        y_pred_hf, y_pred_cm = model.predict(X_test)
    y_pred_hf = y_pred_hf.squeeze()

    hf_thresh = np.extract(y_test, y_pred_hf).mean()
    cm_thresh = [
        np.extract(y_test_cm[:, :, i], y_pred_cm[:, :, i]).mean()
        for i in range(3)
    ]
    
    return hf_thresh, cm_thresh

def predict_X_dhtmc(model, k, X_test, dhtm_hf_thresh, dhtm_cm_thresh):
    X_new = X_test.copy()
    for i in range(k+1):
        with tf.device('/cpu:0'):
            y_pred_hf, y_pred_cm = model.predict(X_new)
        X_new = update_X_dhtm(X_new, y_pred_hf, y_pred_cm, dhtm_hf_thresh, dhtm_cm_thresh)
    return X_new, y_pred_hf

def predict_X(model, k, X_test, thresh):
    X_new = X_test.copy()
    for i in range(k+1):
        with tf.device('/cpu:0'):
            y_pred_hf = model.predict(X_new)
        X_new = update_X_regular(X_new, y_pred_hf, thresh)
    return X_new, y_pred_hf

def compute_ap_roc(y_test_prev, y_pred_prev):
    ap_score = average_precision_score(y_test_prev, y_pred_prev)
    roc_auc = roc_auc_score(y_test_prev, y_pred_prev)
    return ap_score, roc_auc

def compute_traj(model, mode, all_patient_ids, cms, hf_labels, cm_labels, test_id):
    for T_p in tqdm([30]):
        k = 20

        test_cms = []
        traj_labels = []
        prev_labels = []
        test_cm_labels = []

        for patient_id, cm, hf_label, cm_label in zip(all_patient_ids, cms, hf_labels, cm_labels):
            if patient_id in test_id and hf_label.shape[0] > k + T_p:
                test_cms.append(cm[:T_p])
                traj_labels.append(hf_label[T_p:k+T_p])
                prev_labels.append(hf_label[:T_p])
                test_cm_labels.append(cm[:T_p])

        X_test = pad_sequences(test_cms, value=-1).astype(float)
        X_test_cm_labels = pad_sequences(test_cm_labels, value=-1).astype(float)
        y_test_traj = np.array(traj_labels)
        y_test_prev = np.array(prev_labels)  # Label before the trajectory cutoff
        print("="*50)
        print()
        print(y_test_traj.shape)

        # First, compute threshold
        if (mode == 'dhtmc'):
            hf_thresh, cm_thresh = compute_threshold_dhtm(model, X_test, y_test_prev, X_test_cm_labels)
            print("HF Threshold:", hf_thresh)
            print("CM Threshold:", cm_thresh)
            X_new, y_pred_hf = predict_X_dhtmc(model, k, X_test, hf_thresh, cm_thresh)
        else:
            hf_thresh = compute_threshold_regular(model, X_test, y_test_prev)
            print("HF Threshold:", hf_thresh)
            X_new, y_pred_hf = predict_X(model, k, X_test, hf_thresh)

        y_pred_prev = y_pred_hf.squeeze()[:, :-k]
        y_pred_traj = y_pred_hf.squeeze()[:, -k:]

        ap_roc_scores = []
        pr_scores = []
        interval = 2

        # ######################## SINGLE TIME STEP ########################
        print("="*50)
        print("Single Time Step")
        print("T_p:", T_p)
        print('-'*50)

        y_pred_prev = y_pred_prev.reshape(-1)
        y_test_prev = y_test_prev.reshape(-1)

        ap_score, roc_auc = compute_ap_roc(y_test_prev, y_pred_prev)
        print("Score:")
        print('AP Score:', ap_score)
        print("ROC AUC:", roc_auc)

        # ######################## TRAJECTORY ########################
        for i in range(0, k, interval):
            print("="*50)
            print("Trajectory")
            print("T_p:", T_p)
            print("i = ", i, 'to', i+interval)
            print('-'*50)
            
            y_pred_single = y_pred_traj[:, i:i+interval].reshape(-1)
            y_test_single = y_test_traj[:, i:i+interval].reshape(-1)

            ap_score, roc_auc = compute_ap_roc(y_test_single, y_pred_single)
            recall_score, precision_score = compute_precision_recall(i, interval, y_pred_traj, y_test_single, hf_thresh)
            
            print("Score:")
            print('AP Score:', ap_score)
            print("ROC AUC:", roc_auc)
            print('Recall Score:', recall_score)
            print('Precision Score:', precision_score)

            ap_roc_scores.append([i, ap_score, roc_auc])
            pr_scores.append([i, i+2, recall_score, precision_score])

        prediction_store_path = f'predictions/T_p_{T_p}/'
        make_dirs(prediction_store_path, T_p)

        np.save(prediction_store_path + 'y_test_prev.npy', y_test_prev)
        np.save(prediction_store_path + 'y_pred_prev.npy', y_pred_prev)

        np.save(prediction_store_path + 'y_test_traj.npy', y_test_traj)
        np.save(prediction_store_path + 'y_pred_traj.npy', y_pred_traj)

        ap_roc_columns = ['i', 'ap_score', 'roc_auc']
        ap_roc_results = pd.DataFrame(ap_roc_scores, columns=ap_roc_columns)
        ap_roc_results.to_csv(f'results/binned_time_scores_Tp_{T_p}.csv')
        pr_columns = ['timestep_start', 'timestep_end', 'recall_score', 'precision_score']
        pr_results = pd.DataFrame(pr_scores, columns=pr_columns)
        pr_results.to_csv(f'results/trajectory_thresholded_recall_precision_Tp_{T_p}.csv')
        
        return ap_roc_results, pr_results

def make_dirs(prediction_store_path, T_p):
    if not os.path.exists(prediction_store_path):
        os.makedirs(prediction_store_path)
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('plots'):
        os.makedirs('plots')

def plot_ap(results):
    results['ap_score'].plot(kind='bar', figsize=(12, 5))
    plt.xlabel('Number of Years')
    plt.ylabel('AP Score')
    plt.savefig('plots/trajectory_ap_scores_kth_time.pdf')

def plot_roc(results):
    results['roc_auc'].plot(kind='bar', figsize=(12, 5))
    plt.xlabel('Number of Years')
    plt.ylabel('ROC AUC Score')
    plt.savefig('plots/trajectory_roc_auc_kth_time.pdf')

def compute_precision_recall(step, interval, y_pred_traj, y_test_thresholded, hf_thresh):
    y_pred_thresholded = (y_pred_traj[:, step:step+interval].reshape(-1) > hf_thresh).astype(int)
    recall = metrics.recall_score(y_test_thresholded, y_pred_thresholded)
    precision = metrics.precision_score(y_test_thresholded, y_pred_thresholded)
    
    return recall, precision

def plot_recall(results):
    results[['timestep_start'] + ['recall_score']].plot(kind='bar', figsize=(12, 5), x='timestep_start')
    plt.xlabel('Number of years after age of 40')
    plt.ylabel('Recall Score')
    plt.savefig('plots/trajectory_kth_time_recall.pdf')

def plot_precision(results):
    results[['timestep_start'] + ['precision_score']].plot(kind='bar', figsize=(12, 5), x='timestep_start')
    plt.xlabel('Number of years after age of 40')
    plt.ylabel('Precision Score')
    plt.savefig('plots/trajectory_kth_time_precision.pdf')

def parse_arguments(parser):
    parser.add_argument('path', type=str, metavar='<path>', help='Enter the relative path to the folder containing the patient data')
    parser.add_argument('mode', type=str, metavar='<mode>', choices = ['standard', 'dhtm', 'dhtmc'], help='Choose a mode among: standard, dhtm, and dhtmc')
    parser.add_argument('model_path', type=str, metavar='<model_path>', help='Enter the relative path to the model file. E.g. \'./model/dhtmc.h5\'')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parse_arguments(parser)
    args = parser.parse_args()
    path = args.path
    mode = args.mode
    model_path = args.model_path

    cms, hf_labels, train_id, test_id, all_patient_ids, cm_labels = load_data_dhtmc(path)
    model = get_model(mode, model_path)

    ap_roc_results, pr_results = compute_traj(model, mode, all_patient_ids, cms, hf_labels, cm_labels, test_id)
    plot_ap(ap_roc_results)
    plot_roc(ap_roc_results)
    plot_recall(pr_results)
    plot_precision(pr_results)


