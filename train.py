import os
import pickle
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import Model, Sequential, load_model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Input, Masking, concatenate, LayerNormalization, Add, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tqdm import tqdm
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
    train_cms = []
    train_labels = []
    test_cms = []
    test_labels = []

    for patient_id, cm, target in tqdm(zip(all_patient_ids, cms, labels)):
        if patient_id in train_id:
            train_cms.append(cm)
            train_labels.append(target)
        else:
            test_cms.append(cm)
            test_labels.append(target)
    return train_cms, train_labels, test_cms, test_labels

def train_test_preprocess_dhtmc(cms, cm_labels, train_id, test_id, all_patient_ids, hf_labels):    

    train_labels_cm = []
    train_labels_hf = []
    train_cms = []
    test_labels_cm = []
    test_labels_hf = []
    test_cms = []

    for patient_id, cm, hf_label, cm_label in zip(all_patient_ids, cms, hf_labels, cm_labels):
        if patient_id in train_id:
            train_labels_hf.append(hf_label)
            train_labels_cm.append(cm_label)
            train_cms.append(cm)
        else:
            test_labels_hf.append(hf_label)
            test_labels_cm.append(cm_label)
            test_cms.append(cm)
    return train_cms, train_labels_cm, test_cms, test_labels_cm, train_labels_hf, test_labels_hf

def train_test_flat(train_cms, train_labels, test_cms, test_labels):
            
    train_flat_labels = []
    train_flat_cms = []

    for label, cm in zip(train_labels, train_cms):
        train_flat_labels.extend(label)
        train_flat_cms.extend(cm)

    train_flat_cms = np.vstack(train_flat_cms)
    train_flat_labels = np.array(train_flat_labels)
    
    test_flat_labels = []
    test_flat_cms = []

    for label, cm, in zip(test_labels, test_cms):
        test_flat_labels.extend(label)
        test_flat_cms.extend(cm)

    test_flat_cms = np.vstack(test_flat_cms)
    test_flat_labels = np.array(test_flat_labels)
    
    X_train = train_flat_cms
    y_train = train_flat_labels

    X_test = test_flat_cms
    y_test = test_flat_labels
    
    ros = RandomOverSampler(random_state=0)
    train_flat_cms, train_flat_labels = ros.fit_resample(train_flat_cms, train_flat_labels)

    return train_flat_cms, train_flat_labels, test_flat_cms, test_flat_labels

def logistic_regression(path, X_train, y_train, y_test):
    model = LogisticRegressionCV(solver='lbfgs', max_iter=500)
    model.fit(X_train, y_train)
    joblib.dump(model, path + '/models/logistic.joblib')
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

def svm(path, X_train, y_train, y_test):
    svm = LinearSVC(dual=False)
    grid = GridSearchCV(svm, param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100]})
    grid.fit(X_train, y_train)

    joblib.dump(grid, path + '/models/svm.joblib')
    
    y_pred = grid.predict(X_test)
    y_proba = grid.decision_function(X_test)

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

def build_regular_model(input_dim, rnn_fn, n_layers=2, hidden_dim=64):
    input1 = Input(shape=(None, input_dim))
    mask1 = Masking(-1)(input1)
    layers = Dense(hidden_dim)(mask1)
    for i in range(n_layers):
        layers = rnn_fn(hidden_dim, return_sequences=True, name=f'GRU_layer_{i+1}')(layers)
    out = Dense(1, activation='sigmoid')(layers)

    model = Model(inputs=input1, outputs=out)
    
    return model

def build_dhtm(input_dim, rnn_fn, n_layers=2, hidden_dim=64):
    input1 = Input(shape=(None, input_dim))
    mask1 = Masking(-1)(input1)
    layers = Dense(hidden_dim)(mask1)
    for i in range(n_layers):
        fn = rnn_fn(hidden_dim, return_sequences=True, name=f'GRU_L{i+1}')(layers)
        fn = LayerNormalization(name=f'LN_L{i+1}')(fn)
        fn = Dropout(0.2, name=f"drop_L{i+1}")(fn)
        layers = Add(name=f'Residual_L{i+1}')([layers, fn])
    
    out = Dense(1, activation='sigmoid')(layers)

    model = Model(inputs=input1, outputs=out)
    
    return model

def build_dhtmc(input_dim, rnn_fn, n_cm, n_layers, hidden_dim=64):
    input1 = Input(shape=(None, input_dim))
    mask1 = Masking(-1)(input1)
    
    layers1 = Dense(hidden_dim, activation='relu')(mask1)
    for _ in range(n_layers):
        layers1 = rnn_fn(hidden_dim, return_sequences=True)(layers1)
    out_cm = Dense(n_cm, activation='sigmoid', name='cm_pred')(layers1)
    
    layers2 = Dense(hidden_dim)(concatenate([input1, out_cm]))
    for _ in range(n_layers):
        layers2 = rnn_fn(hidden_dim, return_sequences=True)(layers2)
    
    concatenated = concatenate([layers1, layers2, out_cm])
    layers3 = Dense(hidden_dim, activation='relu')(concatenated)
    layers3 = Dense(hidden_dim, activation='relu')(layers3)
    out_hf = Dense(1, activation='sigmoid', name='hf_pred')(layers3)
    
    model = Model(inputs=input1, outputs=[out_hf, out_cm])
    
    return model

def pad(train_cms, train_labels, test_cms, test_labels):
    X_train = pad_sequences(train_cms, value=-1).astype(float)
    y_train = np.expand_dims(pad_sequences(train_labels), axis=-1)

    X_test = pad_sequences(test_cms, value=-1).astype(float)
    y_test = pad_sequences(test_labels)
    
    return X_train, y_train, X_test, y_test

def pad_dhtmc(train_cms, train_labels_cm, test_cms, test_labels_cm, train_labels_hf, test_labels_hf):
    X_train = pad_sequences(train_cms, value=-1).astype(float)
    y_train_hf = np.expand_dims(pad_sequences(train_labels_hf), axis=-1)
    y_train_cm = pad_sequences(train_labels_cm)

    X_test = pad_sequences(test_cms, value=-1).astype(float)
    y_test_hf = np.expand_dims(pad_sequences(test_labels_hf), axis=-1)
    y_test_cm = pad_sequences(test_labels_cm)

    return X_train, y_train_cm, X_test, y_test_cm, y_train_hf, y_test_hf

def make_dir(path, mode, loss_name):
    for dir1 in ['models', 'model_history']:
        if not os.path.exists(path + f'/{dir1}/{mode}_{loss_name}'):
            os.makedirs(path + f'/{dir1}/{mode}_{loss_name}')

def define_structure():
    loss_map = {
        'bce': 'binary_crossentropy',
        'focal_loss': focal_loss,
        'balanced_bce': balanced_crossentropy
    }

    architecture_map = {
        'lstm': LSTM,
        'gru': GRU,
        'rnn': SimpleRNN
    }
   
    return loss_map, architecture_map

def train_baseline(path, mode, loss_name, architecture, n_layers, X_train, y_train):
    loss_map, architecture_map = define_structure()
    loss_fn = loss_map[loss_name]
    rnn_fn = architecture_map[architecture]

    print("="*50)
    print("Mode:", mode)
    print("Architecture:", architecture)
    print('Loss:', loss_name)
    print("Number of layers:", n_layers)
    print('-'*50)

    make_dir(path, mode, loss_name)
    callbacks = [ModelCheckpoint(path + f'/models/{mode}_{loss_name}/{architecture}_{n_layers}_layer.h5', save_best_only=True)]

    with tf.device('/cpu:0'):
        if(mode == 'standard'):
            model = build_regular_model(X_train.shape[-1], rnn_fn, n_layers=n_layers)
        else:
            model = build_dhtm(X_train.shape[-1], rnn_fn, n_layers=n_layers)
        model.compile(Adam(lr=0.0005), loss=loss_fn, metrics=['accuracy'])
        train_hist = model.fit(
            x=X_train, 
            y=y_train, 
            batch_size=32, 
            epochs=30, 
            callbacks=callbacks,
            verbose=0,
            validation_split=0.15
        )

    hist_df = pd.DataFrame(train_hist.history)
    hist_df.to_csv(path + f'/model_history/{mode}_{loss_name}/history_{architecture}_{n_layers}_layer.csv')

def train_dhtmc(path, mode, loss_name, architecture, n_layers, X_train, y_train_cm, y_train_hf):
    loss_map, architecture_map = define_structure()
    loss_fn = loss_map[loss_name]
    rnn_fn = architecture_map[architecture]

    print("="*50)
    print("Mode:", mode)
    print("Architecture:", architecture)
    print('Loss:', loss_name)
    print("Number of layers:", n_layers)
    print('-'*50)

    make_dir(path, mode, loss_name)
    callbacks = [ModelCheckpoint(path + f'/models/{mode}_{loss_name}/{architecture}_{n_layers}_layer.h5', save_best_only=True)]

    with tf.device('/cpu:0'):
        if(mode == 'dhtmc'):
            model = build_dhtmc(X_train.shape[-1], rnn_fn, y_train_cm.shape[-1], n_layers=n_layers)
            model.compile(Adam(lr=0.0005), loss=loss_fn, metrics=['accuracy'])
            train_hist = model.fit(
                x=X_train, 
                y=[y_train_hf, y_train_cm], 
                batch_size=32, 
                epochs=30, 
                callbacks=callbacks,
                verbose=0,
                validation_split=0.15
            )

    hist_df = pd.DataFrame(train_hist.history)
    hist_df.to_csv(path + f'/model_history/{mode}_{loss_name}/history_{architecture}_{n_layers}_layer.csv')

def parse_arguments(parser):
    parser.add_argument('path', type=str, metavar='<path>', help='Enter the relative path to the folder containing the patient data')
    parser.add_argument('mode', type=str, metavar='<mode>', choices = ['lr', 'svm', 'standard', 'dhtm', 'dhtmc'], help='Choose a mode among: lr, svm, standard, dhtm, and dhtmc')
    parser.add_argument('loss_name', type=str, metavar='<loss_name>', choices = ['bce', 'focal_loss', 'balanced_bce', 'none'], help='Choose a loss function in: bce, focal_loss, and balanced_bce')
    parser.add_argument('architecture', type=str, metavar='<architecture>', choices = ['gru', 'lstm', 'rnn', 'none'], help='Choose an architecture in: gru, lstm, and rnn')
    parser.add_argument('n_layers', type=int, metavar='<n_layers>', help='Enter the number of layers in RNN')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parse_arguments(parser)
    args = parser.parse_args()

    path = args.path
    mode = args.mode
    loss_name = args.loss_name
    architecture = args.architecture
    n_layers = args.n_layers

    print("Loading data...")
    if(mode == 'dhtmc'):
        cms, hf_labels, train_id, test_id, all_patient_ids= load_data(path)
        with open(path + '/cm_labels.pickle', 'rb') as f:
            cm_labels = pickle.load(f)
        train_cms, train_labels, test_cms, test_labels, train_labels_hf, test_labels_hf = train_test_preprocess_dhtmc(cms, cm_labels, train_id, test_id, all_patient_ids, hf_labels)
    else:
        cms, labels, train_id, test_id, all_patient_ids = load_data(path)
        train_cms, train_labels, test_cms, test_labels = train_test_preprocess(cms, labels, train_id, test_id, all_patient_ids)
    print("Start training...")
    if(mode == 'lr' or mode == 'svm'):
        if not os.path.exists(path + '/models'):
            os.makedirs(path + '/models')
        if(mode == 'lr'):
            X_train, y_train, X_test, y_test = train_test_flat(train_cms, train_labels, test_cms, test_labels)
            logistic_regression(path, X_train, y_train, y_test)
        else:
            X_train, y_train, X_test, y_test = train_test_flat(train_cms, train_labels, test_cms, test_labels)
            svm(path, X_train, y_train, y_test)
    else:
        if(mode == 'dhtmc'):
            X_train, y_train_cm, X_test, y_test, y_train_hf, y_test_hf = pad_dhtmc(train_cms, train_labels, test_cms, test_labels, train_labels_hf, test_labels_hf)
            train_dhtmc(path, mode, loss_name, architecture, n_layers, X_train, y_train_cm, y_train_hf)
        else:    
            X_train, y_train, X_test, y_test = pad(train_cms, train_labels, test_cms, test_labels)
            train_baseline(path, mode, loss_name, architecture, n_layers, X_train, y_train)
    print("Finish")




