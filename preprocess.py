import os
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Input, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tqdm import tqdm
import argparse

def define_columns():
    surgeries_cols = [f'age_allSX{i}' for i in range(1, 14)]

    birth_cols = ['sex', 'Lesion']

    diagnoses_cols = [
        'CAD',
        'ARR',
        'VARR',
        'PH',
        'DIAB',
        'Stroke',
        'CLD',
        'CKD',
    ]

    temporal_cols = [
        'AMI',
        'IE',
        'Sepsis'
    ]

    ignored_cols = [
        'deces',
        'birth',
        'ADMDATE',
        'timezero',
        'age_congSX1',
        'age_congSX2',
        'age_congSX3',
        'age_congSX4',
        'age_congSX5',
        'age_congSX6',
        'age_congSX7',
        'age_congSX8',
    ]

    diagnoses_time_col = list(map(lambda start_time: f'age_1st{start_time}', diagnoses_cols))
    temporal_time_col = list(map(lambda start_time: f'age_1st{start_time}', temporal_cols))
    return surgeries_cols, birth_cols, diagnoses_time_col, temporal_time_col, ignored_cols

def create_surgery_record(patient_data, surgeries_cols):
    surgeries = patient_data[['nam'] + surgeries_cols].copy()
    surgeries = surgeries.drop_duplicates()
    surgery_record = surgeries.melt(id_vars=['nam'])
    surgery_record = surgery_record.dropna().sort_values('nam')
    surgery_record = surgery_record.reset_index(drop=True).drop(columns=['variable'])
    # Discretize the age to be in a 6-month interval
    surgery_record.value = surgery_record.value.apply(lambda age: round(age * 2) / 2)
    return surgery_record

def create_cm_record(patient_data, diagnoses_time_col):
    diagnoses = patient_data[['nam'] + diagnoses_time_col].drop_duplicates()
    diagnoses_record = diagnoses.melt(id_vars=['nam']).sort_values(['nam', 'variable'])
    diagnoses_record = diagnoses_record.reset_index(drop=True).fillna(np.inf)
    diagnoses_record = diagnoses_record.set_index('nam')
    return diagnoses_record

def create_birth_record(patient_data, birth_cols):
    birth_record = patient_data[['nam'] + birth_cols].drop_duplicates()
    birth_record = birth_record.reset_index(drop=True).sort_values(['nam'])
    birth_record = pd.get_dummies(birth_record, birth_cols)
    birth_record = birth_record.set_index('nam')
    return birth_record
    
def create_hf_record(patient_data):
    non_hf_duration_threshold = 10
    only_select_hf_positive = True
    hf_record = patient_data[['nam', 'HF', 'ageT0', 'age_end']].copy()

    if only_select_hf_positive:
        hf_record = hf_record.dropna()
    else:
        # Now, if patient ageT0 is missing, we automatically set it to a fixed # of years
        # before the end of their studies
        hf_record.ageT0 = hf_record.ageT0.fillna(
            np.maximum(hf_record.age_end - non_hf_duration_threshold, 0)
        )

    hf_record = hf_record.set_index('nam')
    # Round HF age to nearest 6-month interval (0.5 year), and round up the end age
    hf_record.ageT0 = np.round(hf_record.ageT0 * 2.) / 2.
    hf_record.age_end = np.ceil(hf_record.age_end * 2.) / 2.

    return hf_record

def compute_patient_cms_labels(patient_id, patient_hf_rec, patient_diag_rec, patient_birth_rec,
                               patient_temp_rec, surgery_record, age_start, age_end, 
                               min_age=40.):
    patient_cms = []
    patient_hf_labels = []
    patient_cm_labels = []
    
    age_start = max(min_age, age_start)

    for age in np.arange(age_start, age_end+1, 0.5):        
        # Compute the birth records and permanent CMs
        single_birth_cm = patient_birth_rec.values
        single_diag_cm = np.array(patient_diag_rec.value < age, dtype=int)
        
        # Compute temporal CMs and surgery
        single_temp_cm = np.array(patient_temp_rec.value == age, dtype=int)
        single_surg_label = np.any((surgery_record.nam == patient_id) & (surgery_record.value == age)).astype(int)

        # Compute current and next HF
        hf_current = np.any((patient_hf_rec.ageT0 == age) & (patient_hf_rec.HF == 1)).astype(int)
        hf_next = np.any((patient_hf_rec.ageT0 == age + 0.5) & (patient_hf_rec.HF == 1)).astype(int)
        
        # Compute next temporal CMs
        single_temp_cm = np.array(patient_temp_rec.value == age + 0.5, dtype=int)
        
        # Concatenate the CMs
        single_cms = np.concatenate([
            single_birth_cm, single_diag_cm, single_temp_cm,
            np.array([single_surg_label, hf_current, age])
        ])

        patient_cms.append(single_cms)
        patient_hf_labels.append(hf_next)
        patient_cm_labels.append(single_temp_cm)

    patient_cms = np.stack(patient_cms)
    patient_cm_labels = np.stack(patient_cm_labels)
    patient_hf_labels = np.array(patient_hf_labels)

    return patient_cms, patient_cm_labels, patient_hf_labels

def compute_cms_labels(hf_record, birth_record, diagnoses_record, temporal_record, surgery_record):
    all_patient_ids = hf_record.index.unique()
    remaining_patient_ids = []
    cms = []
    hf_labels = []
    cm_labels = []

    for patient_id in tqdm(all_patient_ids):
        patient_birth_rec = birth_record.loc[patient_id]
        patient_diag_rec = diagnoses_record.loc[patient_id]
        patient_hf_rec = hf_record.loc[patient_id]
        patient_temp_rec = temporal_record.loc[patient_id]

        age_start = 40.
        age_end = patient_hf_rec.age_end.max()

        if age_end < 40.:
            continue

        patient_cms, patient_cm_labels, patient_hf_labels = compute_patient_cms_labels(
            patient_id,
            patient_hf_rec, 
            patient_diag_rec, 
            patient_birth_rec, 
            patient_temp_rec,
            surgery_record, 
            age_start, 
            age_end
        )

        cms.append(patient_cms)
        hf_labels.append(patient_hf_labels)
        cm_labels.append(patient_cm_labels)
        remaining_patient_ids.append(patient_id)
    return cms, hf_labels, cm_labels, remaining_patient_ids

def dataset_split(remaining_patient_ids):
    train_id, test_id = train_test_split(remaining_patient_ids, test_size=0.25, random_state=2019)
    train_id = set(train_id)
    test_id = set(test_id)
    return train_id, test_id

def save_processed_data(path, cms, hf_labels, cm_labels, train_id, test_id, remaining_patient_ids):        
    with open(path + '/cms.pickle', 'wb') as f:
        pickle.dump(cms, f)
    with open(path + '/hf_labels.pickle', 'wb') as f:
        pickle.dump(hf_labels, f)
    with open(path + '/cm_labels.pickle', 'wb') as f:
        pickle.dump(cm_labels, f)
    with open(path + '/train_ids.pickle', 'wb') as f:
        pickle.dump(train_id, f)
    with open(path + '/test_ids.pickle', 'wb') as f:
        pickle.dump(test_id, f)
    with open(path + '/remaining_patient_ids.pickle', 'wb') as f:
        pickle.dump(remaining_patient_ids, f)

def parse_arguments(parser):
    parser.add_argument('path', type=str, metavar='<path>', help='Enter the relative path to the folder containing patient_data.csv')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parse_arguments(parser)
    args = parser.parse_args()
    path = args.path
    
    patient_data = pd.read_csv(path + '/patient_data.csv')
    surgeries_cols, birth_cols, diagnoses_time_col, temporal_time_col, ignored_cols = define_columns()
    
    surgery_record = create_surgery_record(patient_data, surgeries_cols)
    diagnoses_record = create_cm_record(patient_data, diagnoses_time_col)
    temporal_record = create_cm_record(patient_data, temporal_time_col)
    birth_record = create_birth_record(patient_data, birth_cols)
    hf_record = create_hf_record(patient_data)
    
    cms, hf_labels, cm_labels, remaining_patient_ids = compute_cms_labels(hf_record, birth_record, diagnoses_record, temporal_record, surgery_record)
    train_id, test_id = dataset_split(remaining_patient_ids)
    save_processed_data(path, cms, hf_labels, cm_labels, train_id, test_id, remaining_patient_ids)











