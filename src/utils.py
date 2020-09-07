import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import sklearn

import copy, pickle, sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
from sklearn.metrics import *

WORK_PATH = '/content/'

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


# I.  Evaluation Metrics, Model Accuracy, p%-value and DI-score
def compute_fairness_score(y_true, y_pred):
  """
  Return Model Accuracy
  """
  acc = accuracy_score(y_true, y_pred)
  return acc


# II. Process datasets

def load_dataset(name, rm_pfeat=False, classes=[0, 1], to_numeric=True):

    if name == 'census':
        dataset = pd.read_csv( WORK_PATH + 'adult.csv', na_values='?', skipinitialspace=True)
        x_feat_c = ['workclass', 'marital.status', 'occupation', 'relationship', 'race',
                    'native.country', 'sex']
        x_feat_n = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
        # p_feat = 'sex'
        y_feat = 'income'
        del dataset['education']

    x_feat = x_feat_n + x_feat_c

    if rm_pfeat:
        del dataset[p_feat]
        del x_feat[x_feat.index(p_feat)]

    dataset = dataset.dropna().reset_index(drop=True)
    # Data Preprocessing
    if to_numeric:
        lb_make = sklearn.preprocessing.LabelEncoder()
        obj_df = dataset.copy()
        for feat in x_feat_c:  # list(obj_df.columns):
            dataset.loc[:, feat] = lb_make.fit_transform(obj_df[feat])

    # Set target to [-1, 1]
    if name in ['census', 'default', 'compas']:
        assert len(classes) == 2
        a, b = min(dataset[y_feat]), max(dataset[y_feat])
        dataset[y_feat] = dataset[y_feat].replace(a, classes[0])
        dataset[y_feat] = dataset[y_feat].replace(b, classes[1])

    # move target values (y) in the first column
    targetcol = dataset[y_feat]
    dataset.drop(labels=[y_feat], axis=1, inplace=True)
    dataset.insert(0, y_feat, targetcol)

    return dataset, x_feat_c, x_feat_n, y_feat


def load_data(file_name):
    if file_name!='bank':
        dataset, x_feat_c, x_feat_n, y_feat = load_dataset(file_name)
        pd00 = copy.deepcopy(dataset)

        label_name = y_feat
        feats = [col for col in x_feat_c + x_feat_n if col not in [label_name]]

    assert (pd00[label_name].min() == 0) and (pd00[label_name].max() == 1)

    return pd00, label_name, feats


def get_data_loader(pd00, feats, label_name, seed=0):
    X_train, X_test, y_train, y_test = train_test_split(pd00[feats].values, pd00[label_name].values,
                                                                         test_size=0.2,
                                                                         stratify=pd00[label_name].values, random_state=seed)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                                         test_size= 0.25,
                                                                         stratify= y_train,
                                                                         random_state= seed)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test =  scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    return X_train, X_val, X_test, y_train, y_val, y_test