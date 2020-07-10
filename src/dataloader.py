""" author Ferdinando Fioretto """
import numpy as np
import pandas as pd
from settings import *
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize

def load(name, classes=[0, 1], to_numeric=True, maxlen=0):
    if name == 'adult':
        dataset = pd.read_csv(DATAPATH + 'adult.csv', na_values='?', skipinitialspace=True)
        x_feat_c = ['workclass', 'marital-status', 'occupation', 'relationship', 'race',
                    'native-country', 'sex']
        x_feat_n = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        #p_feat = 'sex'
        y_feat = 'fnlwgt'
        del dataset['education']

    dataset = dataset.dropna().reset_index(drop=True)
    # Data Preprocessing
    if to_numeric:
        lb_make = LabelEncoder()
        obj_df = dataset.copy()
        for feat in x_feat_c:  # list(obj_df.columns):
            dataset.loc[:, feat] = lb_make.fit_transform(obj_df[feat])

    # Set target to [-1, 1]
    if name in ['adult', 'default', 'compas']:
        assert len(classes) == 2
        a, b = min(dataset[y_feat]), max(dataset[y_feat])
        dataset[y_feat] = dataset[y_feat].replace(a, classes[0])
        dataset[y_feat] = dataset[y_feat].replace(b, classes[1])

    # move target values (y) in the first column
    _maxlen = maxlen if maxlen > 0 else len(dataset)
    X = dataset[x_feat_c + x_feat_n][:_maxlen].values
    Y = dataset[y_feat][:_maxlen].values
    return normalize(X), Y
