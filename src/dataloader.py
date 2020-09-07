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
        x_feat_c = ['workclass', 'marital.status', 'occupation', 'relationship', 'race',
                    'native.country', 'sex']
        x_feat_n = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
        #p_feat = 'sex'
        y_feat = ['income']
        feat_c = x_feat_c+y_feat
        del dataset['education']

    dataset = dataset.dropna().reset_index(drop=True)
    # Data Preprocessing
    if to_numeric:
        lb_make = LabelEncoder()
        obj_df = dataset.copy()
        for feat in feat_c:  # list(obj_df.columns):
            dataset.loc[:, feat] = lb_make.fit_transform(obj_df[feat])

    # move target values (y) in the first column
    _maxlen = maxlen if maxlen > 0 else len(dataset)
    X = dataset[x_feat_c + x_feat_n][:_maxlen].values
    Y = dataset[y_feat][:_maxlen].values
    return normalize(X), Y.astype('int64')