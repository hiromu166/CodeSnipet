import numpy as np
import pandas as pd
from math import *
import tqdm
import random
import os
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler

import setting
from train import Set_fold

# set all seed
def SEED_Everything():
    random.seed(setting.SEED)
    os.environ['PYTHONHASHSEED'] = str(setting.SEED)
    np.random.seed(setting.SEED)

#　二点間の距離を計算
def LatLng_to_xyz(lat, lng):
    rlat, rlng = radians(lat), radians(lng)
    coslat = cos(rlat)
    return coslat*cos(rlng), coslat*sin(rlng), sin(rlat)

def Dist_On_Sphere(pos0_lat, pos0_lng, pos1_lat, pos1_lng, radius=6378.137):
    xyz0, xyz1 = LatLng_to_xyz(pos0_lat, pos0_lng), LatLng_to_xyz(pos1_lat, pos1_lng)
    return acos(sum(x * y for x, y in zip(xyz0, xyz1)))*radius

# 集計をとる
def MakeGroupstat(df, categorical_features=None, quant_features=None, report=True):
    for add_cat_col in tqdm(categorical_features):
        for add_qua_col in quant_features:
            for typ in ['mean', 'max', 'min', 'std']:
                df[add_cat_col +'_'+ typ +'_'+ add_qua_col] = df.groupby([add_cat_col])[add_qua_col].transform(typ)
    return df

# LabeEncoder
def LabelEncoding_separate(train, test):
    cat = []
    for f in train.columns:
        if train[f].dtype == 'object' or test[f].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(train[f].values) + list(test[f].values))
            train[f] = lbl.transform(list(train[f].values))
            test[f] = lbl.transform(list(test[f].values))
            cat.append(f)
    return train, test, cat

def LabelEncoding_total(total):
    cat = []
    for f in total.columns:
        if total[f].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(total[f].values))
            total[f] = lbl.transform(list(total[f].values))
            cat.append(f)
    return total, cat

# target encoding
def TargetEncoding(train, test, target, cat_cols, num_folds=5):
    for c in cat_cols:
        data_tmp = pd.DataFrame({c: train[c], 'target': target})
        target_mean = data_tmp.groupby(c)['target'].mean()
        test[c] = test[c].map(target_mean)

        tmp = np.repeat(np.nan, train.shape[0])

        kf = Set_fold('kfold', num_folds)
        for idx_1, idx_2 in kf.split(train):
            target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
            tmp[idx_2] = train[c].iloc[idx_2].map(target_mean)
        train[c] = tmp
    return train, test

def StandardScale(df):
    scaler = StandardScaler()
    scaler.fit(df)
    df = scaler.transform(df).astype(np.float32)
    return df

def Calculation_Feature(df, col1, col2):
    df[col1 + '+' + col2] = df[col1] + df[col2]
    df[col1 + '-' + col2] = df[col1] - df[col2]
    df[col1 + '*' + col2] = df[col1] * df[col2]
    df[col1 + '/' + col2] = df[col1] / df[col2]
    return df

def Imputation(df, method):
    if method == 'mean':
        df = df.fillna(df.mean())
    elif method == 'median':
        df = df.fillna(df.median())
    elif method == 'mode':
        df = df.fillna(df.mode().iloc[0])
    return df

def Get_Outlier_sigma(df, col, num=2):
    sigma = df[col].mean() + df[col].std() * num
    df[col + '_outlier'] = 0
    df[col + '_outlier'][df[col].map(lambda x: x >= sigma)] = 1
    return df

def Get_Outlier_value_upper(df, col, v):
    df[col + '_outlier'] = 0
    df[col + '_outlier'][df[col].map(lambda x: x >= v)] = 1
    return df

def Get_Outlier_value_bottom(df, col, v):
    df[col + '_outlier'] = 0
    df[col + '_outlier'][df[col].map(lambda x: x <= v)] = 1
    return df