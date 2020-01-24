import numpy as np
import pandas as pd
from math import *
import tqdm
import random
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

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

# 未完成
"""
def svd_feature(prefix, df, traintest, groupby, target,n_comp):
    tfidf_vec = TfidfVectorizer(ngram_range=(1,1), max_features=None)
    df_bag = pd.DataFrame(df[[groupby, target]])
    df_bag = df_bag.groupby(groupby, as_index=False)[target].agg({'list':(lambda x: list(x))}).reset_index()
    df_bag[target + '_list']=df_bag['list'].apply(lambda x: str(x).replace('[','').replace(']','').replace(',',' '))

    df_bag = df_bag.merge(traintest,on=groupby,how='left')
    df_bag_train = df_bag[df_bag['target'].notnull()].reset_index(drop=True)
    df_bag_test = df_bag[df_bag['target'].isnull()].reset_index(drop=True)

    tfidf_full_vector = tfidf_vec.fit_transform(df_bag[target + '_list'])
    tfidf_train_vector = tfidf_vec.transform(df_bag_train[target + '_list'])
    tfidf_test_vector = tfidf_vec.transform(df_bag_test[target + '_list'])

    svd_vec = TruncatedSVD(n_components=5, algorithm='arpack')
    svd_vec.fit(tfidf_full_vector)
    svd_train = pd.DataFrame(svd_vec.transform(tfidf_train_vector))
    svd_test = pd.DataFrame(svd_vec.transform(tfidf_test_vector))

    svd_train.columns = ['svd_%s_%s_%d'%(prefix,target,x) for x in range(n_comp)]
    svd_train[groupby] = df_bag_train[groupby]
    svd_test.columns = ['svd_%s_%s_%d'%(prefix,target,x) for x in range(n_comp)]
    svd_test[groupby] = df_bag_test[groupby]
    #df_svd = pd.concat([svd_train,svd_test],axis=0)
    print ('svd_train:' + str(svd_train.shape))
    print ('svd_test:' + str(svd_test.shape))
    return svd_train,svd_test

def word2vec_feature(prefix, df, groupby, target,size):
    df_bag = pd.DataFrame(df[[groupby, target]])
    df_bag[target] = df_bag[target].astype(str)
    df_bag[target].fillna('NAN', inplace=True)
    df_bag = df_bag.groupby(groupby, as_index=False)[target].agg({'list':(lambda x: list(x))}).reset_index()
    doc_list = list(df_bag['list'].values)
    w2v = Word2Vec(doc_list, size=size, window=3, min_count=1, workers=32)
    vocab_keys = list(w2v.wv.vocab.keys())
    w2v_array = []
    for v in vocab_keys :
        w2v_array.append(list(w2v.wv[v]))
    df_w2v = pd.DataFrame()
    df_w2v['vocab_keys'] = vocab_keys    
    df_w2v = pd.concat([df_w2v, pd.DataFrame(w2v_array)], axis=1)
    df_w2v.columns = [target] + ['w2v_%s_%s_%d'%(prefix,target,x) for x in range(size)]
    print ('df_w2v:' + str(df_w2v.shape))
    return df_w2v
"""

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