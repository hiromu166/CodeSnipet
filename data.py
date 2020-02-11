import pandas as pd
import numpy as np

import Ville.setting

def load_data_separate(target_name):
    train = pd.read_csv(Ville.setting.TRAIN_PATH)
    X_train = train.drop([target_name], axis=1)
    y_train = train[[target_name]]
    X_test = pd.read_csv(Ville.setting.TEST_PATH)
    num_tr = len(train)
    return X_train, y_train, X_test, num_tr

def load_data_total(target_name):
    train = pd.read_csv(Ville.setting.TRAIN_PATH)
    test = pd.read_csv(Ville.setting.TEST_PATH)
    y_train = train[[target_name]]
    total = pd.concat([train.drop(target_name, axis=1), test]).reset_index(drop=True)
    return total, y_train

def separate_data(total, num_tr):
    X_train = total.loc[:num_tr, :].reset_index(drop=True)
    X_test = total.loc[num_tr:, :].reset_index(drop=True)
    return X_train, X_test

def total_data(train, test):
    total = pd.concat([train, test]).reset_index(drop=True)
    return total
