import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import lightgbm as lgb

import setting
from metrics import *

# foldの定義
def Set_fold(type, num_folds):
    if type == 'kfold':
        folds = KFold(n_splits=num_folds, random_state=setting.SEED, shuffle=True)
    elif type == 'stratified':
        folds = StratifiedKFold(n_splits=num_folds, random_state=setting.SEED, shuffle=True)
    elif type == 'group':
        folds = GroupKFold(n_splits=num_folds, random_state=setting.SEED, shuffle=True)
    return folds

def RFR(X_train, y_train, X_test, folds, loss):
    oof_preds = np.zeros(X_train.shape[0])
    test_preds = np.zeros(X_test.shape[0])

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):
        train_x, train_y = X_train.iloc[train_idx], y_train[train_idx]
        valid_x, valid_y = X_train.iloc[valid_idx], y_train[valid_idx]
        test_x = X_test

        print("Fold {}".format(n_fold + 1), train_x.shape, valid_x.shape)

        model = RandomForestRegressor(n_jobs=-1, random_state=setting.SEED)
        model.fit(train_x, train_y)

        oof_preds[valid_idx] = model.predict(valid_x)
        test_preds += model.predict(test_x) / folds.n_splits

        print('Fold %2d LOSS : %.6f' % (n_fold + 1, loss(valid_y, oof_preds[valid_idx])))
        del model, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Valid LOSS : %.6f' % (loss(y_train, oof_preds)))
    return oof_preds, test_preds

def RFC(X_train, y_train, X_test, folds, loss):
    oof_preds = np.zeros(X_train.shape[0])
    test_preds = np.zeros(X_test.shape[0])

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):
        train_x, train_y = X_train.iloc[train_idx], y_train[train_idx]
        valid_x, valid_y = X_train.iloc[valid_idx], y_train[valid_idx]
        test_x = X_test

        print("Fold {}".format(n_fold + 1), train_x.shape, valid_x.shape)

        model = RandomForestClassifier(n_jobs=-1, random_state=setting.SEED)
        model.fit(train_x, train_y)

        oof_preds[valid_idx] = model.predict(valid_x)
        test_preds += model.predict(test_x) / folds.n_splits

        print('Fold %2d LOSS : %.6f' % (n_fold + 1, loss(valid_y, oof_preds[valid_idx])))
        del model, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Valid LOSS : %.6f' % (loss(y_train, oof_preds)))
    return oof_preds, test_preds

def LightGBM(X_train, y_train, X_test, folds, params, loss):
    oof_preds = np.zeros(X_train.shape[0])
    test_preds = np.zeros(X_test.shape[0])

    feature_importance_df = pd.DataFrame()

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):
        train_x, train_y = X_train.iloc[train_idx], y_train[train_idx]
        valid_x, valid_y = X_train.iloc[valid_idx], y_train[valid_idx]
        test_x = X_test

        print("Fold {}".format(n_fold + 1), train_x.shape, valid_x.shape)

        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               free_raw_data=False)

        params = {'application': 'regression',
                  'boosting': 'gbdt',
                  'metric': 'rmse',
                  'max_depth': 5,
                  'learning_rate': 0.01,
                  'bagging_fraction': 0.75,
                  'feature_fraction': 0.6,
                  'lambda_l1': 0.1,
                  'lambda_l2': 0.01,
                  'num_leaves': 2 ** 7,
                  'seed': int(2 ** n_fold),
                  'bagging_seed': int(2 ** n_fold),
                  'drop_seed': int(2 ** n_fold),
                  'data_random_seed': int(2 ** n_fold),
                  'verbosity': -1}

        reg = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_test],
            valid_names=['train', 'test'],
            num_boost_round=99999,
            early_stopping_rounds=150,
            verbose_eval=100
        )

        oof_preds[valid_idx] = reg.predict(valid_x, num_iteration=reg.best_iteration)
        test_preds += reg.predict(test_x) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = train_x.columns
        fold_importance_df["importance"] = np.log1p(
            reg.feature_importance(importance_type='gain', iteration=reg.best_iteration))
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d LOSS : %.6f' % (n_fold + 1, loss(valid_y, oof_preds[valid_idx])))
        del reg, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Valid LOSS : %.6f' % (loss(y_train, oof_preds)))
    return oof_preds, test_preds, feature_importance_df

def Show_lgb_importance(feature_importance_df, num=100):
    cols = (feature_importance_df[["Feature", "importance"]]
            .groupby("Feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:num].index)
    # print(cols)
    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

    plt.figure(figsize=(14, 25))
    sns.barplot(x="importance",
                y="Feature",
                data=best_features.sort_values(by="importance",
                                               ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()

def Cyclic_LightGBM(X_train, y_train, X_test, folds, loss, cycle_list):
    for i, n in enumerate(cycle_list, 1):
        print('Step {}'.format(i))
        oof_preds = np.zeros(X_train.shape[0])
        test_preds = np.zeros(X_test.shape[0])

        feature_importance_df = pd.DataFrame()

        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):
            train_x, train_y = X_train.iloc[train_idx], y_train[train_idx]
            valid_x, valid_y = X_train.iloc[valid_idx], y_train[valid_idx]
            test_x = X_test

            print("Fold {}".format(n_fold + 1), train_x.shape, valid_x.shape)

            lgb_train = lgb.Dataset(train_x,
                                    label=train_y,
                                    free_raw_data=False)
            lgb_test = lgb.Dataset(valid_x,
                                   label=valid_y,
                                   free_raw_data=False)

            params = {'application': 'regression',
                      'boosting': 'gbdt',
                      'metric': 'rmse',
                      'max_depth': 5,
                      'learning_rate': 0.01,
                      'bagging_fraction': 0.75,
                      'feature_fraction': 0.6,
                      'lambda_l1': 0.1,
                      'lambda_l2': 0.01,
                      'num_leaves': 2 ** 7,
                      'seed': int(2 ** n_fold),
                      'bagging_seed': int(2 ** n_fold),
                      'drop_seed': int(2 ** n_fold),
                      'data_random_seed': int(2 ** n_fold),
                      'verbosity': -1}

            reg = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_train, lgb_test],
                valid_names=['train', 'test'],
                num_boost_round=99999,
                early_stopping_rounds=150,
                verbose_eval=100
            )

            oof_preds[valid_idx] = reg.predict(valid_x, num_iteration=reg.best_iteration)
            test_preds += reg.predict(test_x) / folds.n_splits

            fold_importance_df = pd.DataFrame()
            fold_importance_df["Feature"] = train_x.columns
            fold_importance_df["importance"] = np.log1p(
                reg.feature_importance(importance_type='gain', iteration=reg.best_iteration))
            fold_importance_df["fold"] = n_fold + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

            print('Fold %2d LOSS : %.6f' % (n_fold + 1, loss(valid_y, oof_preds[valid_idx])))
            del reg, train_x, train_y, valid_x, valid_y
            gc.collect()

        print('Valid LOSS : %.6f' % (loss(y_train, oof_preds)))

        cols = (feature_importance_df[["Feature", "importance"]]
                .groupby("Feature")
                .mean()
                .sort_values(by="importance", ascending=False)[:cycle_list[i]].index)

        X_train = X_train[cols]
        X_test = X_test[cols]
        #cat_features = [c for c in cat_features if c in cols]
    return X_train, X_test