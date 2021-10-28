import numpy as np
import pandas as pd

import datetime as dt

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

from typing import List


def metric(preds, actuals):
    # As provided as finale metric, DO NOT MODIFY
    """ As provided """
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])


def apply_models(df: pd.DataFrame) -> list:

    print('Start modeling...')

    models = []

    # remove all training and test rows with zero sales
    msk_zero_sales = df.loc[:, 'Sales'] == 0
    full = df.loc[~msk_zero_sales, :].copy()

    # convert to date, remove once cleaning changed
    full.loc[:, 'Date'] = full.loc[:, 'Date'].apply(lambda x: x.date())

    # use last three months of training data as validation set
    val_from = dt.date(2014, 5, 1)
    val_msk = full.loc[:, 'Date'] < val_from

    train = full.loc[val_msk, :]
    val = full.loc[~val_msk, :]

    X_train = train.drop(columns=['Sales', 'Date'])
    y_train = train.loc[:, 'Sales']
    X_val = val.drop(columns=['Sales', 'Date'])
    y_val = val.loc[:, 'Sales']

    for model_type in [RandomForestRegressor, XGBRegressor]:
        metrics = {}
        model = model_type()
        print('Running model', type(model))
        model.fit(X_train, y_train)
        y_hat_rf_train = model.predict(X_train)
        y_hat_rf_val = model.predict(X_val)
        metrics['name'] = type(model)
        metrics['met_train'] = metric(y_hat_rf_train, y_train.to_numpy())
        metrics['met_val'] = metric(y_hat_rf_val, y_val.to_numpy())
        models.append(metrics)
        print('Done', type(model))

    return models
