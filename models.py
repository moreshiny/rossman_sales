import numpy as np
import pandas as pd
import datetime as dt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

from typing import List, Dict, Tuple

MODELS = [RandomForestRegressor, XGBRegressor]


def rmspe(preds: np.array, actuals: np.array) -> float:
    # As provided as finale metric, DO NOT MODIFY
    """ As provided - calculates the root mean square percentage error """
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])


def remove_zero_sales(df: pd.DataFrame) -> pd.DataFrame:
    """ Remove all training and test rows with zero sales """
    msk_zero_sales = df.loc[:, 'Sales'] == 0
    return df.loc[~msk_zero_sales, :]


def convert_date(df: pd.DataFrame) -> pd.DataFrame:
    """ Convert datetime to date, remove once cleaning changed """
    df.loc[:, 'Date'] = df.loc[:, 'Date'].apply(lambda x: x.date())
    return df


def split_validation(df: pd.DataFrame, year: int, month: int, day: int) -> Tuple[pd.DataFrame]:
    val_from = dt.date(year, month, day)
    val_msk = df.loc[:, 'Date'] < val_from

    train = df.loc[val_msk, :]
    val = df.loc[~val_msk, :]

    X_train = train.drop(columns=['Sales', 'Date'])
    y_train = train.loc[:, 'Sales']
    X_val = val.drop(columns=['Sales', 'Date'])
    y_val = val.loc[:, 'Sales']

    return (X_train, y_train, X_val, y_val)


def train_models(X_train: pd.DataFrame, y_train: pd.DataFrame) -> List:
    print('Start modeling...')
    models = []
    for model_type in MODELS:
        model = model_type()
        print('Running model', type(model))
        model.fit(X_train, y_train)
        models.append(model)
        print('Done', type(model))
    print('Modeling done')
    return models


def evaluate_models(models: list, X_val: pd.DataFrame, y_val: pd.DataFrame) -> List[Dict]:
    metrics = []
    for model in models:
        metric = {}
        y_hat = model.predict(X_val)
        metric['model'] = type(model)
        # TODO: is the feature order really correct?
        metric['feat_importance'] = sorted(list(
            zip(list(X_val.columns), list(model.feature_importances_.round(2)))), key=lambda x: x[1], reverse=True)
        metric['rmspe'] = round(rmspe(y_hat, y_val.to_numpy()), 2)
        metrics.append(metric)

    return metrics
