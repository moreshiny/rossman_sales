import numpy as np
import pandas as pd
import datetime as dt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from typing import List, Dict, Tuple

MODELS = [RandomForestRegressor, XGBRegressor]


def rmspe(preds: np.array, actuals: np.array) -> float:
    # As provided as finale metric, DO NOT MODIFY
    """ As provided - calculates the root mean square percentage error """
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])



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


def train_models(X_train: pd.DataFrame, y_train: pd.DataFrame) -> Tuple[Pipeline]:
    print('Start modeling...')

    pipe_rf = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(
            n_estimators=50,
            max_depth=50,
            random_state=42,
            n_jobs=-1,
        )
        ),
    ])

    print('Running model', type(pipe_rf['model']))
    pipe_rf.fit(X_train, y_train)

    pipe_xg = Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBRegressor(
            n_estimators=250,
            max_depth=3,
            random_state=42,
            nthread=-1,
        )
        ),
    ])

    print('Running model', type(pipe_xg['model']))
    pipe_xg.fit(X_train, y_train)

    return (pipe_rf, pipe_xg)


def evaluate_models(models: Tuple[object], X_val: pd.DataFrame, y_val: pd.DataFrame) -> List[Dict]:
    metrics = []
    for model in models:
        metric = {}
        y_hat = model.predict(X_val)
        metric['model'] = type(model['model'])
        # TODO: is the feature order really correct?
        metric['feat_importance'] = sorted(
            list(
                zip(
                    list(X_val.columns),
                    list(model['model'].feature_importances_.round(2))
                )
            ),
            key=lambda x: x[1],
            reverse=True
        )
        metric['rmspe'] = round(rmspe(y_hat, y_val.to_numpy()), 2)
        metrics.append(metric.copy())

    return metrics
