import numpy as np
import pandas as pd
import datetime as dt

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline

from typing import List, Dict, Tuple


def rmspe(preds: np.array, actuals: np.array) -> float:
    # As provided as finale metric, DO NOT MODIFY
    """ As provided - calculates the root mean square percentage error """
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])


def define_pipelines(xg_settings) -> Tuple[Pipeline]:
    # TODO remove scaler as it's not needed?
    # pipe_rf = Pipeline([
    #     (
    #         'model', RandomForestRegressor(
    #             n_estimators=rf_settings['n_estimators'],
    #             max_depth=rf_settings['max_depth'],
    #             random_state=rf_settings['random_state'],
    #             n_jobs=rf_settings['n_jobs'],
    #         )
    #     ),
    # ])

    # TODO remove scaler as it's not needed?
    pipe_xg = Pipeline([
        ('model', XGBRegressor(
            n_estimators=xg_settings['n_estimators'],
            max_depth=xg_settings['max_depth'],
            learning_rate=xg_settings['learning_rate'],
            random_state=xg_settings['random_state'],
            n_jobs=xg_settings['n_jobs'],
        )
        ),
    ])
    return (pipe_xg,)
    # return (pipe_rf, pipe_xg)


def split_validation(df: pd.DataFrame, year: int,
                     month: int, day: int) -> Tuple[pd.DataFrame]:
    val_from = dt.date(year, month, day)
    val_msk = pd.to_datetime(df.loc[:, 'Date']).dt.date < val_from

    train = df.loc[val_msk, :]
    val = df.loc[~val_msk, :]

    X_train = train.drop(columns=['Sales'])
    y_train = train.loc[:, ['Sales']]
    X_val = val.drop(columns=['Sales'])
    y_val = val.loc[:, ['Sales']]

    return (X_train, y_train, X_val, y_val)


def model_metric(X_val, y_val, model):
    metric = {}
    y_hat = model.predict(X_val)
    metric['model'] = type(model['model'])
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
    metric['prediction'] = y_hat
    return metric


def evaluate_models(models: Tuple[object], X_val: pd.DataFrame,
                    y_val: pd.DataFrame) -> List[Dict]:
    metrics = []
    for model in models:
        metric = model_metric(X_val, y_val, model)
        metrics.append(metric.copy())

    return metrics


def features_drop1(pipes, X_train, y_train, X_val, y_val):
    for pipe in pipes:
        scores = {}
        for feature1 in X_train.columns:
            X_train_drop1 = X_train.drop(columns=[feature1])
            X_val_drop1 = X_val.drop(columns=[feature1])
            pipe.fit(X_train_drop1, y_train)
            y_hat = pipe.predict(X_val_drop1)
            scores[feature1] = round(rmspe(y_hat, y_val.to_numpy()), 2)
        print(pipe)
        print(scores)


def hparm_search(X_train, y_train, X_val, y_val, rf_sets, xg_sets):

    best_score = np.inf
    best_xg = {}

    for e_index in range(len(xg_sets['n_estimators'])):
        for d_index in range(len(xg_sets['max_depth'])):
            for l_index in range(len(xg_sets['learning_rate'])):

                xg_settings = xg_sets.copy()
                xg_settings['n_estimators'] = xg_sets['n_estimators'][e_index]
                xg_settings['max_depth'] = xg_sets['max_depth'][d_index]
                xg_settings['learning_rate'] = xg_sets['learning_rate'][l_index]

                # TODO remove scaler as it's not needed?
                pipe_xg = Pipeline([
                    ('model', XGBRegressor(
                        n_estimators=xg_settings['n_estimators'],
                        max_depth=xg_settings['max_depth'],
                        learning_rate=xg_settings['learning_rate'],
                        random_state=xg_settings['random_state'],
                        n_jobs=xg_settings['n_jobs'],
                    )
                    ),
                ])

                print(
                    f'Fitting XG {xg_settings["n_estimators"]}-{xg_settings["max_depth"]}-{xg_settings["learning_rate"]}...')
                pipe_xg.fit(X_train, y_train)
                xg_metric = model_metric(X_val, y_val, pipe_xg)
                print(xg_metric['rmspe'])
                if xg_metric['rmspe'] < best_score:
                    best_score = xg_metric['rmspe']
                    best_xg = dict(
                        model=pipe_xg,
                        validation_metrics=xg_metric,
                        n_estimators=xg_sets['n_estimators'][e_index],
                        depth=xg_sets['max_depth'][d_index],
                        learning_rate=xg_sets['learning_rate'][l_index],
                    )
                print('...done.')

    print(f'Best score for XG: {best_score}')
    for key, value in best_xg.items():
        print(key, value)
    print('')

    # best_score = np.inf
    # best_rf = {}
    # for e_index in range(len(rf_sets['n_estimators'])):
    #     for d_index in range(len(rf_sets['max_depth'])):
    #         rf_settings = rf_sets.copy()
    #         rf_settings['n_estimators'] = rf_sets['n_estimators'][e_index]
    #         rf_settings['max_depth'] = rf_sets['max_depth'][d_index]

    #         # TODO remove scaler as it's not needed?
    #         pipe_rf = Pipeline([
    #             (
    #                 'model', RandomForestRegressor(
    #                     n_estimators=rf_settings['n_estimators'],
    #                     max_depth=rf_settings['max_depth'],
    #                     random_state=rf_settings['random_state'],
    #                     n_jobs=rf_settings['n_jobs'],
    #                 )
    #             ),
    #         ])
    #         print(
    #             f'Fitting RF {rf_settings["n_estimators"]}-{rf_settings["max_depth"]}...')
    #         pipe_rf.fit(X_train, y_train)
    #         rf_metric = model_metric(X_val, y_val, pipe_rf)
    #         print(rf_metric['rmspe'])
    #         if rf_metric['rmspe'] < best_score:
    #             best_score = rf_metric['rmspe']
    #             best_rf = dict(
    #                 model=pipe_rf,
    #                 validation_metrics=rf_metric,
    #                 n_estimators=rf_sets['n_estimators'][e_index],
    #                 depth=rf_sets['max_depth'][d_index],
    #             )
    #         print('...done.')

    # print(f'Best score for RF: {best_score}')
    # for key, value in best_rf.items():
    #     print(key, value)
    # print('')

    # return best_rf, best_xg
    return best_xg


def single_run(pipes, X_train, y_train, X_val, y_val, X_train_full, X_val_full):

    for pipe in pipes:
        print(f'Fitting{pipe}...')
        pipe.fit(X_train, y_train)
        print('...done.')
    # print('')
    # print('Training performance:')
    training_metrics = evaluate_models(
        pipes, X_train, y_train)

    # print(
    #     'Mean as Baseline (RMSPE)',
    #     rmspe(np.full_like(y_train, np.mean(
    #         y_train)), y_train.to_numpy())
    # )

    # for metric in training_metrics:
    #     print('')
    #     for key, values in metric.items():
    #         print(key, values)

    print('')
    print('Validation performance:')
    validation_metrics = evaluate_models(pipes, X_val, y_val)

    print(
        'Mean as Baseline (RMSPE)',
        rmspe(np.full_like(y_val, np.mean(
            y_train)), y_val.to_numpy())
    )

    for metric in validation_metrics:
        print('')
        for key, values in metric.items():
            if key in ['model', 'rmspe']:
                print(key, values)

    X_train.loc[:, 'Date'] = pd.to_datetime(X_train_full.loc[:, 'Date'])
    X_val.loc[:, 'Date'] = pd.to_datetime(X_val_full.loc[:, 'Date'])

    return X_train, y_train, X_val, y_val, training_metrics, validation_metrics
