import numpy as np
import pandas as pd
import datetime as dt

from xgboost import XGBRegressor

from sklearn.pipeline import Pipeline

from typing import List, Dict, Tuple


def rmspe(preds: np.array, actuals: np.array) -> np.float64:
    # As provided as finale metric, DO NOT MODIFY
    """ As provided - calculates the root mean square percentage error """
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])


def define_pipelines(models: List[tuple]) -> Tuple[Pipeline]:
    """ Builds a Pipeline for each model and settings tuple provided.

    Args:
        models (List of tuples): The model class and settings dict for each
        model for which a Piple is to be defined. The settings dict key value
        pairs must be valid model settings, i.e:

        [(sklearn.ensemble.RandomForestegressor, {'n_estimators': 100}]

        Attempts to build an XGBRegressor if just a settings dictionary is
        passed (legacy call type).

    Returns:
        Tuple[Pipeline]: A tuple containing Pipelines.
    """

    # try to translate if a legacy call is made with a dictionary
    if type(models) is dict:
        models = [(XGBRegressor, models)]

    pipes = []
    for model in models:
        pipe = Pipeline([('model', model[0]()), ])
        pipe['model'].set_params(** model[1])
        pipes.append(pipe)

    return tuple(pipes)


def split_validation(df: pd.DataFrame, year: int,
                     month: int, day: int) -> Tuple[pd.DataFrame]:
    """Splits the provided dataframe into a test and validation set and
    separates feature (X) and target (y) columns. Data before the provided
    date is allocated to train, data on or after the provided date is
    allocated to validation.

    Assumes the provided dataframe has a column "Date" in datetime format.

    Args:
        df (pd.DataFrame): Input data, containing a 'Date' column
        year (int): Cut-off year
        month (int): Cut-off month
        day (int): Cut-off day

    Returns:
        Tuple[pd.DataFrame]: (X_train, y_train, X_val, y_val)
    """

    val_from = dt.date(year, month, day)
    val_msk = pd.to_datetime(df.loc[:, 'Date']).dt.date < val_from

    train = df.loc[val_msk, :]
    val = df.loc[~val_msk, :]

    X_train = train.drop(columns=['Sales'])
    y_train = train.loc[:, ['Sales']]
    X_val = val.drop(columns=['Sales'])
    y_val = val.loc[:, ['Sales']]

    return (X_train, y_train, X_val, y_val)


def model_metric(X_val: pd.DataFrame, y_val: pd.DataFrame, pipe: Pipeline) -> dict:
    """ Makes predictions based on X_val and calculates RMSPE. The return
        dictionary also includes (training) feature importance and predicted
        values.

    Args:
        X_val (pd.DataFrame): Features
        y_val (pd.DataFrame): Target
        model (object): A fitted model or pipeline (sklearn style)

    Returns:
        dict: A dictionary containing model evaluation information.
    """
    metric = {}
    y_hat = pipe.predict(X_val)
    metric['model'] = type(pipe['model'])
    metric['feat_importance'] = sorted(
        list(
            zip(
                list(X_val.columns),
                list(pipe['model'].feature_importances_.round(2))
            )
        ),
        reverse=True
    )
    metric['rmspe'] = round(rmspe(y_hat, y_val.to_numpy()), 2)
    metric['prediction'] = y_hat
    return metric


def evaluate_models(pipes: Tuple[Pipeline], X_val: pd.DataFrame,
                    y_val: pd.DataFrame) -> List[Dict]:
    """ Calculates metrics for a set of models and returns them as a list
    of dictionaries.

    Args:
        models (Tuple[object]): A list of fitted model or pipeline (sklearn style)
        X_val (pd.DataFrame): Features
        y_val (pd.DataFrame): Target

    Returns:
        List[Dict]: List of dictionaries containing model evaluation information.
    """
    metrics = []
    for pipe in pipes:
        metric = model_metric(X_val, y_val, pipe)
        metrics.append(metric.copy())

    return metrics


def features_drop1(pipes: List[Pipeline],
                   X_train: pd.DataFrame, y_train: pd.DataFrame,
                   X_val: pd.DataFrame, y_val: pd.DataFrame) -> None:
    """ Calculates the performance of each pipe or model in pipes omiting
    each feature in turn and printing the result.

    Args:
        pipes (List[Pipeline]): Model Pipelines to evaluate
        X_train (pd.DataFrame): Training features
        y_train (pd.DataFrame): Training target
        X_val (pd.DataFrame): Validation features
        y_val (pd.DataFrame): Validation target
    """
    for pipe in pipes:
        scores = {}
        for feature in X_train.columns:
            X_train_drop1 = X_train.drop(columns=[feature])
            X_val_drop1 = X_val.drop(columns=[feature])
            pipe.fit(X_train_drop1, y_train)
            y_hat = pipe.predict(X_val_drop1)
            scores[feature] = round(rmspe(y_hat, y_val.to_numpy()), 2)
        print(pipe)
        print(scores)


def hparm_search(search_models: List[Tuple],
                 X_train: pd.DataFrame, y_train: pd.DataFrame,
                 X_val: pd.DataFrame, y_val: pd.DataFrame) -> List[dict]:
    """Calculate all combinations of provided hyper-parameters for each
       model in search_models. Prints progress and returns evaluation
       information for each model.

    Args:
        search_models (List[Tuple]): A tuple of model class (sklearn type) and
                                    settings dictionary.
        X_train (pd.DataFrame): Training features
        y_train (pd.DataFrame): Training target
        X_val (pd.DataFrame): Validation features
        y_val (pd.DataFrame): Validation target

    Returns:
        List[dict]: A dictionary for each model with evaluation information.
    """
    best_models = []

    for search_model in search_models:
        best_score = np.inf
        best_model = {}
        best_score, best_model = _hparm_search(
            search_model, X_train, y_train, X_val, y_val, best_score, best_model)
        print(
            f'Best score for {search_model[0]}: {best_score}')
        for key, value in best_model.items():
            print(key, value)
        print('')
        best_models.append(best_model)

    return best_models


def _hparm_search(search_model: tuple,
                  X_train: pd.DataFrame, y_train: pd.DataFrame,
                  X_val: pd.DataFrame, y_val: pd.DataFrame,
                  best_score: float, best_model: dict):
    """ Helper for hparm_search """

    list_found = False
    for key in search_model[1].keys():
        if type(search_model[1][key]) is list:
            list_found = True
            for value in search_model[1][key]:
                search_model_freeze = (search_model[0], search_model[1].copy())
                search_model_freeze[1][key] = value
                best_score, best_model =\
                    _hparm_search(search_model_freeze,
                                  X_train, y_train,
                                  X_val, y_val,
                                  best_score, best_model)
            break

    if not list_found:
        pipe = define_pipelines([search_model])[0]
        print(f'Fitting {pipe["model"]}')
        for key, value in search_model[1].items():
            print(key, value)
        pipe.fit(X_train, y_train)
        metric = model_metric(X_val, y_val, pipe)
        print(metric['rmspe'])
        if metric['rmspe'] < best_score:
            best_score = metric['rmspe']
            best_model = dict(
                model=pipe,
                validation_metrics=metric,
            )
            for key, value in search_model[1].items():
                best_model[key] = value
        print('...done.')

    return best_score, best_model


def single_run(pipes: Tuple[Pipeline],
               X_train: pd.DataFrame, y_train: pd.DataFrame,
               X_val: pd.DataFrame, y_val: pd.DataFrame,
               X_train_full: pd.DataFrame, X_val_full: pd.DataFrame) -> Tuple:
    """ Fits the models passed in pipes and reports validation metrics.

        Also returns variables for analysis charts.

    Args:
        pipes (Tuple[Pipeline]): Set of defined models to fit
        X_train (pd.DataFrame): Training features
        y_train (pd.DataFrame): Training target
        X_val (pd.DataFrame): Validation features
        y_val (pd.DataFrame): Validation target
        X_train_full (pd.DataFrame): Full unclean training features
        X_val_full (pd.DataFrame): Full unclrean training target

    Returns:
        Tuple: (X_train, y_train, X_val, y_val, training_metrics, validation_metrics)
    """
    for pipe in pipes:
        print(f'Fitting{pipe}...')
        pipe.fit(X_train, y_train)
        print('...done.')
    print('')
    print('Training performance:')
    training_metrics = evaluate_models(
        pipes, X_train, y_train)

    print(
        'Mean as Baseline (RMSPE)',
        rmspe(np.full_like(y_train, np.mean(
            y_train)), y_train.to_numpy())
    )

    for metric in training_metrics:
        print('')
        for key, values in metric.items():
            print(key, values)

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
