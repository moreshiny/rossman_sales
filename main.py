import pandas as pd
import numpy as np

from models import split_validation
from models import define_pipelines
from models import features_drop1
from models import single_run
from models import single_run2
from models import grid_search

from data_cleaning import DataCleaning

df_train = pd.read_csv('data/train.csv')
df_store = pd.read_csv('data/store.csv')
df_holdout = pd.read_csv('data/holdout.csv')

cleaning_settings = dict(
    hot_encoded_columns=[
        'Open',
        'StateHoliday',
        'Assortment',
        'SchoolHoliday',
        'StoreType',
    ],

    dropped_columns=[
        'Store',
        'CompetitionOpenSinceMonth',
        'CompetitionOpenSinceYear',
        'Promo2SinceWeek',
        'Promo2SinceYear',
        'PromoInterval',
        'Date',
    ],

    filled_in_median=[
        'CompetitionDistance',
    ],

    filled_in_mode=[
        'Promo',
    ],

    target=[
        'Sales',
    ],
)

X_train = df_train.drop(columns='Sales')
y_train = df_train.loc[:, 'Sales']

X_val = df_holdout.drop(columns='Sales')
y_val = df_holdout.loc[:, 'Sales']

# X_train, y_train, X_val, y_val = split_validation(
#     df_train, 2014, 5, 1)

cleaning = DataCleaning(
    store=df_store,
    hot_encoded_columns=cleaning_settings['hot_encoded_columns'],
    dropped_columns=cleaning_settings['dropped_columns'],
    filled_in_median=cleaning_settings['filled_in_median'],
    filled_in_mode=cleaning_settings['filled_in_mode'],
    target=cleaning_settings['target'],
)

X_train_clean, y_train_clean = cleaning.cleaning(
    X_train, y_train, training=True)
X_val_clean, y_val_clean = cleaning.cleaning(
    X_val, y_val, training=False)

#X_val_clean.loc[:, ['StateHoliday_0', 'StateHoliday_b', 'StateHoliday_c']] = 0

RANDOM_SEED = 42
CORES = -1

rf_settings = dict(
    n_estimators=50,
    max_depth=50,
    random_state=RANDOM_SEED,
    n_jobs=CORES,
)

xg_settings = dict(
    n_estimators=250,
    max_depth=3,
    random_state=RANDOM_SEED,
    n_jobs=CORES,
)

pipes = define_pipelines(df_store, cleaning_settings, rf_settings, xg_settings)

# single_run(pipes, X_train_clean, y_train_clean, X_val_clean, y_val_clean)

single_run2(pipes, X_train_clean, y_train_clean,
            X_val_clean, y_val_clean, X_train)


#features_drop1(pipes, X_train_clean, y_train_clean, X_val_clean, y_val_clean)

# rf_sets = dict(
#     n_estimators=[16, 48, 64, 96],
#     max_depth=[4, 16, 64, 128],
#     random_state=RANDOM_SEED,
#     n_jobs=CORES,
# )

# xg_sets = dict(
#     n_estimators=[24, 48, 96, 192],
#     max_depth=[1, 3, 5, 7],
#     random_state=RANDOM_SEED,
#     n_jobs=CORES,
# )

# grid_search(X_train_clean, y_train_clean, X_val_clean,
#             y_val_clean, rf_sets, xg_sets)
