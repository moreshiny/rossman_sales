import pandas as pd

from models import split_validation
from models import define_pipelines
from models import single_run
from models import features_drop1
from models import hparm_search

from data_cleaning import DataCleaning

TRAINING_DATA = 'data/train.csv'
HOLDOUT_DATA = 'data/holdout.csv'
STORE_DATA = 'data/store.csv'
TEST_DATA = ''

RANDOM_SEED = 42
CORES = -1

try:
    df_test = pd.read_csv(TEST_DATA)
except FileNotFoundError:
    print('Test data file not found, using holdout as validation set')
    df_test = pd.read_csv(HOLDOUT_DATA)
    df_train = pd.read_csv(TRAINING_DATA)
else:
    print('Test data loaded, using full training data for model training')
    df_train = pd.concat([
        pd.read_csv(TRAINING_DATA),
        pd.read_csv(HOLDOUT_DATA)
    ])
finally:
    df_store = pd.read_csv(STORE_DATA)

# TODO add asserts to check assumptions on test data


X_train = df_train.drop(columns='Sales')
y_train = df_train.loc[:, 'Sales']
X_val = df_test.drop(columns='Sales')
y_val = df_test.loc[:, 'Sales']

# Feature cleaning and transformation defined below - see data_cleaning.py
# One hot encoded features are dropped
# Target is returned as y
# TODO 'Store' is mean encoded (hardcoded)
# Date is split, converted, and sin/cos transformed where applicable
cleaning_settings = dict(
    hot_encoded_columns=[
        'StateHoliday',
        'Assortment',
        'SchoolHoliday',
    ],
    dropped_columns=[
        'Store',
        'CompetitionOpenSinceMonth',
        'CompetitionOpenSinceYear',
        'Promo2SinceWeek',
        'Promo2SinceYear',
        'PromoInterval',
        'Date',
        'Open',
        'StoreType',
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

cleaning = DataCleaning(
    store=df_store,
    hot_encoded_columns=cleaning_settings['hot_encoded_columns'],
    dropped_columns=cleaning_settings['dropped_columns'],
    filled_in_median=cleaning_settings['filled_in_median'],
    filled_in_mode=cleaning_settings['filled_in_mode'],
    target=cleaning_settings['target'],
)

X_train_clean, y_train_clean =\
    cleaning.cleaning(X_train, y_train, training=True)
X_val_clean, y_val_clean =\
    cleaning.cleaning(X_val, y_val, training=False)


# Use this for a single run of the model with current parameters
rf_settings = dict(
    n_estimators=50,
    max_depth=50,
    random_state=RANDOM_SEED,
    n_jobs=CORES,
)

xg_settings = dict(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.2,
    random_state=RANDOM_SEED,
    n_jobs=CORES,
)

pipes = define_pipelines(xg_settings)
single_run(pipes, X_train_clean, y_train_clean,
           X_val_clean, y_val_clean, X_train, X_val)


# Use this to drop each feature in turn and return the score without it
#features_drop1(pipes, X_train_clean, y_train_clean, X_val_clean, y_val_clean)


# Use this to search over hyper-parameter ranges definde above and print scores

# rf_sets = dict(
#     n_estimators=[50, 100, 200, 400],
#     max_depth=[4, 16, 64, 128],
#     random_state=RANDOM_SEED,
#     n_jobs=CORES,
# )

# xg_sets = dict(
#     n_estimators=[400, 500, 600, 700],
#     max_depth=[2, 3, 4],
#     learning_rate=[.1, .2],
#     random_state=RANDOM_SEED,
#     n_jobs=CORES,
# )

# best_xg = hparm_search(X_train_clean, y_train_clean, X_val_clean,
#                        y_val_clean, rf_sets, xg_sets)

# #print(best_rf)
# print(best_xg)
