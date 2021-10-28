import pandas as pd
import numpy as np
from data_cleaning import merge_data
from data_cleaning import ohe
from data_cleaning import date_treatment
from data_cleaning import filling

from models import convert_date, remove_zero_sales
from models import split_validation, train_models, evaluate_models

TESTING = False

df_train = pd.read_csv('data/train.csv')
df_store = pd.read_csv('data/store.csv')
df_holdout = pd.read_csv('data/holdout.csv')

hot_encoded_columns = [
    'Open',
    'StateHoliday',
    'StoreType',
    'Assortment',
]

dropped_columns = [
    'Store',
    'CompetitionOpenSinceMonth',
    'CompetitionOpenSinceYear',
    'Promo2SinceWeek',
    'Promo2SinceYear',
    'PromoInterval',
]

filled_in_median = [
    'CompetitionDistance',
]

filled_in_mode = [
    'Promo',
    'SchoolHoliday',
]

target = [
    'Sales',
]

df_p = merge_data(df_train, df_store)
df_p = df_p.drop(columns=dropped_columns)
df_p = date_treatment(df_p)
df_p = ohe(df_p, hot_encoded_columns)
df_p = filling(df_p, filled_in_median, np.median)
df_p = filling(df_p, filled_in_mode, np.min)

df_p = remove_zero_sales(df_p)
df_p = convert_date(df_p)

if TESTING:
    print('WARNING - test run, using just 10k data points!')
    df_p = df_p[:10000]

X_train, y_train, X_val, y_val = split_validation(df_p, 2014, 5, 1)

models = train_models(X_train, y_train)

print('')
print('Training performance:')
if TESTING:
    print('WARNING - test run, using just 10k data points!')
training_metrics = evaluate_models(models, X_train, y_train)
print(training_metrics)

print('')
print('Validation performance:')
if TESTING:
    print('WARNING - test run, using just 10k data points!')
validation_metrics = evaluate_models(models, X_val, y_val)
print(validation_metrics)
