import pandas as pd
import numpy as np
import datetime as dt
from data_cleaning import merge_data
from data_cleaning import ohe
from data_cleaning import date_treatment
from data_cleaning import filling

from models import apply_models

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

models = apply_models(df_p)

for model in models:
    print('Model Name', model['name'])
    print('Metric (Validation)', model['met_val'])
    print('Metric (Train)', model['met_train'])
