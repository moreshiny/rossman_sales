#!/usr/bin/env python
import pandas as pd 
import numpy as np 
import datetime as dt

def merge_data(train, store):
        """ Takes two dataframes,
            creates two copies
            drop the customers axis
            drop the nan for sale and stores
            make sure the store coumns are of the same type. 
            inner merge on the column store.
        """
        train_copy = train.copy()
        store_copy = store.copy()
        train_copy = train_copy.drop(columns = ['Customers'])
        train_copy = train_copy.dropna(axis = 0, how = 'any', subset = ['Sales', 'Store'])
        train_copy['Store'] = train_copy['Store'].astype(int)
        store_copy['Store'] = store_copy['Store'].astype(int)
        df_p = pd.merge(train_copy, store_copy, how = 'inner', on = 'Store')
    
        return df_p


def date_treatment(df):
    """
    Input: A dataframe with a date columns  
    Deal with the date as follows: format the Date, extract the day, month, year weekday, 
    Introduce cyclical encoding for weekday and month month, drop then month weekday, day of the week.
    """
    import datetime as dt
    df_copy = df.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy['day'] = df_copy['Date'].dt.day
    df_copy['month'] = df_copy['Date'].dt.month
    df_copy['year'] = df_copy['Date'].dt.year
    df_copy['weekday'] = df_copy['Date']\
        .apply(lambda x: x.weekday())
    mask_missing_day = df_copy['DayOfWeek'].isna()  
    df_copy['month_sin'] = df_copy['month'].\
        apply(lambda x: np.sin(2 * np.pi * (x-1) / 12))
    df_copy['month_cos'] = df_copy['month'].\
    apply(lambda x: np.cos(2 * np.pi * (x-1) / 12))
    df_copy['weekday_sin'] = df_copy['weekday'].\
        apply(lambda x: np.sin(2 * np.pi * x / 7))
    df_copy['weekday_cos'] = df_copy['weekday'].\
        apply(lambda x: np.cos(2 * np.pi * x / 7))
    df_copy['day_sin'] = df_copy['day'].\
        apply(lambda x: np.sin(2 * np.pi * (x-1) / 30.5))
    df_copy['day_cos'] = df_copy['day'].\
        apply(lambda x: np.cos(2 * np.pi * (x-1) / 30.5))
    df_copy['Date'] = df_copy['Date'].dt.date
    
    df_copy = df_copy.drop(columns = ['DayOfWeek', 'month', 'weekday','day'])
    
    return df_copy

def ohe(df, hot_encoded_columns):
    """
    Input: A dataframe df, and a list of strings which are columns of the data frame
    Perform one hot encodind the list of inputed columns using the get_dummy.
    """
    df_copy = df.copy()
    df_copy = pd.get_dummies(df, prefix = hot_encoded_columns,\
        dummy_na = True, columns = hot_encoded_columns, drop_first = True)    
    return df_copy

def filling(df, filled_columns, filling_function):
    """
    Input: A dataframe, a list of strings which are name of columns of the dataframe, a filling function
    Fill the NaNs of the given columns, using the filling function
    """
    df_copy = df.copy()
    for col in filled_columns:
        value = filling_function(df[col].dropna())
        df_copy[col] = df_copy[col].fillna(value = value)      
    return df_copy

def storetype_replacing(df):
    """
    Take a dataframe, with a store column, erase the type dac, and keep the type b only.
    """
    df_copy = df.copy()
    mask_b = df.loc[:, 'StoreType'] == 'b'
    df_copy.loc[mask_b, 'StoreType'] = 1
    df_copy.loc[~mask_b, 'StoreType'] = 0
    return df_copy 

def remove_zero_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all training and test rows with zero sales
    """
    msk_zero_sales = df.loc[:, 'Sales'] == 0
    return df.loc[~msk_zero_sales, :]


