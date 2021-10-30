#!/usr/bin/env python
import pandas as pd
import numpy as np
import datetime as dt


class DataCleaning():

    def __init__(self, store, hot_encoded_columns, dropped_columns,
                 filled_in_median, filled_in_mode, target) -> None:
        self.store = store
        self.hot_encoded_columns = hot_encoded_columns
        self.dropped_columns = dropped_columns
        self.filled_in_median = filled_in_median
        self.filled_in_mode = filled_in_mode
        self.target = target
        pass

    def merge_data(self, train):
        """ Takes two dataframes,
            creates two copies
            drop the customers axis
            drop the nan for sale and stores
            make sure the store coumns are of the same type.
            inner merge on the column store.
        """
        train_copy = train.copy()
        store_copy = self.store.copy()
        train_copy = train_copy.drop(columns=['Customers'])
        train_copy['Store'] = train_copy['Store'].astype(int)
        store_copy['Store'] = store_copy['Store'].astype(int)
        df_p = pd.merge(train_copy, store_copy, how='inner', on='Store')
        return df_p

    def date_treatment(self, df):
        """
        Input: A dataframe with a date columns
        Deal with the date as follows: format the date, extract the day, month, year weekday,
        Introduce cyclical encoding for weekday and month month, drop then month weekday, day of the week.
        """
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        df_copy['day'] = df_copy['Date'].dt.day
        df_copy['month'] = df_copy['Date'].dt.month
        df_copy['year'] = df_copy['Date'].dt.year
        df_copy['weekday'] = df_copy['Date']\
            .apply(lambda x: x.weekday())
        mask_missing_day = df_copy['DayOfWeek'].isna()
 #       df_copy['month_sin'] = df_copy['month'].\
 #           apply(lambda x: np.sin(2 * np.pi * (x-1) / 12))
 #       df_copy['month_cos'] = df_copy['month'].\
 #           apply(lambda x: np.cos(2 * np.pi * (x-1) / 12))
 #       df_copy['weekday_sin'] = df_copy['weekday'].\
 #           apply(lambda x: np.sin(2 * np.pi * x / 7))
 #       df_copy['weekday_cos'] = df_copy['weekday'].\
 #           apply(lambda x: np.cos(2 * np.pi * x / 7))
 #       df_copy['day_sin'] = df_copy['day'].\
 #           apply(lambda x: np.sin(2 * np.pi * (x-1) / 30.5))
 #       df_copy['day_cos'] = df_copy['day'].\
 #           apply(lambda x: np.cos(2 * np.pi * (x-1) / 30.5))
 #       df_copy['Date'] = df_copy['Date'].dt.date
        df_copy = df_copy.drop(
            columns=['DayOfWeek',
                     #            'month',
                     #             'weekday',
                     #              'day'
                     ])
        return df_copy

    def cleaning_state_holiday(self, df):
        """change the state holiday type 0 to no"""
        df_copy = df.copy()
        mask_0 = (df_copy.loc[:,
                              'StateHoliday'] == '0') | (df_copy.loc[:, 'StateHoliday'] == 0.0)
        df_copy.loc[mask_0, 'StateHoliday'] = 'no'
        return df_copy

    def store_type_encoding(self, df):
        """
        Add two new features: store type median average and standard deviation.
        """

    def ohe(self, df):
        """
        Input: A dataframe df, and a list of strings which are columns of the data frame
        Perform one hot encodind the list of inputed columns using the get_dummy.
        """
        df_copy = df.copy()
        df_copy = pd.get_dummies(df, prefix=self.hot_encoded_columns,
                                 dummy_na=True, columns=self.hot_encoded_columns, drop_first=True)
        return df_copy

    def filling(self, df, filling_function, filled_columns):
        """
        Input: A dataframe, a list of strings which are name of columns of the dataframe, a filling function
        Fill the NaNs of the given columns, using the filling function
        """
        df_copy = df.copy()
        for col in filled_columns:
            value = filling_function(df[col].dropna())
            df_copy[col] = df_copy[col].fillna(value=value)
        return df_copy

    def remove_zero_sales(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove all training and test rows with zero sales
        """
        mask_zero_sales = df.loc[:, 'Sales'] == 0
        return df.loc[~mask_zero_sales, :]

    def mean_encoding_dictionary(self, X, Y):
        """
        Perform the mean encoding on the store on the entire past of the serie.
        """
        list_store = set(X.loc[:, 'Store'])
        list_type = set(X.loc[:, 'StoreType'])
        self.list_store = list_store
        self.list_type = list_type
        dict_store_mean = dict()
        dict_store_std = dict()
        dict_store_iqr = dict()
        dict_type_median = dict()
        dict_type_std = dict()

        for store in list_store:
            mask_store = X.loc[:, 'Store'] == store
            mean_store = np.median(Y.loc[mask_store])
            std_store = np.std(Y.loc[mask_store])
            iqr_store = np.subtract(
                *np.percentile(Y.loc[mask_store], [75, 25]))
            dict_store_mean[store] = mean_store
            dict_store_std[store] = std_store
            dict_store_iqr[store] = iqr_store
        for type in list_type:
            mask_type = X.loc[:, 'StoreType'] == type
            median_type = np.mean(Y.loc[mask_type])
            std_type = np.std(Y.loc[mask_type])
            dict_type_median[type] = median_type
            dict_type_std[type] = std_type
        self.dict_store_mean = dict_store_mean
        self.dict_store_std = dict_store_std
        self.dict_store_iqr = dict_store_iqr
        self.dict_type_median = dict_type_median
        self.dict_type_std = dict_type_std
        self.dict_mean = np.median(list(dict_store_mean.values()))
        self.dict_std = np.std(list(dict_store_std.values()))
        self.dict_iqr = np.subtract(
            *np.percentile(list(dict_store_iqr.values()), [75, 25]))
        self.dict_type_median_v = np.median(list(dict_type_median.values()))
        self.dict_type_std_v = np.median(list(dict_type_std.values()))
        pass

    def mean_encoding(self, df):
        """Perform the mean encoding based on the mean sales dictionary per store"""
        df.loc[:, 'median_sales'] = self.dict_mean
        df.loc[:, 'std_sales'] = self.dict_std
        df.loc[:, 'type_median'] = self.dict_type_median_v
        df.loc[:, 'type_std'] = self.dict_type_std_v
        #df.loc[:, 'irq'] = self.dict_iqr
        for store in self.list_store:
            mask_store = df.loc[:, 'Store'] == store
            df.loc[mask_store, 'median_sales'] = self.dict_store_mean[store]
            df.loc[mask_store, 'std_sales'] = self.dict_store_std[store]
            #df.loc[mask_store, 'iqr'] = self.dict_store_iqr[store]
        for type in self.list_type:
            mask_type = df.loc[:, 'StoreType'] == type
            df.loc[mask_type, 'type_median'] = self.dict_type_median[type]
            df.loc[mask_type, 'type_std'] = self.dict_type_std[type]
        return df

    def cleaning(self, X, Y, training=True):
        """
        Take the full data
        """
        if training:
            print(
                "Cyclicality removed. Added columns: log median, std, encoding for the store, somethin, and added standard deviation, best score for the boost")
            df_p = pd.concat([X, Y], axis=1)
            df_p = df_p.dropna(subset=['Store', 'Sales'])
            df_p = self.merge_data(df_p)
            df_p = self.cleaning_state_holiday(df_p)
            training = df_p.drop(columns=['Sales'])
            self.mean_encoding_dictionary(training, df_p['Sales'])
            df_p = self.remove_zero_sales(df_p)
            df_p = self.date_treatment(df_p)
            df_p = self.mean_encoding(df_p)
            df_p = self.ohe(df_p)
            df_p = self.filling(df_p, np.median, self.filled_in_median)
            df_p = self.filling(df_p, np.min, self.filled_in_mode)
            df_p = df_p.drop(columns=self.dropped_columns)
            print("cleaning on training done succesfully")
            return df_p.drop(columns=['Sales']), df_p['Sales']
        else:
            df_p = pd.concat([X, Y], axis=1)
            df_p = df_p.dropna(subset=['Store', 'Sales'])
            df_p = self.merge_data(df_p)
            df_p = self.cleaning_state_holiday(df_p)
            df_p = self.remove_zero_sales(df_p)
            df_p = self.date_treatment(df_p)
            df_p = self.mean_encoding(df_p)
            df_p = self.ohe(df_p)
            df_p = self.filling(df_p, np.median, self.filled_in_median)
            df_p = self.filling(df_p, np.min, self.filled_in_mode)
            df_p = df_p.drop(columns=self.dropped_columns)
            print("cleaning on test done succesfully")
            return df_p.drop(columns=['Sales']), df_p['Sales']
