import numpy as np
import pandas as pd
import datetime
def preprocess(data:pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess a DataFrame containing financial data.

    Parameters:
    - data: pd.DataFrame
        The input DataFrame containing financial data.

    Returns:
    - pd.DataFrame
        A preprocessed DataFrame with the 'datetime' column converted to datetime,
        sorted by 'datetime', and numeric columns ('open', 'high', 'low', 'close',
        'volume') converted to float.

    Example:
    data = preprocess(data)
    """

    data['datetime'] = pd.to_datetime(data['datetime'])
    data.sort_values(by = 'datetime', inplace=True)

    # Convert 'open', 'high', 'low', 'close' , 'volume' columns to float
    data['open'] = pd.to_numeric(data['open'], errors='coerce')
    data['high'] = pd.to_numeric(data['high'], errors='coerce')
    data['low'] = pd.to_numeric(data['low'], errors='coerce')
    data['close'] = pd.to_numeric(data['close'], errors='coerce')
    data['volume'] = pd.to_numeric(data['volume'], errors='coerce')

    data.drop(columns=['symbol_id', 'symbol'], inplace=True)

    return data

import pandas as pd

def train_test_split_by_time(data: pd.DataFrame, 
                             train_start: datetime, 
                             train_end: datetime, 
                             test_start: datetime, 
                             test_end: datetime,
                             ) -> (pd.DataFrame, pd.DataFrame):
    
    """
    Split a DataFrame by time into training and testing DataFrames.

    Parameters:
    - data: pd.DataFrame
        The input DataFrame containing time-based data with a 'datetime' index.

    - train_start: str
        The start date and time for the training data in 'YYYY-MM-DD HH:MM:SS' format.

    - train_end: str
        The end date and time for the training data in 'YYYY-MM-DD HH:MM:SS' format.

    - test_start: str
        The start date and time for the testing data in 'YYYY-MM-DD HH:MM:SS' format.

    - test_end: str
        The end date and time for the testing data in 'YYYY-MM-DD HH:MM:SS' format.

    Returns:
    - train_data: pd.DataFrame
        The training DataFrame containing data within the specified time range.

    - test_data: pd.DataFrame
        The testing DataFrame containing data within the specified time range.

    Example:
    train_data, test_data = split_dataframe_by_time(data, '2021-01-01 00:00:00', '2021-01-02 00:00:00',
                                                '2021-01-02 00:00:00', '2021-01-03 00:00:00', 1)
    """
    # Ensure the 'datetime' column is in datetime format
    data.index = pd.to_datetime(data.index)

    # Set the index to 'datetime' for the input DataFrame
    data.set_index('datetime', inplace=True)

    # Split the DataFrame into training and testing based on the specified time ranges
    train_data = data.loc[train_start:train_end]
    test_data = data.loc[test_start:test_end]


    return train_data, test_data


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df[indices_to_keep].astype(np.float64)
