# Importing required libraries

import pandas as pd
import numpy as np
import talib as tb



# Calculate SMA for given periods

def SMA(close:pd.Series,short_period:int,long_period:int) -> pd.DataFrame:

    """
    Here we calculate the SMA value for given long and short time period.
    close:pd.Series = Price of the stock
    short_period:int = Short time period
    long_period:int = Long time period

    """
    # shortSMA is the sma of short_period
    shortSMA = tb.SMA(close,short_period)
    # longSMA is the sma of the long_period
    longSMA = tb.SMA(close,long_period)
    # Generating SMA signal : 1 when the difference between shortSMA and longSMA is positive else it is -1.
    smaSignal = [1 if (short - long) > 0 else (-1 if(short - long) < 0 else 0) for long, short in zip(longSMA, shortSMA)]
    

    # setting column names
    cols = ['close','Signal', 'shortSMA', 'longSMA']
    df  =  pd.DataFrame(columns=cols)
    # Assigning Close price to df['Close']
    df['close'] = close
    # Assigning SMA to df['Signal']
    df['Signal'] = smaSignal
    # Assigning Short SMA to df['shortSMA']
    df['shortSMA'] = shortSMA
    # Assigning Long SMA to df['longSMA']
    df['longSMA'] = longSMA
    df = df.fillna(0)
    return df



# Calculate EMA for given periods

def EMA(close:pd.Series,short_period:int,long_period:int) -> pd.DataFrame:

    """
    Here we calculate the EMA value for given long and short time period.
    close:pd.Series = Price of the stock
    short_period:int = Short time period
    long_period:int = Long time period

    """
    # shortEMA is the sma of short_period
    shortEMA = tb.EMA(close,short_period)
    # longEMA is the sma of the long_period
    longEMA = tb.EMA(close,long_period)
    # Generating EMA signal : 1 when the difference between shortEMA and longEMA is positive else it is -1.
    emaSignal = [1 if (short - long) > 0 else (-1 if (short - long) < 0 else 0) for long, short in zip(longEMA, shortEMA)]

    # setting column names
    cols = ['close','Signal', 'shortEMA', 'longEMA']
    df  =  pd.DataFrame(columns=cols)
    # Assigning Close price to df['Close']
    df['close'] = close
    # Assigning EMA to df['Signal']
    df['Signal'] = emaSignal
    # Assigning Short EMA to df['shortEMA']
    df['shortEMA'] = shortEMA
    # Assigning Long EMA to df['longEMA']
    df['longEMA'] = longEMA
    df = df.fillna(0)
    return df



# Calculate RSI for given periods

def RSI(close:pd.Series,period:int) -> pd.DataFrame:

    """
    Here we calculate the RSI value for given long and short time period.
    close:pd.Series = Price of the stock
    """
    Overbought = 80
    Oversold = 20 
    value_RSI = tb.RSI(close,timeperiod=period)
    rsiSignal = [-1 if value > Overbought else (1 if value < Oversold else np.nan) for value in value_RSI]

    # setting column names
    cols = ['close','Signal', 'RSI_Value']
    df  =  pd.DataFrame(columns=cols)
    # Assigning Close price to df['Close']
    df['close'] = close
    df['Signal'] = rsiSignal
    df['RSI_Value'] = value_RSI
    df = df.fillna(method='ffill')
    return df



# Calculate ROC for given periods

def ROC(close:pd.Series,period:int) -> pd.DataFrame:

    """
    Here we calculate the ROC value for given long and short time period.
    close:pd.Series = Price of the stock
    short_period:int = Short time period
    long_period:int = Long time period

    """
    value_ROC = tb.ROC(close,timeperiod=period)
    rsiSignal = [1 if value > 0 else -1 for value in value_ROC]

    # setting column names
    cols = ['close','Signal', 'ROC_Value']
    df  =  pd.DataFrame(columns=cols)
    # Assigning Close price to df['Close']
    df['close'] = close
    df['Signal'] = rsiSignal
    df['ROC_Value'] = value_ROC
    df = df.fillna(method='ffill')
    return df


def optmizer_oscillator(price,min_period, max_period, TI):
    opt_period = -1
    max_pnl = 0
    for i in range(min_period, max_period):
        if TI == 'RSI':
            df = RSI(price,i)
        elif TI == 'ROC':
            df = ROC(price,i)
        pnl = PnL(df)
        if (opt_period == -1) or (opt_period  == -1) or (opt_period  == 0):
            max_pnl = pnl
            opt_period = i
        if max_pnl < pnl:
            max_pnl = pnl
            opt_period = i
    return opt_period


# Profit and Loss calculator

def PnL(data:pd.DataFrame) -> float:
    
    """
    Here we calculate the total profit and loss, given the Price and the Signal.
    data:pd.DataFrame = Price and Signal dataframe

    """
    # Calculating the P/L by multiplying the price difference with the signal and summing up to get the whole P/L.
    pnl = (data['close'].diff(1)*data['Signal'].shift(1)).drop(0).sum()
    
    return pnl



# Technical Indicator Optimizer 

def optimize_trend(price:pd.Series, min_long:int, max_long:int, min_short:int, max_short:int,TI:str )-> int:

    """
    Here we optimize the long and short time period given, the price, long range - (min ,max) and short range - (min ,max).
    price:pd.Series = Price of the stock
    min_long:int = minimum value of the long time period
    max_long:int = maximum value of the long time period
    min_short:int = minimum value of the short time period
    max_short:int = maximum value of the short time period
    TI:str = Technical indicator that you want to optimize

    """
    # opt_short = optimized short time period
    opt_short = -1
    # opt_long = optimized long time period
    opt_long = -1
    # initializing maximum P/L as 0
    max_pnl = 0

    # iterating over the long range.
    for i in range(min_long, max_long):

        # iterating over the short range.
        for j in range(min_short, max_short):
            # Checking short value
            if j > i :
                # Skipping if short value is more than long value
                continue
            else:
                # Calculate TI for given periods
                if TI == 'SMA':
                    df = SMA(price, j, i)
                elif TI == 'EMA':
                    df = EMA(price, j, i)

                # Calculating P/L based on calculated TI
                pnl = PnL(df)

                if (opt_short == -1) or (opt_long  == -1):
                    # Updating Optimized short and long periods
                    max_pnl = pnl
                    opt_short = j
                    opt_long = i
                    
                if max_pnl < pnl:
                    # Updating Optimized short and long periods
                    max_pnl = pnl
                    opt_short = j
                    opt_long = i
                    
    return opt_long, opt_short
