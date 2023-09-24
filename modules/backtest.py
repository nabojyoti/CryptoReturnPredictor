import pandas as pd
import os


def single_crypto_backtest(data, initial_balance = 1000000, rebalance_interval_minutes=15):
    """
    Perform a backtest for single cryptocurrency trading with portfolio rebalancing.

    Parameters:
    - data (pd.DataFrame): DataFrame with columns 'datetime', 'close', and 'prediction'.
    - rebalance_interval_minutes (int): Time interval for portfolio rebalancing in minutes.
    - initial_balance (int) # Starting balance
    
    Returns:
    - final_value (float): Final portfolio value.
    - portfolio_df (pd.DataFrame): DataFrame containing portfolio information.
    """

    # Backtesting logic
    
    balance = initial_balance
    shares_held = 0
    buy_price = None
    last_rebalance_time = None
    portfolio_info = []

    for index, row in data.iterrows():
        if last_rebalance_time is None or (index - last_rebalance_time).total_seconds() >= (rebalance_interval_minutes * 60):
            # Time to rebalance the portfolio
            if shares_held > 0:
                # Sell all shares to rebalance
                sell_quantity = shares_held
                balance += sell_quantity * row['close']
                portfolio_info.append({'Action': 'Sell', 'Quantity': sell_quantity, 'Price': row['close'], 'Time': index,
                                       'Portfolio Value': shares_held * row['close'], 'Remaining Balance': balance})
                shares_held = 0  # Update shares_held after selling

        if row['signal'] == 1 and balance > row['close']:
            # Buy signal
            if last_rebalance_time is None or (index - last_rebalance_time).total_seconds() >= (rebalance_interval_minutes * 60):
                # Buy only if it's time to rebalance
                buy_quantity = int(balance // row['close'])  # Buy as many shares as possible
                shares_held += buy_quantity
                balance -= buy_quantity * row['close']
                portfolio_info.append({'Action': 'Buy', 'Quantity': buy_quantity, 'Price': row['close'], 'Time': index,
                                       'Portfolio Value': shares_held * row['close'], 'Remaining Balance': balance})
                buy_price = row['close']

        last_rebalance_time = index

    # Sell any remaining shares at the end of the backtest
    if shares_held > 0:
        sell_quantity = shares_held
        balance += sell_quantity * data['close'][-1]
        portfolio_info.append({'Action': 'Sell', 'Quantity': sell_quantity, 'Price': data['close'][-1], 'Time': data.index[-1],
                               'Portfolio Value': shares_held * data['close'][-1], 'Remaining Balance': balance})
        shares_held = 0

    # Calculate final portfolio value
    final_value = balance

    # Create a DataFrame from the portfolio information
    portfolio_df = pd.DataFrame(portfolio_info)
    
    return final_value, portfolio_df

def backtestCalculator(cash:float, backtest_df:pd.DataFrame, weights:dict) -> tuple:

    """

    Given cash value, backtest_df and weights it calculates the value invested, current value, balance left and shares.

 

    parameters

        cash: Cash value before trading

        backtest_df: Contains price information of t and t+1 th days

        weights: weight to assign to each asset

    returns

        cash_invested, current_value_of_investment, shares

    """

    amount_allocation = {}

    shares, total, balance, total_invested = {}, 0, 0, 0

    prices = backtest_df.iloc[0].to_dict()

    new_prices = backtest_df.iloc[len(backtest_df)-1].to_dict()

 

    for keys in backtest_df.columns:

        amount_allocation[keys] = weights[keys] * cash

 

    for keys in backtest_df.columns:

        shares[keys] = (amount_allocation[keys] // prices[keys])

 

    for keys in backtest_df.columns:

        total_invested = total_invested + (shares[keys] * prices[keys])

 

    balance = cash - total_invested

 

    for keys in backtest_df.columns:

        total = total + (shares[keys] * new_prices[keys])

 

    return total_invested, total, balance, shares[keys]