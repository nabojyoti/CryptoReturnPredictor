import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from itertools import cycle

# plot candle stick chart 
def candle_stick_chart(data:pd.DataFrame) -> go.Figure:
    """
    Create a candlestick chart with volume for financial data.

    Parameters:
    - data: pd.DataFrame
        The input DataFrame containing financial data with columns 'datetime', 'open', 'high',
        'low', 'close', and 'volume'.

    Returns:
    - go.Figure
        A Plotly Figure object displaying the candlestick chart with volume.

    Example:
    fig = candlestick_chart(df)
    fig.show()
    """
        
    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)

    # Create candlestick charts for different timeframes
    candlestick_trace = go.Candlestick(
        x=data['datetime'],
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='15Min',
    )

    # Customize the layout (optional)
    candlestick_trace.update(
        increasing_line_color='green',
        decreasing_line_color='red'
    )

    # Add the candlestick trace to the first subplot
    fig.add_trace(candlestick_trace, row=1, col=1)

    # Add x-axis labels
    fig.update_xaxes(title_text='Date', row=1, col=1)

    # Create and add the volume trace to the second subplot (you can customize this as needed)
    volume_trace = go.Bar(
        x=data.index.to_list(),
        y=data['volume'],  # Use 'Close' for volume for this example
        name='volume',
        marker_color='black'
    )

    # Add the volume trace to the second subplot
    fig.add_trace(volume_trace, row=2, col=1)


    # Customize the layout (optional)
    fig.update_layout(
        title='Candlestick Chart of ETHUSDT',
        xaxis_title='Timestamp',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=False,
        xaxis=dict(
        showgrid=False,  # Remove x-axis grid lines
        showline=False,  # Remove x-axis line
        ),
        yaxis=dict(
            showgrid=False,  # Remove y-axis grid lines
            showline=False,  # Remove y-axis line
        ),
        paper_bgcolor='white',  # Set white background
        font=dict(family = "Courier New, monospace",size = 10, color = "RebeccaPurple")
    )

    return fig

#plot monthwise open and close price
def plot_monthly_open_close_comparison(data:pd.DataFrame) -> go.Figure:
    """
    Create a bar chart comparing the monthly open and close prices of ETHUSDT.

    Parameters:
    - data: pd.DataFrame
        The input DataFrame containing financial data with 'datetime', 'open', and 'close' columns.

    Returns:
    - go.Figure
        A Plotly Figure object displaying the month-wise comparison.

    Example:
    fig = plot_monthly_open_close_comparison(df)
    fig.show()
    """
    # Group data by month and calculate the mean open and close prices
    monthwise = data.groupby(data['datetime'].dt.strftime('%B'))[['open', 'close']].mean()

    # Define the order of months
    new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                 'September', 'October', 'November', 'December']

    # Reindex the grouped data to ensure the order of months
    monthwise = monthwise.reindex(new_order, axis=0)

    # Create a Plotly figure
    fig = go.Figure()

    # Add bar traces for open and close prices
    fig.add_trace(go.Bar(
        x=monthwise.index,
        y=monthwise['open'],
        name='ETHUSDT Open Price',
        marker_color='crimson'
    ))
    fig.add_trace(go.Bar(
        x=monthwise.index,
        y=monthwise['close'],
        name='ETHUSDT Close Price',
        marker_color='lightsalmon'
    ))

    # Customize the layout
    fig.update_layout(
        xaxis_title='Months',
        yaxis_title='Price (USD)',
        barmode='group',
        xaxis_tickangle=-45,
        title='Monthwise Comparison between Open and Close Price of ETHUSDT',
        font=dict(family="Courier New, monospace", size=10, color="RebeccaPurple")
    )

    return fig

#plot monthwise high and low price
def plot_monthly_high_low_comparison(data):
    """
    Create a bar chart comparing the monthly high and low prices of ETHUSD.

    Parameters:
    - data: pd.DataFrame
        The input DataFrame containing financial data with 'datetime', 'high', and 'low' columns.

    Returns:
    - go.Figure
        A Plotly Figure object displaying the month-wise comparison.

    Example:
    fig = plot_monthly_high_low_comparison(df)
    fig.show()
    """
    # Define the order of months
    new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                 'September', 'October', 'November', 'December']

    # Group data by month and calculate the minimum low and maximum high prices
    monthwise_high = data.groupby(data['datetime'].dt.strftime('%B'))['high'].max()
    monthwise_low = data.groupby(data['datetime'].dt.strftime('%B'))['low'].min()

    # Reindex the grouped data to ensure the order of months
    monthwise_high = monthwise_high.reindex(new_order, axis=0)
    monthwise_low = monthwise_low.reindex(new_order, axis=0)

    # Create a Plotly figure
    fig = go.Figure()

    # Add bar traces for high and low prices
    fig.add_trace(go.Bar(
        x=monthwise_high.index,
        y=monthwise_high,
        name='ETH High Price',
        marker_color='rgb(0, 153, 204)'
    ))
    fig.add_trace(go.Bar(
        x=monthwise_low.index,
        y=monthwise_low,
        name='ETH Low Price',
        marker_color='rgb(255, 128, 0)'
    ))

    # Customize the layout
    fig.update_layout(
        xaxis_title='Months',
        yaxis_title='Price (USD)',
        barmode='group',
        xaxis_tickangle=-45,
        title='Monthwise Comparison between ETHUSDT High and Low Price of ETHUSDT',
        font=dict(family="Courier New, monospace", size=10, color="RebeccaPurple")
    )

    return fig


def plot_ohlc_price_chart(data):
    """
    Create an OHLC price chart for ETHUSD.

    Parameters:
    - data: pd.DataFrame
        The input DataFrame containing financial data with 'datetime', 'open', 'close', 'high', and 'low' columns.

    Returns:
    - go.Figure
        A Plotly Figure object displaying the OHLC price chart.

    Example:
    fig = plot_ohlc_price_chart(df)
    fig.show()
    """
    # Create an infinite cycle of trace names
    names = cycle(['ETHUSDT Open Price', 'ETHUSDT Close Price', 'ETHUSDT High Price', 'ETHUSDT Low Price'])

    # Create the OHLC price chart
    fig = px.line(data, x=data['datetime'], y=[data['open'], data['close'], data['high'], data['low']],
                  labels={'Date': 'Date', 'value': 'Price (USD)'})

    # Customize the layout
    fig.update_layout(
        title_text='ETHUSDT OHLC Price Chart',
        font_size=15,
        font_color='black',
        legend_title_text='Parameters',
        font=dict(family="Courier New, monospace", size=10, color="RebeccaPurple")
    )

    # Update trace names using the cycle
    fig.for_each_trace(lambda t: t.update(name=next(names)))

    # Remove grid lines from the x and y axes
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig

def prediction_visualization(train:pd.DataFrame, test:pd.DataFrame, time_period:str = '15 mins') -> go.Figure:
    """
    Create a visualization of actual vs. predicted returns for a given time period.

    Parameters:
    train (pd.DataFrame): DataFrame containing training data with columns 'datetime', 'predicted_return', and 'label'.
    test (pd.DataFrame): DataFrame containing test data with columns 'datetime', 'predicted_return', and 'label'.
    time_period (str, optional): The time period for which the visualization is created (default is '15 mins').

    Returns:
    go.Figure: A Plotly Figure object containing the visualization.

    This function creates a Plotly Figure to visualize actual and predicted returns over time for both training and test datasets.
    It plots the actual returns and predicted returns on the same graph for comparison.

    Example:
    ```
    import pandas as pd
    import plotly.graph_objs as go

    # Create train and test DataFrames
    train_data = pd.DataFrame(...)
    test_data = pd.DataFrame(...)

    # Generate the visualization
    figure = prediction_visualization(train_data, test_data, time_period='30 mins')
    figure.show()
    ```

    Note:
    - Make sure to have the 'datetime', 'predicted_return', and 'label' columns in both train and test DataFrames.
    - You need to have the Plotly library (imported as 'go') installed to use this function.

    """
    train.reset_index(inplace = True)
    test.reset_index(inplace = True)
    #historical
    fig = go.Figure()
    fig.add_trace(
      go.Scatter(
          x=train['datetime'],
          y=train['prediction'],
          mode='lines',
#           line=dict(color='orange', width=2.5),
          name='Predicted Returns (train data)'
      )
    )
    #historical
    fig.add_trace(
      go.Scatter(
          x=train['datetime'],
          y=train['label'],
          mode='lines',
#           line=dict(color='black', width=1.5),
          name='Actual Returns (train data)',
      )
    )

    
    #prediction
    fig.add_trace(
      go.Scatter(
          x=test['datetime'],
          y=test['prediction'],
          mode='lines',
#           line=dict(color='navy', width=2.5),
          name='Predicted Returns (test_data)'
      )
    )
    #historical
    fig.add_trace(
      go.Scatter(
          x=test['datetime'],
          y=test['label'],
          mode='lines',
#           line=dict(color='grey', width=1.5),
          name='Actual Returns (test data)',
      )
    )
    # fig.update_layout(plot_bgcolor='white')
    fig.update_layout(
      title=f"Actual vs Predicted Returns | ETH USD | {time_period} duration ",
      xaxis_title="Datetime",
      yaxis_title="Returns",
      # legend_title="Legend Title",
      plot_bgcolor='white'
    )

    return fig
