import pandas as pd
import multiprocessing as mp
from database import *
from logger import create_logger
import plotly.graph_objects as go


def build_candlesticks(start_date: str, end_date: str = None, debug: bool = False, cores: int = mp.cpu_count()):
    logger = create_logger(__name__, '', 'Coinbase Pro - PostgreSQL ETL Pipeline')
    db = Database(debug=debug, logger=logger)
    ticker_dict = db.build_candlesticks(start_date=start_date, end_date=end_date, logger=logger, cores=cores)
    return ticker_dict


def create_candlestick_chart(df):
    # Ensure the DataFrame has the required columns
    required_columns = ["Time_Stamp", "Date", "Time", "Open", "High", "Low", "Close", "Volume"]
    if not all(column in df.columns for column in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    # Create the candlestick figure
    fig = go.Figure()

    # Add candlestick trace
    fig.add_trace(go.Candlestick(
        x=df['Time_Stamp'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        name='Candlesticks'
    ))

    # Add volume bars with full opacity and secondary y-axis
    fig.add_trace(go.Bar(
        x=df['Time_Stamp'],
        y=df['Volume'],
        name='Volume',
        marker=dict(
            color=['green' if open_ < close_ else 'red' for open_, close_ in zip(df['Open'], df['Close'])],
            line=dict(width=0)  # Remove borders for a cleaner look
        ),
        yaxis='y2',
        opacity=1.0,  # Ensure full opacity
        showlegend=False
    ))

    # Set up layout
    fig.update_layout(
        title='Candlestick Chart with Volume',
        xaxis_title='Time',
        yaxis_title='Price',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False,
            position=0.95,  # Adjust the position to make volume bars more prominent
            anchor='x',
            titlefont=dict(color='blue'),  # Change title font color for clarity
            tickfont=dict(color='blue'),  # Change tick font color for clarity
        ),
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Customize hover labels for both traces
    fig.update_traces(hoverinfo='x+y+text')

    # Show the figure
    fig.write_html('test_plot.html')
    fig.show()


if __name__ == '__main__':
    ticker_dict = build_candlesticks(start_date='2024-05-11', debug=True)
    create_candlestick_chart(pd.DataFrame(ticker_dict['BTC-USDT']['data']))
    add_new_data(ticker='BTC-USDT', start_date='2024-05-08-00-00', end_date='2024-05-12-00-00')
