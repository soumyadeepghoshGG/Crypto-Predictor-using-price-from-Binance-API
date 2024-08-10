import os
import pandas as pd
import pickle
import datetime
import plotly
import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots as sp
from prophet.plot import plot_components_plotly
from .base import CryptoBase

class Plotter(CryptoBase):
    def __init__(self, 
                 data: pd.DataFrame, 
                 forecast: pd.DataFrame, 
                 crypto: str, 
                 interval: str, 
                 n_days_past: int, 
                 model: str):
        super().__init__(crypto=crypto,
                         interval=interval,
                         ohlcv='open',     # Placeholder
                         n_days_past=n_days_past, 
                         n_days_future=1,  # Placeholder
                         model=model,      # Placeholder
                         override=False)   # Placeholder
        self.data = data
        self.forecast = forecast
        self.plot_dir = 'plots'
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

    def plot_trend(self) -> None:
        # Create candlestick plot for actual data
        actual = go.Candlestick(
            x=self.data['timestamp'],
            open=self.data['open'],
            high=self.data['high'],
            low=self.data['low'],
            close=self.data['close'],
            increasing={'line': {'color': '#54b686'}},
            decreasing={'line': {'color': '#a64149'}},
            name='Candlestick'
        )

        # Create forecast line plot
        yhat = go.Scatter(
            x=self.forecast['ds'],
            y=self.forecast['yhat'],
            mode='lines',
            line={'color': '#3bbed7', 'width': 3},
            name='Forecast'
        )

        # Confidence interval for forecast
        yhat_upper = go.Scatter(
            x=self.forecast['ds'],
            y=self.forecast['yhat_upper'],
            hoverinfo='none',
            mode='lines',
            line={'color': 'rgba(255, 255, 255, 0)'},
            fill='tonexty',
            fillcolor='rgba(255, 255, 255, 0.15)',
            name='Confidence Interval'
        )

        yhat_lower = go.Scatter(
            x=self.forecast['ds'],
            y=self.forecast['yhat_lower'],
            hoverinfo='none',
            mode='lines',
            line={'color': 'rgba(255, 255, 255, 0)'},
            showlegend=False
        )

        # Define layout with annotations
        layout = go.Layout(
            title={
                'text': f'<b>{self.crypto} trend using {self.interval} interval data for {self.n_days_past} days - {self.model}</b>',
                'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
                'font': {'size': 24, 'color': '#800000'}
            },
            xaxis={'title': '', 'showgrid': True, 'gridcolor': '#262b32', 'gridwidth': 1},
            yaxis={'title': '', 'showgrid': True, 'gridcolor': '#262b32', 'gridwidth': 1},
            hovermode='x',
            margin={'t': 50, 'b': 0, 'l': 0, 'r': 0},
            legend={
                'x': 0.05, 'y': 0.95, 'bgcolor': 'rgba(147, 149, 151, 0.8)', 'borderwidth': 1, 'bordercolor': '#000000',
                'font': {'size': 16, 'color': 'black'}
            },
            width=1900,
            height=950,
            plot_bgcolor='#161a1e',
            paper_bgcolor='#161a1e',
            annotations=[
                dict(
                    text=f"Date: {datetime.datetime.now().strftime('%d-%m-%Y')}",
                    x=0.99,
                    y=1,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16, color="black"),
                    bgcolor="rgba(147, 149, 151, 0.8)",
                    bordercolor="black",
                    borderwidth=1
                )
            ]
        )

        fig = go.Figure(data=[actual, yhat, yhat_lower, yhat_upper], layout=layout)

        # Save the plot to HTML
        try:
            plotly.offline.plot(
                fig, 
                filename=f"{self.plot_dir}/{self.crypto} trend using {self.interval} interval data for {self.n_days_past} days - {self.model} ({datetime.datetime.now().strftime('%d-%m-%Y')}).html"
            )
        except Exception as e:
            print(f"Error saving trend plot: {e}")

    def plot_components(self) -> None:
        pio.templates.default = "plotly_dark"
        
        # Load model instance for plotting components
        try:
            with open(f"model/Prophet_model_for_{self.crypto}-{self.interval}-{self.n_days_past}d.pkl", 'rb') as file:
                model_instance = pickle.load(file)

            fig = plot_components_plotly(m=model_instance, fcst=self.forecast)

            for axis in fig.layout:
                if 'xaxis' in axis:
                    fig.layout[axis].update(showgrid=True)
                if 'yaxis' in axis:
                    fig.layout[axis].update(showgrid=True)

            fig.update_layout(
                title={
                    'text': f"<b>{self.crypto} components using {self.interval} interval data for {self.n_days_past} days - Prophet</b>",
                    'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
                    'font': {'size': 24, 'color': '#800000'}
                },
                showlegend=False,
                width=1900,
                height=1200,
                plot_bgcolor='#161a1e',
                paper_bgcolor='#161a1e',
                annotations=[
                    dict(
                        text=f"Date: {datetime.datetime.now().strftime('%d-%m-%Y')}",
                        x=0.01,
                        y=1,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=16, color="black"),
                        bgcolor="rgba(147, 149, 151, 0.8)",
                        bordercolor="black",
                        borderwidth=1
                    )
                ]
            )

            plotly.offline.plot(
                fig, 
                filename=f"{self.plot_dir}/{self.crypto} components using {self.interval} interval data for {self.n_days_past} days - Prophet ({datetime.datetime.now().strftime('%d-%m-%Y')}).html"
            )
        except Exception as e:
            print(f"Error saving components plot: {e}")
