import os
from typing import Any, List

import pandas as pd
import plotly.graph_objs as go
from plotly import tools as pytls

from utils import load_configs


class Plot:

    def __init__(self) -> None:
        self.configs = load_configs()

    def _define_layout(self, title: str, y_t1: str, y_t2: str) -> go.Layout:
        layout = go.Layout(
            title=title,
            titlefont=dict(family='Courier New, monospace'),
            legend=dict(orientation='h'),
            xaxis=dict(type='date'),
            yaxis=dict(
                domain=[0, 0.3],
                title=y_t1,
                titlefont=dict(family='Courier New, monospace', size=16),
                hoverformat='.4f',
                tickformat='.f'
            ),
            yaxis2=dict(
                domain=[0.4, 1],
                title=y_t2,
                titlefont=dict(family='Courier New, monospace', size=16),
                hoverformat='.8f',
                tickformat='.6f'
            )
        )

        return layout

    def _define_results_layout(self, title: str, x_t: str, y_t: str) -> go.Layout:
        layout = go.Layout(
            title=title,
            titlefont=dict(family='Courier New, monospace'),
            legend=dict(orientation='h'),
            xaxis=dict(
                title=x_t,
                titlefont=dict(family='Courier New, monospace', size=16)
            ),
            yaxis=dict(
                title=y_t,
                titlefont=dict(family='Courier New, monospace', size=16),
                hoverformat='.8f',
                tickformat='.6f'
            )
        )

        return layout

    def scale_plot(self, data_col: pd.Series) -> pd.Series:
        # min-max scaling (values between 0 and 1)
        scaled = (data_col - min(data_col)) / (max(data_col) - min(data_col))
        return scaled

    def plot_price_and_volume(self, data: pd.DataFrame, tag: str) -> None:
        trace_price = go.Scatter(
            x=data['Open Time'],
            y=data['Close'],
            name=tag[:3]
        )

        trace_volume = go.Scatter(
            x=data['Open Time'],
            y=data['Volume'],
            xaxis='x',
            yaxis='y2',
            name=tag[:3]
        )

        ty1 = 'Volume [BTC]'
        ty2 = 'Closing Price [BTC]'
        layout = self._define_layout(tag, ty1, ty2)

        fig = pytls.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01)
        fig.append_trace(trace_price, 2, 1)
        fig.append_trace(trace_volume, 1, 1)

        fig = go.Figure(fig, layout=layout)
        fig.show()

    def plot_loss(self, history: Any, tag: str) -> None:
        title = f'Model Loss {tag}'
        ty1 = 'Number of Epochs'
        ty2 = 'Loss'

        trace_loss = go.Scatter(
            x=history.epoch,
            y=history.history['loss'],
            name='Loss'
        )

        trace_val_loss = go.Scatter(
            x=history.epoch,
            y=history.history['val_loss'],
            name='Validation Loss'
        )

        layout = self._define_results_layout(title, ty1, ty2)
        data = [trace_loss, trace_val_loss]

        fig = go.Figure(data=data, layout=layout)
        fig.show()

    def plot_predictions(self, predicted_df: pd.DataFrame, tag: str) -> None:
        title = f'Prediction {tag}'
        ty1 = 'Date'
        ty2 = 'Price'

        trace_target = go.Scatter(
            x=predicted_df['Open Time'],
            y=predicted_df['Target'],
            name='Actual'
        )

        trace_predicted = go.Scatter(
            x=predicted_df['Open Time'],
            y=predicted_df['Results'],
            name='Predicted'
        )

        layout = self._define_results_layout(title, ty1, ty2)
        # process timestamps as date
        layout.update(xaxis=dict(type='date'))

        data = [trace_target, trace_predicted]

        fig = go.Figure(data=data, layout=layout)
        fig.show()

    def plot_episodes(self, episodes: List[int], balances: List[str], tag: str) -> None:
        title = f'Model {tag} balance vs training episodes'
        ty1 = 'Episodes'
        ty2 = 'Balance'

        trace = go.Scatter(x=episodes, y=balances)

        layout = self._define_results_layout(title, ty1, ty2)
        # fix to two decimal places
        layout.update(yaxis=dict(hoverformat='.2f', tickformat='.2f'))

        fig = go.Figure(data=[trace], layout=layout)
        fig.show()

    def plot_evaluation(self, fname: str) -> None:
        path = os.path.join(self.configs['model']['models_dir'], fname)
        ep_on_save = 10
        balances = None
        with open(path + '.txt', 'r', encoding='UTF8') as f:
            balances = f.read().split(',')

        balances.pop(-1)  # last entry is a comma
        ep = (len(balances) - 1) * ep_on_save
        xaxis = list(range(0, ep + ep_on_save, ep_on_save))

        self.plot_episodes(xaxis, balances, fname)
