import os
from typing import Any, Tuple

from keras.layers.core import Activation, Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model

from utils import load_configs


class Model:

    def __init__(self) -> None:
        configs = load_configs()
        self.neurons = configs['model']['neurons']  # number of hidden units in the LSTM layer
        self.activation_function = configs['model']['activation_function']
        self.loss_function = configs['model']['loss_function']  # loss function for calculating the gradient
        self.optimizer = configs['model']['optimizer']  # optimizer for applying gradient descent
        self.dropout = configs['model']['dropout']  # dropout rate used after each LSTM layer to avoid overfitting
        self.model_dir = configs['model']['models_dir']

    def build_network(self, shape: Tuple[int, int], output_size: int, reinforcement: bool) -> Sequential:
        # start stacking layers
        model = Sequential()
        model.add(LSTM(
            units=self.neurons,
            input_shape=(shape[1], shape[2]),  # Shape X (1105, 50, 9), Shape Y (1105, )
            activation=self.activation_function
        ))
        model.add(Dropout(self.dropout))

        model.add(Dense(units=output_size))  # 1 (price) or 3 (hold, long, short)
        if reinforcement:
            model.add(Activation('linear'))
        else:
            model.add(Activation(self.activation_function))

        model.compile(loss=self.loss_function, optimizer=self.optimizer)
        model.summary()

        return model

    def save_network(self, model: Any, model_name: str) -> None:
        fpath = self.model_dir
        if not os.path.exists(fpath):
            os.makedirs(fpath)

        filename = os.path.join(fpath, model_name + '.h5')
        model.save(filename)

    def load_network(self, model_name: str) -> Any:
        filename = os.path.join(self.model_dir, model_name + '.h5')
        if not os.path.isfile(filename):
            print(f'Model {model_name} does not exist...')
            return False

        return load_model(filename)
