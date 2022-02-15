import gc
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import plot
from model import Model
from preprocess import Preprocess
from utils import load_configs


def build_and_run_lstm() -> None:
    configs = load_configs()
    x_window_size = configs['data']['x_window_size']
    y_window_size = configs['data']['y_window_size']
    input_scaling = configs['data']['input_scaling']
    train_set_size = configs['data']['train_set_size']
    batch_size = configs['model']['batch_size']  # default 32
    epochs = configs['model']['epochs']
    label = configs['data']['chart']

    p = Preprocess()
    hist_data = p.load_processed_data(label)

    # split to training and testing data
    train_set, test_set = p.split_train_test(hist_data, train_set_size=train_set_size)

    if input_scaling == 'minmax':
        # creates input data and labels for supervised learning
        print('Generating training inputs and lables (X_train, Y_train)...')
        X_train, Y_train, _ = p.create_inputs_minmax(train_set, x_win_size=x_window_size, y_win_size=y_window_size)
        # creates validation set to check for overfitting
        print('Generating testing inputs and lables (X_test, Y_test)...')
        X_test, Y_test, _ = p.create_inputs_minmax(test_set, x_win_size=x_window_size, y_win_size=y_window_size)
    elif input_scaling == 'zerobase':
        # creates input data and labels for supervised learning
        print('Generating training inputs and lables (X_train, Y_train)...')
        X_train, Y_train, _ = p.create_inputs_zero_base(train_set, x_win_size=x_window_size, y_win_size=y_window_size)
        # creates validation set to check for overfitting
        print('Generating testing inputs and lables (X_test, Y_test)...')
        X_test, Y_test, _ = p.create_inputs_zero_base(test_set, x_win_size=x_window_size, y_win_size=y_window_size)
    else:
        raise ValueError(f'Invalid option: input_scaling=\'{input_scaling}\'. Aborting process ...')

    model_name = f'lstm_{label}_e{epochs}_{input_scaling}'

    # clean up memory
    gc.collect()
    # fix random seed for reproducibility
    np.random.seed(202)

    # create model architecture
    lstm = Model()
    lstm_model = lstm.build_network(shape=X_train.shape, output_size=1, reinforcement=False)

    # train model
    history = lstm_model.fit(x=X_train,
                             y=Y_train,
                             batch_size=batch_size,
                             epochs=epochs,
                             verbose=2,
                             validation_data=(X_test, Y_test),
                             shuffle=True)

    # save the model to a file
    lstm.save_network(lstm_model, model_name)
    # plot model loss
    plot.plot_loss(history, label)

    # predict
    single_step_p = one_step_prediction(X_test, Y_test, hist_data, label, lstm_model)

    rmse = np.sqrt(mean_squared_error(single_step_p, Y_test))
    print(f'Test RMSE: {rmse:.3f}')

    print('Done')


def one_step_prediction(X_test: np.ndarray,
                        Y_test: np.ndarray,
                        data: pd.DataFrame,
                        label: str,
                        lstm_model: Any) -> pd.DataFrame:
    predictions = lstm_model.predict(X_test, verbose=2)

    # keep date for plotting
    size_test_set = X_test.shape[0]
    predicted_df = pd.DataFrame(data['Open Time'].iloc[-size_test_set:])

    # need to reset the index to start from 0
    predicted_df.index = range(len(predicted_df))
    predicted_df['Results'] = pd.DataFrame(predictions)
    predicted_df['Target'] = pd.DataFrame(Y_test)

    # plot prediction results
    plot.plot_prediction(predicted_df, label)

    return predictions


if __name__ == '__main__':
    build_and_run_lstm()
