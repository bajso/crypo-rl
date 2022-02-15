import numpy as np

from dqn import DQNAgent
from model import Model
from preprocess import Preprocess
from utils import load_configs

_PRETRAINED_MODEL = ''


def evaluate():
    configs = load_configs()
    train_set_size = configs['data']['train_set_size']
    label = configs['data']['chart']
    window_size = configs['reinforcement']['window_size']
    option = configs['reinforcement']['option']

    p = Preprocess()
    model = Model()

    df = p.load_processed_data(label)
    _, test_set = p.split_train_test(df, train_set_size=train_set_size)

    print('\nGenerating inputs ...')
    inputs, closing_prices = p.create_inputs_reinforcement(test_set, x_win_size=window_size)

    agent = DQNAgent(np.shape(inputs), option)

    # load pretrained model
    network = model.load_network(_PRETRAINED_MODEL)
    agent.model = network
    if network is False:
        raise ValueError(f'Model {_PRETRAINED_MODEL} not found')

    print(f'---------------------------- {option.upper()} --- {label} ----------------------------')
    # reset environment (first window [0:10])
    state = inputs[0]
    state = np.reshape(state, (1, agent.state_shape[0], agent.state_shape[1]))  # (1, 10, 9)
    total_reward = 0
    balance = 1000
    long = False
    short = False
    action = 0
    agent.inventory = []

    time_steps = len(inputs) - 1
    for t in range(time_steps):
        print(f'Step {t}/{time_steps}')
        print(f"Inventory: {agent.inventory}\tAction: {action}\tBalance: {balance}\t Loss: {agent.history}")

        # compute action based on current state
        action = agent.compute_action(state, evaluation=False, explore_p=0)

        # calculate reward
        reward, balance, long, short = agent.calculate_reward(action, closing_prices[t], long, short, balance)
        total_reward += reward

        # get next state
        next_state = inputs[t + 1]
        next_state = np.reshape(next_state, state.shape)

        state = next_state
