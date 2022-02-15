import numpy as np

from dqn import DQNAgent
from model import Model
from preprocess import Preprocess
from utils import load_configs

_PRETRAINED_MODEL = ''


def run_rfl() -> None:
    configs = load_configs()
    train_set_size = configs['data']['train_set_size']
    label = configs['data']['chart']
    window_size = configs['reinforcement']['window_size']
    episodes = configs['reinforcement']['episodes']
    option = configs['reinforcement']['option']

    p = Preprocess()
    model = Model()

    df = p.load_processed_data(label)
    train_set, _ = p.split_train_test(df, train_set_size=train_set_size)

    print('\nGenerating inputs ...')
    inputs, closing_prices, _ = p.create_inputs_reinforcement(train_set, x_win_size=window_size)

    agent = DQNAgent(np.shape(inputs), option)

    # load trained model if exists
    network = model.load_network(_PRETRAINED_MODEL)
    if network is not False:
        agent.model = network

    # update weights of target network with dqn network weights
    agent.update_target_model()

    evaluation = False  # switch on for testing

    rewards = []
    losses = []
    decay_step = 0

    time_steps = len(inputs) - 1
    print(f'---------------------------- {option.upper()} --- {label} ----------------------------')
    for e in range(episodes + 1):
        # reset environment (first window [0:10])
        state = inputs[0]
        state = np.reshape(state, (1, agent.state_shape[0], agent.state_shape[1]))  # (1, 10, 9)
        total_reward = 0
        balance = 1000
        long = False
        short = False
        action = 0
        agent.inventory = []

        for t in range(time_steps):
            print(f'Step {t}/{time_steps}')
            print(f"Inventory: {agent.inventory}\tAction: {action}\tBalance: {balance}\t Loss: {agent.history}")

            decay_step += 1
            # calculate exploration probability
            explore_p = agent.epsilon_min + (agent.epsilon - agent.epsilon_min) * \
                np.exp(-agent.epsilon_decay * decay_step)

            # compute action based on current state
            action = agent.compute_action(state, evaluation, explore_p)

            # calculate reward
            reward, balance, long, short = agent.calculate_reward(action, closing_prices[t], long, short, balance)
            total_reward += reward

            done = True if t == time_steps - 1 else False

            if done:
                rewards.append(total_reward)
                losses.append(agent.history.history)

                agent.update_target_model()

                print(
                    f"Episode: {e}/{episodes}\tTotal Reward: {total_reward}\tStart Balance: 1000\tEnd Balance: {balance}\tLoss: {agent.history}")
                break

            # get next state
            next_state = inputs[t + 1]
            next_state = np.reshape(next_state, state.shape)

            # store in memory (SARS')
            agent.remember(state, action, reward, next_state, done)

            # recall after memory has enough samples in it
            if len(agent.memory) > agent.batch_size:
                agent.experience_replay(agent.batch_size)

            state = next_state

        if e % 10 == 0:
            model.save_network(agent.model, f'{label}_ep{e}')
