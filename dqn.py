import random
from collections import deque

import numpy as np

from model import Model
from utils import load_configs


class DQNAgent:

    def __init__(self, state_shape: np.shape, version: str) -> None:
        if not version in ['dqn',  'ddqn']:
            raise ValueError(f'Invalid agent version: \'{version}\'. Aborting process ...')
        self.version = version
        configs = load_configs()
        self.action_size = configs['reinforcement']['action_size']  # hold, long. short
        self.gamma = configs['reinforcement']['gamma']  # reward discount rate
        self.epsilon = configs['reinforcement']['epsilon']  # exploration rate
        self.epsilon_min = configs['reinforcement']['epsilon_min']
        self.epsilon_decay = configs['reinforcement']['epsilon_decay']
        self.batch_size = configs['reinforcement']['batch_size']

        memory_size = configs['reinforcement']['memory_size']
        self.memory = deque(maxlen=memory_size)  # deque automatically removes oldest memory once the maxlen is reached
        self.state_shape = state_shape[1:]  # (_, window_len, features)
        self.inventory = []
        self.trades = []
        self.history = 0

        m = Model()
        self.model = m.build_network(shape=state_shape, output_size=self.action_size, reinforcement=True)
        self.target_model = m.build_network(shape=state_shape, output_size=self.action_size, reinforcement=True)

    def compute_action(self, state: np.ndarray, evaluation: bool, explore_p: float) -> int:
        # get random action based on epsilon for exploration
        if not evaluation and explore_p > np.random.rand():
            return random.randrange(self.action_size)

        options = self.model.predict(state)
        # returns the index of the max value in this state [_, _, _]
        return np.argmax(options[0])  # returns action

    def calculate_reward(self,
                         action: int,
                         current_price: float,
                         long: bool,
                         short: bool,
                         balance: float) -> float | float | bool | bool:
        reward = 0  # percentage reward

        if action == 1:  # buy
            # if new trade
            if not long and not short:
                self.inventory.append(current_price)
                long = True
                short = False
            # if already short
            elif short:
                # close position
                prev_price = self.inventory.pop()
                reward = -1 * (current_price - prev_price) / prev_price  # short is inverse
                long = False
                short = False
            elif long:
                # already in long position so do nothing
                pass

        elif action == 2:  # sell
            # if new trade
            if not long and not short:
                self.inventory.append(current_price)
                long = False
                short = True
            # if already long
            elif long:
                # close position
                prev_price = self.inventory.pop()
                reward = (current_price - prev_price) / prev_price
                long = False
                short = False
            elif short:
                # already in short position so do nothing
                pass

        elif action == 3:  # hold
            # skip this step and continue with existing position
            pass

        balance = balance + balance*reward

        # clip reward
        if reward > 0:
            reward = 1
        else:
            reward = 0

        return reward, balance, long, short

    def remember(self, state: np.ndarray, action: int, reward: int, next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def experience_replay(self, batch_size: int) -> None:

        # fill minibatch with random states from the memory
        minibatch = random.sample(self.memory, batch_size)

        states = []
        targets = []
        for state, action, reward, next_state, done in minibatch:
            # predict Q values for current state [[_, _, _]]
            target = self.model.predict(state)

            if done:
                Q_update = reward
            else:
                if self.version == 'dqn':
                    # for less variation use target model to predict the Q values and pick max
                    maxQ = np.amax(self.target_model.predict(next_state)[0])

                    # Q(s,a) = r(s,a) + y*max(Q(s',a))
                    # Q target is reward of taking action at current state plus
                    # discounted max Q of all possible actions from the next state
                    Q_update = reward + self.gamma * maxQ

                elif self.version == 'ddqn':
                    # select action
                    a = np.argmax(self.model.predict(next_state)[0])
                    # predict next state
                    q_target_next_state = self.target_model.predict(next_state)
                    # evaluate action
                    # Q(s,a) = r(s,a) + y*Q(s',argmax(Q(s',a)
                    Q_update = reward + self.gamma * q_target_next_state[0][a]

            # replace Q value of action with best Q value for actions from next state
            target[0][action] = Q_update

            state = np.reshape(state, self.state_shape)  # (10, 9)
            target = np.reshape(target, self.action_size)  # 3
            states.append(state)
            targets.append(target)

        # train the model towards updated prediction
        history = self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        # save loss history
        self.history = history

    def update_target_model(self) -> None:
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
