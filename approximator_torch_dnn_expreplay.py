from collections import deque
import numpy as np
import random
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, state_size, action_size):
        super(Model, self).__init__()
        # begin answer
        self.layer1 = torch.nn.Linear(state_size, 128)
        self.layer2 = torch.nn.Linear(128, 32)
        self.layer3 = torch.nn.Linear(32, action_size)
        # end answer
        pass

    def forward(self, state):
        qvalues = None
        # begin answer
        qvalues = F.relu(self.layer3(F.relu(self.layer2(F.relu(self.layer1(state))))))
        # end answer
        return qvalues


class Approximator:
    '''Approximator for Q-Learning in reinforcement learning.
    
    Note that this class supports for solving problems that provide
    gym.Environment interface.
    '''

    def __init__(self,
                 state_size,
                 action_size,
                 learning_rate=0.001,
                 gamma=0.95,
                 init_epsilon=1.0,
                 epsilon_decay=0.995,
                 min_epsilon=0.01,
                 batch_size=32,
                 memory_pool_size=10000,
                 double_QLearning=False):
        '''Initialize the approximator.

        Args:
            state_size (int): the number of states for this environment. 
            action_size (int): the number of actions for this environment.
            learning_rate (float): the learning rate for training optimzer for approximator.
            gamma (float): the gamma factor for reward decay.
            init_epsilon (float): the initial epsilon probability for exploration.
            epsilon_decay (float): the decay factor each step for epsilon.
            min_epsilon (float): the minimum epsilon in training.
            batch_size (int): the batch size for training, only applicable for experience replay.
            memory_pool_size (int): the maximum size for memory pool for experience replay.
            double_QLearning (bool): whether to use double Q-learning.
        '''

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = init_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.memory_pool_size = memory_pool_size
        self.memory = deque(maxlen=memory_pool_size)
        self.batch_size = batch_size
        self.double_QLearning = double_QLearning
        # save the approximator model in self.model
        self.model = None
        # implement your approximator below
        self.model = Model(self.state_size, self.action_size)
        # begin answer
        self.optimzer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # end answer
    
    def add_to_memory(self, state, action, reward, new_state, done):
        """Add the experience to memory pool.

        Args:
            state (int): the current state.
            action (int): the action to take.
            reward (int): the reward corresponding to state and action.
            new_state (int): the new state after taking action.
            done (bool): whether the decision process ends in this state.
        """
        # begin answer
        if len(self.memory) > self.memory_pool_size:
            self.memory.popleft()
        self.memory.append((torch.Tensor(state), action, reward, torch.Tensor(new_state), done))
        # end answer
        pass
    
    def take_action(self, state):
        """Determine the action for state according to Q-value and epsilon-greedy strategy.
        
        Args:
            state (int): the current state.

        Returns:
            action (int): the action to take.
        """
        if isinstance(state, np.ndarray):
            state = torch.Tensor(state)
        action = 0
        # begin answer
        if self.epsilon < np.random.rand():
            with torch.no_grad():
                action = np.argmax(self.model(state).detach().numpy())
        else:
            action = np.random.choice(self.action_size)
        # end answer
        return int(action)
    
    def online_training(self, state, action, reward, new_state, done):
        """Train the approximator with a batch.

        Args:
            state (tuple(Tensor)): the current state.
            action (tuple(int)): the action to take.
            reward (tuple(float)): the reward corresponding to state and action.
            new_state (tuple(Tensor)): the new state after taking action.
            done (tuple(bool)): whether the decision process ends in this state.
        """
        states = torch.stack(state)  # BatchSize x StateSize
        next_states = torch.stack(new_state)  # BatchSize x StateSize
        actions = torch.Tensor(action).long()  # BatchSize
        rewards = torch.Tensor(reward)  # BatchSize
        masks = torch.Tensor(done)  # BatchSize. Note that 1 means done

        # begin answer
        loss = F.smooth_l1_loss(rewards + self.gamma * self.model(torch.Tensor(next_states)).max(axis=1).values, self.model(torch.Tensor(states)).gather(1, actions.reshape(-1,1)).flatten())
        self.optimzer.zero_grad()
        loss.backward()
        self.optimzer.step()
        # end answer
        pass
    
    def experience_replay(self):
        """Use experience replay to train the approximator.
        """
        # HINT: You may find `zip` is useful.
        # begin answer
        len_mem = len(self.memory)
        if len_mem < self.batch_size:
            return
        state, action, reward, new_state, done = zip(*((np.array(self.memory, dtype=object)[np.random.randint(0, len_mem, self.batch_size)]).tolist()))
        self.online_training(state, action, reward, new_state, done)
        # end answer
        pass
    
    def train(self, env, total_episode):
        """Train the approximator.

        Args:
            env (gym.Environment): the environment that provides gym.Environment interface.
            total_episode (int): the number of episodes to train.
        """
        # save the rewards for each training episode in self.reward_list.
        self.reward_list = []
        # Hint: you need to change the reward returned by env to be -1
        #   if the decision process ends at one step.
        # begin answer
        for episode in range(total_episode):
            total_reward = 0
            state = env.reset()
            for step in range(300):
            # begin answer
                action = self.take_action(state)
                new_state, reward, done, info = env.step(action)
                # done = 0 reward = 1
                # done = 1 reward = -1
                reward = - done * 2 + 1
                self.add_to_memory(state, action, reward, new_state, done)
                self.experience_replay()
                # self.bellman_equation_update(state, action, reward, new_state)
                total_reward += reward
                state = new_state
                if done:
                    break
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.min_epsilon:
                self.epsilon = self.min_epsilon
            # end answer
            # all_rewards += total_reward
            # all_steps += step + 1
            self.reward_list.append(total_reward)
        # end answer
        pass

    def eval(self, env, total_episode):
        """Evaluate the approximator.

        Args:
            env (gym.Environment): the environment that provides gym.Environment interface.
            total_episode (int): the number of episodes to evaluate.

        Returns:
            reward_list (list[float]): the list of rewards for every episode.
        """
        reward_list = []
        # Training has ended; thus agent does not need to explore.
        # However, you can leave it unchanged and it may not make much difference here.
        self.epsilon = 0.0
        # begin answer
        for episode in range(total_episode):
            total_reward = 0
            state = env.reset()
            for step in range(300):
                action = self.take_action(state)
                new_state, reward, done, info = env.step(action)
                total_reward += reward
                state = new_state
                if done:
                    break
            reward_list.append(total_reward)
        # end answer
        print('Average reward per episode is {}'.format(sum(reward_list) / total_episode))
        # change epsilon back for training
        self.epsilon = self.min_epsilon
        return reward_list