import math
import time

import gym
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


class Session:

    def __init__(self, states, actions, total_reward, max_x):
        self.states = states
        self.actions = actions
        self.total_reward = total_reward
        self.max_x = max_x


class CrossEntropyAgent(nn.Module):

    def __init__(self, state_dim, action_n):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.ReLU(),
            nn.Linear(100, action_n)
        )
        self.softmax = nn.Softmax()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, input):
        return self.network(input)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.network(state)
        action_prob = self.softmax(logits).detach().numpy()
        action = np.random.choice(len(action_prob), p=action_prob)
        return action

    def update_policy(self, elite_sessions):
        elite_states, elite_actions = [], []
        for session in elite_sessions:
            elite_states.extend(session.states)
            elite_actions.extend(session.actions)

        elite_states = torch.FloatTensor(elite_states)
        elite_actions = torch.LongTensor(elite_actions)

        loss = self.loss(self.network(elite_states), elite_actions)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return None


def get_session(env, agent, session_len, visual=False):
    states, actions = [], []
    total_reward = 0
    max_x = -math.inf

    state = env.reset()
    for _ in range(session_len):
        states.append(state)
        action = agent.get_action(state)
        actions.append(action)

        if visual:
            env.render()
            time.sleep(0.02)

        state, reward, done, _ = env.step(action)
        total_reward += reward

        if max_x < state[0]:
            max_x = state[0]

        if done:
            break

    return Session(states, actions, total_reward, max_x)


def get_elite_sessions(sessions, q_param):

    total_rewards = np.array([session.total_reward for session in sessions])
    quantile = np.quantile(total_rewards, q_param)

    elite_sessions = []
    for session in sessions:
        if session.total_reward > quantile:
            elite_sessions.append(session)

    return elite_sessions


def show_progress(mean_total_rewards, max_x, min_x):
    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    ax1.plot(mean_total_rewards)
    ax1.set_ylabel('mean reward')

    ax2 = fig.add_subplot(212)
    ax2.plot(max_x)
    ax2.plot(min_x)
    ax2.set_ylabel('min/max x')

    plt.show()


def main():
    path = 'CE.n'

    env = gym.make("MountainCar-v0")
    agent = CrossEntropyAgent(2, 3)

    episode_n = 1000
    session_n = 100
    session_len = 500
    q_param = 0.8
    mean_total_rewards = []
    max_x = []
    min_x = []
    t = time.time()

    for episode in range(episode_n):
        sessions = [get_session(env, agent, session_len) for _ in range(session_n)]

        mean_total_reward = np.mean([session.total_reward for session in sessions])

        elite_sessions = get_elite_sessions(sessions, q_param)

        if len(elite_sessions) > 0:
            agent.update_policy(elite_sessions)

        episode_time = time.time() - t
        print('Episode:', episode, 'Mean_total_reward =', mean_total_reward, 'Time for episode:', episode_time)
        t = time.time()

        if episode > 0 and episode % 100 == 0:
            show_progress(mean_total_rewards, max_x, min_x)
            torch.save(agent.state_dict(), path)

        mean_total_rewards.append(mean_total_reward)
        max_x.append(max([session.max_x for session in sessions]))
        min_x.append(min([session.max_x for session in sessions]))

    show_progress(mean_total_rewards, max_x, min_x)
    torch.save(agent.state_dict(), path)


if __name__ == '__main__':
    main()
