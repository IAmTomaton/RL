import time
import gym
import numpy as np
import torch
from torch import nn


class CrossEntropyAgent(nn.Module):

    def __init__(self, state_dim, action_n):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_n)
        )
        self.state_dim = state_dim
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

        logits = self.network(elite_states)
        loss = self.loss(logits, elite_actions)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return None


class Session:

    def __init__(self, states, actions, total_reward, done):
        self.states = states
        self.actions = actions
        self.total_reward = total_reward
        self.done = done

    def __str__(self):
        return 'states: {0} actions: {1} reward: {2}'.format(self.states, self.actions, self.total_reward)


def get_session(env, agent, session_len, visual=False):
    states, actions = [], []
    total_reward = 0

    state = env.reset()
    done = False
    for _ in range(session_len):
        states.append(state)
        action = agent.get_action(state)
        actions.append(action)

        if visual:
            env.render()
            time.sleep(0.05)

        state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    return Session(states, actions, total_reward, done)


def get_elite_sessions(sessions, q_param):
    total_rewards = np.array([session.total_reward for session in sessions])
    quantile = np.quantile(total_rewards, q_param)

    elite_sessions = []
    for session in sessions:
        if session.total_reward > quantile:
            elite_sessions.append(session)

    return elite_sessions


def main():

    env = gym.make("Acrobot-v1")
    state = env.reset()

    agent = CrossEntropyAgent(6, 3)
    episode_n = 100
    session_n = 20
    session_len = 500
    q_param = 0.9

    for episode in range(episode_n):
        sessions = [get_session(env, agent, session_len) for _ in range(session_n)]

        mean_total_reward = np.mean([session.total_reward for session in sessions])
        print('mean_total_reward:', mean_total_reward)

        elite_sessions = get_elite_sessions(sessions, q_param)

        if elite_sessions:
            agent.update_policy(elite_sessions)

    print('next')
    input()
    session = get_session(env, agent, session_len, True)
    print(session.total_reward, session.done)


if __name__ == '__main__':
    main()
