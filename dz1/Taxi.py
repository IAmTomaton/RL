import time
from json import JSONEncoder
import gym
import gym_maze
import numpy as np
import json
import matplotlib.pyplot as plt

"""
В методе train агент обучается, гафик обучения лежит рядом с файлом.
В файле data.json лежит уже сгенерированный агент.
В методе test можно проверить работоспособость уже обученного агента.
"""


class Session:

    def __init__(self, states, actions, total_reward, done):
        self.states = states
        self.actions = actions
        self.total_reward = total_reward
        self.done = done


class Agent:

    def __init__(self, state_n, action_n):
        self.state_n = state_n
        self.action_n = action_n
        self.policy = np.ones((state_n, action_n)) / action_n

    def get_action(self, state):
        prob = self.policy[state]
        action = np.random.choice(np.arange(self.action_n), p=prob)
        return int(action)

    def update_policy(self, elite_sessions):
        new_policy = np.zeros((self.state_n, self.action_n))

        for session in elite_sessions:
            for state, action in zip(session.states, session.actions):
                new_policy[state][action] += 1

        for state in range(self.state_n):
            if sum(new_policy[state]) == 0:
                new_policy[state] += 1 / self.action_n
            else:
                new_policy[state] /= sum(new_policy[state])

        self.policy = new_policy

    def to_json(self, path):
        with open(path, "w") as write_file:
            json.dump(self.policy, write_file, cls=NumpyArrayEncoder)

    def from_json(self, path):
        with open(path, "r") as write_file:
            decodedArrays = json.load(write_file)
            self.policy = np.asarray(decodedArrays)


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def get_state(obs):
    return obs


def get_session(env, agent, session_len, visual=False):
    states, actions = [], []
    total_reward = 0

    obs = env.reset()
    done = False
    for _ in range(session_len):
        state = get_state(obs)
        states.append(state)
        action = agent.get_action(state)
        actions.append(action)

        if visual:
            env.render()
            time.sleep(0.1)
        obs, reward, done, _ = env.step(action)
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


def train():
    env = gym.make("Taxi-v3")

    agent = Agent(500, 6)
    episode_n = 1000
    session_n = 300
    session_len = 50
    q_param = 0.1

    rewards = []

    for episode in range(episode_n):
        sessions = [get_session(env, agent, session_len) for _ in range(session_n)]

        mean_total_reward = np.mean([session.total_reward for session in sessions])
        rewards.append(mean_total_reward)
        print('episode:', episode, 'mean_total_reward:', mean_total_reward)

        elite_sessions = get_elite_sessions(sessions, q_param)

        if elite_sessions:
            agent.update_policy(elite_sessions)

    plt.plot(range(episode_n), rewards)
    plt.show()

    # agent.to_json('data.json')


def test():
    env = gym.make("Taxi-v3")
    session_len = 50
    agent = Agent(500, 6)
    agent.from_json('data.json')
    session = get_session(env, agent, session_len, True)
    print('total_reward:', session.total_reward)


if __name__ == '__main__':
    test()
