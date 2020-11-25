import json
import math
import time
from json import JSONEncoder

import numpy as np
import gym
import matplotlib.pyplot as plt


def get_epsilon_greedy_action(q_values, epsilon, action_n):
    prob = np.ones(action_n) * epsilon / action_n
    argmax_action = np.argmax(q_values)
    prob[argmax_action] += 1 - epsilon
    action = np.random.choice(np.arange(action_n), p=prob)
    return action


def convert_state(raw_state):
    return int(round(raw_state[0] * 100)) + 200, int(round(raw_state[1] * 1000)) + 100


def get_episode(env, Q, epsilon, action_n, gamma, t_max, alpha):
    total_reward = 0
    total_x = -math.inf
    state = convert_state(env.reset())

    for t in range(t_max):
        action = get_epsilon_greedy_action(Q[state[0]][state[1]], epsilon, action_n)
        next_state, reward, done, _ = env.step(action)
        next_state = convert_state(next_state)

        Q[state[0]][state[1]][action] += alpha * (reward + gamma * np.max(Q[next_state[0]][next_state[1]]) -
                                                  Q[state[0]][state[1]][action])

        total_reward += reward
        if total_x < next_state[0]:
            total_x = next_state[0]

        if done:
            break

        state = next_state
    return total_reward, total_x


def QLearning(env, episode_n, noisy_episode_n, Q=None, gamma=0.99, t_max=500, alpha=0.5):
    action_n = env.action_space.n

    if Q is None:
        Q = np.zeros((300, 200, action_n))
    epsilon = 1

    total_rewards = []
    max_xs = []
    t = time.time()
    for episode in range(episode_n):
        total_reward, max_x = get_episode(env, Q, epsilon, action_n, gamma, t_max, alpha)

        epsilon = max(0, epsilon - 1 / noisy_episode_n)

        total_rewards.append(total_reward)
        max_xs.append(max_x)

        if episode > 0 and episode % 100 == 0:
            total_time = time.time() - t
            print('Episode number:', episode, 'Time for 100 episodes:', total_time)
            t = time.time()

        if episode > 0 and episode % 10000 == 0:
            show_progress(total_rewards, max_xs)
            to_json('Q.json', Q)

    return total_rewards, max_xs, Q


def to_json(path, Q):
    with open(path, "w") as write_file:
        json.dump(Q, write_file, cls=NumpyArrayEncoder)


def from_json(path):
    with open(path, "r") as read_file:
        decodedArrays = json.load(read_file)
        return np.asarray(decodedArrays)


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def show_progress(total_rewards, max_x):
    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    ax1.plot(total_rewards)
    ax1.set_ylabel('reward')

    ax2 = fig.add_subplot(212)
    ax2.plot(max_x)
    ax2.set_ylabel('max x')

    plt.show()


def create():
    env = gym.make("MountainCar-v0")

    total_rewards, max_x, Q = QLearning(env, episode_n=100000, noisy_episode_n=99000, t_max=1000, gamma=0.999, alpha=0.5)
    to_json('Q.json', Q)

    show_progress(total_rewards, max_x)


def test():
    env = gym.make("MountainCar-v0")

    Q = from_json('Q.json')
    action_n = env.action_space.n
    total_reward = 0
    max_x = -math.inf
    state = convert_state(env.reset())

    for t in range(1000):
        action = get_epsilon_greedy_action(Q[state[0]][state[1]], 0, action_n)
        next_state, reward, done, _ = env.step(action)
        next_state = convert_state(next_state)

        if done:
            break

        env.render()
        time.sleep(0.02)

        state = next_state

        total_reward += reward
        if max_x < next_state[0]:
            max_x = next_state[0]

    print('Total_reward:', total_reward, 'Max_x:', max_x)


if __name__ == '__main__':
    test()
