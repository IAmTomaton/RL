import time
import gym
import gym_maze
import numpy as np


class Session:

    def __init__(self, states, actions, total_reward):
        self.states = states
        self.actions = actions
        self.total_reward = total_reward

    def __str__(self):
        return 'states: {0} actions: {1} reward: {2}'.format(self.states, self.actions, self.total_reward)


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


def get_state(obs):
    return int(obs[0] * 5 + obs[0])


def get_session(env, agent, session_len, visual=False):
    states, actions = [], []
    total_reward = 0

    obs = env.reset()
    for _ in range(session_len):
        state = get_state(obs)
        states.append(state)
        action = agent.get_action(state)
        actions.append(action)

        if visual:
            env.render()
            time.sleep(0.2)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    return Session(states, actions, total_reward)


def get_elite_sessions(sessions, q_param):
    total_rewards = np.array([session.total_reward for session in sessions])
    quantile = np.quantile(total_rewards, q_param)

    elite_sessions = []
    for session in sessions:
        if session.total_reward > quantile:
            elite_sessions.append(session)

    return elite_sessions


def main():
    env = gym.make('maze-sample-5x5-v0')

    agent = Agent(25, 4)
    episode_n = 50
    session_n = 100
    session_len = 100
    q_param = 0.9

    for episode in range(episode_n):
        sessions = [get_session(env, agent, session_len) for _ in range(session_n)]

        mean_total_reward = np.mean([session.total_reward for session in sessions])
        print('mean_total_reward:', mean_total_reward)

        elite_sessions = get_elite_sessions(sessions, q_param)

        if elite_sessions:
            agent.update_policy(elite_sessions)

    get_session(env, agent, session_len, True)


if __name__ == '__main__':
    main()
