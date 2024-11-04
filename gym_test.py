import time

import gymnasium as gym

env = gym.make('FrozenLake-v1', render_mode='human')
obs, info = env.reset()
env.render()

episode_over = False
while not episode_over:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs, reward, terminated, truncated, info)

    episode_over = terminated or truncated
    env.render()
    time.sleep(2)

env.close()
