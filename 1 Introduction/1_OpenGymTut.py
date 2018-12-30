import gym
env = gym.make('MsPacman-v0')
env.reset()
for _ in range(2000):
	env.render()
	env.step(env.action_space.sample())