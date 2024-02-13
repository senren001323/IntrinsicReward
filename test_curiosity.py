import numpy as np
import time
import torch
import matplotlib.pyplot as plt
import utils
from custom_envs.Explore import Explore
from curiosity.model import DDPGCuriosity

env = Explore()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

curiosity = False
agent_0 = DDPGCuriosity(state_dim, 
                        128, 
                        action_dim,
                        8,
                        action_bound,
                        0.00003, 
                        0.00003,
                        0.0001,
                        0.0003, 
                        0.0003, 
                        curiosity,
                        0.95, 
                        0.01, 
                        0.01, 
                        5.0, 
                        0.2, 
                        device)

curiosity = True
agent_1 = DDPGCuriosity(state_dim, 
                        128, 
                        action_dim,
                        8,
                        action_bound,
                        0.00003, 
                        0.00003,
                        0.0001,
                        0.0003, 
                        0.0003, 
                        curiosity,
                        0.95, 
                        0.01, 
                        0.01, 
                        5.0, 
                        0.2, 
                        device)

agent_0_path = './param_dict/' + 'ddpg_actor.pth'
agent_1_path = './param_dict/' + 'ddpg_actor_curiosity.pth'
agent_0.actor.load_state_dict(torch.load(agent_0_path))
agent_1.actor.load_state_dict(torch.load(agent_1_path))

env_test = Explore(gui=True,
                    record=False,
                    )
obs = env_test.reset()
for i in range(50*env_test.SIM_FREQ):
    action = agent_0.take_action(obs)
    obs, reward, done, info = env_test.step(action)
    if i%env.SIM_FREQ == 0:
        env.render()
        print(done)
    if done:
        obs = env_test.reset()
env_test.close()

env_test = Explore(gui=True,
                    record=False,
                    )
obs = env_test.reset()
for i in range(500*env_test.SIM_FREQ):
    action = agent_1.take_action(obs)
    obs, reward, done, info = env_test.step(action)
    if i%env.SIM_FREQ == 0:
        env.render()
        print(done)
    if done:
        obs = env_test.reset()
env_test.close()