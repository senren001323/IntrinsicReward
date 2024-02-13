import numpy as np
import torch
import utils
from train_function.train import PPOTrainCausal
from custom_envs.CausalImitationA import CasualEnv
from causal.model import IPPO

env = CasualEnv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dims = []
action_dims = []
action_bounds = []
for i in range(env.NUM_DRONES):
    state_dims.append(env.observation_space[i].shape[0])
    action_dims.append(env.action_space[i].shape[0])
    action_bounds.append(env.action_space[i].high[0])

agents = IPPO(state_dims, 
            256, 
            action_dims,
            action_bounds,
            0.001, 
            0.001,
            0.92,
            10,
            0.2,
            0.92,
            device,
            num_agents=2)

agents.load_model(0)
agents.load_model(1)

env_test = CasualEnv(gui=True,
                    record=False
                    )
agents.agents[0].set_mode('test')
agents.agents[1].set_mode('test')

obs = env_test.reset()
for i in range(10*env_test.SIM_FREQ):
    action = agents.take_actions(obs)
    obs, reward, done, info = env_test.step(action)
    done = done["__all__"]
    if i%env.SIM_FREQ == 0:
        env.render()
        print(done)
    if done:
        obs = env_test.reset()
env_test.close()

# without causal reward
env = CasualEnv()

policy = PPOTrainCausal(env, causal=False)
agents, return_list = policy.learn(num_episodes=1000)

env_test = CasualEnv(gui=True,
                    record=False
                    )
agents.agents[0].set_mode('test')
agents.agents[1].set_mode('test')

obs = env_test.reset()
for i in range(100*env_test.SIM_FREQ):
    action = agents.take_actions(obs)
    obs, reward, done, info = env_test.step(action)
    done = done["__all__"]
    if i%env.SIM_FREQ == 0:
        env.render()
        print(done)
    if done:
        obs = env_test.reset()
env_test.close()