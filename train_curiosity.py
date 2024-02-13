import torch
import matplotlib.pyplot as plt
import utils
from custom_envs.Explore import Explore
from train_function.train import DDPGTrainCuriosity
import os

# nothing used
env = Explore()

policy = DDPGTrainCuriosity(env, curiosity=False)
agent, return_list = policy.learn(num_episodes=10000)

file_dir = './return_result/'
file_path = os.path.join(file_dir, "Normal Gaussian")

episodes = list(range(len(return_list)))
mv_return = utils.moving_average(return_list, 9)
plt.plot(episodes, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Normal Gaussian')
plt.savefig(file_path)
plt.close()

file_name = './param_dict/' + 'ddpg_actor.pth'
torch.save(agent.actor.state_dict(), file_name)

# curiosity
env = Explore()

policy = DDPGTrainCuriosity(env)
agent, return_list = policy.learn(num_episodes=6000)


file_name = './param_dict/' + 'ddpg_actor_curiosity.pth'
torch.save(agent.actor.state_dict(), file_name)

