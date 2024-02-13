import utils
import gymnasium as gym
import json
import torch
import random
import numpy as np
from .trainer import Trainer
from curiosity.model import DDPGCuriosity
from causal.model import IPPO

class DDPGTrainCuriosity:
    """Off-policy DDPG training setting
    Function:
     -learn(): Creat agent class and learning policy
    """
    def __init__(self, 
                 env,
                 curiosity=True):
        with open('./train_function/curiosity_config.json', 'r') as file:
            config = json.load(file)
        self.pre_lr = config['pre_lr']
        self.inver_lr = config['inver_lr']
        self.encode_lr = config['encode_lr']
        self.actor_lr = config['actor_lr']
        self.critic_lr = config['critic_lr']
        self.num_episodes = config['num_episodes']
        self.hidden_dim = config['hidden_dim']
        self.feature_dim = config['feature_dim']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.sigma = config['sigma']
        self.theta = config['theta']
        self.lambdA = config['lambdA']
        self.beta = config['beta']
        self.buffer_size = config['buffer_size']
        self.minimal_size = config['minimal_size']
        self.batch_size = config['batch_size']
        self.update_interval = config['update_interval']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_seeds()
        self.replaybuffer = utils.ReplayBuffer(self.buffer_size)
        self.env = env
        self.curiosity = curiosity

    def set_seeds(self):
        random.seed(24)
        np.random.seed(24)
        torch.manual_seed(24)

    def learn(self, num_episodes=None, curiosity=True):
        """Learning policy
        """
        if num_episodes is None:
            num_episodes = self.num_episodes
        curiosity = self.curiosity

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_bound = self.env.action_space.high[0]

        agent = DDPGCuriosity(state_dim, 
                              self.hidden_dim, 
                              action_dim,
                              self.feature_dim,
                              action_bound,
                              self.pre_lr, 
                              self.inver_lr,
                              self.encode_lr,
                              self.actor_lr, 
                              self.critic_lr, 
                              curiosity, 
                              self.gamma, 
                              self.tau, 
                              self.sigma, 
                              self.lambdA, 
                              self.beta, 
                              self.device)
        
        trainer = Trainer(self.env, 
                          agent,
                          num_episodes,
                          self.minimal_size,
                          self.update_interval,
                          self.replaybuffer,
                          self.theta, 
                          self.batch_size,
                          curiosity)
        
        return_list = trainer.train_off_policy()
        return  agent, return_list

class PPOTrainCausal:
    """On-policy PPO training setting
    """
    def __init__(self, 
                 env,
                 causal=True):
        with open('./train_function/causal_config.json', 'r') as file:
            config = json.load(file)
        self.actor_lr = config['actor_lr']
        self.critic_lr = config['critic_lr']
        self.num_episodes = config['num_episodes']
        self.hidden_dim = config['hidden_dim']
        self.epochs = config['epochs']
        self.episilon = config['episilon']
        self.lmbda = config['lmbda']
        self.gamma = config['gamma']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_seeds()
        self.env = env
        self.causal = causal

    def set_seeds(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

    def learn(self, num_episodes=None):
        if num_episodes is None:
            num_episodes = self.num_episodes

        state_dims = []
        action_dims = []
        action_bounds = []
        for i in range(self.env.NUM_DRONES):
            state_dims.append(self.env.observation_space[i].shape[0])
            action_dims.append(self.env.action_space[i].shape[0])
            action_bounds.append(self.env.action_space[i].high[0])

        agents = IPPO(state_dims, 
                    self.hidden_dim, 
                    action_dims,
                    action_bounds,
                    self.actor_lr, 
                    self.critic_lr,
                    self.lmbda,
                    self.epochs,
                    self.episilon,
                    self.gamma,
                    self.device,
                    num_agents=2)
        
        trainer = Trainer(self.env, 
                          agents, 
                          num_episodes, # on-poilcy only need 3 args in training process
                          minimal_size=None, 
                          update_interval=None, 
                          replaybuffer=None, 
                          theta=None,
                          batch_size=None,
                          curiosity=False)
        
        return_list = trainer.train_on_policy_causal(self.causal)
        return  agents, return_list






