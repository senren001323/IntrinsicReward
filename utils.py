import torch
import numpy as np
from torch.distributions import Normal
import collections
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

class ReplayBuffer:
    """ReplyBuffer

    Buffer: (state, action, reward, next_state, done)
    """
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Return touples of states, actions, next_states, rewards, dones 
        """
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def clear(self):
        self.buffer.clear()
    
    def size(self):
        return len(self.buffer)

def stack_array(x, nums):
    """Reshape the sampled tensor to (batch_size, ...)
    """
    aranged = [[sub[i] for sub in x]
               for i in range(nums)]
    batch_stack = [
        torch.tensor(np.vstack(a), dtype=torch.float).to(device)
        for a in aranged
    ]
    return batch_stack

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def margin_probs(obs, actor_net, action_range, action_size=4, nums_sample=100):
    """Calculate counter factual actions' margin porbs
        The counter factual actions are sampled from Uniform distribution
        Using Monte Carlo sampling to estimate marginal probability distribution
            p(action_i | obs_i) = Σp(action_i | counter_action, obs_i)

    Return:
     -Normal(mu, std), Gaussian distribution for marginal probs
    """
    counter_obs = obs.clone()
    mu = 0
    std = 0
    for _ in range(nums_sample):
        counter_act = torch.rand(counter_obs.shape[0], action_size)
        counter_act = counter_act * (action_range[1] - action_range[0]) + action_range[0]
        counter_obs[:, -action_size:] = counter_act
        counter_mu, counter_std = actor_net(counter_obs)
        mu += counter_mu.detach()
        std += counter_std.detach()
        
    mu /= nums_sample
    std /= nums_sample
    
    return Normal(mu, std)

def compute_advantage(gamma, lmbda, td_delta):
    """Coumpute Advantage function for PPO algorithm
    """
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

def evaluate(env, agents, num_episodes):
    """Policy evaluation for UAVs
    Return:
     -returns: List of drones' reward
         Calculated form average episodes' returns
    """
    env = env
    returns = np.zeros(env.NUM_DRONES)
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            actions = agents.take_actions(obs)
            obs, rew, done, info = env.step(actions)
            rew = np.array(list(rew.values()))
            returns += rew / num_episodes
            done = done["__all__"]
    return returns.tolist()
        
        
        
        
        


