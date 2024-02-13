import torch
import torch.nn.functional as F
import numpy as np
import utils
from torch.distributions import Normal, kl_divergence
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:

    def __init__(self, 
                 env, 
                 agent,
                 num_episodes, 
                 minimal_size,
                 update_interval,
                 replaybuffer,
                 theta,
                 batch_size,
                 curiosity):
        self.env = env
        self.agent = agent
        self.theta = theta
        self.curiosity = curiosity
        self.num_episodes = num_episodes
        self.replaybuffer = replaybuffer
        self.minimal_size = minimal_size
        self.batch_size = batch_size
        self.update_interval = update_interval
    
    def train_off_policy(self):
        """Off-Policy training process
            Use like DDPG algorithm
        Return：
         -return_list: List of each episode's return in training process
         -intrinsic_list: List of curiosity influence
        """
        return_list = []
        cnt = 0
        for i in range(10):
            with tqdm(total=int(self.num_episodes/10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(self.num_episodes/10)):
                    episode_return = 0
                    obs = self.env.reset()
                    done = False
                    while not done:
                        action = self.agent.take_action(obs)
                        next_obs, rew, done, info = self.env.step(action)
                        # curiosity reward
                        if self.curiosity:
                            curr_fea = self.agent.FeatureEncoder(torch.FloatTensor(obs).unsqueeze(0)[:, :12]) # [1, obs_dim]
                            curr_act = torch.FloatTensor(action).unsqueeze(0) # [1, act_dim]
                            next_fea = self.agent.FeatureEncoder(torch.FloatTensor(next_obs).unsqueeze(0)[:, :12]) # [1, obs_dim]
                            intrinsic_rew = F.mse_loss(self.agent.predictor(curr_fea, curr_act), next_fea).item() 
                            rew += self.theta * intrinsic_rew
                        
                        self.replaybuffer.add(obs, action, rew, next_obs, done)
                        obs = next_obs
                        episode_return += rew
                        cnt +=1
                        
                        if self.replaybuffer.size() > self.minimal_size and cnt % self.update_interval == 0:
                            b_s, b_a, b_r, b_ns, b_dn = self.replaybuffer.sample(self.batch_size)
                            dict = {'states': b_s, 'actions': b_a, 'rewards': b_r,
                                               'next_states': b_ns, 'dones': b_dn}
                            self.agent.update(dict)
                    return_list.append(episode_return)
                    if (i_episode+1) % 10 == 0:
                        pbar.set_postfix({'episode': '%d' % (self.num_episodes/10 * i + i_episode+1),
                                          'return': '%.5f' % np.mean(return_list[-10:])})
                    pbar.update(1)
        return return_list
    
    def train_on_policy_causal(self):
        return_list = []
        kl_reward = []
        self.agent.load_model(0)
        self.agent.agents[0].set_mode('test')
        for i in range(10):
            with tqdm(total=int(self.num_episodes/10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(self.num_episodes/10)):
                    episode_return = 0
                    dict_0 = {'obs': [], 'actions': [], 'rewards': [], 'next_obs': [], 'dones':[]}
                    dict_1 = {'obs': [], 'actions': [], 'rewards': [], 'next_obs': [], 'dones':[]}
                    obs = self.env.reset()
                    done = False
                    cnt = 0
                    while not done:
                        actions = self.agent.take_actions(obs)
                        next_obs, rew, done, _ = self.env.step(actions)
                        # causal reward for agent 1
                        causal_obs = torch.FloatTensor(obs[1]).unsqueeze(0).to(device) # [1, obs1_dim]
                        causal_act = torch.FloatTensor(actions[1]).unsqueeze(0).to(device)
                        margin_dist = utils.margin_probs(causal_obs, self.agent.agents[1].actor, action_range=(-0.5, 0.5))
                        mu, std = self.agent.agents[1].actor(causal_obs) # compute conditional probs
                        condi_dist = Normal(mu.detach(), std.detach())
                        kl = kl_divergence(condi_dist, margin_dist).mean()
                        rew[1] += kl
                        '''
                        dict_0['obs'].append(obs[0])
                        dict_0['actions'].append(actions[0])
                        dict_0['rewards'].append(rew[0])
                        dict_0['next_obs'].append(next_obs[0])
                        dict_0['dones'].append(done[0])
                        '''
                        dict_1['obs'].append(obs[1])
                        dict_1['actions'].append(actions[1])
                        dict_1['rewards'].append(rew[1])
                        dict_1['next_obs'].append(next_obs[1])
                        dict_1['dones'].append(done[1])
                        # update done and obs
                        episode_return += rew[1].item()
                        kl_reward.append(kl)
                        done = done["__all__"]
                        obs = next_obs
                        cnt += 1
                        
                    # update agent 1
                    return_list.append(episode_return/cnt)
                    self.agent.update(dict_1, agent_i=1)
                    if (i_episode+1) % 10 == 0:
                        pbar.set_postfix({'episode': '%d' % (self.num_episodes/10 * i + i_episode+1),
                                          'return': '%.5f' % np.mean(return_list[-10:])})
                    pbar.update(1)
        return return_list
    
    
