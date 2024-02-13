import numpy as np
import torch
import utils
import torch.nn.functional as F
from torch.distributions import Normal
from .net import PolicyNet, ValueNet


class PPO:
    
    def __init__(self, 
                 state_dim, 
                 hidden_dim, 
                 action_dim,
                 action_bound,
                 actor_lr,
                 critic_lr,
                 lmbda, 
                 epochs, 
                 epsilon, 
                 gamma,
                 agent_idx,
                 device,
                 mode = 'train',
                 chkpt_file='./param_dict/'):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound, agent_idx, chkpt_file).to(device)
        self.critic = ValueNet(state_dim, hidden_dim, agent_idx, chkpt_file).to(device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        '''
        self.log_alpha = torch.tensor(np.log(0.0001), dtype=torch.float)
        self.log_alpha.requires_grad = False
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                              lr=0.0001)
        '''
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.epsilon = epsilon
        self.mode = mode
        self.device = device

    def take_action(self, state):
        mu, sigma = self.actor(state)
        action_dist = Normal(mu, sigma)
        action = action_dist.sample()
        if self.mode == 'train':
            return action.cpu().detach().numpy()
        elif self.mode == 'test':
            mu, _ = self.actor(state)
            return mu.cpu().detach().numpy()

    def set_mode(self, mode):
        self.mode = mode


class IPPO:

    def __init__(self, 
                 state_dims, 
                 hidden_dim, 
                 action_dims,
                 action_bound,
                 actor_lr,
                 critic_lr,
                 lmbda, 
                 epochs, 
                 episilon, 
                 gamma,
                 device,
                 num_agents):
        self.agents = []
        for i in range(num_agents):
            self.agents.append(
                PPO(state_dims[i], hidden_dim, action_dims[i], action_bound[i], actor_lr, critic_lr,
                    lmbda, epochs, episilon, gamma, i, device)
            )
        self.num_agents = num_agents
        self.lmbda = lmbda
        self.epochs = epochs
        self.episilon = episilon
        self.gamma = gamma
        self.device = device

    def take_actions(self, states):
        """
        Args:
         -states: Shape of (nums_agents, state_dim)
        Return:
         -array(num_drones, action_dim)
             For gym-pybullet-drone frame need
        """
        states = [
            torch.tensor([states[i]], dtype=torch.float).to(self.device)
        for i in range(self.num_agents)
        ] # List of [(1, state_dim)_1, ..., (1, state_dim)_i]
        actions = [agent.take_action(state)
                   for agent, state in zip(self.agents, states)]
        actions = {i: actions[i] for i in range(self.num_agents)}
        return actions

    def test_actions(self, states):
        """For testing policy
        """
        states = [
            torch.tensor([states[i]], dtype=torch.float).to(self.device)
        for i in range(self.num_agents)
        ]
        actions = [agent.test_action(state)
                   for agent, state in zip(self.agents, states)]
        actions = {i: actions[i] for i in range(self.num_agents)}
        return actions

    def update(self, dict, agent_i):
        states = torch.tensor(dict['obs'], dtype=torch.float).to(self.device)
        actions = torch.tensor(dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device) 
        next_states = torch.tensor(dict['next_obs'], dtype=torch.float).to(self.device)
        dones = torch.tensor(dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        #rewards = (rewards+50) / 50

        agent = self.agents[agent_i]
        
        td_target = rewards + self.gamma * agent.critic(next_states) * (1 - dones)
        td_delta = td_target - agent.critic(states)
        advantage = utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        mu, std = agent.actor(states)
        action_dists = Normal(mu.detach(), std.detach())
        old_log_probs = action_dists.log_prob(actions)

        mu, std = agent.actor(states)
        action_dists = torch.distributions.Normal(mu, std)
        #entropy = action_dists.entropy().sum(dim=1, keepdim=True) # entropy
        log_probs = action_dists.log_prob(actions)
        ratio = (log_probs - old_log_probs) + 1 # First-order Maclaurin series
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.episilon, 1 + self.episilon) * advantage
        actor_loss = torch.mean(-torch.min(surr1, surr2))
        critic_loss = torch.mean(
            F.mse_loss(agent.critic(states), td_target.detach())
        )
        agent.actor_optimizer.zero_grad()
        agent.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
        #torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.actor_optimizer.step()
        agent.critic_optimizer.step()
        '''
        alpha_loss = torch.mean(
            (entropy - (-1)).detach() * agent.log_alpha.exp())
        agent.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        agent.log_alpha_optimizer.step()
        '''
    def save_model(self, agent_i):
        self.agents[agent_i].actor.save_checkpoint()
        self.agents[agent_i].critic.save_checkpoint()

    def load_model(self, agent_i):
        self.agents[agent_i].actor.load_checkpoint()
        self.agents[agent_i].critic.load_checkpoint()



