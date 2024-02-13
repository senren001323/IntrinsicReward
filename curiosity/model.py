import numpy as np
import torch
import torch.nn.functional as F
from curiosity.net import FeatureEncoder, PredictionNet, InverseNet, PolicyNet, QValueNet

class DDPGCuriosity:

    def __init__(self, 
                 state_dim, 
                 hidden_dim, 
                 action_dim,
                 feature_dim,
                 action_bound,
                 pre_lr,
                 inver_lr,
                 encode_lr,
                 actor_lr,
                 critic_lr,
                 curiosity,
                 gamma, 
                 tau, 
                 sigma,
                 lambdA,
                 beta,
                 device):
        """
        Parameters:
         -gamma: discount factor
         -tau: soft update factor
         -sigma: gausian noise factor
         -lambdA: weight of policy loss in joint optimazation
         -beta: weight of predict_loss and inverse_loss in joint optimization
        """
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound, sigma).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound, sigma).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)

        self.FeatureEncoder = FeatureEncoder(state_dim, feature_dim).to(device)
        self.predictor  = PredictionNet(feature_dim, action_dim, hidden_dim).to(device)
        self.inverse = InverseNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.FeatureEncoder_optimizer = torch.optim.Adam(self.FeatureEncoder.parameters(), lr=encode_lr)
        self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=pre_lr)
        self.inverse_optimizer = torch.optim.Adam(self.inverse.parameters(), lr=inver_lr)
        
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.curiosity = curiosity
        self.gamma = gamma
        self.tau = tau
        self.sigma = sigma
        self.lambdA = lambdA
        self.beta = beta
        self.action_dim = action_dim
        self.device = device


    def take_action(self, state):
        state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
        action = self.actor(state)
        action = action.cpu().detach().numpy()
        return action

    def soft_update(self, target_net, net):
        for target_param, param in zip(target_net.parameters(), 
                                       net.parameters()):
            target_param.data.copy_(target_param.data * (1-self.tau) + param.data * self.tau)
    
    def update(self, dict):
        """Update nets
            If curiosity is True, 
            then the rewards plus 'prediction loss' which is curiosity reward
        """
        states = torch.tensor(dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        
        # Update critic net
        next_qvalue = self.target_critic(next_states, self.target_actor(next_states))
        td_target = rewards + self.gamma * next_qvalue * (1-dones)
        critic_loss = torch.mean(F.mse_loss(td_target, self.critic(states, actions)))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Optimize joint loss under curiosity added
        # Or just update policy net without curiosity
        predictor_loss = self.beta * F.mse_loss(self.predictor(self.FeatureEncoder(states[:, :12]), actions), 
                                    self.FeatureEncoder(next_states[:, :12]))
        inverse_loss = (1-self.beta) * F.mse_loss(self.inverse(states[:, :12], next_states[:, :12]), 
                                                  actions)
        
        actor_loss = -torch.mean(self.critic(states, self.actor(states))) # curr actor
        
        if self.curiosity:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step() 
            
            self.inverse_optimizer.zero_grad()
            self.FeatureEncoder_optimizer.zero_grad()
            inverse_loss.backward()
            self.FeatureEncoder_optimizer.step()
            self.inverse_optimizer.step()
            
            self.predictor_optimizer.zero_grad()
            predictor_loss.backward()
            self.predictor_optimizer.step()
        else:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()      
        
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)


