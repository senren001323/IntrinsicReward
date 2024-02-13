import torch
import os
import torch.nn.functional as F


class PolicyNet(torch.nn.Module):
    def __init__(self, 
                 state_dim, 
                 hidden_dim, 
                 action_dim,
                 action_bound,
                 agent_idx,
                 chkpt_file):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound
        
        self.chkpt_file = os.path.join(chkpt_file + f'agent{agent_idx}_actor.pth')

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) * self.action_bound # Normalized
        std = F.softplus(self.fc_std(x))
        return mu, std

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))
    
class ValueNet(torch.nn.Module):
    def __init__(self, 
                 state_dim, 
                 hidden_dim,
                 agent_idx,
                 chkpt_file):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)
        
        self.chkpt_file = os.path.join(chkpt_file + f'agent{agent_idx}_cirtic.pth')

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))


