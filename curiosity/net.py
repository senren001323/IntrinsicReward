import torch
import torch.nn.functional as F

class FeatureEncoder(torch.nn.Module):
    def __init__(self, 
                 state_dim,
                 feature_dim):
        super(FeatureEncoder, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 12)
        self.fc2 = torch.nn.Linear(12, 12)
        self.fc3 = torch.nn.Linear(12, feature_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PredictionNet(torch.nn.Module):
    """Predict s_t+1
        For curiosity reward
    """
    def __init__(self, 
                 feature_dim, 
                 action_dim, 
                 hidden_dim):
        super(PredictionNet, self).__init__()
        self.fc1 = torch.nn.Linear(feature_dim+action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, feature_dim)
     
    def forward(self, x, a):
        """ Return: s' """
        cat = torch.cat([x, a], dim=-1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class InverseNet(torch.nn.Module):
    """Inverse model
        Predict a_t from inputing s_t and s_t+1
    """
    def __init__(self, 
                 state_dim, 
                 hidden_dim, 
                 action_dim,
                 action_bound):
        super(InverseNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim+state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x, next_x):
        """Return: (batch_size, action_dim)
        """
        cat = torch.cat([x, next_x], dim=-1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) * self.action_bound

class PolicyNet(torch.nn.Module):
    """DDPG Policy Net
    """
    def __init__(self, 
                 state_dim, 
                 hidden_dim, 
                 action_dim, 
                 action_bound,
                 sigma):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound
        self.sigma = sigma

    def forward(self, x):
        """Return: (batch_size, action_dim)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = self.fc3(x)
        noise = self.sigma * torch.randn_like(action)
        action = action + noise
        return torch.tanh(action) * self.action_bound



class QValueNet(torch.nn.Module):
    """DDPG Qvalue Net
    """
    def __init__(self, 
                 state_dim, 
                 hidden_dim, 
                 action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


