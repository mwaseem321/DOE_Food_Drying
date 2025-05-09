import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
"""
Notice:
        ActorNetwork part, network output needs to be modified according to different tasks
Netwroks Args:
    CriticNetwork:
        beta: CriticNetwork learning rate
        critic_state_dims: The input state part of the critic network
        fc1_out_dims: net 1 out_dims
        fc2_out_dims: net 2 out_dims
        n_agents:
        n_actions: list Contains all action spaces
        name: eg: agent_1_actor.pth
        chkpt_dir: eg: tmp/maddpg/
    
    ActorNetwork:
        alpha: ActorNetwork learning rate
        actor_state_dims: For the state input of a single agent, values ​​are extracted based on a list in the same way.
        n_actions: list Contains all action spaces
        n_actions_single: number ,The action space of the current agent, size
        fc1_out_dims: net 1 out_dims
        fc2_out_dims: net 2 out_dims
        name: eg: agent_1_actor.pth
        chkpt_dir: eg: tmp/maddpg/ save_module/maddpg/
"""
class CriticNetwork(nn.Module):
    def __init__(self, beta, critic_state_dims, fc1_out_dims, fc2_out_dims, fc3_out_dims,
                    n_agents, n_actions, name, chkpt_dir, init_w=3e-3):
        super(CriticNetwork, self).__init__()

        self.beta = beta
        self.critic_state_dims = critic_state_dims
        self.fc1_out_dims = fc1_out_dims
        self.fc2_out_dims = fc2_out_dims
        self.fc3_out_dims = fc3_out_dims
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.name = name
        self.chkpt_dir = chkpt_dir

        # n_actions is defined as list although action_dims may be the same in the environment
        # CriticNetwork_input_dims is critic_states_dims + critic_actions_dims
        self.critic_actions_dims = sum(self.n_actions)

        self.chkpt_file = os.path.join(self.chkpt_dir, self.name)

        self.fc1 = nn.Linear(self.critic_state_dims + self.critic_actions_dims, self.fc1_out_dims)
        self.fc2 = nn.Linear(self.fc1_out_dims, self.fc2_out_dims)
        self.q = nn.Linear(self.fc2_out_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr = self.beta)

        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, actor_state_dims, fc1_out_dims, fc2_out_dims, fc3_out_dims,
                 n_actions, n_actions_single, name, chkpt_dir, init_w = 3e-3):
        super(ActorNetwork, self).__init__()

        self.alpha = alpha
        self.actor_state_dims = actor_state_dims
        self.fc1_out_dims = fc1_out_dims
        self.fc2_out_dims = fc2_out_dims
        self.fc3_out_dims = fc3_out_dims
        self.n_actions = n_actions
        self.n_actions_single = n_actions_single
        self.name = name
        self.chkpt_dir = chkpt_dir

        self.chkpt_file = os.path.join(chkpt_dir, self.name)

        self.fc1 = nn.Linear(self.actor_state_dims, self.fc1_out_dims)
        self.fc2 = nn.Linear(self.fc1_out_dims, self.fc2_out_dims)
        self.pi = nn.Linear(self.fc2_out_dims, self.n_actions_single)

        self.optimizer = optim.Adam(self.parameters(), lr = self.alpha)

        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, actor_data):
        x = self.fc1(actor_data)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.pi(x)
        pi = torch.sigmoid(x)

        return pi

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))