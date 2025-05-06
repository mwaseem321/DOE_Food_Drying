import torch
import numpy as np
from .networks import ActorNetwork, CriticNetwork




class Agent:
    def __init__(self, alpha, actor_state_dims, actor_fc1, actor_fc2, actor_fc3,n_actions_single, 
                 beta, critic_state_dims, critic_fc1, critic_fc2, critic_fc3,
                 n_agents, n_actions, agent_idx, chkpt_dir, gamma, tau):
        
        self.gamma = gamma
        self.tau = tau
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.n_actions_single = n_actions_single
        self.agent_idx = agent_idx
        self.chkpt_dir = chkpt_dir

        self.agent_name = 'agent_{}'.format(self.agent_idx)

        self.actor= ActorNetwork(alpha=alpha, actor_state_dims=actor_state_dims,
                                 fc1_out_dims=actor_fc1, fc2_out_dims=actor_fc2, fc3_out_dims=actor_fc3,
                                 n_actions=self.n_actions, n_actions_single=self.n_actions_single,
                                 name=self.agent_name+'_actor.pth', chkpt_dir=self.chkpt_dir)
        self.critic = CriticNetwork(beta=beta, critic_state_dims=critic_state_dims,
                                    fc1_out_dims=critic_fc1, fc2_out_dims=critic_fc2, fc3_out_dims=critic_fc3,
                                    n_agents=self.n_agents, n_actions=self.n_actions,
                                    name=self.agent_name+'_critic.pth', chkpt_dir=self.chkpt_dir)
        self.target_actor = ActorNetwork(alpha=alpha, actor_state_dims=actor_state_dims,
                                 fc1_out_dims=actor_fc1, fc2_out_dims=actor_fc2, fc3_out_dims=actor_fc3,
                                 n_actions=self.n_actions, n_actions_single=self.n_actions_single,
                                 name=self.agent_name+'_target_actor.pth', chkpt_dir=self.chkpt_dir)
        self.target_critic = CriticNetwork(beta=beta, critic_state_dims=critic_state_dims,
                                    fc1_out_dims=critic_fc1, fc2_out_dims=critic_fc2, fc3_out_dims=critic_fc3,
                                    n_agents=self.n_agents, n_actions=self.n_actions,
                                    name=self.agent_name+'_target_critic.pth', chkpt_dir=self.chkpt_dir)

        self.update_network_parameters(tau_=0.01)


        # self.ou_noise = OUNoise(action_dim=self.n_actions_single)

    # This step performs soft update
    def update_network_parameters(self, tau_=None):
        if tau_ is None:
            tau_ = self.tau
        
        # soft update target_actor
        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)

        for name in actor_state_dict:
            actor_state_dict[name] = tau_*actor_state_dict[name].clone() + \
                    (1-tau_)*target_actor_state_dict[name].clone()
            
        self.target_actor.load_state_dict(actor_state_dict)

        #  soft update target_critic
        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau_*critic_state_dict[name].clone() + \
                    (1-tau_)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    # def choose_action(self, obs):
    #     state = torch.tensor([obs], dtype=torch.float).to(self.actor.device)
    #     actions = self.actor.forward(state)
    #     # noise : noise
    #     # noise = torch.rand(self.n_actions).to(self.actor.device)
    #     # actions = actions + noise
    #     # return actions.detach().cpu().to(self.actor.device)
    #     return actions.detach().cpu()
    
    # def choose_action(self, obs, ou_noise):
    #     state = torch.tensor(obs, dtype=torch.float).to(self.actor.device)
    #     actions = self.actor.forward(state)
    #     # print(f'choose actions: {actions}')
    #     # noise : noise
    #     # noise = torch.randn(1).to(self.actor.device) *0.0
    #
    #     #OU noise
    #     noise =  torch.tensor(ou_noise.sample(), dtype=torch.float32).to(self.actor.device)
    #     print('action noise', noise)
    #
    #     # noise = torch.random.normal(0, self.max_action * self.noise, size=self.action_dim)
    #     # print(f'noise: {noise}')
    #     actions = (actions + noise).clip(0,1)
    #     # return actions.detach().cpu().to(self.actor.device)
    #     return actions.detach().cpu().numpy()

    def choose_action(self, obs, epsilon=0.1):
        # epsilon=0.1
        state = torch.tensor(obs, dtype=torch.float).to(self.actor.device)
        actions = self.actor.forward(state)
        # print(f'choose actions: {actions}')
        # noise : noise
        noise = torch.randn(1).to(self.actor.device) *epsilon

        #OU noise
      #  noise =  torch.tensor(ou_noise.sample(), dtype=torch.float32).to(self.actor.device)
      #  print('action noise', noise)

        # noise = torch.random.normal(0, self.max_action * self.noise, size=self.action_dim)
        # print(f'noise: {noise}')
        actions = (actions + noise).clip(0,1)
        # return actions.detach().cpu().to(self.actor.device)
        return actions.detach().cpu().numpy()
    
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()