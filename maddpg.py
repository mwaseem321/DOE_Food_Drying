import torch
import torch.nn.functional as F
from .agent import Agent
from .replay_buffer import MultiAgentReplayBuffer
import numpy as np

"""
Notice:
    MADDPG version written here
    The structure of the middle layer of Actor and Critic of each agent is set to be the same.
    Please note that you can modify it as required
"""


class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2):
        self.action_dim = action_dim
        self.mu = mu * np.ones(self.action_dim)  # mean of noise
        self.theta = theta  # the rate at which the noise reverts to the mean
        self.sigma = sigma  # the scale of the noise
        self.dt = dt  # time interval
        self.reset()

    def reset(self):
        """Reset the noise process (typically called at the start of each episode)."""
        self.state = np.copy(self.mu)

    def sample(self):
        """Generate a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.action_dim)
        self.state = x + dx
        return self.state

class MADDPG:
    def __init__(self, parse_args):
        # initialization device
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        # Stores the actor critic's gradient for uniform gradient updates
        self.gradient_list = []
        # hyperparameters
        self.alpha = parse_args.alpha
        self.beta = parse_args.beta
        self.gamma = parse_args.gamma
        self.tau = parse_args.tau

        self.actor_states_dims = parse_args.actor_states_dims
        self.critic_state_dims = sum(self.actor_states_dims)

        self.n_agents = parse_args.n_agents
        self.n_actions = parse_args.n_actions

        # Pass the experience reply parameters. Pay attention to the size of the parameters passed. Pay attention to the dimension changes of the matrix data during the analysis process.
        self.buffer_max_size = parse_args.buffer_max_size
        self.buffer_critic_state_dims = self.critic_state_dims
        self.buffer_actor_state_dims = self.actor_states_dims
        self.buffer_n_actions = self.n_actions
        self.buffer_n_agents = self.n_agents
        self.buffer_batch_size = parse_args.buffer_batch_size
        # Define experience replay
        self.buffer = MultiAgentReplayBuffer(
            max_size = self.buffer_max_size,
            critic_state_dims = self.buffer_critic_state_dims,
            actor_state_dims = self.buffer_actor_state_dims,
            n_actions = self.buffer_n_actions,
            n_agents = self.buffer_n_agents,
            batch_size = self.buffer_batch_size
        )
        # initialization agents
        self.agents = []
        for idx in range(self.n_agents):
            self.agents.append(
                Agent(alpha = self.alpha,
                      actor_state_dims = self.actor_states_dims[idx],
                      actor_fc1 = parse_args.actor_fc1,
                      actor_fc2 = parse_args.actor_fc2,
                      actor_fc3 = parse_args.actor_fc3,
                      n_agents = self.n_agents,
                      n_actions = self.n_actions,
                      n_actions_single = self.n_actions[idx],
                      agent_idx = idx,
                      chkpt_dir = parse_args.chkpt_dir,
                      gamma = self.gamma,
                      tau = self.tau,
                      beta = self.beta,
                      critic_state_dims = self.critic_state_dims,
                      critic_fc1 = parse_args.critic_fc1,
                      critic_fc2 = parse_args.critic_fc2,
                      critic_fc3 = parse_args.critic_fc3
                      )
            )

            # initialization ou noise
            # self.ou_noises = [OUNoise(action_dim=self.n_actions[idx]) for idx in range(self.n_agents)]
            self.epsilon=0.5

    def save_checkpoint(self):
        print('... saving checkpoint models ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint models ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, actor_state_all):
        # The accepted data is the state two-dimensional matrix of all Agents
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            # action = agent.choose_action(actor_state_all[agent_idx], self.ou_noises[agent_idx])
            action = agent.choose_action(actor_state_all[agent_idx],self.epsilon)

            actions.append(action)
        return actions

    def reset_noise(self):
        for noise_idx in self.ou_noises:
            noise_idx.reset()

        print('OU noise reset')

    def learn(self, writer, step):
        if not self.buffer.ready():
            return

        print(f'step: start learning')
        critic_states, actor_states, actions, rewards, \
        critic_states_next, actor_states_next, terminal = self.buffer.sample_buffer()

        critic_states = torch.tensor(critic_states, dtype=torch.float).to(self.device)

        actions_list = []
        for idx in range(self.n_agents):
            actions_list.append(torch.tensor(actions[idx], dtype=torch.float).to(self.device))

        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        critic_states_next = torch.tensor(critic_states_next, dtype=torch.float).to(self.device)
        terminal = torch.tensor(terminal).to(self.device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = torch.tensor(actor_states_next[agent_idx], dtype=torch.float).to(self.device)
            new_pi = agent.target_actor.forward(new_states)
            all_agents_new_actions.append(new_pi)

            mu_states = torch.tensor(actor_states[agent_idx], dtype=torch.float).to(self.device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)

            old_agents_actions.append(actions_list[agent_idx])

        new_actions = torch.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = torch.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = torch.cat([acts for acts in old_agents_actions],dim=1)
        
        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(critic_states_next, new_actions).flatten()
            critic_value_[terminal[:, agent_idx]] = 0.0
            critic_value = agent.critic.forward(critic_states, old_actions).flatten()

            target = rewards[:, agent_idx] + agent.gamma * critic_value_
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)


            if agent.agent_name == "agent_0":
                writer.add_scalar('loss/agent_0_critic_loss', critic_loss.item(), step)
            if agent.agent_name == "agent_1":
                writer.add_scalar('loss/agent_1_critic_loss', critic_loss.item(), step)
            if agent.agent_name == "agent_2":
                writer.add_scalar('loss/agent_2_critic_loss', critic_loss.item(), step)
            # if agent.agent_name == "agent_3":
            #     writer.add_scalar('loss/agent_3_critic_loss', critic_loss.item(), step)
            # if agent.agent_name == "agent_4":
            #     writer.add_scalar('loss/agent_4_critic_loss', critic_loss.item(), step)
            # if agent.agent_name == "agent_5":
            #     writer.add_scalar('loss/agent_5_critic_loss', critic_loss.item(), step)


            # Save the gradient of the model
            gradient_dict = {}
            for name, param in agent.critic.named_parameters():
                if param.grad is not None:
                    gradient_dict[name] = param.grad.clone()
            # Save gradients to list
            self.gradient_list.append(gradient_dict)

            actor_loss = agent.critic.forward(critic_states, mu).flatten()
            actor_loss = -torch.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)

            if agent.agent_name == "agent_0":
                writer.add_scalar('loss/agent_0_actor_loss', actor_loss.item(), step)
            if agent.agent_name == "agent_1":
                writer.add_scalar('loss/agent_1_actor_loss', actor_loss.item(), step)
            if agent.agent_name == "agent_2":
                writer.add_scalar('loss/agent_2_actor_loss', actor_loss.item(), step)
            # if agent.agent_name == "agent_3":
            #     writer.add_scalar('loss/agent_3_actor_loss', actor_loss.item(), step)
            # if agent.agent_name == "agent_4":
            #     writer.add_scalar('loss/agent_4_actor_loss', actor_loss.item(), step)
            # if agent.agent_name == "agent_5":
            #     writer.add_scalar('loss/agent_5_actor_loss', actor_loss.item(), step)



            # Save the gradient of the model
            gradient_dict = {}
            for name, param in agent.actor.named_parameters():
                if param.grad is not None:
                    gradient_dict[name] = param.grad.clone()
            # Save gradients to list
            self.gradient_list.append(gradient_dict)

        for agent_idx, agent in enumerate(self.agents):
            # Set gradients for model parameters
            for name, param in agent.critic.named_parameters():
                if name in self.gradient_list[agent_idx * 2]:
                    param.grad = self.gradient_list[agent_idx * 2][name]

            # Set gradients for model parameters
            for name, param in agent.actor.named_parameters():
                if name in self.gradient_list[agent_idx * 2 + 1]:
                    param.grad = self.gradient_list[agent_idx * 2 + 1][name]

            # Update model parameters
            agent.critic.optimizer.step()
            agent.actor.optimizer.step()
        """
        Set self.gradient_list to zero
        Otherwise, the loaded gradient is always the initial gradient value.
        self.gradient_list will always accumulate
        """
        # print(len(self.gradient_list))
        self.gradient_list = []

        # Finally, perform a soft update on target_networks
        for idx, agent in enumerate(self.agents):
            agent.update_network_parameters()

    def learn_prior(self, writer, step):
        if not self.buffer.ready():
            return

        print(f'step: start learning')
        critic_states, actor_states, actions, rewards, \
            critic_states_next, actor_states_next, terminal = self.buffer.sample_buffer_prior()

        critic_states = torch.tensor(critic_states, dtype=torch.float).to(self.device)

        actions_list = []
        for idx in range(self.n_agents):
            actions_list.append(torch.tensor(actions[idx], dtype=torch.float).to(self.device))

        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        critic_states_next = torch.tensor(critic_states_next, dtype=torch.float).to(self.device)
        terminal = torch.tensor(terminal).to(self.device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = torch.tensor(actor_states_next[agent_idx], dtype=torch.float).to(self.device)
            new_pi = agent.target_actor.forward(new_states)
            all_agents_new_actions.append(new_pi)

            mu_states = torch.tensor(actor_states[agent_idx], dtype=torch.float).to(self.device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)

            old_agents_actions.append(actions_list[agent_idx])

        new_actions = torch.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = torch.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = torch.cat([acts for acts in old_agents_actions], dim=1)

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(critic_states_next, new_actions).flatten()
            critic_value_[terminal[:, agent_idx]] = 0.0
            critic_value = agent.critic.forward(critic_states, old_actions).flatten()

            target = rewards[:, agent_idx] + agent.gamma * critic_value_
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)

            if agent.agent_name == "agent_0":
                writer.add_scalar('loss/agent_0_critic_loss', critic_loss.item(), step)
            if agent.agent_name == "agent_1":
                writer.add_scalar('loss/agent_1_critic_loss', critic_loss.item(), step)
            if agent.agent_name == "agent_2":
                writer.add_scalar('loss/agent_2_critic_loss', critic_loss.item(), step)
            # if agent.agent_name == "agent_3":
            #     writer.add_scalar('loss/agent_3_critic_loss', critic_loss.item(), step)
            # if agent.agent_name == "agent_4":
            #     writer.add_scalar('loss/agent_4_critic_loss', critic_loss.item(), step)
            # if agent.agent_name == "agent_5":
            #     writer.add_scalar('loss/agent_5_critic_loss', critic_loss.item(), step)

            # Save the gradient of the model
            gradient_dict = {}
            for name, param in agent.critic.named_parameters():
                if param.grad is not None:
                    gradient_dict[name] = param.grad.clone()
            # Save gradients to list
            self.gradient_list.append(gradient_dict)

            actor_loss = agent.critic.forward(critic_states, mu).flatten()
            actor_loss = -torch.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)

            if agent.agent_name == "agent_0":
                writer.add_scalar('loss/agent_0_actor_loss', actor_loss.item(), step)
            if agent.agent_name == "agent_1":
                writer.add_scalar('loss/agent_1_actor_loss', actor_loss.item(), step)
            if agent.agent_name == "agent_2":
                writer.add_scalar('loss/agent_2_actor_loss', actor_loss.item(), step)
            # if agent.agent_name == "agent_3":
            #     writer.add_scalar('loss/agent_3_actor_loss', actor_loss.item(), step)
            # if agent.agent_name == "agent_4":
            #     writer.add_scalar('loss/agent_4_actor_loss', actor_loss.item(), step)
            # if agent.agent_name == "agent_5":
            #     writer.add_scalar('loss/agent_5_actor_loss', actor_loss.item(), step)

            # Save the gradient of the model
            gradient_dict = {}
            for name, param in agent.actor.named_parameters():
                if param.grad is not None:
                    gradient_dict[name] = param.grad.clone()
            # Save gradients to list
            self.gradient_list.append(gradient_dict)

        for agent_idx, agent in enumerate(self.agents):
            # Set gradients for model parameters
            for name, param in agent.critic.named_parameters():
                if name in self.gradient_list[agent_idx * 2]:
                    param.grad = self.gradient_list[agent_idx * 2][name]

            # Set gradients for model parameters
            for name, param in agent.actor.named_parameters():
                if name in self.gradient_list[agent_idx * 2 + 1]:
                    param.grad = self.gradient_list[agent_idx * 2 + 1][name]

            # Update model parameters
            agent.critic.optimizer.step()
            agent.actor.optimizer.step()
        """
        Set self.gradient_list to zero
        Otherwise, the loaded gradient is always the initial gradient value.
        self.gradient_list will always accumulate
        """
        # print(len(self.gradient_list))
        self.gradient_list = []

        # Finally, perform a soft update on target_networks
        for idx, agent in enumerate(self.agents):
            agent.update_network_parameters()