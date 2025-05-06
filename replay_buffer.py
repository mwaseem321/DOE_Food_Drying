import numpy as np

class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_state_dims, actor_state_dims, 
                 n_actions, n_agents, batch_size):
        """
        Args:
            max_size: Maximum experience replay capacity
            critic_state_dims:  critic_state_dims = sum(actor_state_dims)
            actor_state_dims: Data format list n_agents data eg: [6, 6, 6] means that the state space of the three agents is 6 6 6
            n_actions: Action space dimension data format list n_agents data eg: [3, 3, 3] means the action space of the three agents is 3 3 3
            n_agents: The number of agents
            batch_size: batch_size
        """
        self.mem_size = max_size
        self.critic_state_dims = critic_state_dims
        self.actor_state_dims = actor_state_dims
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.batch_size = batch_size


        # Define counter
        self.mem_cntr = 0

        # Define current target type
        self.current_target = None

        # Define initialization Critic experience replay
        self.critic_state_memory = np.zeros((self.mem_size, self.critic_state_dims))
        self.critic_new_state_memory = np.zeros((self.mem_size, self.critic_state_dims))

        # define reward terminal
        self.reward_memory = np.zeros((self.mem_size, self.n_agents))
        self.terminal_memory = np.zeros((self.mem_size, self.n_agents), dtype=bool)

        # Initialize Actor storage experience replay
        """
        Actor experience replay logic
            The MADDPG algorithm means that each Actor's network accepts the self-agent obs
            So in order to be compatible, initialize the Actor's experience pool to n_agents * self.mem_size * self.actor_state_dims[i]
        """
        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.action_memory = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                            np.zeros((self.mem_size, self.actor_state_dims[i])))
            self.actor_new_state_memory.append(
                            np.zeros((self.mem_size, self.actor_state_dims[i])))
            self.action_memory.append(
                            np.zeros((self.mem_size, self.n_actions[i])))

    def store_transition(self, critic_state, actor_state, action, reward, 
                            critic_state_next, actor_state_next,  terminal):
        # Define index storage data logic
        index = self.mem_cntr % self.mem_size

        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = actor_state[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = actor_state_next[agent_idx]
            self.action_memory[agent_idx][index] = action[agent_idx]

        self.critic_state_memory[index] = critic_state
        self.critic_new_state_memory[index] = critic_state_next
        self.reward_memory[index] = reward
        self.terminal_memory[index] = terminal

        # counting logic variable + 1
        # self.mem_cntr += 1

        self.mem_cntr = (self.mem_cntr+1) % self.mem_size


    def sample_buffer(self):
        # notice： What comes out of the buffer is batchz_size dimension data
        # logical control variable
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        critic_states = self.critic_state_memory[batch]
        rewards = self.reward_memory[batch]
        critic_states_next = self.critic_new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        # Request data by n_agents
        actor_states = []
        actor_states_next = []
        actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_states_next.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.action_memory[agent_idx][batch])

        return (critic_states, actor_states, actions, rewards, 
                critic_states_next, actor_states_next, terminal)

    def sample_buffer_prior(self,recency_bias=1.0, target_bias=2.0):
        # notice： What comes out of the buffer is batchz_size dimension data
        # logical control variable
        max_mem = min(self.mem_cntr, self.mem_size)

        """Samples a batch of transitions, factoring in recency and current target relevance."""
        current_time = self.mem_cntr
        priorities = []
        for i in range(current_time):
            # Recency factor
            recency_factor = (current_time - i) ** recency_bias
            # Target factor - Increase priority if it's from the time of the current target
            target_factor = 1  # Default to 1; adjust as needed
            if self.current_target_start_index is not None and i >= self.current_target_start_index:
                target_factor = (1.0 + (i - self.current_target_start_index) / current_time) ** target_bias
                # This increases as the relative recency to the target start increases

            priority = recency_factor * target_factor
            priorities.append(priority)

        probabilities = np.array(priorities) / np.sum(priorities)
        batch = np.random.choice(max_mem, self.batch_size, p=probabilities, replace=False)




        # batch = np.random.choice(max_mem, self.batch_size, replace=False)

        critic_states = self.critic_state_memory[batch]
        rewards = self.reward_memory[batch]
        critic_states_next = self.critic_new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        # Request data by n_agents
        actor_states = []
        actor_states_next = []
        actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_states_next.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.action_memory[agent_idx][batch])

        return (critic_states, actor_states, actions, rewards,
                critic_states_next, actor_states_next, terminal)

    def update_current_target(self, target_start_index):
        """Updates the index at which the current target's experiences start."""
        self.current_target_start_index = target_start_index


    def ready(self):
        # training start identifier
        if self.mem_cntr >= self.batch_size:
            return True