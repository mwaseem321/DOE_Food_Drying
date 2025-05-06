import math
import torch
import random
import numpy as np

# Create reused functions
class functions():

    # Process obs data
    @staticmethod
    def obs_dict_to_array(obs_dict):
        # The number of agents set is 2
        agent_0_state = obs_dict["agent_0"]
        agent_1_state = obs_dict["agent_1"]
        agent_2_state = obs_dict["agent_2"]
        agent_3_state = obs_dict["agent_3"]
        # agent_4_state = obs_dict["agent_4"]
        # agent_5_state = obs_dict["agent_5"]

        # actor_state = np.vstack((agent_0_state, agent_1_state, agent_2_state, agent_3_state, agent_4_state,
        #                          agent_5_state), dtype=np.float32)
        # critic_state = np.concatenate((agent_0_state, agent_1_state, agent_2_state, agent_3_state, agent_4_state,
        #                                agent_5_state), axis=0, dtype=np.float32)

        # actor_state = np.vstack((agent_0_state, agent_1_state, agent_2_state, agent_3_state), dtype=np.float32)

        actor_state = [agent_0_state, agent_1_state, agent_2_state, agent_3_state]
        critic_state = np.concatenate((agent_0_state, agent_1_state, agent_2_state, agent_3_state), axis=0, dtype=np.float32)

        return critic_state, actor_state
    

    # Process actons into action data suitable for MPE
    @staticmethod
    def action_array_to_dict(actions):
        action_env = {}
        action_env["agent_0"] = actions[0]
        action_env["agent_1"] = actions[1]
        action_env["agent_2"] = actions[2]
        action_env["agent_3"] = actions[3]
        # action_env["agent_4"] = actions[4]
        # action_env["agent_5"] = actions[5]

        return action_env








