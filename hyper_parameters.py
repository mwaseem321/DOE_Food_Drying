import argparse
"""
Notice:
    Supplement parameter data by yourself
    DOE food drying project
"""
class hyper_parameters():

    def parse_args_maddpg(self):
        parser = argparse.ArgumentParser("MADDGP Framworks Hyper Parasmeters")
        parser.add_argument("--alpha", type=float, default=0.001, help="ActorNetwork learning rate")
        parser.add_argument("--beta", type=float, default=0.001, help="CriticNetwork learning rate")
        parser.add_argument("--actor_states_dims", type=list, default=[4,8,4,4], help="The dimensions of the input ActorNetwork of all agents are eg:[3, 3]. Among them, the dimensions of the state information of 2 agents' acquisition actions are 3 3 respectively.")
        parser.add_argument("--actor_fc1", type=int, default=64, help="ActorNetwork linear 1 output dims")
        parser.add_argument("--actor_fc2", type=int, default=32, help="ActorNetwork linear 2 output dims")
        parser.add_argument("--actor_fc3", type=int, default=32, help="ActorNetwork linear 3 output dims")
        parser.add_argument("--critic_fc1", type=int, default=64, help="CriticNetwork linear 1 output dims")
        parser.add_argument("--critic_fc2", type=int, default=32, help="CriticNetwork linear 2 output dims")
        parser.add_argument("--critic_fc3", type=int, default=32, help="CriticNetwork linear 3 output dims")
        parser.add_argument("--n_actions", type=int, default=[1,1,1,1], help="The action space dimensions of all agents are eg:[2,3]. There are two agents whose action spaces are 2 3")
        parser.add_argument("--n_agents",type=int, default=4, help="number of agents")
        parser.add_argument("--chkpt_dir", type=str, default='model/maddpg/20250321multi/', help="model save/load chkpt_dir eg':model/maddpg/'")
        parser.add_argument("--gamma", type=float, default=0.95, help="attenuation factor gamma gamma Need to consider carefully because different gamma needs different effects")
        parser.add_argument("--tau", type=float, default=0.01, help="soft update parameters")
        parser.add_argument("--buffer_max_size", type=int, default=10000, help="Maximum data capacity of experience replay")
        parser.add_argument("--buffer_batch_size", type=int , default=1024, help="maddpg learn batch_size")

        return parser.parse_args()
    
