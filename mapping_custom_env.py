import datetime
import os
import numpy as np
import gymnasium as gym
import networkx as nx
import matplotlib.pyplot as plt
from gymnasium.spaces import Discrete, MultiDiscrete, Dict, Box

DEBUG = False
NEGATIVE_INF = -float('inf')


class CustomRLEnvironment(gym.Env):

    def __init__(self, render_freq=1000, seed=None):
        super(CustomRLEnvironment, self).__init__()
        self.seed = seed
        
        self.MAX_PROCS = 256
        self.MAX_NODES = 32
        self.MAX_MATRIX_SHAPE = (self.MAX_PROCS, self.MAX_PROCS)
        
        # Save init array for environment reset
        #self.node_capacity_init = node_capacity
        
        # ADJ MATRIX OF ENV
        # self.adj_matrix = np.pad(adj_matrix,
        #                         pad_width=[(0, self.MAX_PROCS - adj_matrix.shape[0]), (0,self.MAX_PROCS-adj_matrix.shape[1])],
        #                         mode='constant',
        #                         constant_values=-1)
        
        self.RENDER_FREQ = render_freq

        obs_len = np.prod(self.MAX_MATRIX_SHAPE) + self.MAX_PROCS
        
        # Initialize all processes unassigned
        self.NOT_ASSIGNED = self.MAX_NODES + 1
        self.action_space = Discrete(self.MAX_NODES)
        self.observation_space = Dict(
                {
                    "processes": Discrete(self.MAX_PROCS),
                    "nodes": Discrete(self.MAX_NODES),
                    "adjacency_matrix": Box(low=0, high=np.inf, dtype=np.float32, shape=(self.MAX_PROCS, self.MAX_PROCS)),
                    
                    "node_capacity": Box(low=0, high=(self.MAX_PROCS//self.MAX_NODES), dtype=np.int32, shape=(self.MAX_NODES,)),
                    "current_assignment": Box(low=0, high=self.MAX_NODES, dtype=np.int32, shape=(self.MAX_PROCS,)),
                    
                    "best": Box(low=0, high=np.inf)
                }
        )
        
        Box(low=1, high=np.inf, shape=(obs_len,), dtype=np.float32)



        # Don't panic, adj matrix is not the amplified matrix after concatenation
        self.current_process = 0
        self.last_reward = 0
        self.best = 0

        self.n_episode = 0
        self.n_render = 0
        self.ingraph_node_pos = None
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.renders_dir = os.path.join("./renders", timestamp)
        if not os.path.exists(self.renders_dir):
            os.makedirs(self.renders_dir)
            

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        
        self.P = np.random.randint(low=0, high=self.MAX_PROCS)
        self.M = np.random.randint(low=4, high=self.MAX_NODES)

        self.adj_matrix = np.random.random_integers(0, high=999999, size=(self.P, self.P))
        np.fill_diagonal(self.adj_matrix, 0)
        self.adj_matrix = np.pad(array=self.adj_matrix,
                                 pad_width=[(0,self.MAX_PROCS-self.P), (0,self.MAX_PROCS-self.P)],
                                 mode='constant', constant_values=0)

        self.total_comms = np.sum(self.adj_matrix)


        # Reset node capacities
        # Node cap array is node cap array from config with remaining of nodes (until max) capacities to 0
        self.node_capacity = np.random.randint(2, high=16 + 1, size=self.M)
        pad_width = (0, self.MAX_NODES - len(self.node_capacity))
        self.node_capacity = np.pad(self.node_capacity, pad_width, mode="constant", constant_values=0) # 0 constant by default

        self.current_assignment = np.full(self.MAX_PROCS, self.NOT_ASSIGNED)
        self.current_process = 0

        return self.current_observation, {}
        
    @property
    def current_observation(self):
        current = {
            "processes": self.P,
            "nodes": self.M,
            "adjacency_matrix": self.adj_matrix,
            "node_capacity": self.node_capacity,
            "current_assignment": self.current_assignment,
            "best": self.best
        }
        return current
        

    def step(self, action):

        process_id = self.current_process
        node_id = action
        
        if self.current_process>0:
            pass
        
        proc_assigned = self.current_assignment[process_id] != self.NOT_ASSIGNED
        node_full = self.node_capacity[node_id] == 0

        truncated = False

        if not proc_assigned and not node_full:
            self.current_assignment[process_id] = node_id
            self.node_capacity[node_id] -= 1
        else:
            print("truncate")
            truncated = True
        
        self.current_process += 1

        reward = count_communications(self.current_assignment, self.adj_matrix, self.NOT_ASSIGNED, self.total_comms, self.P)

        if reward > self.best:
            self.best = reward

        # Check if all processes have been assigned and done with an episode
        full_capacity = np.all(self.node_capacity == 0)
        #all_assigned = np.all(self.current_assignment != (self.M + 1))
        #only until P processes
        all_assigned = np.all(self.current_assignment[:self.P] != (self.MAX_NODES + 1))
        done = full_capacity or all_assigned

        if done:
            print(f"reward: {reward} obs: {self.current_assignment}")
            # if(self.n_episode % self.RENDER_FREQ == 0):
            #     self.render(reward)
            self.n_episode +=1

        info = {"Action": action,
                "Current Assignment": self.current_observation,
                "Node Capacities": self.node_capacity,
                "Reward": reward,
                "Done": done,
                "Full nodes": full_capacity,
                "All procs assigned": all_assigned}


        #return self.current_assignment, reward, done, truncated, info
        return self.current_observation, reward, done, truncated, info

    def valid_action_mask(self):

        node_capacity = self.node_capacity.copy()
        M = self.MAX_NODES

        valid_actions = np.zeros(len(node_capacity), dtype=bool)

        for node in range(M):
            if node_capacity[node] > 0:
                valid_actions[node] = True

        return np.array(valid_actions)

    def render(self, reward, mode="human"):
        if mode != "human":
            raise NotImplementedError(f"Render mode {mode} not implemented")

        G = nx.Graph()

        for proc in range(self.P):
            G.add_node(f"P{proc}")

        # Communication Volume (graph inter-node cost) between processes
        for proc in range(self.P):
            for proc2 in range(proc + 1, self.P):
                # Add only edges for processes that actually communicates
                if self.adj_matrix[proc][proc2] != 0:
                    G.add_edge(f"P{proc}", f"P{proc2}", weight=self.adj_matrix[proc][proc2])

        # FIXME Generar los colores en el constructor para que sean los mismos siempre?
        color_map = []
        for proc in range(self.P):
            if self.current_assignment[proc] != self.NOT_ASSIGNED:
                # Random color for each process in a machine
                color_map.append('C{}'.format(self.current_assignment[proc]))
            else:
                # Unassigned color. We will never hit this condition, just in case...
                color_map.append('grey') 

        
        plt.clf()
        plt.figure(figsize=(20, 20))
        if self.ingraph_node_pos is None:
            # self.ingraph_node_pos = nx.spring_layout(G, k=1.5)
            self.ingraph_node_pos = nx.spring_layout(G, k=4)
        nx.draw(G, self.ingraph_node_pos, with_labels=True, node_color=color_map, node_size=200, edge_color="black")

        # Etiquetas de peso en las aristas
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, self.ingraph_node_pos, edge_labels=edge_labels)

        plt.text(0.95, 0.95, f'Rwd={reward}', horizontalalignment='right', verticalalignment='top', transform=plt.gcf().transFigure)

        #plt.draw()
        file_path = os.path.join(self.renders_dir, f"plt_{self.n_episode}_{self.n_render}")
        plt.savefig(file_path)
        plt.savefig(os.path.join(self.renders_dir, "plt_last"))
        

        self.n_render += 1

    def close(self):
        plt.close('all')

def mask(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()

def count_communications(positions, adjacency_matrix, not_assigned, total_comms, P):
    positions_nulled = positions[:P].copy()
    positions_nulled[positions_nulled == not_assigned] = -1
    positions_nulled += 1
    
    adjacency_matrix = adjacency_matrix[:P, :P]

    total_assigned = np.count_nonzero(positions_nulled)
    unassigned = len(positions_nulled) - total_assigned

    mask_0 = np.logical_and(positions_nulled[:, None], positions_nulled)
    mask_1 = positions_nulled[:, None] != positions_nulled

    mask = mask_0 & mask_1

    communications_matrix = adjacency_matrix * mask
    
    volume_count = np.sum(communications_matrix)


    #reward = reward * (total_assigned/len(positions_nulled))
    

    if total_assigned < len(positions_nulled):
        reward =  0
    else:
        reward = total_comms/(volume_count+1)
        
    #reward = reward + total_assigned
    
    return reward

def lrsched(lr0=10, lr1=0.000001, decay_rate=2.0):
    def reallr(progress):
        norm_progress = 1.0 - min(max(progress, 0.0), 1.0)
        # See https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
        lr = (lr0 - lr1) * np.exp(-decay_rate * norm_progress) + lr1
        return lr

    return reallr

