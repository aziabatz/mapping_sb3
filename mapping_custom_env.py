import datetime
import os
import numpy as np
import gymnasium as gym
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from gymnasium.spaces import Discrete, MultiDiscrete, Dict, Box
from embed import EmbedGraph

DEBUG = False
NEGATIVE_INF = -float('inf')


class CustomRLEnvironment(gym.Env):

    def __init__(self, P, M, node_capacity, adj_matrix: np.ndarray, n_msgs, embedded_adj_matrix: np.ndarray = None, render_freq=1000, seed=None):
        super(CustomRLEnvironment, self).__init__()
        self.seed = seed
        self.P = P
        self.M = M
        # Save init array for environment reset
        self.node_capacity = self.node_capacity_init = node_capacity
        self.adj_matrix = adj_matrix
        self.n_msgs = n_msgs
        self.RENDER_FREQ = render_freq
        
        if embedded_adj_matrix is not None:
            self.embedded_adj_matrix = embedded_adj_matrix
        else:
            self.embedded_adj_matrix = self.adj_matrix

        obs_len = np.prod(adj_matrix.shape) + self.P
        
        # Initialize all processes unassigned
        self.NOT_ASSIGNED = self.M + 1
        self.current_assignment = np.full(self.P, self.NOT_ASSIGNED)
        self.best_volume = np.full(self.P, np.inf)
        self.action_space = Discrete(self.M)
        #self.observation_space = Box(low=0, high=np.inf, shape=(obs_len,), dtype=np.float32)
        
        self.observation_space = Dict({
            'communication_matrix': Box(low=0, high=np.inf, shape=(self.embedded_adj_matrix.shape), dtype=np.float32),
            # 'current_assignment': MultiDiscrete([self.M+2] * P),
            # 'node_capacities': MultiDiscrete(self.node_capacity_init + 2),
            'current_assignment': Box(low=0, high=self.M+2, shape=(self.current_assignment.shape), dtype=np.float32),
            'node_capacities': Box(low=0, high=self.node_capacity_init+2, shape=(self.node_capacity_init.shape), dtype=np.float32),
            # 'total_processes': Discrete(1, start=self.P),
            # 'total_nodes': Discrete(1, start=self.M),
            #'current_process': Box(low=0, high=self.P+90)
        })
        
        self.total_comms = np.sum(adj_matrix) #len(self.adj_matrix)
        self.current_process = 0
        self.best_reward= 0
        self.best_found = None

        self.n_episode = 0
        self.n_render = 0
        self.ingraph_node_pos = None
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.renders_dir = os.path.join("./renders", timestamp)
        # if not os.path.exists(self.renders_dir):
        #     os.makedirs(self.renders_dir)
            
    @property
    def current_observation(self):
        current = {
            'communication_matrix': self.embedded_adj_matrix,
            
            'current_assignment': self.current_assignment,
            'node_capacities': self.node_capacity,
            
            # 'total_processes': self.P,
            # 'total_nodes': self.M,
            
            #'current_process': self.current_process
        }
        # current = np.concatenate(
        #     [self.current_assignment, self.adj_matrix.flatten()]
        # )
        return current
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        
        self.current_assignment = np.full(self.P, self.NOT_ASSIGNED)
        self.node_capacity = self.node_capacity_init.copy()  # Reset node capacities
        self.current_process = 0
        
        return self.current_observation, {}
        

    def step(self, action):

        process_id = self.current_process
        node_id = action

        proc_assigned = self.current_assignment[process_id] != self.NOT_ASSIGNED
        node_full = self.node_capacity[node_id] == 0

        truncated = False

        if not proc_assigned and not node_full:
            self.current_assignment[process_id] = node_id
            self.node_capacity[node_id] -= 1
        else:
            print("truncate")
            truncated = True
            return self.current_observation, -10, False, truncated, {}
        
        self.current_process += 1

        reward = count_communications(self.current_assignment, self.adj_matrix, self.NOT_ASSIGNED, self.total_comms, self.best_volume)
        
        if(self.best_reward < reward):
            self.best_found = self.current_assignment.copy()
            self.best_reward = reward

        # Check if all processes have been assigned and done with an episode
        full_capacity = np.all(self.node_capacity == 0)
        all_assigned = np.all(self.current_assignment != (self.M + 1))
        done = full_capacity or all_assigned
        #truncated = True if reward==0 else False

        if done:
            print(f"reward: {reward} obs: {self.current_assignment}")
            if(self.n_episode % self.RENDER_FREQ == 0):
                self.render(reward)
            self.n_episode +=1

        info = {"Action": action,
                "Current Assignment": self.current_observation,
                "Node Capacities": self.node_capacity,
                "Reward": reward,
                "Done": done,
                "Full nodes": full_capacity,
                "All procs assigned": all_assigned}


        return self.current_observation, reward, done, truncated, info

    def valid_action_mask(self):

        node_capacity = self.node_capacity.copy()
        M = self.M

        valid_actions = np.zeros(len(node_capacity), dtype=bool)

        for node in range(M):
            if node_capacity[node] > 0:
                valid_actions[node] = True

        return np.array(valid_actions)

    def render(self, reward, mode="human"):
        if mode != "human":
            raise NotImplementedError(f"Render mode {mode} not implemented")
        
        return

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

        
    
        fig = plt.figure(figsize=(20, 20))
        if self.ingraph_node_pos is None:
            self.ingraph_node_pos = nx.spring_layout(G, k=1.5)
            #self.ingraph_node_pos = nx.planar_layout(G, scale=2)
        nx.draw(G, self.ingraph_node_pos, with_labels=True, node_color=color_map, node_size=200, edge_color="black")

        # Etiquetas de peso en las aristas
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, self.ingraph_node_pos, edge_labels=edge_labels)

        fig.text(0.95, 0.95, f'Rwd={reward}', horizontalalignment='right', verticalalignment='top', transform=plt.gcf().transFigure)

        #plt.draw()
        file_path = os.path.join(self.renders_dir, f"plt_{self.n_episode}_{self.n_render}")
        fig.savefig(file_path)
        fig.savefig(os.path.join(self.renders_dir, "plt_last"))
        
        fig.clf()
        self.close()

        self.n_render += 1

    def close(self):
        plt.close('all')

def mask(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()

def count_communications(positions, adjacency_matrix, not_assigned, total_comms, best_volume):
    positions_nulled = positions.copy()
    positions_nulled[positions_nulled == not_assigned] = -1
    positions_nulled += 1

    total_assigned = np.count_nonzero(positions_nulled)
    unassigned = len(positions_nulled) - total_assigned

    mask_0 = np.logical_and(positions_nulled[:, None], positions_nulled)
    mask_1 = positions_nulled[:, None] != positions_nulled

    mask = mask_0 & mask_1

    communications_matrix = adjacency_matrix * mask
    
    volume_count = np.sum(communications_matrix)
    
    current = total_assigned -1
    #print(f"current mapping at {current} volume is {volume_count}, best is {best_volume[current]}")
    if best_volume[current] > volume_count:
        if best_volume[current] == np.inf:
            reward = 0
        else:
            reward = 3
        best_volume[current] = volume_count
    elif best_volume[current] == volume_count:
        reward = 1
    else:
        reward = 0
        
    return reward #* total_assigned
    

    if total_assigned < len(positions_nulled):
        reward =  total_comms/(volume_count+1)  #0
    else:
        reward = total_comms/(volume_count+1) 
        
    #reward = reward + total_assigned
    
    return reward ** total_assigned

def lrsched(lr0=10, lr1=0.000001, decay_rate=2.0):
    def reallr(progress):
        norm_progress = 1.0 - min(max(progress, 0.0), 1.0)
        # See https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
        lr = (lr0 - lr1) * np.exp(-decay_rate * norm_progress) + lr1
        return lr

    return reallr

