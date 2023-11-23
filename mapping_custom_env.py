import datetime
import os
import numpy as np
import gymnasium as gym
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from gymnasium.spaces import Discrete, MultiDiscrete

DEBUG = False
NEGATIVE_INF = -float('inf')


class CustomRLEnvironment(gym.Env):

    def __init__(self, P, M, node_capacity, adj_matrix, n_msgs, seed=None):
        super(CustomRLEnvironment, self).__init__()
        self.seed = seed
        self.P = P
        self.M = M
        # Save init array for environment reset
        self.node_capacity = self.node_capacity_init = node_capacity
        self.adj_matrix = adj_matrix
        self.n_msgs = n_msgs

        # Initialize all processes unassigned
        self.NOT_ASSIGNED = self.M + 1
        self.current_assignment = np.full(self.P, self.NOT_ASSIGNED)
        self.action_space = Discrete(self.M)
        self.observation_space = MultiDiscrete([self.M + 2] * self.P)
        self.total_comms = len(self.adj_matrix)
        self.current_process = 0

        self.n_episode = 0
        self.n_render = 0
        self.ingraph_node_pos = None
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.renders_dir = os.path.join("./renders", timestamp)
        if not os.path.exists(self.renders_dir):
            os.makedirs(self.renders_dir)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.node_capacity = self.node_capacity_init.copy()  # Reset node capacities
        self.current_assignment = np.full(self.P, self.NOT_ASSIGNED)
        self.aux = 0
        self.current_process = 0
        return self.current_assignment, {}

    def step(self, action):

        # process_id, node_id = action // self.M, action % self.M
        process_id = self.current_process
        node_id = action
        #print(f"{process_id} to node {node_id}")
        # print(action in self.valid_action_mask())

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

        reward = count_communications(self.current_assignment, self.adj_matrix, self.NOT_ASSIGNED, self.total_comms)

        # Check if all processes have been assigned and done with an episode
        full_capacity = np.all(self.node_capacity == 0)
        all_assigned = np.all(self.current_assignment != (self.M + 1))
        done = full_capacity or all_assigned

        if done:
            print(f"reward: {reward} obs: {self.current_assignment}")
            self.n_episode +=1
            if(self.n_episode % 1000 == 0):
                self.render(reward)

        info = {"Action": action,
                "Current Assignment": self.current_assignment,
                "Node Capacities": self.node_capacity,
                "Reward": reward,
                "Done": done,
                "Full nodes": full_capacity,
                "All procs assigned": all_assigned}


        return self.current_assignment, reward, done, truncated, info

    def valid_action_mask(self):

        current_assignment = self.current_assignment.copy()
        node_capacity = self.node_capacity.copy()
        M = self.M

        valid_actions = np.zeros(len(node_capacity), dtype=bool)

        for node in range(M):
            if node_capacity[node] > 0:
                valid_actions[node] = True


        # for process_index in range(len(current_assignment)):
        #     # Process is not assigned yet
        #     if current_assignment[process_index] == M + 1:
        #         for node_index in range(M):
        #             # Node still can place at least one more process
        #             if node_capacity[node_index] > 0:  # Nodo tiene capacidad
        #                 # Set valid action as true in the array
        #                 action = process_index * M + node_index
        #                 valid_actions[action] = True

        return np.array(valid_actions)

    def render(self, reward, mode="human"):
        if mode is not "human":
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

        
        plt.figure(figsize=(10, 10))
        if self.ingraph_node_pos is None:
            # self.ingraph_node_pos = nx.spring_layout(G, k=1.5)
            self.ingraph_node_pos = nx.planar_layout(G)
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

def count_communications(positions, adjacency_matrix, not_assigned, total_comms):
    positions_nulled = positions.copy()
    positions_nulled[positions_nulled == not_assigned] = -1
    positions_nulled += 1

    total_assigned = np.count_nonzero(positions_nulled)
    unassigned = len(positions_nulled) - total_assigned

    # Create a mask for positions where processes are different
    mask_0 = np.logical_and(positions_nulled[:, None], positions_nulled)
    mask_1 = positions_nulled[:, None] != positions_nulled

    mask = mask_0 & mask_1

    # Use the mask to filter the adjacency_matrix
    communications_matrix = adjacency_matrix * mask

    # Count the number of non-zero elements (corresponding to communications) in the communications_matrix
    communications_count = np.count_nonzero(communications_matrix)

    # print(f"total {total_assigned} comms {communications_count}(+1= {communications_count+1}) = {total_assigned/(communications_count+1)}")
    reward = total_comms / (communications_count + 1)
    #reward = reward + total_assigned

    return reward

def lrsched(lr0=10, lr1=0.000001, decay_rate=2.0):
    def reallr(progress):
        norm_progress = 1.0 - min(max(progress, 0.0), 1.0)
        # See https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
        lr = (lr0 - lr1) * np.exp(-decay_rate * norm_progress) + lr1
        return lr

    return reallr

