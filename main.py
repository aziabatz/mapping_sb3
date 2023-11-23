import json
import time
import numpy as np
import gymnasium as gym


from stable_baselines3 import DQN, PPO, A2C
from sb3_contrib import RecurrentPPO, MaskablePPO
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


from mapping_custom_env import CustomRLEnvironment, lrsched, mask

# Load the JSON data from the file
with open('./binomial_16_8.json', 'r') as file:
    data = json.load(file)

# Extract the relevant information
P = data["Graph"]["P"]
M = data["Graph"]["M"]
node_names = data["Graph"]["node_names"]
edges = data["Graph"]["comms"]["edges"]
volume = data["Graph"]["comms"]["volume"]
n_msgs = data["Graph"]["comms"]["n_msgs"]

# Extract node capacities from the JSON data
node_capacity = data["Graph"]["capacity"]

np_node_capacity = np.array(node_capacity)

# Extract the number of nodes (M)
M = data["Graph"]["M"]

# Initialize the adjacency matrix with zeros
adj_matrix = np.zeros((P, P))

seed = 42

# Populate the adjacency matrix with edge weights (volume)
edges = data["Graph"]["comms"]["edges"]
volume = data["Graph"]["comms"]["volume"]


for edge, msg_volume in zip(edges, volume):
    node1, node2 = edge
    adj_matrix[node1][node2] = msg_volume
    adj_matrix[node2][node1] = msg_volume  # Since it's an undirected graph


# Create the Gym environment with the adjacency matrix and node capacities
env = CustomRLEnvironment(P, M, np_node_capacity, adj_matrix, n_msgs)
unwrapped = env
env = ActionMasker(env, action_mask_fn=mask)

# params = {"P": P, "M": M}

# model = MaskablePPO(
#     policy=MaskableActorCriticPolicy,
#     env=env,
#     #learning_rate=lrsched(lr0=0.0003, lr1=0.0000003, decay_rate=0.2),
#     tensorboard_log="./mask_ppo_new/",
#     verbose=1,
#     device="cpu"
# )

# trained = model.learn(total_timesteps=1500000, log_interval=1)

# trained.save("last.model.maskablepolicy")

trained = MaskablePPO.load("last.model.maskablepolicy")

terminated = False
truncated = False

obs, info = env.reset()
while not terminated:
    obs, info = env.reset()
    truncated = False
    while not (terminated or truncated):
        action, _ = trained.predict(observation=obs, deterministic=True, action_masks=env.action_masks())
        obs, reward, terminated, truncated, info = env.step(action)
        print("Predict observation:", obs)
        unwrapped.render(reward, "human")

placement = {}
for proc, node in enumerate(obs, start=0):
    if node not in placement:
        placement[node] = []
    placement[node].append(proc)

env.close()

for node, processes in placement.items():
    print(f'Processes in node {node}: {processes}')

print("Final assignm", obs)