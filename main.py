import json
import argparse
import numpy as np
import signal
import sys

from sb3_contrib import MaskablePPO, RecurrentPPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from mapping_custom_env import CustomRLEnvironment, lrsched, mask, count_communications

### END IMPORTS ###

signal_count = 0
model = None

def signal_handler(sig, frame):
    global signal_count
    global model
    signal_count += 1
    if signal_count == 1:
        if model is None:
            print("Agent has not started learning yet. Exiting...")
            sys.exit(0)
        name = input("Interrupt agent training. Introduce name for file:")
        model.save(name)
    else:
        print("ABORTED")
    sys.exit(0)

def load_config(config_file):
    with open(config_file, 'r') as file:
        data = json.load(file)
        return data
    
def read_config_paths(config_path_file):
    with open(config_path_file, 'r') as file:
        return file.read().splitlines()
    
def init_matrix(P, edges, volume, n_msgs, reward_type="volume"):
    adj_matrix = np.zeros((P, P))
    factor = volume if reward_type == "volume" else n_msgs
    for edge, factor in zip(edges, volume):
        node1, node2 = edge
        cost = factor
        adj_matrix[node1][node2] = cost

    return adj_matrix

def get_render_config(data):
    config = data["Config"]
    reward_type = config["reward_type"]
    verbosity = config["verbosity"]
    verbose_freq = config["verbosity_interval"]
    return reward_type, verbosity, verbose_freq

def init_env(data):
    P = data["Graph"]["P"]
    M = data["Graph"]["M"]
    reward_type, _, render_freq = get_render_config(data)
    edges = data["Graph"]["comms"]["edges"]
    volume = data["Graph"]["comms"]["volume"]
    n_msgs = data["Graph"]["comms"]["n_msgs"]

    adj_matrix = init_matrix(P, edges, volume, n_msgs, reward_type)

    optimal = data["Benchmark"]["optimal_mapping"]
    optimal = np.array(optimal)
    
    node_capacity = data["Graph"]["capacity"]
    np_node_capacity = np.array(node_capacity)    

    env = CustomRLEnvironment(P, M, np_node_capacity, adj_matrix, n_msgs, optimal, render_freq)

    unwrapped = env
    env = ActionMasker(env, action_mask_fn=mask)
    return env, unwrapped

def get_total_steps(data):
    P = data["Graph"]["P"]
    episodes = data["Hyperparameters"]["n_episodes"]
    return episodes * P

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = "main.py"
    )
    parser.add_argument(
        "configs"
    )
    args = parser.parse_args()
    args = vars(args)
    all_config = args["configs"]
    
    configs_paths = read_config_paths(all_config)
    
    for config_path in configs_paths:
        data = load_config(config_path)
        env, unwrapped = init_env(data)
    
        P = data["Graph"]["P"]
        total_steps = get_total_steps(data=data)
        
        if model is None:
            # model = MaskablePPO(
            #     #policy=MaskableActorCriticPolicy,
            #     policy="MlpPolicy",
            #     env=env,
            #     learning_rate=lrsched(lr0=0.0003, lr1=0.00000001, decay_rate=5),
            #     tensorboard_log="./newobs",
            #     verbose=1,
            #     device="cpu",
            #     #vf_coef=0.3,
            #     #normalize_advantage=False,
            #     ent_coef= 0.85,
            #     gamma=0.99,
            #     n_steps=8192,
            #     n_epochs=40,
            #     gae_lambda=0.97,
            #     batch_size=256,
            #     clip_range=lrsched(lr0=3, lr1=0.05, decay_rate=2.5)
            # )
            model = MaskablePPO(
                policy=MaskableActorCriticPolicy,
                env = env,
                learning_rate=lrsched(lr0=0.0003, lr1=0.00000001, decay_rate=5),
                tensorboard_log="./newobs",
                verbose=1
            )
        else:
            # del model
            # model = MaskablePPO.load("whatever.model")
            model.set_env(env=env)
            model.learning_rate = lrsched(lr0=0.0003, lr1=0.00000001, decay_rate=5)
        
        model.learn(total_timesteps=total_steps, log_interval=1, reset_num_timesteps=False, progress_bar=True)
        model.save("whatever.model")
    
    # ### PREDICTION ###

    # terminated = False
    # truncated = False

    # obs, info = env.reset()
    # while not terminated:
    #     obs, info = env.reset()
    #     truncated = False
    #     while not (terminated or truncated):
    #         action, _ = trained.predict(observation=obs, deterministic=True, action_masks=env.action_masks())
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         #print("Predict observation:", obs)

    # ### PREDICTION END ###
    
    # ### MODEL EVAL ###

    # placement = {}
    # for proc, node in enumerate(unwrapped.current_assignment, start=0):
    #     if node not in placement:
    #         placement[node] = []
    #     placement[node].append(proc)

    # unwrapped.render(reward, "human")
    # env.close()

    # for node, processes in placement.items():
    #     print(f'Processes in node {node}: {processes}')

    # print("Final assignm", unwrapped.current_assignment)

    # optimal = data["Benchmark"]["optimal_mapping"]
    # optimal = np.array(optimal)
    # optimal_reward = count_communications(
    #     positions=optimal,
    #     adjacency_matrix=unwrapped.adj_matrix,
    #     not_assigned=unwrapped.NOT_ASSIGNED,
    #     total_comms=unwrapped.total_comms,
    #     P=unwrapped.P
    # )

    # print(f"Optimal placement was {optimal} with reward {optimal_reward}")