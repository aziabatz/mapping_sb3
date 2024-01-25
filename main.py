import json
import argparse
import numpy as np
import signal
import sys
import time

from sb3_contrib import MaskablePPO, RecurrentPPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from mapping_custom_env import CustomRLEnvironment, lrsched, mask, count_communications
from callbacks.entropy_callback import EntropyCallback
from callbacks.tb_log_callback import TensorboardLogCallback

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv

from results.write_results import JsonManager

from stable_baselines3.common.callbacks import CallbackList

### END IMPORTS ###

signal_count = 0
model = None

NUM_ENVS = 4

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

    
    node_capacity = data["Graph"]["capacity"]
    np_node_capacity = np.array(node_capacity)
    
    def vectorize():
        env = CustomRLEnvironment(P, M, np_node_capacity, adj_matrix, n_msgs, render_freq)
        env = ActionMasker(env, action_mask_fn=mask)
        return Monitor(env)
    
    
    env = DummyVecEnv([lambda: vectorize() for _ in range(4)])
    # env = VecFrameStack(env, n_stack=4)
    
    

    return env, env.unwrapped

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
        "graph_config"
    )
    args = parser.parse_args()
    args = vars(args)
    config_file = args["graph_config"]
    


    data = load_config(config_file)
    env, unwrapped = init_env(data)
    
    
    
    P = data["Graph"]["P"]
    total_steps = get_total_steps(data=data)
    
    ent_start, ent_end = 0.25, 0.0
    entropy_scheduler = EntropyCallback(start=ent_start, end=ent_end, steps=total_steps, verbose=2)
    logger_callback = TensorboardLogCallback(verbose=0)
    
    callbacks = CallbackList([entropy_scheduler, logger_callback])

    model = RecurrentPPO(
        #policy=MaskableActorCriticPolicy,
        policy="MultiInputLstmPolicy",
        env=env,
        #learning_rate=lrsched(lr0=0.0003, lr1=0.00000001, decay_rate=5),
        tensorboard_log="./newobs",
        verbose=1,
        device="cpu",

        
        clip_range= lrsched(lr0=1, lr1=0.01, decay_rate=0.5),
        #ent_coef= ent_start,
        
        gamma=0.998,
        #n_steps=4,
        #n_epochs=40,
        #gae_lambda=0.97,
        #batch_size=256
        
    )
    
    # model = MaskablePPO(
    #     policy="MlpPolicy",
    #     env=env,
    #     tensorboard_log="./newobs",
    #     verbose=1
    # )

    # model = MaskablePPO.load(f"last.{config_file}.model")
    
    # model.set_env(env, force_reset=True)
    # model.learning_rate = lrsched(lr0=0.0003, lr1=0.0000003, decay_rate=0.3)
    # model.ent_coef = 0.01 # coeficiente de entropia (mas alto mas exploracion)
    # #model.gamma = 0.98 # Factor de descuento de recompensas futuras

    start_time = time.time()
    trained = model.learn(total_timesteps=total_steps, log_interval=1, 
                        callback= callbacks, 
                        tb_log_name="PPO_clip_entropy_sched")
    end_time = time.time()
    
    
    trained.save(f"./models/newobs/last.{config_file}.model")

    del trained

    #trained = MaskablePPO.load(f"./models/newobs/last.{config_file}.model")
    
    ### PREDICTION ###

    terminated = False
    truncated = False
    
    # P = data["Graph"]["P"]
    # M = data["Graph"]["M"]
    # reward_type, _, render_freq = get_render_config(data)
    # edges = data["Graph"]["comms"]["edges"]
    # volume = data["Graph"]["comms"]["volume"]
    # n_msgs = data["Graph"]["comms"]["n_msgs"]
    
    # node_capacity = data["Graph"]["capacity"]
    # np_node_capacity = np.array(node_capacity)

    # adj_matrix = init_matrix(P, edges, volume, n_msgs, reward_type)
    
    # env = CustomRLEnvironment(P, M, np_node_capacity, adj_matrix, n_msgs, render_freq)
    # unwrapped = env
    # env = ActionMasker(env, action_mask_fn=mask)
    
    terminated = [False] * env.num_envs

    obs = env.reset()
    while not all(terminated):
        #obs = env.reset()
        action_masks = [env.envs[i].action_masks() for i in range(env.num_envs)]
        action, _ = trained.predict(observation=obs, deterministic=True)#, action_masks=action_masks)
        obs, reward, terminated, info = env.step(action)
        #print("Predict observation:", obs["current_assignment"])
        for i, done in enumerate(terminated):
            if done:
                final_obs = {key: value[i] for key, value in obs.items()}
                print(f"Final observation for environment {i}: {final_obs['current_assignment']}")

    ### PREDICTION END ###
    
    exit()
    
    ### MODEL EVAL ###

    placement = {}
    for proc, node in enumerate(unwrapped.current_assignment, start=0):
        if node not in placement:
            placement[node] = []
        placement[node].append(proc)

    unwrapped.render(reward, "human")
    env.close()

    for node, processes in placement.items():
        print(f'Processes in node {node}: {processes}')

    print("Final assignm", obs)

    optimal = data["Benchmark"]["optimal_mapping"]
    optimal = np.array(optimal)
    optimal_reward = count_communications(
        positions=optimal,
        adjacency_matrix=unwrapped.adj_matrix,
        not_assigned=unwrapped.NOT_ASSIGNED,
        total_comms=unwrapped.total_comms
    )

    print(f"Optimal placement was {optimal} with reward {optimal_reward}")
    
    
    exec_result = {
        'config': f"PPO_{config_file}",
        'episodes': data["Hyperparameters"]["n_episodes"],
        'time': end_time-start_time,
        'best_found': unwrapped.best_found.tolist(),
        'optimal': data["Benchmark"]["optimal_mapping"],
        'best_found_reward': unwrapped.best_reward,
        'optimal_reward': optimal_reward
    }
    
    json_manager = JsonManager("results_collection.json")
    json_manager.update_data(None, exec_result)