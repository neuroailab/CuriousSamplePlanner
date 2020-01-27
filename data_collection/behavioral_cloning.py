#!/usr/bin/env python
from __future__ import print_function
import pybullet as p
import time
import random
import math
import os
import shutil
import pickle
import collections
from CuriousSamplePlanner.planning_pybullet.motion.motion_planners.discrete import astar
import sys
from CuriousSamplePlanner.planning_pybullet.pybullet_tools.utils import WorldSaver, enable_gravity, connect, dump_world, set_pose, \
    Pose, Point, Euler, set_default_camera, stable_z, \
    BLOCK_URDF, load_model, wait_for_user, disconnect, DRAKE_IIWA_URDF, user_input, update_state, disable_real_time,inverse_kinematics,end_effector_from_body,approach_from_grasp, get_joints, get_joint_positions


# Planners
from CuriousSamplePlanner.trainers.state_estimation_planner import StateEstimationPlanner
from CuriousSamplePlanner.trainers.random_search_planner import RandomSearchPlanner
from CuriousSamplePlanner.trainers.effect_prediction_planner import EffectPredictionPlanner
from CuriousSamplePlanner.trainers.random_state_embedding_planner import RandomStateEmbeddingPlanner
from CuriousSamplePlanner.scripts.utils import *
from CuriousSamplePlanner.agent.planning_agent import PlanningAgent


def main(exp_id="no_expid", load_id="no_loadid"):  # control | execute | step

    # load = "found_pah.pkl"
    load = None

    
    # Set up the hyperparameters
    experiment_dict = {
        # Hyps
        "task": "ThreeBlocks",
        "learning_rate": 5e-5,  
        "sample_cap": 1e7, 
        "batch_size": 128,
        "return_on_solution":True,
        'actor_lr': 1e-4,
        'critic_lr': 1e-3,
        "node_sampling": "uniform",
        "mode": "RandomStateEmbeddingPlanner",
        "feasible_training": True,
        "nsamples_per_update": 1024,
        "training": True,
        'exploration_end': 100, 
        "exp_id": exp_id,
        "load_id": load_id,
        'noise_scale': 0.3,
        'final_noise_scale': 0.05,
        'update_interval' : 1,
        "enable_asm": False, 
        "growth_factor": 10,
        "detailed_gmp": False, 
        "adaptive_batch": True,
        "num_training_epochs": 30,
        "infeasible_penalty" : 0,
        'tau': 0.001,
        'reward_size': 100,
        'hidden_size': 64,
        'use_splitter': False, # Can't use splitter on ppo or a2c because they are on-policy algorithms
        'split': 0.2,
        'gamma': 0.9,
        'ou_noise': True,
        'param_noise': False,
        'updates_per_step': 1,
        'replay_size': 100000,
        # Stats
        "world_model_losses": [],
        "feasibility":[],
        "rewards": [],
        "num_sampled_nodes": 0,
        "num_graph_nodes": 0,
    }

    experiment_dict['exp_path'] = "./solution_data/" + experiment_dict["exp_id"]
    if (not os.path.isdir("./solution_data")):
        os.mkdir("./solution_data")
        
    if (os.path.isdir(experiment_dict['exp_path'])):
        # shutil.rmtree(experiment_dict['exp_path'])
        pass
    else:
        os.mkdir(experiment_dict['exp_path'])
    
    g_states = []
    g_lens = []
    g_actions = []
    g_rewards = []


    for solution_index in range(0, 4096):
        print(solution_index)
        experiment_dict['exp_path'] = "./solution_data/" + experiment_dict["exp_id"]
        experiment_dict['load_path'] = "./solution_data/" + experiment_dict["load_id"]
        if (not os.path.isdir("./solution_data")):
            os.mkdir("./solution_data")
        #experiment_dict['exp_path'] = "example_images/" + experiment_dict["exp_id"]
        #experiment_dict['load_path'] = 'example_images/' + experiment_dict["load_id"]
        adaptive_batch_lr = {
            "StateEstimationPlanner": 0.003,
            "RandomStateEmbeddingPlanner": 0.00005,
            "EffectPredictionPlanner": 0.001,
            "RandomSearchPlanner": 0 
        }
        experiment_dict["loss_threshold"] = adaptive_batch_lr[experiment_dict["mode"]]
        PC = getattr(sys.modules[__name__], experiment_dict['mode'])
        planner = PC(experiment_dict)

        graph, plan, experiment_dict = planner.plan()

        # Save the graph so we can load it back in later
        if(graph is not None):
            s_states = [np.expand_dims(plan[i].config.numpy(), axis=0) for i in range(len(plan)-1)]
            s_actions = [plan[i].action for i in range(1, len(plan))]        
            s_rewards = [float(1) for _ in range(1, len(plan))]
            s_len = len(plan)-1
            g_states.append(np.expand_dims(np.concatenate(s_states, axis=0), axis=0))
            g_actions.append(np.expand_dims(np.concatenate(s_actions, axis=0), axis=0))
            g_rewards.append(np.array(s_rewards))
            g_lens.append(float(s_len))

        disconnect()

    # Finished collecting expert trajectories, time to save the file as a pt record
    data_states = np.concatenate(g_states, axis=0)
    data_actions = np.concatenate(g_actions, axis=0)
    data_rewards = np.array(g_rewards)
    data_lens = np.array(g_lens)

    print(data_states.shape)
    print(data_actions.shape)
    print(data_rewards.shape)
    print(data_lens.shape)
    data = {
        'states': torch.from_numpy(data_states).float(),
        'actions': torch.from_numpy(data_actions).float(),
        'rewards': torch.from_numpy(data_rewards).float(),
        'lengths': torch.from_numpy(data_lens).float()
    }
    torch.save(data, "./rl_utils/gail_experts/trajs_threeblocks.pt")


if __name__ == '__main__':
    exp_id = str(sys.argv[1])
    if(len(sys.argv)>3):
        load_id = str(sys.argv[3])
    else:
        load_id = ""

    main(exp_id=exp_id, load_id=load_id)
