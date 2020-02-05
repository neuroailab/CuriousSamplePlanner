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

def new_exp_dict(exp_id, load_id, model):
    # Set up the hyperparameters
    experiment_dict = {
        # Hyps
        "task": "ThreeBlocks",
        "policy": "FixedPolicy",
        "policy_path": model,
        "return_on_solution": True,
        "learning_rate": 5e-5,  
        "sample_cap": 1e7, 
        "batch_size": 128,
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
        'use_splitter': True, # Can't use splitter on ppo or a2c because they are on-policy algorithms
        'split': 0.5,
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
    adaptive_batch_lr = {
        "StateEstimationPlanner": 0.003,
        # "RandomStateEmbeddingPlanner": 0.00005,
        "RandomStateEmbeddingPlanner": 1,
        "EffectPredictionPlanner": 0.001,
        "RandomSearchPlanner": 0 
    }
    experiment_dict["loss_threshold"] = adaptive_batch_lr[experiment_dict["mode"]]

    if(torch.cuda.is_available()):
        prefix = "/mnt/fs0/arc11_2/solution_data/"
    else:
        prefix = "./solution_data/"

    experiment_dict['exp_path'] = prefix + experiment_dict["exp_id"]
    experiment_dict['load_path'] = prefix + experiment_dict["load_id"]

    return experiment_dict

def main(exp_id="no_expid", load_id="no_loadid"):  # control | execute | step


    # load = "found_path.pkl"
    load = None
    NUM_SAMPLES = 50
    models = ["/mnt/fs0/arc11_2/policy_data_new/single_step_4_update="+str(i)+"/" for i in range(100, 110, 5)]
    experiment_dict = new_exp_dict(exp_id, load_id, models[0])

    if (os.path.isdir(experiment_dict['exp_path'])):
        shutil.rmtree(experiment_dict['exp_path'])

    os.mkdir(experiment_dict['exp_path'])

    means = []
    stds = []
    for model in models:
        print("Collecting from model: "+str(model))
        samples = []
        for i in range(NUM_SAMPLES):
            print("Num: "+str(i))
            experiment_dict = new_exp_dict(exp_id, load_id, model)
            PC = getattr(sys.modules[__name__], experiment_dict['mode'])
            planner = PC(experiment_dict)
            graph, plan, experiment_dict = planner.plan()
            samples.append(experiment_dict['num_sampled_nodes'])
            disconnect()
        print(samples)
        means.append(str(np.mean(np.array(samples))))
        stds.append(str(np.std(np.array(samples))))


        print("Mean:"+str(means[-1])+", Std:"+str(stds[-1]))

    print("Final Results")
    print(means)
    print(stds)






if __name__ == '__main__':
    exp_id = str(sys.argv[1])
    if(len(sys.argv)>3):
        load_id = str(sys.argv[3])
    else:
        load_id = ""

    main(exp_id=exp_id, load_id=load_id)
