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


# Planners
from CuriousSamplePlanner.trainers.state_estimation_planner import StateEstimationPlanner
from CuriousSamplePlanner.trainers.random_search_planner import RandomSearchPlanner
from CuriousSamplePlanner.trainers.effect_prediction_planner import EffectPredictionPlanner
from CuriousSamplePlanner.trainers.RRTPlanner import RRTPlanner
from CuriousSamplePlanner.trainers.DRLPlanner import DRLPlanner

from CuriousSamplePlanner.trainers.random_state_embedding_planner import RandomStateEmbeddingPlanner
from CuriousSamplePlanner.scripts.utils import *
from CuriousSamplePlanner.agent.planning_agent import PlanningAgent


def main(exp_id="no_expid", load_id="no_loadid"):  # control | execute | step


    # load = "found_path.pkl"
    load = None

    # Set up the hyperparameters
    experiment_dict = {
        # Hyps
        "task": "TwoBlocks",
        "policy": "ACLearningPolicy",
        "policy_path": "/mnt/fs0/arc11_2/policy_data_new/normalize_returns_4_update=1/",
        "return_on_solution": True,
        "learning_rate": 5e-4,  
        "wm_learning_rate": 5e-5,
        "sample_cap": 1e7, 
        "batch_size": 128,
        'actor_lr': 1e-4,
        'critic_lr': 1e-3,
        "node_sampling": "uniform",
        "mode": "DRLPlanner",
        "feasible_training": True,
        "nsamples_per_update": 1024,
        "training": False,
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
        'gamma': 0.5,
        'ou_noise': True,
        'param_noise': False,
        'updates_per_step': 1,
        'replay_size': 100000,
        # DRL-Specific
        'recurrent_policy': False,
        'algo': 'a2c',
        'value_loss_coef': 0.5,
        'reward_alpha': 1,
        'eps': 1e-5,
        'entropy_coef': 0,
        'alpha': 0.99,
        'max_grad_norm': 0.5,
        'num_steps': 128,
        'num_env_steps': 1e7,
        'use_linear_lr_decay': False,
        'reset_frequency':1e-3,
        'terminate_unreachable': False,
        'use_gae': False,
        'use_proper_time_limits': False,
        'log_interval': 1,
        'clip_param': 0.2, 
        # stats
        "world_model_losses": [],
        "feasibility":[],
        "rewards": [],
        "num_sampled_nodes": 0,
        "num_graph_nodes": 0,
    }


    if(torch.cuda.is_available()):
        prefix = "/mnt/fs0/arc11_2/solution_data/"
    else:
        prefix = "./solution_data/"

    experiment_dict['exp_path'] = prefix + experiment_dict["exp_id"]
    experiment_dict['load_path'] = prefix + experiment_dict["load_id"]
    #experiment_dict['exp_path'] = "example_images/" + experiment_dict["exp_id"]
    #experiment_dict['load_path'] = 'example_images/' + experiment_dict["load_id"]

    adaptive_batch_lr = {
        "StateEstimationPlanner": 0.003,
        "RandomStateEmbeddingPlanner": 0.0005,
        # "RandomStateEmbeddingPlanner": 1,
        "EffectPredictionPlanner": 0.001,
        "RandomSearchPlanner": 100 
    }
    if(experiment_dict["mode"] in adaptive_batch_lr.keys()):
        experiment_dict["loss_threshold"] = adaptive_batch_lr[experiment_dict["mode"]]

    PC = getattr(sys.modules[__name__], experiment_dict['mode'])
    planner = PC(experiment_dict)
    
    if (load == None):
        if (os.path.isdir(experiment_dict['exp_path'])):
            shutil.rmtree(experiment_dict['exp_path'])

        os.mkdir(experiment_dict['exp_path'])
        
        graph, plan, experiment_dict = planner.plan()

        # Save the graph so we can load it back in later
        if(graph is not None):
            graph_filehandler = open(experiment_dict['exp_path'] + "/found_graph.pkl", 'wb')
            filehandler = open(experiment_dict['exp_path'] + "/found_path.pkl", 'wb')

            pickle.dump(graph, graph_filehandler)
            pickle.dump(plan, filehandler)

        stats_filehandler = open(experiment_dict['exp_path'] + "/stats.pkl", 'wb')
        pickle.dump(experiment_dict, stats_filehandler)

    else:
        # Find the plan and execute it
        filehandler = open(experiment_dict['exp_path'] + '/' + load, 'rb')
        plan = pickle.load(filehandler)

        agent = PlanningAgent(planner.environment)
        agent.multistep_plan(plan)


if __name__ == '__main__':
    exp_id = str(sys.argv[1])
    if(len(sys.argv)>3):
        load_id = str(sys.argv[3])
    else:
        load_id = ""

    main(exp_id=exp_id, load_id=load_id)
