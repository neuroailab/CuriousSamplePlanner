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
from CuriousSamplePlanner.trainers.CSPPlanner import CSPPlanner

from CuriousSamplePlanner.trainers.random_state_embedding_planner import RandomStateEmbeddingPlanner
from CuriousSamplePlanner.scripts.utils import *
from CuriousSamplePlanner.agent.planning_agent import PlanningAgent


def main(exp_id="no_expid"):  # control | execute | step


	# load = "found_path.pkl"
	load = None


	# Set up the hyperparameters
	experiment_dict = {
		# Hyps
		"task": "FiveBlocks",
		"mode": "RandomStateEmbeddingPlanner",
		"policy_path": "/mnt/fs0/arc11_2/policy_data_new/normalize_returns_4_update=1/",
		"return_on_solution": True,
		"learning_rate": 1e-5,
		"wm_learning_rate": 3e-4,
		"sample_cap": 1e7, 
		"batch_size": 128,
		'actor_lr': 1e-5,
		'critic_lr': 1e-3,
		"node_sampling": "softmax",
		"policy": "RandomPolicy",
		"feasible_training": True,
		"training": False,
		'exploration_end': 100, 
		"exp_id": exp_id,
		'noise_scale': 0.9,
		'final_noise_scale': 0.8,
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
		'debug_timing': True,
		'wm_batch_timings': [],
		'policy_update_timings': [],
		'action_selection_timings': [],

		# DRL-Specific
		'recurrent_policy': False,
		'algo': 'a2c',
		'value_loss_coef': 0.5,
		'reward_alpha': 1,
		'eps': 5e-5,
		'entropy_coef': 0.01,
		'alpha': 0.99,
		'max_grad_norm': 0.5,
		'num_steps': 5,
		'num_env_steps': 2560000,
		'use_linear_lr_decay': True,
		'reset_frequency': 1e-4,
		'terminate_unreachable': False,
		'use_gae': False,
		'use_proper_time_limits': False,
		'log_interval': 1,
		'save_interval': 10,
		'clip_param': 0.2, 
		'ppo_epoch': 5,
		'num_mini_batch': 8,

		# HER-Specific
		'n_batches': 1, # Number of times to update the network
		'noise_eps': 0.2,
		'random_eps': 0.3,
		'polyak': 0.95,
		'clip_obs': 200,
		'action_l2': 1,
		'clip_range': 5,
		'replay_k': 4,
		'replay_strategy': "future",
		'buffer_size': int(1e6),
 
		# stats
		"world_model_losses": [],
		"feasibility":[],
		"rewards": [],
		"num_sampled_nodes": 0,
		"num_graph_nodes": 0,

		# Global Env variables		
		"max_reach_horiz": 0.65,
		"min_reach_horiz": 0.45,
		"reachable_max_height": 0.8,

		# Environment render
		"render": False
	}


	if(torch.cuda.is_available()):
		prefix = "/mnt/fs0/arc11_2/solution_data/"
	else:
		prefix = "./solution_data/"

	experiment_dict['exp_path'] = prefix + experiment_dict["exp_id"]
	#experiment_dict['exp_path'] = "example_images/" + experiment_dict["exp_id"]

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
	
	if (os.path.isdir(experiment_dict['exp_path'])):
		shutil.rmtree(experiment_dict['exp_path'])

	os.mkdir(experiment_dict['exp_path'])
	graph, plan, experiment_dict = planner.plan()
	print("Planning Complete")
	if(len(experiment_dict['wm_batch_timings'])>0):
		print("wm_batch_timings: "+str(np.mean(np.array(experiment_dict['wm_batch_timings'])))+", "+ str(np.std(np.array(experiment_dict['wm_batch_timings']))))
	if(len(experiment_dict['policy_update_timings'])>0):
		print("policy_update_timings: "+str(np.mean(np.array(experiment_dict['policy_update_timings'])))+", "+str(np.std(np.array(experiment_dict['policy_update_timings']))))
	if(len(experiment_dict['action_selection_timings'])>0):
		print("action_selection_timings: "+str(np.mean(np.array(experiment_dict['action_selection_timings'])))+", "+str(np.std(np.array(experiment_dict['action_selection_timings']))))
	# Save the graph so we can load it back in later
	if(graph is not None):
		graph_filehandler = open(experiment_dict['exp_path'] + "/found_graph.pkl", 'wb')
		path_filehandler = open(experiment_dict['exp_path'] + "/found_path.pkl", 'wb')
		expdict_filehandler = open(experiment_dict['exp_path'] + "/experiment_dict.pkl", 'wb')
		pickle.dump(graph, graph_filehandler)
		pickle.dump(plan, path_filehandler)
		pickle.dump(experiment_dict, expdict_filehandler)

	stats_filehandler = open(experiment_dict['exp_path'] + "/stats.pkl", 'wb')
	pickle.dump(experiment_dict, stats_filehandler)
	

if __name__ == '__main__':
	exp_id = str(sys.argv[1])
	main(exp_id=exp_id)
