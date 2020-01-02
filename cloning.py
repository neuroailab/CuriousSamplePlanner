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
from CuriousSamplePlanner.trainers.ACPlanner import ACPlanner
from CuriousSamplePlanner.scripts.utils import *
from CuriousSamplePlanner.agent.planning_agent import PlanningAgent


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# First, we need to load in the existing paths
NUM_SOLUTIONS = 1000
paths = []
path_path = "./solution_data/testbk"
for path_index in range(1, NUM_SOLUTIONS):
	# Load the file in
	path_file = open(path_path+"/found_path_"+str(path_index)+".pkl", 'rb')
	path_data = pickle.load(path_file)
	paths.append(path_data)
	path_file.close()

state_action_pairs = []
# Collect the state/action pairs
for path in paths:
	for ni in range(1, len(path)):
		state_action_pairs.append((path[ni-1].config, path[ni].action))


(X, y) = [s[0] for s in state_action_pairs], [s[1] for s in state_action_pairs]

regressor = LinearRegression()  
regressor.fit(X, y) # training the algorithm


# Now that we have a model, we are ready to test out how good it is.

# First, spin up an environment
experiment_dict = {
	# Hyps
	"task": "ThreeBlocks",
	"learning_rate": 5e-5,
	"sample_cap": 1e7,
	"batch_size": 128,
	"node_sampling": "softmax",
	"mode": "RandomStateEmbeddingPlanner",
	"feasible_training": True,
	"nsamples_per_update": 1024,
	"training": True,
	"exp_id": exp_id,
	"load_id": load_id,
	"enable_asm": False,
	"growth_factor": 10,
	"detailed_gmp": False,
	"adaptive_batch": True,
	"num_training_epochs": 30,
	"infeasible_penalty" : 0,
	# Stats
	"world_model_losses": [],
	"feasibility":[],
	"num_sampled_nodes": 0,
	"num_graph_nodes": 0,
}

PC = getattr(sys.modules[__name__], experiment_dict['mode'])
planner = PC(experiment_dict)
graph, plan, experiment_dict = planner.plan()

