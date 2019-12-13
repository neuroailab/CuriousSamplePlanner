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
from CuriousSamplePlanner.tasks.three_block_stack import ThreeBlocks
from CuriousSamplePlanner.tasks.ball_ramp import BallRamp
from CuriousSamplePlanner.tasks.pulley import PulleySeesaw
from CuriousSamplePlanner.tasks.bookshelf import BookShelf
from CuriousSamplePlanner.trainers.architectures import PolicyModel
from CuriousSamplePlanner.tasks.five_block_stack import FiveBlocks
import pybullet as p
import numpy as np
import time
import random
import math
import imageio
import matplotlib.pyplot as plt
import os
import shutil
import h5py
import imageio
import pickle
import collections
from CuriousSamplePlanner.planning_pybullet.motion.motion_planners.discrete import astar
import sys

from CuriousSamplePlanner.scripts.utils import *
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader



def score_policy(policy):
	time_limit = 1000
	total_actions = 2000
	actions_sofar = 0
	reward = 0
	with torch.set_grad_enabled(False):
		rewards = []
		while(True):
			if(actions_sofar>total_actions):
				break
			if(actions_sofar%100==0):
				environment.reset()
				rewards.append(reward)
				reward = 0

			config = environment.get_current_config()

			action = policy(torch.tensor([config]).type(torch.FloatTensor))
			environment.take_action(torch.tensor(action))
			actions_sofar+=1
			t = 0
			while (True):
				for _ in range(5):
					p.stepSimulation(physicsClientId=0)
				config_t = environment.get_current_config()
				if (dist(config_t, config) < 3e-5 or t>time_limit):
					break
				config = config_t
				t+=1
			config = config_t
			if(environment.check_goal_state(config)):
				reward += 1
				environment.reset()

	nprewards = np.array(rewards)
	print("Mean: "+str(np.mean(nprewards))+", Std: "+str(np.std(nprewards)))
	return np.mean(nprewards), np.std(nprewards)


class Teacher(Dataset):
	def __init__(self, files = [0, 0]):
		# First, we need to load in the existing paths
		NUM_SOLUTIONS = 5000
		paths = []
		path_path = "./solution_data/testbk2"
		for path_index in range(files[0], files[1]):
			# Load the file in
			path_file = open(path_path+"/found_path_"+str(path_index)+".pkl", 'rb')
			path_data = pickle.load(path_file)
			paths.append(path_data)
			path_file.close()

		state_action_pairs = []
		# Collect the state/action pairs
		for path in paths:
			for ni in range(1, len(path)):
				state_action_pairs.append([path[ni-1].config, path[ni].action])


		(self.X, self.y) = torch.tensor(np.array([s[0] for s in state_action_pairs])).type(torch.FloatTensor), torch.tensor(np.array([s[1] for s in state_action_pairs])).type(torch.FloatTensor)

	def __len__(self):
		return self.y.shape[0]

	def __getitem__(self, index):
		return self.X[index, :], self.y[index, :]



# Get the data for the model 
NUM_DATA_POINTS = 8192
train_teacher = Teacher(files = [1, NUM_DATA_POINTS])
# val_teacher = Teacher(files = [2600, 2600+128])

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
	"exp_id": "bk",
	"load_id": None,
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


# Create the environment
EC = getattr(sys.modules[__name__], experiment_dict["task"])
environment = EC(experiment_dict)

# Create the regression model
policy = PolicyModel(environment.config_size, environment.action_space_size)
policy_optimizer = optim.Adam(policy.parameters(), lr=5e-3)
criterion = nn.MSELoss()

# Fit the regression model
NUM_EPOCHS = 1000
tot_index = 0	
train_losses = []
# val_losses = []
scores = []
scores_std = []

for epoch in range(NUM_EPOCHS):
	train_loader = DataLoader(train_teacher, batch_size=256, shuffle=True, num_workers=0,)
	# val_loader = DataLoader(val_teacher, batch_size=128, shuffle=True, num_workers=0)

	# with torch.set_grad_enabled(False):
	# 	for X, y in val_loader:
	# 		actions = policy(X)
	# 		loss = criterion(actions, y)
	# 		Lw = loss.mean()
	# 		val_losses.append((tot_index, Lw.item()))


	if(epoch%10 == 0):
		mean, std = score_policy(policy)
		scores.append((tot_index, mean))
		scores_std.append((tot_index, std))


	for idx, (X, y) in enumerate(train_loader):
		policy_optimizer.zero_grad()
		actions = policy(X)
		loss = criterion(actions, y)
		Lw = loss.mean()
		Lw.backward()
		policy_optimizer.step()	
		tot_index += X.shape[0]
		train_losses.append((tot_index, Lw.item()))

np.save("scores"+str(NUM_DATA_POINTS), scores)
np.save("scores_std"+str(NUM_DATA_POINTS), scores_std)
# plt.plot([v[0] for v in train_losses],[v[1] for v in train_losses])
# plt.plot([v[0] for v in val_losses],[v[1] for v in val_losses])
# plt.plot([v[0] for v in scores],[v[1] for v in scores])
# plt.show()









