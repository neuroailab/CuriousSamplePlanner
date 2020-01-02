import copy
import glob
import os
import time
from collections import deque
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from CuriousSamplePlanner.rl_ppo_rnd.a2c_ppo_acktr import algo, utils
from CuriousSamplePlanner.rl_ppo_rnd.a2c_ppo_acktr.model import Policy
from CuriousSamplePlanner.rl_ppo_rnd.a2c_ppo_acktr.storage import RolloutStorage
from gym import spaces
from CuriousSamplePlanner.scripts.utils import *
from CuriousSamplePlanner.trainers.architectures import WorldModel, DynamicsModel, ConvWorldModel
import shutil
from CuriousSamplePlanner.tasks.three_block_stack import ThreeBlocks
from CuriousSamplePlanner.tasks.five_block_stack import FiveBlocks
from CuriousSamplePlanner.tasks.ball_ramp import BallRamp
from CuriousSamplePlanner.tasks.bookshelf import BookShelf
from CuriousSamplePlanner.tasks.pulley import PulleySeesaw
from CuriousSamplePlanner.trainers.dataset import ExperienceReplayBuffer
import sys
import h5py





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
	"exp_id": "test",
	"load_id": None,
	"enable_asm": False, 
	"growth_factor": 10,
	"detailed_gmp": False, 
	"adaptive_batch": True,
	"num_training_epochs": 30,
	"terminate_unreachable": False,
	"infeasible_penalty" : 0,
	# Stats
	"world_model_losses": [],
	"feasibility":[],
	"num_sampled_nodes": 0,
	"num_graph_nodes": 0,
}

PC = getattr(sys.modules[__name__], experiment_dict['task'])
env = PC(experiment_dict)


obj_file = open("models/box_obj/tinker.obj","r")
obj_mesh = obj_file.read()

f = h5py.File('/mnt/fs0/arc11_2/HRN_Data_2.hdf5', 'w')
picture, view_matrix, projection_matrix = take_picture(45, -45, 0, size=2048)
f["camera_view_matrix"] = view_matrix
f["camera_projection_matrix"] = projection_matrix
f["object_1"] = obj_mesh
f["object_2"] = obj_mesh
f["object_3"] = obj_mesh

record_index = 0
for record_index in range(10000):
	if(record_index%100 == 0):
		print("Record Index: "+str(record_index))
	obs = env.reset()
	# run the three-stack loop to steady-state, looping on convergence 
	action = torch.unsqueeze(torch.tensor(np.random.uniform(low=-1, high=1, size=env.action_space_size)), dim=0)
	_ = env.take_action(action)
	
	time_limit = 200
	config = env.get_current_config()
	dconfig = env.get_current_detailed_config()

	t = 0
	while (True):
		for _ in range(5):
			p.stepSimulation(physicsClientId=0)
		config_t = env.get_current_config()
		dconfig_t = env.get_current_detailed_config()
		if(record_index<10):
			picture, _, _ = take_picture(45, -45, 0, size=512)
			f["record_"+str(record_index)+"/image_"+str(t)] = picture

		for key, val in dconfig_t.items():
			f["record_"+str(record_index)+"/"+str(t)+"/"+str(key)] = val

		if (dist(config_t, config) < 3e-5 or t>time_limit):
			break
		config = config_t
		t+=1
f.close()

		





