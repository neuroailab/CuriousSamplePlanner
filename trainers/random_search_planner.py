#!/usr/bin/env python
from __future__ import print_function
import pybullet as p
import numpy as np
import random
import math
import imageio
import matplotlib.pyplot as plt
import os
import shutil
import pickle
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys

from CuriousSamplePlanner.tasks.three_block_stack import ThreeBlocks

from CuriousSamplePlanner.tasks.ball_ramp import BallRamp
from CuriousSamplePlanner.tasks.pulley import PulleySeesaw
from CuriousSamplePlanner.tasks.bookshelf import BookShelf
from CuriousSamplePlanner.scripts.utils import *
from CuriousSamplePlanner.trainers.planner import Planner
from CuriousSamplePlanner.trainers.dataset import ExperienceReplayBuffer
from CuriousSamplePlanner.trainers.ACPlanner import ACPlanner


class RandomSearchPlanner(ACPlanner):
	def __init__(self, *args):
	   super(RandomSearchPlanner, self).__init__(*args)


	def update_novelty_scores(self):
		pass	

	def save_params(self):
		pass
		
	def train_world_model(self, run_index):
		pass

	def calc_novelty(self):
		whole_losses = []
		whole_indices = []
		whole_feasibles = []
		whole_losses += [random.uniform(-1, 1) for _ in range(len(self.experience_replay))]
		whole_indices += [i for i in range(len(self.experience_replay))]
		whole_feasibles += [1 for i in range(len(self.experience_replay))]
		return whole_losses, whole_indices, whole_feasibles

	# def expand_graph(self, run_index):
	# 	high_loss_indices = list(range(self.num_training_epochs-1))
	# 	np.random.shuffle(high_loss_indices)
	# 	added_base_count = 0
	# 	for en_index, hl_index in enumerate(high_loss_indices):
	# 		input, target, pretarget, action, parent_index, _ = self.experience_replay.__getitem__(hl_index)
	# 		ntarget = target.cpu().numpy()
	# 		npretarget = pretarget.cpu().numpy()
	# 		if(not self.graph.is_node(ntarget)):
	# 			self.environment.set_state(ntarget)
	# 			for perspective in self.environment.perspectives:
	# 				imageio.imwrite(self.exp_path+'/run_index='+str(run_index)+',index='+str(en_index)+'perpective='+str(perspective)+'.jpg', take_picture(perspective[0], perspective[1], 0, size=512))
	# 			self.graph.add_node(ntarget, npretarget, action.cpu().numpy(), torch.squeeze(parent_index).item())
	# 			added_base_count+=1
	# 		if(added_base_count == self.growth_factor):
	# 			break
	# 	del self.experience_replay
	# 	self.experience_replay = ExperienceReplayBuffer()
	# 	self.experiment_dict["num_graph_nodes"]+=self.growth_factor

