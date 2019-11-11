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
import imageio
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
from CuriousSamplePlanner.trainers.architectures import WorldModel, SkinnyWorldModel
from CuriousSamplePlanner.trainers.dataset import ExperienceReplayBuffer
from CuriousSamplePlanner.trainers.ACPlanner import ACPlanner


class RandomStateEmbeddingPlanner(ACPlanner):
	def __init__(self, *args):
		super(RandomStateEmbeddingPlanner, self).__init__(*args)
		self.worldModelTarget = opt_cuda(SkinnyWorldModel(config_size=self.environment.config_size))
		self.worldModel = opt_cuda(WorldModel(config_size=self.environment.config_size))
		def weights_init_uniform(m):
			classname = m.__class__.__name__            
			# apply a uniform distribution to the weights and a bias=0
			if classname.find('Linear') != -1:
				m.weight.data.uniform_(0.0, 0.06)
				m.bias.data.fill_(0)

		self.worldModelTarget.apply(weights_init_uniform)
		for param in self.worldModelTarget.parameters():
			param.requires_grad = False
		self.worldModel = opt_cuda(WorldModel(config_size=self.environment.config_size))
		self.criterion = nn.MSELoss()
		self.optimizer_world = optim.Adam(self.worldModel.parameters(), lr=self.experiment_dict["learning_rate"])


	def update_novelty_scores(self):
		if(len(self.graph)>0 and self.experiment_dict["node_sampling"] == "softmax"):
			for _, (inputs, labels, prestates, node_key, index) in enumerate(DataLoader(self.graph, batch_size=self.batch_size, shuffle=True, num_workers=0)):
				# TODO: Turn this into a data provider
				outputs = opt_cuda(self.worldModel(opt_cuda(labels)).type(torch.FloatTensor))
				#target_outputs = opt_cuda(self.worldModelTarget(opt_cuda(labels)).type(torch.FloatTensor))
				target_outputs = opt_cuda(labels)
				losses = []
				states = []
				for node in range(outputs.shape[0]):
					losses.append(torch.unsqueeze(self.criterion(outputs[node, self.environment.predict_mask], target_outputs[node, self.environment.predict_mask]), dim=0))
					states.append(labels[node, self.environment.predict_mask])
				self.graph.set_novelty_scores(index, losses)
				
	def save_params(self):
		# Save the updated models
		with open(self.exp_path + "/worldModel.pkl", 'wb') as fw:
			pickle.dump(self.worldModel, fw)
		with open(self.exp_path + "/actor_critic.pkl", "wb") as fa:
			pickle.dump(self.environment.actor_critic, fa)

	def train_world_model(self, run_index):
		for epoch in range(self.num_training_epochs):
			for next_loaded in enumerate(DataLoader(self.experience_replay, batch_size=self.batch_size, shuffle=True, num_workers=0)):
				_, batch = next_loaded
				inputs, labels, prestates, acts, acts_log_probs, values, feasible, _, index = batch
				self.optimizer_world.zero_grad()
				outputs = self.worldModel(labels)
				loss = torch.mean((outputs[:, self.environment.predict_mask] - labels[:, self.environment.predict_mask]) ** 2, dim=1).reshape(-1, 1)
				Lw = loss.mean()
				Lw.backward()
				self.optimizer_world.step()
				loss = loss.detach()
				if(self.experiment_dict["enable_asm"] and epoch==0):
					if(self.experiment_dict["feasible_training"]):
						loss = loss*feasible-(1-feasible)
					done = [False]
					infos = [{}]
					recurrent_hidden_states = opt_cuda(torch.tensor([]))
					masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
					bad_masks = torch.FloatTensor([[1.0] for info in infos])
					for index in range(labels.shape[0]):
						self.rollouts.insert(labels[index, :], recurrent_hidden_states, torch.unsqueeze(acts[index, :], 0), torch.unsqueeze(acts_log_probs[index, :], 0), values[index, :], loss[index, :], masks, bad_masks)

	def calc_novelty(self):
		whole_losses = []
		whole_indices = []
		whole_feasibles = []
		for _, batch in enumerate(
				DataLoader(self.experience_replay, batch_size=self.batch_size, shuffle=True, num_workers=0)):
			inputs, labels, prestates, acts, _, _, feasible, _, index = batch
			outputs = self.worldModel(labels)
			losses = []
			for i in range(self.batch_size):
				losses.append(torch.unsqueeze(
					self.criterion(outputs[i, self.environment.predict_mask], labels[i, self.environment.predict_mask]),
					dim=0))
				whole_feasibles.append(feasible[i].item())

			whole_losses += [l.item() for l in losses]
			whole_indices += [l.item() for l in index]
		return whole_losses, whole_indices, whole_feasibles

	# def expand_graph(self, run_index):
	# 	print(self.num_training_epochs)
	# 	for epoch in range(self.num_training_epochs):
	# 		for next_loaded in enumerate(DataLoader(self.experience_replay, batch_size=self.batch_size, shuffle=True, num_workers=0)):
	# 			self.optimizer.zero_grad()
	# 			_, batch = next_loaded
	# 			inputs, labels, prestates, _, _, _, index = batch
	# 			outputs = opt_cuda(self.worldModel(labels).type(torch.FloatTensor))
	# 			#outputs_target = opt_cuda(self.worldModelTarget(labels).type(torch.FloatTensor))
	# 			outputs_target=opt_cuda(labels)

	# 			loss = self.criterion(outputs[:, self.environment.predict_mask],
	# 								 outputs_target[:, self.environment.predict_mask])
	# 			loss.backward()
	# 			self.experiment_dict["world_model_losses"].append(loss.item())
	# 			self.optimizer.step()
	# 			print('world loss:', loss.item())


	# 	# Get the losses from all observations
	# 	whole_losses = []
	# 	whole_indices = []
	# 	for _, batch in enumerate(DataLoader(self.experience_replay, batch_size=self.batch_size, shuffle=True, num_workers=0)):
	# 		inputs, labels, prestates, _, _, index = batch

	# 		outputs = opt_cuda(self.worldModel(labels).type(torch.FloatTensor))
	# 		#target_outputs = opt_cuda(self.worldModelTarget(labels).type(torch.FloatTensor))
	# 		target_outputs = opt_cuda(labels)

	# 		losses = []
	# 		for i in range(self.batch_size):
	# 			losses.append(torch.unsqueeze(self.criterion(outputs[i, self.environment.predict_mask], target_outputs[i, self.environment.predict_mask]), dim=0))

	# 		whole_losses+=[l.item() for l in losses]
	# 		whole_indices+=[l.item() for l in index]

	# 	sort_args = np.array(whole_losses).argsort()[::-1]
	# 	high_loss_indices = [whole_indices[p] for p in sort_args]

	# 	average_loss = sum(whole_losses)/len(whole_losses) 
	# 	print("Average Loss: "+str(average_loss))

	# 	if(average_loss <= self.loss_threshold):
	# 		print("Adding to bases and restarting experience replay")
	# 		added_base_count = 0
	# 		for en_index, hl_index in enumerate(high_loss_indices):
	# 			input, target, pretarget, action, parent_index, _ = self.experience_replay.__getitem__(hl_index)
	# 			ntarget = target.cpu().numpy()
	# 			npretarget = pretarget.cpu().numpy()
	# 			if(not self.graph.is_node(ntarget)):
	# 				self.environment.set_state(ntarget)
	# 				for perspective in self.environment.perspectives:
	# 					imageio.imwrite(self.exp_path
	# 									+ '/run_index=' + str(run_index)
	# 									+ ',index=' + str(en_index)
	# 									+ ',parent_index=' + str(int(parent_index.item()))
	# 									+ ',node_index=' + str(self.graph.node_key) + '.jpg',
	# 									take_picture(perspective[0], perspective[1], 0, size=512))
	# 				self.graph.add_node(ntarget, npretarget, action.cpu().numpy(), torch.squeeze(parent_index).item())

	# 				added_base_count+=1

				
	# 			if(added_base_count == self.growth_factor):
	# 				break
	# 		del self.experience_replay
	# 		self.experience_replay = ExperienceReplayBuffer()
	# 		self.experiment_dict["num_graph_nodes"]+=self.growth_factor

	# 	total_losses = []
	# 	total_losses_states = []
	# 	if(len(self.graph)>0):
	# 		for _, (inputs, labels, prestates, node_key, index) in enumerate(DataLoader(self.graph, batch_size=self.batch_size, shuffle=True, num_workers=0)):
	# 			# TODO: Turn this into a data provider
	# 			outputs = opt_cuda(self.worldModel(opt_cuda(labels)).type(torch.FloatTensor))
	# 			#target_outputs = opt_cuda(self.worldModelTarget(opt_cuda(labels)).type(torch.FloatTensor))
	# 			target_outputs = opt_cuda(labels)
	# 			losses = []
	# 			states = []
	# 			for node in range(outputs.shape[0]):
	# 				losses.append(torch.unsqueeze(self.criterion(outputs[node, self.environment.predict_mask], target_outputs[node, self.environment.predict_mask]), dim=0))
	# 				states.append(labels[node, self.environment.predict_mask])
	# 			self.graph.set_novelty_scores(index, losses)

	# 			total_losses+=losses
	# 			total_losses_states+=states

	
