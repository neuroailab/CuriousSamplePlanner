#!/usr/bin/env python
from __future__ import print_function
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import time
from CuriousSamplePlanner.scripts.utils import *
from CuriousSamplePlanner.trainers.CSPPlanner import CSPPlanner
from CuriousSamplePlanner.trainers.architectures import WorldModel, SkinnyWorldModel


class RandomStateEmbeddingPlanner(CSPPlanner):
	def __init__(self, *args):
		super(RandomStateEmbeddingPlanner, self).__init__(*args)
		self.worldModel = opt_cuda(WorldModel(config_size=self.environment.config_size))
		self.transform = list(self.environment.predict_mask)
		random.shuffle(self.transform)
		self.criterion = nn.MSELoss()
		self.optimizer_world = optim.Adam(self.worldModel.parameters(), lr=self.experiment_dict["wm_learning_rate"])

	def reset_world_model(self):
		self.worldModel = opt_cuda(WorldModel(config_size=self.environment.config_size))
		self.criterion = nn.MSELoss()
		self.optimizer_world = optim.Adam(self.worldModel.parameters(), lr=self.experiment_dict["wm_learning_rate"])

	def update_novelty_scores(self):
		if(len(self.graph)>0 and self.experiment_dict["node_sampling"] == "softmax"):
			for _, (inputs, labels, prestates, node_key, index, _) in enumerate(DataLoader(self.graph, batch_size=self.experiment_dict['batch_size'], shuffle=True, num_workers=0)):
				# TODO: Turn this into a data provider
				outputs = opt_cuda(self.worldModel(opt_cuda(labels)).type(torch.FloatTensor))
				targets = opt_cuda(labels.type(torch.FloatTensor))[:, self.transform]
				#target_outputs = opt_cuda(self.worldModelTarget(opt_cuda(labels)).type(torch.FloatTensor))
				losses = []
				states = []
				for node in range(outputs.shape[0]):
					losses.append(torch.unsqueeze(self.criterion(outputs[node, self.environment.predict_mask], targets[node, :]), dim=0))
					states.append(labels[node, self.environment.predict_mask])
				self.graph.set_novelty_scores(index, losses)
				

	def train_world_model(self, run_index):
		start_time = time.time()
		for epoch in range(self.experiment_dict['num_training_epochs']):
			for next_loaded in enumerate(DataLoader(self.experience_replay, batch_size=self.experiment_dict['batch_size'], shuffle=True, num_workers=0)):
				_, batch = next_loaded
				inputs, labels, prestates, acts, feasible, _, index = batch
				labels = torch.squeeze(labels)
				self.optimizer_world.zero_grad()
				outputs = self.worldModel(labels)
				targets = labels[:, self.transform]
				loss = torch.mean((outputs[:, self.environment.predict_mask] - targets[:, :]) ** 2, dim=1).reshape(-1, 1)
				Lw = loss.mean()
				Lw.backward()
				self.optimizer_world.step()
				loss = loss.detach()

		if(self.experiment_dict['debug_timing']):
			self.experiment_dict['wm_batch_timings'].append(time.time()-start_time)


	def calc_novelty(self):
		whole_losses = []
		whole_indices = []
		whole_feasibles = []
		for _, batch in enumerate(
				DataLoader(self.experience_replay, batch_size=self.experiment_dict['batch_size'], shuffle=True, num_workers=0)):
			inputs, labels, prestates, acts, feasible, _, index = batch

			#Convert to cuda
			inputs = opt_cuda(inputs)
			labels = opt_cuda(labels)
			prestate = opt_cuda(prestates)
			acts = opt_cuda(acts)

			labels = torch.squeeze(labels)
			targets = labels[:, self.transform]
			outputs = self.worldModel(labels)
			losses = []
			for i in range(self.experiment_dict['batch_size']):
				l = self.criterion(outputs[i, self.environment.predict_mask], targets[i,:])
				
				losses.append(torch.unsqueeze(l,dim=0))
				whole_feasibles.append(feasible[i].item())


			whole_losses += [l.item() for l in losses]
			whole_indices += [l.item() for l in index]
		return whole_losses, whole_indices, whole_feasibles


	
