#!/usr/bin/env python
from __future__ import print_function
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Utils
from CuriousSamplePlanner.scripts.utils import *

# Planners/Arch
from CuriousSamplePlanner.trainers.CSPPlanner import CSPPlanner
from CuriousSamplePlanner.trainers.architectures import ConvWorldModel

class StateEstimationPlanner(CSPPlanner):
	def __init__(self, *args):
		super(StateEstimationPlanner, self).__init__(*args)
		self.worldModel = opt_cuda(ConvWorldModel(config_size=self.environment.config_size, num_perspectives=len(self.environment.perspectives)))
		self.criterion = nn.MSELoss()
		self.optimizer_world = optim.Adam(self.worldModel.parameters(), lr=self.experiment_dict["wm_learning_rate"])


	def update_novelty_scores(self):
		if(len(self.graph)>0 and self.experiment_dict["node_sampling"] == "softmax"):
			for _, (inputs, labels, prestates, node_key, index, _) in enumerate(DataLoader(self.graph, batch_size=self.experiment_dict['batch_size'], shuffle=True, num_workers=0)):
				# TODO: Turn this into a data provider
				outputs = opt_cuda(self.worldModel(opt_cuda(inputs)).type(torch.FloatTensor))
				target_outputs = opt_cuda(labels)
				losses = []
				states = []
				for node in range(outputs.shape[0]):
					losses.append(torch.unsqueeze(self.criterion(outputs[node, self.environment.predict_mask], target_outputs[node, self.environment.predict_mask]), dim=0))
					states.append(labels[node, self.environment.predict_mask])
				self.graph.set_novelty_scores(index, losses)

	# def save_params(self):
	# 	# Save the updated models
	# 	with open(self.exp_path + "/worldModel.pkl", 'wb') as fw:
	# 		pickle.dump(self.worldModel, fw)
	# 	with open(self.exp_path + "/actor_critic.pkl", "wb") as fa:
	# 		pickle.dump(self.environment.actor_critic, fa)

	def train_world_model(self, run_index):
		start_time = time.time()
		for epoch in range(self.experiment_dict['num_training_epochs']):
			for next_loaded in enumerate(DataLoader(self.experience_replay, batch_size=self.experiment_dict['batch_size'], shuffle=True, num_workers=0)):
				_, batch = next_loaded
				inputs, labels, prestates, acts, feasible, _, index = batch
				self.optimizer_world.zero_grad()
				outputs = self.worldModel(torch.squeeze(inputs))
				labels = torch.squeeze(labels)
				loss = torch.mean((outputs[:, self.environment.predict_mask] - labels[:, self.environment.predict_mask]) ** 2, dim=1).reshape(-1, 1)
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
			labels = torch.squeeze(labels)
			outputs = self.worldModel(torch.squeeze(inputs))
			losses = []
			for i in range(self.experiment_dict['batch_size']):
				losses.append(torch.unsqueeze(
					self.criterion(outputs[i, self.environment.predict_mask], labels[i, self.environment.predict_mask]),
					dim=0))
				whole_feasibles.append(feasible[i].item())

			whole_losses += [l.item() for l in losses]
			whole_indices += [l.item() for l in index]
		return whole_losses, whole_indices, whole_feasibles



	
