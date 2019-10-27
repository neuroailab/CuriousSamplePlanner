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
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import collections
import sys

# Tasks
from CuriousSamplePlanner.tasks.five_block_stack import FiveBlocks
from CuriousSamplePlanner.tasks.three_block_stack import ThreeBlocks
from CuriousSamplePlanner.tasks.ball_ramp import BallRamp
from CuriousSamplePlanner.tasks.pulley import PulleySeesaw
from CuriousSamplePlanner.tasks.bookshelf import BookShelf

# Utils
from CuriousSamplePlanner.scripts.utils import *

# Planners/Arch
from CuriousSamplePlanner.trainers.planner import Planner
from CuriousSamplePlanner.trainers.architectures import ConvWorldModel
from CuriousSamplePlanner.trainers.dataset import ExperienceReplayBuffer

class StateEstimationPlanner(Planner):
    def __init__(self, *args):
        super(StateEstimationPlanner, self).__init__(*args)

        self.worldModel = opt_cuda(ConvWorldModel(config_size=self.environment.config_size, num_perspectives=len(self.environment.perspectives)))
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.worldModel.parameters(), lr=self.experiment_dict["learning_rate"])

    def expand_graph(self, run_index):
        for epoch in range(self.num_training_epochs):
            for next_loaded in enumerate(DataLoader(self.experience_replay, batch_size=self.batch_size, shuffle=True, num_workers=0)):
                self.optimizer.zero_grad()
                _, batch = next_loaded
                inputs, labels, _, _, _, index = batch
                outputs = opt_cuda(self.worldModel(inputs).type(torch.FloatTensor))

                loss = self.criterion(outputs[:, self.environment.predict_mask], labels[:, self.environment.predict_mask])
                loss.backward()
                self.experiment_dict["world_model_losses"].append(loss.item())
                self.optimizer.step()
                print(loss)

        # Get the losses from all observations
        whole_losses = []
        whole_indices = []
        for _, batch in enumerate(DataLoader(self.experience_replay, batch_size=self.batch_size, shuffle=True, num_workers=0)):
            inputs, labels, _, _, _, index = batch
            outputs = opt_cuda(self.worldModel(inputs).type(torch.FloatTensor))
            losses = []
            for i in range(self.batch_size):
                losses.append(torch.unsqueeze(self.criterion(outputs[i, self.environment.predict_mask], labels[i, self.environment.predict_mask]), dim=0))
            loss = torch.mean(torch.cat(losses))

            whole_losses+=[l.item() for l in losses]
            whole_indices+=[l.item() for l in index]


        sort_args = np.array(whole_losses).argsort()[::-1]
        high_loss_indices = [whole_indices[p] for p in sort_args]

        average_loss = sum(whole_losses)/len(whole_losses) 

        if(average_loss <= self.loss_threshold):
            added_base_count = 0
            for en_index, hl_index in enumerate(high_loss_indices):
                input, target, pretarget, action, parent_index, _ = self.experience_replay.__getitem__(hl_index)
                ntarget = target.cpu().numpy()
                npretarget = target.cpu().numpy()
                if(not self.graph.is_node(ntarget)):
                    self.environment.set_state(ntarget)
                    for perspective in self.environment.perspectives:
                        imageio.imwrite(self.exp_path+'/run_index='+str(run_index)+',index='+str(en_index)+'perpective='+str(perspective)+'.jpg', take_picture(perspective[0], perspective[1], 0, size=512))
                    self.graph.add_node(ntarget, pretarget, action.cpu().numpy(), torch.squeeze(parent_index).item())
                    added_base_count+=1
                if(added_base_count == self.growth_factor):
                    break
            del self.experience_replay
            self.experience_replay = ExperienceReplayBuffer()
            self.experiment_dict["num_graph_nodes"]+=self.growth_factor


        total_losses = []
        total_losses_states = []
        if(len(self.graph)>0):
            for _, (inputs, labels, prestates, node_key, index) in enumerate(DataLoader(self.graph, batch_size=self.batch_size, shuffle=True, num_workers=0)):
                # TODO: Turn this into a data provider
                outputs = opt_cuda(self.worldModel(inputs).type(torch.FloatTensor))
                losses = []
                states = []
                for node in range(outputs.shape[0]):
                    losses.append(torch.unsqueeze(self.criterion(outputs[node, self.environment.predict_mask], opt_cuda(labels)[node, self.environment.predict_mask]), dim=0))
                    states.append(labels[node, self.environment.predict_mask])
                self.graph.set_novelty_scores(index, losses)

                total_losses+=losses
                total_losses_states+=states

    
