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

from tasks.three_block_stack import ThreeBlocks

from tasks.ball_ramp import BallRamp
from tasks.pulley import PulleySeesaw
from tasks.bookshelf import BookShelf

from scripts.utils import *
from trainers.planner import Planner
from trainers.architectures import DynamicsModel
from trainers.dataset import ExperienceReplayBuffer
from trainers.ACPlanner import ACPlanner


class EffectPredictionPlanner(ACPlanner):
    def __init__(self, *args):
        super(EffectPredictionPlanner, self).__init__(*args)
        self.worldModel = opt_cuda(
            DynamicsModel(config_size=self.environment.config_size, action_size=self.environment.action_space_size))
        self.criterion = nn.MSELoss()
        self.optimizer_world = optim.Adam(self.worldModel.parameters(), lr=self.experiment_dict["learning_rate"])

    def save_params(self):
        # Save the updated models
        with open(self.exp_path + "/worldModel.pkl", 'wb') as fw:
            pickle.dump(self.worldModel, fw)
        with open(self.exp_path + "/actor_critic.pkl", "wb") as fa:
            pickle.dump(self.environment.actor_critic, fa)

    def update_novelty_scores(self):
        if len(self.graph) > 0 and self.experiment_dict["node_sampling"] == "softmax":
            for _, (inputs, labels, prestates, node_key, index, actions) in enumerate(
                    DataLoader(self.graph, batch_size=self.batch_size, shuffle=True, num_workers=0)):
                # TODO: Turn this into a data provider
                outputs = opt_cuda(self.worldModel(opt_cuda(prestates, actions)).type(torch.FloatTensor))
                target_outputs = opt_cuda(labels)
                losses = []
                states = []
                for node in range(outputs.shape[0]):
                    losses.append(torch.unsqueeze(self.criterion(outputs[node, self.environment.predict_mask],
                                                                 target_outputs[node, self.environment.predict_mask]),
                                                  dim=0))
                    states.append(labels[node, self.environment.predict_mask])
                self.graph.set_novelty_scores(index, losses)

    def train_world_model(self, run_index):
        for epoch in range(self.num_training_epochs):
            for next_loaded in enumerate(
                    DataLoader(self.experience_replay, batch_size=self.batch_size, shuffle=True, num_workers=0)):
                _, batch = next_loaded
                inputs, labels, prestates, acts, acts_log_probs, values, feasible, _, index = batch
                self.optimizer_world.zero_grad()
                outputs = self.worldModel(prestates, acts)
                loss = torch.mean(
                    (outputs[:, self.environment.predict_mask] - labels[:, self.environment.predict_mask]) ** 2,
                    dim=1).reshape(-1, 1)
                Lw = loss.mean()
                Lw.backward()
                self.optimizer_world.step()
                loss = loss.detach()
                if self.experiment_dict["enable_asm"] and epoch == 0:
                    if self.experiment_dict["feasible_training"]:
                        loss = loss * feasible - (1 - feasible)
                    done = [False]
                    infos = [{}]
                    recurrent_hidden_states = opt_cuda(torch.tensor([]))
                    masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                    bad_masks = torch.FloatTensor([[1.0] for info in infos])
                    for index in range(labels.shape[0]):
                        self.rollouts.insert(labels[index, :], recurrent_hidden_states,
                                             torch.unsqueeze(acts[index, :], 0),
                                             torch.unsqueeze(acts_log_probs[index, :], 0), values[index, :],
                                             loss[index, :], masks, bad_masks)

    def calc_novelty(self):
        whole_losses = []
        whole_indices = []
        whole_feasibles = []
        for _, batch in enumerate(
                DataLoader(self.experience_replay, batch_size=self.batch_size, shuffle=True, num_workers=0)):
            inputs, labels, prestates, actions, _, _, feasible, _, index = batch
            outputs = self.worldModel(prestates, actions)
            losses = []
            for i in range(self.batch_size):
                losses.append(torch.unsqueeze(
                    self.criterion(outputs[i, self.environment.predict_mask], labels[i, self.environment.predict_mask]),
                    dim=0))
                whole_feasibles.append(feasible[i].item())

            whole_losses += [l.item() for l in losses]
            whole_indices += [l.item() for l in index]
        return whole_losses, whole_indices, whole_feasibles
