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
from CuriousSamplePlanner.rl_ppo_rnd.a2c_ppo_acktr.model import Policy
from CuriousSamplePlanner.tasks.ball_ramp import BallRamp
from CuriousSamplePlanner.tasks.pulley import PulleySeesaw
from CuriousSamplePlanner.tasks.bookshelf import BookShelf
from CuriousSamplePlanner.rl_ppo_rnd.a2c_ppo_acktr.storage import RolloutStorage
from CuriousSamplePlanner.rl_ppo_rnd.a2c_ppo_acktr import algo, utils
from CuriousSamplePlanner.scripts.utils import *
from CuriousSamplePlanner.trainers.planner import Planner
from CuriousSamplePlanner.trainers.architectures import FocusedWorldModel
from CuriousSamplePlanner.trainers.dataset import ExperienceReplayBuffer


class RecycleACPlanner(Planner):
    def __init__(self, *args):
        super(RecycleACPlanner, self).__init__(*args)

        # if self.experiment_dict['recycle']:
        #     with open(self.load_path + "/worldModel.pkl", "rb") as wf:
        #         self.worldModel = pickle.load(wf)
        #     with open(self.load_path + "/reward.pkl", 'rb') as rf:
        #         self.environment.reward = pickle.load(rf)
        #     with open(self.load_path + "/actor.pkl", 'rb') as af:
        #         self.environment.actor = pickle.load(af)
        # else:
        self.worldModel = opt_cuda(FocusedWorldModel(config_size=self.environment.config_size))
        self.value_loss_coef = 0.5
        self.entropy_coef = 0
        self.lr = 1e-3
        self.eps = 1e-5
        self.alpha = 0.99
        self.max_grad_norm = 0.5
        self.agent = algo.A2C_ACKTR(
            self.environment.actor_critic,
            self.value_loss_coef,
            self.entropy_coef,
            lr=self.lr,
            eps=self.eps,
            alpha=self.alpha,
            max_grad_norm=self.max_grad_norm)

        # self.environment.epsilon = self.experiment_dict['initial_epsilon']
        self.criterion = nn.MSELoss()
        self.optimizer_world = optim.Adam(self.worldModel.parameters(), lr=self.experiment_dict["learning_rate"])
        # self.optimizer_actor = optim.Adam(self.environment.actor_critic.parameters(), lr=self.experiment_dict["actor_learning_rate"])

    def expand_graph(self, run_index):

        self.num_steps = self.experiment_dict["nsamples_per_update"]

        rollouts = RolloutStorage(self.num_steps, 1, [self.environment.config_size], self.environment.action_space, 0)
        rollouts.opt_cuda()
        for epoch in range(self.num_training_epochs):
            for next_loaded in enumerate(
                    DataLoader(self.experience_replay, batch_size=self.batch_size, shuffle=True, num_workers=0)):

                _, batch = next_loaded
                inputs, labels, prestates, acts, acts_log_probs, values, feasible, _, index = batch
                self.optimizer_world.zero_grad()
                outputs = self.worldModel(labels)
                loss = torch.mean((outputs[:, self.environment.predict_mask] - labels[:, self.environment.predict_mask]) ** 2, dim=1).reshape(-1, 1)
                Lw = loss.mean()
                Lw.backward()
                self.optimizer_world.step()
                loss = loss.detach()
                if(self.experiment_dict["enable_asm"]):
                    if(self.experiment_dict["feasible_training"]):
                        loss = loss*feasible-(1-feasible)
                    done = [False]
                    infos = [{}]
                    recurrent_hidden_states = opt_cuda(torch.tensor([]))
                    masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                    bad_masks = torch.FloatTensor([[1.0] for info in infos])
                    for index in range(labels.shape[0]):
                        rollouts.insert(labels[index, :], recurrent_hidden_states, torch.unsqueeze(acts[index, :], 0), torch.unsqueeze(acts_log_probs[index, :], 0), values[index, :], loss[index, :], masks, bad_masks)

        if(self.experiment_dict["enable_asm"]):
            with torch.no_grad():
                next_value = self.environment.actor_critic.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, False, 0, 0, False)
            value_loss, action_loss, dist_entropy = self.agent.update(rollouts)
            rollouts.after_update()

        # Get the losses from all observations
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

        sort_args = np.array(whole_losses).argsort()[::-1]
        high_loss_indices = [whole_indices[p] for p in sort_args]

        # print("Feasible: "+str(sum(feasible)/len(feasible)))
        average_loss = sum(whole_losses) / len(whole_losses)
        self.experiment_dict['world_model_losses'].append(average_loss)
        self.print_exp_dict(verbose=False)

        if (average_loss <= self.loss_threshold):
            added_base_count = 0
            for en_index, hl_index in enumerate(high_loss_indices):
                input, target, pretarget, action, _, _, _, parent_index, _ = self.experience_replay.__getitem__(hl_index)
                print(target)
                time.sleep(1)
                ntarget = target.cpu().numpy()
                npretarget = pretarget.cpu().numpy()
                if (not self.graph.is_node(ntarget)):
                    self.environment.set_state(ntarget)
                    for perspective in self.environment.perspectives:
                        imageio.imwrite(self.exp_path
                                        + '/run_index=' + str(run_index)
                                        + ',index=' + str(en_index)
                                        + ',parent_index=' + str(int(parent_index.item()))
                                        + ',node_index=' + str(self.graph.node_key) + '.jpg',
                                        take_picture(perspective[0], perspective[1], 0, size=512))
                    self.graph.add_node(ntarget, npretarget, action.cpu().numpy(), torch.squeeze(parent_index).item())
                    added_base_count += 1
                if (added_base_count == self.growth_factor):
                    break
            del self.experience_replay
            self.experience_replay = ExperienceReplayBuffer()
            self.experiment_dict["num_graph_nodes"] += self.growth_factor

        with open(self.exp_path + "/worldModel.pkl", 'wb') as fw:
            pickle.dump(self.worldModel, fw)
        with open(self.exp_path + "/actor_critic.pkl", "wb") as fa:
            pickle.dump(self.environment.actor_critic, fa)


