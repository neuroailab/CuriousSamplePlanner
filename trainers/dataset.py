#!/usr/bin/env python
from __future__ import print_function
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

class ExperienceReplayBuffer(Dataset):
	def __init__(self):
		self.buffer = []
		self.target_buffer = []
		self.parents = []
		self.actions = []
		self.pretarget_buffer = []
		self.action_log_probs = []
		self.feasible = []
		self.values = []
		self.commands = []
		
	def __len__(self):
		return len(self.buffer)

	def bufferadd_single(self, item, target, pretarget, actions, action_log_probs, values, feasible, parent, command):  
		self.buffer.append(item)
		self.target_buffer.append(target)
		self.pretarget_buffer.append(pretarget)
		self.actions.append(actions)
		self.parents.append(parent)
		self.feasible.append(feasible)
		self.action_log_probs.append(action_log_probs)
		self.values.append(values)
		self.commands.append(command)

	def bufferadd(self, item, target, pretarget, actions, action_log_probs, values, feasible, parent, command):
		for i in range(target.shape[0]):
			self.buffer.append(item[i, :])
			self.target_buffer.append(target[i, :])
			self.pretarget_buffer.append(pretarget[i, :])
			self.actions.append(actions[i, :])
			self.parents.append(parent[i, :])
			self.action_log_probs.append(action_log_probs[i, :])
			self.values.append(values[i, :])
			self.feasible.append(feasible[i, :])
			self.commands.append(command[i])

	def get_command(self, index):
		return self.commands[index]
		
	def __getitem__(self, index):
		return self.buffer[index], self.target_buffer[index], self.pretarget_buffer[index], self.actions[index], self.action_log_probs[index], self.values[index], self.feasible[index], self.parents[index], torch.tensor(index)
