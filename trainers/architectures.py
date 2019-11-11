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
import sys

from CuriousSamplePlanner.tasks.three_block_stack import ThreeBlocks

from CuriousSamplePlanner.tasks.ball_ramp import BallRamp
from CuriousSamplePlanner.tasks.pulley import PulleySeesaw
from CuriousSamplePlanner.tasks.bookshelf import BookShelf
from CuriousSamplePlanner.scripts.utils import *

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)


class ConvWorldModel(nn.Module):
	def __init__(self, config_size=9, num_perspectives=1):
		super(ConvWorldModel, self).__init__()
		self.conv = nn.Sequential(nn.Conv2d(3*num_perspectives, 32, 8, stride=4), nn.ReLU(),
								  nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
								  nn.Conv2d(64, 32, 3, stride=1), nn.ReLU(), Flatten(),
								  nn.Linear(1568*num_perspectives, 128), nn.ReLU(), nn.Linear(128, config_size))
	def forward(self, inputs):
		return self.conv(inputs)


class SkinnyWorldModel(nn.Module):
	def __init__(self, config_size=9):
		super(SkinnyWorldModel, self).__init__()
		hidden = 128
		self.mlp = nn.Linear(config_size, hidden)

	def forward(self, config):
		l = self.mlp(config)
		return l

class WorldModel(nn.Module):
	def __init__(self, config_size=9):
		super(WorldModel, self).__init__()
		hidden = 128
		self.mlp = nn.Sequential(nn.Linear(config_size, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, config_size))

	def forward(self, config):
		l = self.mlp(config)
		return l

