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

from tasks.three_block_stack import ThreeBlocks

from tasks.ball_ramp import BallRamp
from tasks.pulley import PulleySeesaw
from tasks.bookshelf import BookShelf
from scripts.utils import *
from trainers.planner import Planner
from trainers.dataset import ExperienceReplayBuffer
from trainers.ACPlanner import ACPlanner


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
