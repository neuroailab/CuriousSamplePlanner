#!/usr/bin/env python
from __future__ import print_function
import random
from CuriousSamplePlanner.scripts.utils import *
from CuriousSamplePlanner.trainers.CSPPlanner import CSPPlanner


class RandomSearchPlanner(CSPPlanner):
	def __init__(self, *args):
		super(RandomSearchPlanner, self).__init__(*args)

	def update_novelty_scores(self):
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



