#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import torch
import sys

from CuriousSamplePlanner.tasks.two_block_stack import TwoBlocks
from CuriousSamplePlanner.tasks.three_block_stack import ThreeBlocks
from CuriousSamplePlanner.tasks.ball_ramp import BallRamp
from CuriousSamplePlanner.tasks.pulley import PulleySeesaw
from CuriousSamplePlanner.tasks.bookshelf import BookShelf
from CuriousSamplePlanner.tasks.five_block_stack import FiveBlocks
from CuriousSamplePlanner.tasks.four_block_stack import FourBlocks
from CuriousSamplePlanner.tasks.simple_2d import Simple2d

from CuriousSamplePlanner.scripts.utils import *
from CuriousSamplePlanner.trainers.plan_graph import PlanGraph

from CuriousSamplePlanner.policies.fixed import FixedPolicy
from CuriousSamplePlanner.policies.random import RandomPolicy
from CuriousSamplePlanner.policies.DDPGLearning import DDPGLearningPolicy
from CuriousSamplePlanner.policies.PPOLearning import PPOLearningPolicy
from CuriousSamplePlanner.policies.HERLearning import HERLearningPolicy


class Planner():
	def __init__(self, experiment_dict):
		self.experiment_dict = experiment_dict

		# Create the environment
		EC = getattr(sys.modules[__name__], self.experiment_dict["task"])
		self.environment = EC(experiment_dict)

		# Create the policy
		PC = getattr(sys.modules[__name__], self.experiment_dict["policy"])
		self.policy = PC(experiment_dict, self.environment)

		# Init the plan graph 
		self.graph = PlanGraph(environment=self.environment, node_sampling = self.experiment_dict['node_sampling'])
		super(Planner, self).__init__()