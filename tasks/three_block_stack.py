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
from CuriousSamplePlanner.planning_pybullet.pybullet_tools.kuka_primitives import BodyPose, BodyConf, Command, get_grasp_gen, \
	get_ik_fn, get_free_motion_gen, get_holding_motion_gen
from CuriousSamplePlanner.planning_pybullet.pybullet_tools.utils import WorldSaver, enable_gravity, connect, dump_world, set_pose, \
	Pose, Point, Euler, set_default_camera, stable_z, \
	BLOCK_URDF, load_model, wait_for_user, disconnect, DRAKE_IIWA_URDF, user_input, update_state, disable_real_time,inverse_kinematics,end_effector_from_body,approach_from_grasp, get_joints, get_joint_positions


from CuriousSamplePlanner.planning_pybullet.pybullet_tools.utils import get_pose, set_pose, get_movable_joints, \
	set_joint_positions, add_fixed_constraint, enable_real_time, disable_real_time, joint_controller, \
	enable_gravity, get_refine_fn, user_input, wait_for_duration, link_from_name, get_body_name, sample_placement, \
	end_effector_from_body, approach_from_grasp, plan_joint_motion, GraspInfo, Pose, INF, Point, \
	inverse_kinematics, pairwise_collision, remove_fixed_constraint, Attachment, get_sample_fn, \
	step_simulation, refine_path, plan_direct_joint_motion, get_joint_positions, dump_world, get_link_pose,control_joints

from CuriousSamplePlanner.planning_pybullet.pybullet_tools.kuka_primitives import BodyPath, Attach, Detach
import pickle
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import collections
from motion_planners.discrete import astar
import sys
from gym import spaces
from CuriousSamplePlanner.tasks.environment import Environment
from CuriousSamplePlanner.scripts.utils import *
from CuriousSamplePlanner.tasks.macroactions import PickPlace, AddLink, MacroAction
from CuriousSamplePlanner.tasks.state import State
from CuriousSamplePlanner.rl_ppo_rnd.a2c_ppo_acktr.model import Policy



class ThreeBlocks(Environment):
	def __init__(self, *args):
		super(ThreeBlocks, self).__init__(*args)
		connect(use_gui=False)
		self.INFEASIBLE = -1

		if(self.detailed_gmp):
			self.robot = p.loadURDF(DRAKE_IIWA_URDF, useFixedBase=True,  globalScaling=1.2) # KUKA_IIWA_URDF | DRAKE_IIWA_URDF
		else:
			self.robot = None

		self.floor = p.loadURDF('models/short_floor.urdf', useFixedBase=True)
		set_default_camera()
		self.green_block = p.loadURDF("models/box_green.urdf", useFixedBase=False)
		self.red_block = p.loadURDF("models/box_red.urdf", useFixedBase=False)
		self.blue_block = p.loadURDF("models/box_blue.urdf", useFixedBase=False)
		self.perspectives = [(0, -90)]

		self.objects = [self.green_block, self.red_block, self.blue_block]
		# Set up the state space and action space for this task
		self.break_on_timeout = True
		self.macroaction = MacroAction([
								PickPlace(objects = self.objects, robot=self.robot, fixed=self.fixed, gmp=self.detailed_gmp),
								# AddLink(objects = self.objects, robot=self.robot, fixed=self.fixed, gmp=self.detailed_gmp),
							])
		self.action_space_size = self.macroaction.action_space_size
		state = State(len(self.objects), len(self.macroaction.link_status))
		self.config_size =  state.config_size # (4 for links)
		print(self.config_size)
		self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space_size,))

		self.actor_critic = opt_cuda(Policy([self.config_size], self.action_space, base_kwargs={'recurrent': False}))

 
 
		# Select out positional degrees of freedom
		self.predict_mask = [0,1,2]+[6,7,8]+[12,13,14]
		p.setGravity(0, 0, -10, physicsClientId=0)

		# One step of the simulation to add everything to the scene
		p.stepSimulation(physicsClientId=0)


	@property
	def fixed(self):
		return [self.floor]

	def set_state(self, conf):
		i = 0
		for block in self.objects:
			set_pose(block, Pose(Point(x = conf[i], y = conf[i+1], z=conf[i+2]), Euler(roll = conf[i+3], pitch = conf[i+4], yaw=conf[i+5])))
			i+=6

	def check_goal_state(self, config):
		# collect the y values
		vals = [config[2], config[8], config[14]]
		vals.sort()

		# Two stack
		if( (vals[0] > 0.06) or (vals[1] > 0.06) or (vals[2] > 0.06)):
			return True

		# Three stack
		# if(vals[0]<0.06 and (vals[1] < 0.16 and vals[1] > 0.06) and (vals[2] > 0.16)):
		# 	return True
		return False


	def get_current_config(self):
		gpos, gquat = p.getBasePositionAndOrientation(self.green_block, physicsClientId=0)
		rpos, rquat  = p.getBasePositionAndOrientation(self.red_block, physicsClientId=0)
		ypos, yquat = p.getBasePositionAndOrientation(self.blue_block, physicsClientId=0)
		# Convert quat to euler
		geuler = p.getEulerFromQuaternion(gquat)
		reuler = p.getEulerFromQuaternion(rquat)
		yeuler = p.getEulerFromQuaternion(yquat)
		return np.concatenate([gpos, geuler, rpos, reuler, ypos, yeuler]+[self.macroaction.link_status])

	def get_start_state(self):
		collision = True
		z = stable_z(self.green_block, self.floor)
		while(collision):
			pos1, pos2, pos3 = self.reachable_pos(z=0), self.reachable_pos(z=0), self.reachable_pos(z=0)
			conf = np.array([pos1[0], pos1[1], z, 0, 0, 0, pos2[0], pos2[1], z, 0, 0, 0, pos3[0], pos3[1], z, 0, 0, 0]+self.macroaction.link_status)
			self.set_state(conf)
			collision = check_pairwise_collisions([self.green_block, self.red_block, self.green_block])
		return conf


