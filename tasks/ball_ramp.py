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
from CuriousSamplePlanner.tasks.environment import Environment
from CuriousSamplePlanner.scripts.utils import *
from CuriousSamplePlanner.tasks.macroactions import PickPlace, AddLink, MacroAction
from CuriousSamplePlanner.rl_ppo_rnd.a2c_ppo_acktr.model import Policy
from gym import spaces
from CuriousSamplePlanner.tasks.state import State


class BallRamp(Environment):
	def __init__(self, *args):
		super(BallRamp, self).__init__(*args)  

		connect(use_gui=False)
		if(self.detailed_gmp):
			self.robot = p.loadURDF(DRAKE_IIWA_URDF, useFixedBase=True,  globalScaling=1) # KUKA_IIWA_URDF | DRAKE_IIWA_URDF
		else:
			self.robot = None

		set_default_camera()
		self.blue_ball = p.loadURDF("models/blue_ball.urdf", useFixedBase=False)
		self.red_block = p.loadURDF("models/box_red.urdf", useFixedBase=False)
		self.green_block = p.loadURDF("models/box_green.urdf", useFixedBase=False)
		self.purple_rod = p.loadURDF("models/purple_rod.urdf", useFixedBase=False)
		self.floor = p.loadURDF('models/short_floor.urdf', useFixedBase=True)
		
		self.objects = [self.blue_ball, self.red_block, self.green_block, self.purple_rod]
		self.static_objects = []

		self.perspectives = [(0, -90)]
		
		self.break_on_timeout = True
		self.macroaction = MacroAction([
								PickPlace(objects = self.objects, robot=self.robot, fixed=self.fixed, gmp=self.detailed_gmp),
								# AddLink(objects = self.objects, robot=self.robot, fixed=self.fixed, gmp=self.detailed_gmp),
							])

		# Config state attributes
		self.config_state_attrs()

		p.setGravity(0, 0, -10)
		p.stepSimulation(physicsClientId=0)

	@property
	def fixed(self):
		return [self.floor]

	def check_goal_state(self, config):
		# collect the y values
		ball_distance = abs(math.sqrt(config[0]**2+config[1]**2))
		if(ball_distance>=1):
			return True
		return False

	def set_state(self, conf):
		i = 0
		for block in self.objects:
			set_pose(block, Pose(Point(x = conf[i], y = conf[i+1], z=conf[i+2]), Euler(roll = conf[i+3], pitch = conf[i+4], yaw=conf[i+5])))
			i+=6

	def get_current_config(self):
		bpos, bquat = p.getBasePositionAndOrientation(self.blue_ball)
		rpos, rquat  = p.getBasePositionAndOrientation(self.red_block)
		gpos, gquat = p.getBasePositionAndOrientation(self.green_block)
		ppos, pquat = p.getBasePositionAndOrientation(self.purple_rod)
		beuler = p.getEulerFromQuaternion(bquat)
		reuler = p.getEulerFromQuaternion(rquat)
		geuler = p.getEulerFromQuaternion(gquat)
		peuler = p.getEulerFromQuaternion(pquat)
		return np.concatenate([bpos, beuler, rpos, reuler, gpos, geuler, ppos, peuler])

	def get_start_state(self):
		collision = True
		while(collision):
			z = stable_z(self.green_block, self.floor)
			pos1, pos2, pos3, pos4 = self.reachable_pos(z=0), self.reachable_pos(z=0), self.reachable_pos(z=0), self.reachable_pos(z=0)
			state = State(len(self.objects), len(self.static_objects), len(self.macroaction.link_status))
			state.set_position(0, pos1[0], pos1[1], z)
			state.set_position(1, pos2[0], pos2[1], z)
			state.set_position(2, pos3[0], pos3[1], z)
			state.set_position(3, pos4[0], pos4[1], z)
			self.set_state(state.config)
			collision = check_pairwise_collisions(self.objects)
		return state.config
