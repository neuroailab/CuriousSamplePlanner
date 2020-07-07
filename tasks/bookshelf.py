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
from CuriousSamplePlanner.planning_pybullet.pybullet_tools.utils import multiply, invert, get_pose, set_pose, get_movable_joints, \
	set_joint_positions, add_fixed_constraint_2, enable_real_time, disable_real_time, joint_controller, \
	enable_gravity, get_refine_fn, user_input, wait_for_duration, link_from_name, get_body_name, sample_placement, \
	end_effector_from_body, approach_from_grasp, plan_joint_motion, GraspInfo, Pose, INF, Point, \
	inverse_kinematics, pairwise_collision, remove_fixed_constraint, Attachment, get_sample_fn, \
	step_simulation, refine_path, plan_direct_joint_motion, get_joint_positions, dump_world, get_link_pose,control_joints, interpolate_poses

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


class BookShelf(Environment):
	def __init__(self, *args):
		super(BookShelf, self).__init__(*args)
		connect(use_gui=self.experiment_dict['render'])

		self.arm_size = 1
		if(self.detailed_gmp):
			self.robot = p.loadURDF(DRAKE_IIWA_URDF, useFixedBase=True,  globalScaling=1) # KUKA_IIWA_URDF | DRAKE_IIWA_URDF
		else:
			self.robot = None

		self.current_constraint_id = None
		self.default_links = [0]
		self.y_offset = 0.5
		self.floor = p.loadURDF('models/short_floor.urdf', useFixedBase=True)
		self.ARM_LEN = 0.8
		set_default_camera()
		
		self.shelf = p.loadURDF("models/kiva_shelf/bookcase.urdf", globalScaling=3.0, useFixedBase=True)
		set_pose(self.shelf, Pose(Point(x= -0.2 , y=-1.6, z=-0.010), Euler(yaw=0, pitch=0, roll=math.pi/2)))

		self.book = p.loadURDF("models/book/book.urdf", globalScaling=0.15, useFixedBase=False)
		self.book_pos = [0.06, -1.43, 0.488]
		self.book_rot = [math.pi/2, 0, 0]
		set_pose(self.book, Pose(Point(x=self.book_pos[0], y=self.book_pos[1], z=self.book_pos[2]), Euler(roll=self.book_rot[0], pitch=self.book_rot[1], yaw=self.book_rot[2])))

		self.blue_rod_1 = p.loadURDF("models/blue_rod.urdf", globalScaling=2.0, useFixedBase=False)
		set_pose(self.blue_rod_1, Pose(Point(x=0.7, y=0, z=1.2), Euler(yaw=0, pitch=0, roll=0)))

		self.blue_rod_2 = p.loadURDF("models/blue_rod.urdf", globalScaling=2.0, useFixedBase=False)
		set_pose(self.blue_rod_2, Pose(Point(x=-0.7, y=0, z=1.2), Euler(yaw=0, pitch=0, roll=0)))

		self.objects = [self.blue_rod_1, self.blue_rod_2]
		self.static_objects = [self.book]

		self.perspectives = [(0, -90)]
		self.break_on_timeout = False

		self.macroaction = MacroAction(macroaction_list = [
								PickPlace(objects = self.objects, robot = self.robot, fixed = self.fixed, gmp = self.detailed_gmp),
								AddLink(objects = self.objects, robot = self.robot, fixed = self.fixed, gmp = self.detailed_gmp),
							], experiment_dict = self.experiment_dict)

		# Config state attributes
		self.config_state_attrs(linking=True)

		p.setGravity(0, 0, -10, physicsClientId=0)

		self.break_on_timeout = True

		self.actor_critic = opt_cuda(Policy([self.config_size], self.action_space, base_kwargs={'recurrent': False}))



	def check_goal_state(self, conf):
		if(conf[14]<0.3):
			return True
		return False
		

	def set_state(self, conf):

		self.remove_constraints()
		set_pose(self.blue_rod_1, Pose(Point(x = conf[0], y = conf[1], z=conf[2]), 
									   Euler(roll=conf[3], pitch=conf[4], yaw=conf[5])))
		set_pose(self.blue_rod_2, Pose(Point(x = conf[6], y = conf[7], z=conf[8]), 
									   Euler(roll=conf[9], pitch=conf[10], yaw=conf[11])))
		set_pose(self.book, Pose(Point(x = self.book_pos[0], y =self.book_pos[1], z=self.book_pos[2]), 
								 Euler(roll=self.book_rot[0], pitch=self.book_rot[1], yaw=self.book_rot[2])))
		self.add_constraints(conf)
		time.sleep(0.001)


	def get_macroaction_params(self, obj, conf):
		if(int(obj) == 0):
			return (self.blue_rod_1, Pose(Point(x = conf[0], y = conf[1], z=conf[2]), 
									   Euler(roll=conf[3], pitch=conf[4], yaw=conf[5])))
		elif(int(obj) == 1):
			return (self.blue_rod_2, Pose(Point(x = conf[6], y = conf[7], z=conf[8]), 
									   Euler(roll=conf[9], pitch=conf[10], yaw=conf[11])))
		else:
			return [self.blue_rod_1, self.blue_rod_2]


	def get_current_config(self):
		b1pos, b1quat = p.getBasePositionAndOrientation(self.blue_rod_1)
		b2pos, b2quat = p.getBasePositionAndOrientation(self.blue_rod_2)
		bookpos, bookquat = p.getBasePositionAndOrientation(self.book)

		# Reduce configuration redundancy/complexity
		b1e = p.getEulerFromQuaternion(b1quat)
		b2e = p.getEulerFromQuaternion(b2quat)
		booke = p.getEulerFromQuaternion(bookquat)

		returning_state =  np.array(list(b1pos+b1e+b2pos+b2e+bookpos+booke)+self.macroaction.link_status)
		return returning_state

	@property
	def fixed(self):
		return [self.floor, self.shelf]
  
	def get_start_state(self):
		collision = True
		z = 0.02
		while(collision):
			poses = [self.macroaction.reparameterize(self.objects[0], np.random.uniform(low=-1, high=1, size=4)) for _ in range(2)]
			pos1, pos2 = [pose[0] for pose in poses]
			state = State(len(self.objects), len(self.static_objects), len(self.macroaction.link_status))
			state.set_position(0, pos1[0], pos1[1], z)
			state.set_position(1, pos2[0], pos2[1], z)
			self.set_state(state.config)
			collision = check_pairwise_collisions(self.objects)

		return state.config 
		  

