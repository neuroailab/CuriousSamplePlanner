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
	BLOCK_URDF, load_model, wait_for_user, disconnect, DRAKE_IIWA_URDF, user_input, update_state, disable_real_time, inverse_kinematics,end_effector_from_body,approach_from_grasp, get_joints, get_joint_positions

from CuriousSamplePlanner.planning_pybullet.pybullet_tools.utils import multiply, invert, get_pose, set_pose, get_movable_joints, \
	set_joint_positions, add_fixed_constraint_2, enable_real_time, disable_real_time, joint_controller, \
	enable_gravity, get_refine_fn, user_input, wait_for_duration, link_from_name, get_body_name, sample_placement, \
	end_effector_from_body, approach_from_grasp, plan_joint_motion, GraspInfo, Pose, INF, Point, \
	inverse_kinematics, pairwise_collision, remove_fixed_constraint, Attachment, get_sample_fn, \
	step_simulation, refine_path, plan_direct_joint_motion, get_joint_positions, dump_world, get_link_pose,control_joints

from CuriousSamplePlanner.planning_pybullet.pybullet_tools.kuka_primitives import BodyPath, Attach, Detach, ApplyForce
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

class MacroAction():
	def __init__(self, macroaction_list=[], *args):
		self.reachable_max_height = 0.8
		self.max_reach_horiz = 0.45
		self.min_reach_horiz = 0.4
		self.macroaction_list = macroaction_list
		self.link_status = []
		self.links = []
		for macroaction in self.macroaction_list:
			self.link_status += [0 for _ in range(macroaction.num_links)]
			self.links += [None for _ in range(macroaction.num_links)]
			self.objects = macroaction.objects

	def add_arm(self, arm):
		self.robot = arm
		for m in self.macroaction_list:
			m.robot = arm

	@property
	def action_space_size(self):
		return sum([macro.num_selectors+macro.num_params for macro in self.macroaction_list])

	def reparameterize(self, block_to_move, pos):
		r = reparameterize(pos[1].item(), self.min_reach_horiz, self.max_reach_horiz)
		height = reparameterize(pos[2].item(), 0.1, self.reachable_max_height)
		theta = reparameterize(pos[0].item(), -math.pi, math.pi)
		yaw = reparameterize(pos[3].item(), -math.pi, math.pi)
		_, orig_quat = p.getBasePositionAndOrientation(block_to_move, physicsClientId=0)
		orig_euler = p.getEulerFromQuaternion(orig_quat)
		teleport_pose = Pose(Point(x = r*math.cos(theta), y = r*math.sin(theta), z=min(height, 1)), Euler(roll=orig_euler[0], pitch=orig_euler[1], yaw=yaw))
		return teleport_pose

	# def reparameterize(self, block_to_move, pos):
	# 	_, orig_quat = p.getBasePositionAndOrientation(block_to_move, physicsClientId=0)
	# 	height = reparameterize(pos[2].item(), 0.1, self.reachable_max_height)

	# 	orig_euler = p.getEulerFromQuaternion(orig_quat)
	# 	teleport_pose = Pose(Point(x = pos[0], y = pos[1], z=height), Euler(roll=orig_euler[0], pitch=orig_euler[1], yaw=pos[3]))
	# 	return teleport_pose			


	def object_unreachable(self, obj):
		margin = 0.1
		obj_pose = get_pose(obj)
		obj_pos = obj_pose[0]
		distance = math.sqrt(obj_pos[0]**2+obj_pos[1]**2)
		if(distance>(self.max_reach_horiz+margin) or distance<(self.min_reach_horiz-margin)):
			return True
		return False

	def execute(self, embedding, config, sim=False):

		"""
			Output: (feasible, planning command)

		"""
		total_selectors = sum([self.macroaction_list[macro_idx].num_selectors for macro_idx in range(len(self.macroaction_list))])
		max_element = np.argmax(embedding[0:total_selectors])
		prev = 0
		for ma_index, ma in enumerate(self.macroaction_list):
			if(max_element >= prev and max_element < ma.num_selectors + prev ):
				macroaction_index = ma_index
			prev += ma.num_selectors


		# Select out the nodes of the network responsible for that macroaction

		if(isinstance(self.macroaction_list[macroaction_index], PickPlace)):
			# Need to do some extra preprocessing to account for links
			# If there is a link, we need to transport objects at the same time with the same dynamics to avoid explosion
			# First, get the block to move
			object_index = np.argmax(embedding[0:len(self.objects)], axis=0)
			block_to_move = self.objects[int(object_index)]

			# Then, get all of the blocks connected to the block_to_move
			connected_blocks = []
			num_blocks = int(math.sqrt(len(self.link_status)))
			for block_index in range(num_blocks):
				if(block_index != block_to_move and self.link_status[int(object_index)*num_blocks+block_index]):
					connected_blocks.append(self.objects[block_index])

			# Then we need to get the goal pose of the moving object
			start_index = len(self.objects)
			end_index = start_index+self.macroaction_list[macroaction_index].object_params[object_index]
			pos = embedding[start_index: end_index]
			teleport_pose = self.reparameterize(block_to_move, pos)

			# Now we need to perform forward dynamics on the connected objects
			for connected_block in connected_blocks:
				b1pos, b1quat = p.getBasePositionAndOrientation(block_to_move)
				b1e = p.getEulerFromQuaternion(b1quat)
				b2pos, b2quat = p.getBasePositionAndOrientation(connected_block)
				b2e = p.getEulerFromQuaternion(b2quat)
				start_pose = Pose(Point(*b1pos), Euler(*b1e))
				other_pose = Pose(Point(*b2pos), Euler(*b2e))
				collision=False
				index = 0
				other_goal_pose = multiply(teleport_pose, multiply(invert(start_pose), other_pose))
				set_pose(connected_block, other_goal_pose)

			(feasible, planning_commands, _) = self.macroaction_list[macroaction_index].execute(block_to_move, teleport_pose, sim)
			return (feasible, planning_commands)

		elif(isinstance(self.macroaction_list[macroaction_index], AddLink)):
			mask_start = sum([self.macroaction_list[macro_idx].num_selectors for macro_idx in range(macroaction_index)])
			mask_end = mask_start+self.macroaction_list[macroaction_index].num_selectors
			(feasible, planning_commands, link_status) = self.macroaction_list[macroaction_index].execute(embedding[mask_start:mask_end], self.link_status, sim)
			if(link_status is not None):
				self.link_status = link_status

			return (feasible, planning_commands)

class PickPlace(MacroAction):
	def __init__(self, objects=[], robot=None, fixed = [], gmp=False, *args):
		self.gmp = gmp
		self.fixed = fixed
		self.objects = objects 
		self.robot = robot
		self.teleport = True
		# Three positional and 1 rotational degree of freedom
		self.object_params = [4 for _ in range(len(self.objects))]
		super(PickPlace, self).__init__(*args)


	@property
	def num_links(self):
		return 0
	
	@property
	def num_params(self):
		return 4

	@property
	def num_selectors(self):
		return len(self.objects)


	def feasibility_check(self, block_to_move, goal_pose, sim=False):
		"""
			Output: (planning steps, feasible)
		"""

		# Rough feasibility check
		# pos, _ = p.getBasePositionAndOrientation(block_to_move, physicsClientId=0)
		# if(math.sqrt((pos[0]**2+pos[1]**2)) > self.reachable_max_height+0.15):
		# 	return (None, False)

		# # Quick bookend collision checking
		# notcs = [i for i in self.objects if i not in [block_to_move]]
		# saved_world = WorldSaver()
		# for notc in notcs:
		# 	contact = p.getClosestPoints(bodyA=block_to_move, bodyB=notc, distance=0, physicsClientId=0)
		# 	if(len(contact) > 0):
		# 		if(contact[0][5][2]>pos[2]):
		# 			return (None, False)

		# set_pose(block_to_move, goal_pose)
		# for notc in notcs:
		# 	contact = p.getClosestPoints(bodyA=block_to_move, bodyB=notc, distance=0, physicsClientId=0)
		# 	if(len(contact)>0):
		# 		return (None, False)
		# saved_world.restore()

		if(not self.gmp and not sim):
			return (None, True)

		saved_world = WorldSaver()
		# Approach from different angles
		ik_fn = get_ik_fn(self.robot, fixed=self.fixed, teleport=self.teleport) # These are functions which generate sequences of actions
		free_motion_fn = get_free_motion_gen(self.robot, fixed=([block_to_move] + self.fixed), teleport=self.teleport)
		holding_motion_fn = get_holding_motion_gen(self.robot, fixed=self.fixed, teleport=self.teleport)
		block1_pose0 = BodyPose(block_to_move)
		conf0 = BodyConf(self.robot)
		for grasp_gen in [get_grasp_gen(self.robot, direction) for direction in ['top', 'side']]: # Can put in other angles here, leaving as top for now
			for (grasp,) in grasp_gen(block_to_move):
				grasp2 = grasp
				saved_world.restore()
				result1 = ik_fn(block_to_move, block1_pose0, grasp)
				if result1 is None:
					continue
				conf1, path2 = result1
				gripper_pose = end_effector_from_body(goal_pose, grasp2.grasp_pose)
				grasp2.approach_pose = Pose(0.1*Point(z=1))
				approach_pose = approach_from_grasp(grasp2.approach_pose, gripper_pose)
				movable_joints = get_movable_joints(self.robot)
				starting_joint_positions = get_joint_positions(self.robot, movable_joints)

				for i in range(5): # Infeasible if it cannot reach in 5 tries
					sample_fn = get_sample_fn(self.robot, movable_joints)
					set_joint_positions(self.robot, movable_joints, sample_fn())
					q_approach = inverse_kinematics(self.robot, grasp2.link, approach_pose)
					if(q_approach is not None):
						break

				block2_conf = BodyConf(self.robot, q_approach)
				block1_pose0.assign()
				saved_world.restore()
				result2 = free_motion_fn(conf0, conf1)
				if result2 is None:
					continue
				path1, = result2
				result3 = holding_motion_fn(conf1, block2_conf, block_to_move, grasp)
				if result3 is None:
					continue
				path3, = result3
				path4 = Command([Detach(block_to_move, self.robot, grasp2.link)])
				return (Command(path1.body_paths +
								  path2.body_paths +
								  path3.body_paths +
								  path4.body_paths), True)

		return (None, False)

	def execute(self, block_to_move, teleport_pose, sim=False):
		"""
			Output: (feasible, planning commands, auxiliary output)
		"""
		# Get the selected position
		(feas_command, feasible) = self.feasibility_check(block_to_move, teleport_pose, sim)
		if(feasible == True):
			set_pose(block_to_move, teleport_pose)
			return (True, feas_command, None)
		else:
			return (False, None, None)

class Link(ApplyForce):
	def __init__(self, link_point1):
		self.sphere = None
		self.link_point1 = link_point1

	def iterator(self):
		self.sphere = p.loadURDF("./models/small_yellow_ball.urdf", self.link_point1)
		return []

class AddLink(MacroAction):
	def __init__(self, objects, robot=None, fixed=[], gmp=False, *args):
		self.objects = objects

		self.gmp = gmp
		self.fixed = fixed
		self.robot = robot
		self.teleport = True
		# Three positional and 1 rotational degree of freedom

		self.links = [None for i in range(len(objects)**2)]
		super(AddLink, self).__init__(*args)

	
	def feasibility_check(self, block1, block2, sim=False):
		"""
			Output: (planning steps, feasible)
		"""
		if(block1 != block2 and check_state_collision(block1, block2)):
			"""
				This function plans a path for the object to reach the goal position
			"""

			# Rough feasibility check
			if(not self.gmp and not sim):
				return (None, True)

			close_points = p.getClosestPoints(bodyA=block1, bodyB=block2, distance=0.01, physicsClientId=0)
			link_point1 = close_points[0][5]
			bpos, bquat = p.getBasePositionAndOrientation(block1)
			be = p.getEulerFromQuaternion(bquat)
			poseb = Pose(Point(*bpos), Euler(*be))
			local_link_transform = multiply(invert(poseb), Pose(Point(*link_point1)))
			objobjlink = add_fixed_constraint_2(block1, block2)
			guide_cube = p.loadURDF("./models/guide_cube.urdf", link_point1)
			link_object = Link(link_point1)


			grasp_gen = get_grasp_gen(self.robot, 'top')
			ik_fn = get_ik_fn(self.robot, fixed=self.fixed, teleport=self.teleport) # These are functions which generate sequences of actions
			free_motion_fn = get_free_motion_gen(self.robot, fixed=([] + self.fixed), teleport=self.teleport)
			holding_motion_fn = get_holding_motion_gen(self.robot, fixed=self.fixed, teleport=self.teleport)

			block1_pose0 = BodyPose(guide_cube)
			conf0 = BodyConf(self.robot)
			saved_world = WorldSaver()
			grasp, = next(grasp_gen(guide_cube))

			saved_world.restore()
			result1 = ik_fn(guide_cube, block1_pose0, grasp)
			if result1 is None:
				return (None, False)

			conf1, path2 = result1
			goal_pose = block1_pose0.pose
			gripper_pose = end_effector_from_body(goal_pose, grasp.grasp_pose)
			grasp.approach_pose = Pose(0.1*Point(z=1))
			approach_pose = approach_from_grasp(grasp.approach_pose, gripper_pose)

			movable_joints = get_movable_joints(self.robot)
			starting_joint_positions = get_joint_positions(self.robot, movable_joints)
			for i in range(1000):
				sample_fn = get_sample_fn(self.robot, movable_joints)
				set_joint_positions(self.robot, movable_joints, sample_fn())
				q_approach = inverse_kinematics(self.robot, grasp.link, approach_pose)
				if(q_approach is not None):
					break
			block2_conf = BodyConf(self.robot, q_approach)
			result2 = free_motion_fn(conf0, conf1)
			if result2 is None:
				return (None, False)
			path1, = result2
			return ([link_object, block1, local_link_transform, objobjlink], Command(path1.body_paths+[link_object]))
		
		else:
			return (None, False)

	@property
	def total_objects(self):
		return self.objects+self.fixed

	@property
	def num_links(self):
		return (len(self.total_objects))**2
	@property
	def num_params(self):
		return 0 # Pairwise object groupings
	@property
	def num_selectors(self):
		return (len(self.total_objects))**2 # Pairwise object groupings


	
	def execute(self, embedding, link_status, sim=False):
		"""
			Output: (feasible, planning commands, auxiliary output)
			Note planning commands
		"""
		object_pair_index = np.argmax(embedding, axis=0)
		object1_index = int(float(object_pair_index)/len(self.total_objects))
		object2_index = int(float(object_pair_index)%len(self.total_objects))
		# Just check to make sure I have this calculation right
		assert object1_index*len(self.total_objects)+object2_index == object_pair_index or object2_index*len(self.total_objects)+object1_index == object_pair_index
		# Not already linked
		(feas_command, feasible) = self.feasibility_check(self.total_objects[object1_index], self.total_objects[object2_index], sim)
		if(feasible == True):
			if(link_status[object_pair_index] == 1):
				# Already linked, must unlink
				# Matrix should by symmetric
				link_status[object1_index*len(self.total_objects)+object2_index] = 0
				link_status[object2_index*len(self.total_objects)+object1_index] = 0
				return (True, feas_command, link_status)

			else:
				# Matrix should by symmetric
				link_status[object1_index*len(self.total_objects)+object2_index] = 1
				link_status[object2_index*len(self.total_objects)+object1_index] = 1
				return (True, feas_command, link_status)
		else:
			return (False, None, None)


