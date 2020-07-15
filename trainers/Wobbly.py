#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import time
import imageio
import pickle
import torch
import sys

from CuriousSamplePlanner.scripts.utils import *
from CuriousSamplePlanner.trainers.plan_graph import PlanGraph
from CuriousSamplePlanner.trainers.dataset import ExperienceReplayBuffer
from CuriousSamplePlanner.trainers.planner import Planner

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
from CuriousSamplePlanner.scripts.utils import *


class Wobbly(Planner):
	def __init__(self, experiment_dict):

		super(Wobbly, self).__init__(experiment_dict)

		# Create the replay buffer for training world models
		self.experience_replay = ExperienceReplayBuffer()

	def print_exp_dict(self, verbose=False):
		mean_reward = sum(self.experiment_dict['rewards'][-128:])/len(self.experiment_dict['rewards'][-128:])
		print("Sampled: "+str(self.experiment_dict['num_sampled_nodes'])+"\t Graph Size: "+str(self.experiment_dict['num_graph_nodes'])+"\t WM Loss: "+str(self.experiment_dict['world_model_losses'][-1])+"\t Feasibility: "+str(self.experiment_dict['feasibility'][-1])+"\t Reward: "+str(mean_reward))
		# print("Sampled: "+str(self.experiment_dict['num_sampled_nodes'])+"\t Graph Size: "+str(self.experiment_dict['num_graph_nodes'])+"\t WM Loss: "+str(self.experiment_dict['world_model_losses'][-1]))
	
	def save_params(self):
		# Save the updated models
		# with open(self.exp_path + "/worldModel.pkl", 'wb') as fw:
		# 	pickle.dump(self.worldModel, fw)
		# with open(self.exp_path + "/actor_critic.pkl", "wb") as fa:
		# 	pickle.dump(self.environment.actor_critic, fa)
		
		with open(self.experiment_dict['exp_path'] + "/exp_dict.pkl", "wb") as fa:
			pickle.dump(self.experiment_dict, fa)
			fa.close()

		graph_filehandler = open(self.experiment_dict['exp_path'] + "/found_graph.pkl", 'wb')
		pickle.dump(self.graph, graph_filehandler)

	def plan(self):
		self.environment.reset()
		while(True):
			conf0 = BodyConf(self.environment.robot)
			conf1 = BodyConf(self.environment.robot, configuration = conf0.configuration+np.random.uniform(-0.1, 0.1, len(conf0.configuration)))# TODO Generate random configuration
			free_motion_fn = get_free_motion_gen(self.environment.robot, fixed=[], teleport=True)
			result2, = free_motion_fn(conf0, conf1)
			path_command = Command(result2.body_paths)
			path_command.refine(num_steps=100).execute()






