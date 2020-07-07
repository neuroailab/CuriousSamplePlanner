# The robot is not able to reach certain points and its reachability is not a simple geometric shape. 
# Here I analyze where it can reach by trying to reach different points for each z value and seeing the appropriate range

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
from planning_pybullet.pybullet_tools.kuka_primitives import BodyPose, BodyConf, Command, get_grasp_gen, \
    get_ik_fn, get_free_motion_gen, get_holding_motion_gen
from planning_pybullet.pybullet_tools.utils import WorldSaver, enable_gravity, connect, dump_world, set_pose, \
    Pose, Point, Euler, set_default_camera, stable_z, \
    BLOCK_URDF, load_model, wait_for_user, disconnect, DRAKE_IIWA_URDF, user_input, update_state, disable_real_time,inverse_kinematics,end_effector_from_body,approach_from_grasp, get_joints, get_joint_positions


from planning_pybullet.pybullet_tools.utils import get_pose, set_pose, get_movable_joints, \
    set_joint_positions, add_fixed_constraint_2, enable_real_time, disable_real_time, joint_controller, \
    enable_gravity, get_refine_fn, user_input, wait_for_duration, link_from_name, get_body_name, sample_placement, \
    end_effector_from_body, approach_from_grasp, plan_joint_motion, GraspInfo, Pose, INF, Point, \
    inverse_kinematics, pairwise_collision, remove_fixed_constraint, Attachment, get_sample_fn, \
    step_simulation, refine_path, plan_direct_joint_motion, get_joint_positions, dump_world, get_link_pose,control_joints

from planning_pybullet.pybullet_tools.kuka_primitives import BodyPath, Attach, Detach
import pickle
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import collections
from motion_planners.discrete import astar
import sys

from CuriousSamplePlanner.tasks.three_block_stack_AC import ThreeBlocks
from CuriousSamplePlanner.tasks.ball_ramp import BallRamp
sfrom CuriousSamplePlanner.tasks.pulley import PulleySeesaw
from CuriousSamplePlanner.tasks.bookshelf import BookShelf
from CuriousSamplePlanner.tasks.five_block_stack_AC import FiveBlocks
from CuriousSamplePlanner.scripts.utils import *

from CuriousSamplePlanner.trainers.plan_graph import PlanGraph
from CuriousSamplePlanner.trainers.dataset import ExperienceReplayBuffer

# First, spawn the robot in with fixed base
connect(use_gui=True)

floor = p.loadURDF('models/short_floor.urdf', useFixedBase=True)

robot = p.loadURDF(DRAKE_IIWA_URDF, useFixedBase=True, globalScaling=1) # KUKA_IIWA_URDF | DRAKE_IIWA_URDF
obj = p.loadURDF("models/box_green.urdf")

# 0.1 : 0.42-0.81
# 0.2 : 0.40-0.81
# 0.3 : 0.35-0.79
# 0.4 : 0.27-0.75
# 0.5 : 0.10-0.70
# 0.6 : 0.10-0.62
# 0.7 : 0.29-0.40

for i in range(0, 100):
	grasp_gen = get_grasp_gen(robot, 'top')
	grasp, = next(grasp_gen(obj))
	goal_pose = Pose(Point(x=i/100.0, y=0, z=0))
	gripper_pose = end_effector_from_body(goal_pose, grasp.grasp_pose)
	grasp.approach_pose = Pose(0.1*Point(z=1))
	approach_pose = approach_from_grasp(grasp.approach_pose, gripper_pose)
	q_approach = inverse_kinematics(robot, grasp.link, approach_pose)
	print(str(i/100.0)+": "+str(q_approach))








