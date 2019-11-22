
#!/usr/bin/env python
from __future__ import print_function
import pybullet as p
import numpy as np
import time
import random
import math
import imageio
import os.path as osp
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


from planning_pybullet.pybullet_tools.utils import multiply, invert, get_pose, set_pose, get_movable_joints, \
    set_joint_positions, add_fixed_constraint_2, enable_real_time, disable_real_time, joint_controller, \
    enable_gravity, get_refine_fn, user_input, wait_for_duration, link_from_name, get_body_name, sample_placement, \
    end_effector_from_body, approach_from_grasp, plan_joint_motion, GraspInfo, Pose, INF, Point, \
    inverse_kinematics, pairwise_collision, remove_fixed_constraint, Attachment, get_sample_fn, \
    step_simulation, refine_path, plan_direct_joint_motion, get_joint_positions, dump_world, get_link_pose,control_joints

from planning_pybullet.pybullet_tools.kuka_primitives import BodyPath, Attach, Detach, ApplyForce
import pickle
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import collections
from planning_pybullet.motion.motion_planners.discrete import astar
import sys

from tasks.three_block_stack import ThreeBlocks

from tasks.ball_ramp import BallRamp
from tasks.pulley import PulleySeesaw
from tasks.bookshelf import BookShelf
from tasks.five_block_stack import FiveBlocks
from scripts.utils import *

from trainers.plan_graph import PlanGraph
from trainers.dataset import ExperienceReplayBuffer
import copy


class Link(ApplyForce):

    def __init__(self, link_point1):
        self.sphere = None
        self.link_point1 = link_point1

    def iterator(self):
        self.sphere = p.loadURDF("./models/yellow_sphere.urdf", self.link_point1)
        return []

class PlanningAgent:

    def __init__(self, environment, out):
        self.environment = environment
        self.out_path = out
        self.links = []
        if self.environment.robot != None:
            self.robot = self.environment.robot
        else:
            self.add_arm(self.environment.arm_size)

    def multistep_plan(self, plan):

        current_config = plan[0].config
        self.environment.set_state(current_config)
        start_world = WorldSaver()
        commands = []
        for i in range(1, len(plan)):
            if plan[i].command != None:
                commands.append(plan[i].command)
                self.execute(plan[i].command)
            else:
                for perspective in self.environment.perspectives:
                    imageio.imwrite(osp.join(self.out_path, '{}.jpg'.format(i)),
                                    take_picture(perspective[0], perspective[1], 0, size=512))

                macroaction = self.environment.macroaction
                macroaction.add_arm(self.robot)
                macroaction.gmp = True
                macroaction.teleport = False
                # command, aux = macroaction.execute(config = self.environment.get_current_config(), embedding = plan[i].action, sim=True)
                feasible, command = macroaction.execute(config=self.environment.get_current_config(), embedding=plan[i].action, sim=True)
                # if aux != None:
                #     self.links.append(aux)
                # Restore the state
                if command is not None:
                    self.execute(command)
                    self.environment.run_until_stable(hook=self.hook, dt=0.01)
                commands.append(command)

        # # Remove all the links
        # for (link_object, link, link_transform, objobjlink) in self.links:
        #     p.removeBody(link_object.sphere)
        #     p.removeConstraint(link)
        #     p.removeConstraint(objobjlink)

        # self.links = []
        # self.environment.set_state(plan[0].config)
        # for command in commands:
        #     self.execute(command)
        #     self.environment.run_until_stable()

        for i in range(1000):
            p.stepSimulation()
            self.hook()
            time.sleep(0.01)



    def hook(self):
        for (link_object, link, link_transform, _) in self.links:
            if link_object.sphere != None:
                lpos, lquat = p.getBasePositionAndOrientation(link)
                le = p.getEulerFromQuaternion(lquat)
                lpose = Pose(Point(*lpos), Euler(*le))
                set_pose(link_object.sphere, multiply(lpose, link_transform))

    def execute(self, command):
        command.refine(num_steps=10).execute(time_step=0.001, hook=self.hook)

    def hide_arm(self):
        p.removeBody(self.robot)

    def add_arm(self, arm_size):
        print(arm_size)
        arm_size=1.1
        self.robot = p.loadURDF(DRAKE_IIWA_URDF, useFixedBase=True, globalScaling=arm_size) # KUKA_IIWA_URDF | DRAKE_IIWA_URDF




