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
    set_joint_positions, add_fixed_constraint, add_fixed_constraint_2, enable_real_time, disable_real_time, joint_controller, \
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
from CuriousSamplePlanner.tasks.state import State
from CuriousSamplePlanner.scripts.utils import *
from gym import spaces
from CuriousSamplePlanner.rl_ppo_rnd.a2c_ppo_acktr.model import Policy

class Spec():
    def __init__(self, id ):
        self.id = id

class Environment():
    def __init__(self, experiment_dict):
        self.dt = 0.01
        self.nsamples_per_update = experiment_dict['nsamples_per_update']
        self.detailed_gmp = experiment_dict['detailed_gmp']
        self.training = experiment_dict['training']
        if(experiment_dict['mode'] == "EffectPredictionPlanner" or experiment_dict['mode'] == "RandomStateEmbeddingPlanner"):
            self.image_based = False
        else:
            self.image_based = True

        self.arm_size=1

    def config_state_attrs(self, linking=False):
        self.state = State(len(self.objects), len(self.static_objects),len(self.macroaction.link_status))
        self.action_space_size = self.macroaction.action_space_size
        self.config_size =  self.state.config_size
        self.reward_range = [0, 1]
        self.metadata=None
        self.spec = Spec(0)

        self.observation_space = spaces.Box(low=-1, high=1, shape=( self.config_size,))
        
        if(not linking):
            self.predict_mask = self.state.positions
        else:
            self.predict_mask = self.state.positions+self.state.links

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space_size,))
        self.actor_critic = opt_cuda(Policy([self.config_size], self.action_space, base_kwargs={'recurrent': False}))


    def get_random_config(self):
        sample_config = []
        for obj in self.objects:
            random_vector=opt_cuda(torch.tensor(np.random.uniform(low=-1, high=1, size=self.action_space_size))).type(torch.FloatTensor)
            random_state=self.macroaction.reparameterize(obj, random_vector)
            pos, quat=random_state
            euler=p.getEulerFromQuaternion(quat)
            sample_config+=list(pos)+list(euler)
        for obj in self.static_objects:
            pos, quat = p.getBasePositionAndOrientation(obj)
            euler = p.getEulerFromQuaternion(quat)
            sample_config+=list(pos)+list(euler)
        for link in range(len(self.macroaction.link_status)):
            link_status = random.choice([0, 1])
            sample_config.append(link_status)

        return sample_config


    def take_action(self, action):
        # Get the macroaction that is being executed
        action = action
        config = self.get_current_config()
        feasible, command =  self.macroaction.execute(action, config)
        return (action, int(feasible), command)


    def remove_constraints(self):
        if(len(self.macroaction.link_status)>0):
            for obj1_index in range(len(self.objects)):
                for obj2_index in range(obj1_index, len(self.objects)):
                    link_index = obj1_index*len(self.objects)+obj2_index
                    if(self.macroaction.links[link_index] is not None):
                        p.removeConstraint(self.macroaction.links[link_index])
                        self.macroaction.links[link_index] = None

    def add_constraints(self, conf):
        if(len(self.macroaction.link_status)>0):
            for obj1_index in range(len(self.objects)):
                for obj2_index in range(obj1_index, len(self.objects)):
                    link_index = obj1_index*len(self.objects)+obj2_index
                    if(conf[self.state.links[link_index]]==1):
                        self.macroaction.links[link_index] = add_fixed_constraint_2(self.objects[obj1_index], self.objects[obj2_index])
            # Update link status
            self.macroaction.link_status = list(conf[-len(self.macroaction.link_status):len(conf)])

    # These function are overwritten in subclasses
    @property
    def fixed(self):
        raise NotImplementedError

    def set_state(self, configuration):
        raise NotImplementedError

    def get_object_pose(self, object, action):
        raise NotImplementedError

    def check_goal_state(self, config):
        raise NotImplementedError

    # Roll out for a fixed period of time or until stable
    def run_until_stable(self, hook=None, dt=0.001):
        time_limit = 1000
        config = self.get_current_config()
        t = 0
        while (True):
            for _ in range(5):
                p.stepSimulation(physicsClientId=0)
            time.sleep(dt)
            config_t = self.get_current_config()
            if (dist(config_t, config) < 3e-5 or t>time_limit):
                break

            if(hook is not None):
                hook()
            # config_t = self.set_legal_config(config_t)
            config = config_t
            t+=1

    def is_stable(self, old_conf, conf, count):
        return (dist(old_conf, conf)<5e-4, count>500)

    # Special Gym wrappers
    def step(self, action, terminate_unreachable=False, state_estimation=False):
        _, feasible, command = self.take_action(action)
        reward = -0.2
        done = False
        pre_stable_state = self.get_current_config()
        self.run_until_stable(dt=self.dt)
        time.sleep(0.01)
        post_stable_state = self.get_current_config()
        next_state = torch.unsqueeze(torch.tensor(post_stable_state), 0).type(torch.FloatTensor)
        goal_state = next_state
        if(terminate_unreachable and any([self.macroaction.object_unreachable(obj) for obj in self.objects])):
            print("Block is out of reach. Planning will never complete.")
            sys.exit(1)
        if(self.check_goal_state(post_stable_state)):
            reward = 1.0
            done = True
            next_state = self.reset()
            
        if(state_estimation):
            inputs = torch.unsqueeze(torch.cat([torch.tensor(take_picture(yaw, pit, 0)).type(torch.FloatTensor).permute(2, 0, 1) for yaw, pit in self.perspectives]), dim=0)
        else:
            inputs = torch.tensor([0])

        # reward = opt_cuda(torch.unsqueeze(torch.tensor(reward), 0).type(torch.FloatTensor))
        # print(reward)

        return next_state, reward, done, {"episode": {"r": reward} , "inputs": inputs, "prestable": torch.unsqueeze(torch.tensor(pre_stable_state), 0), "feasible":feasible, "command":command, "goal_state":goal_state }

    def reset(self):
        start_config = self.get_start_state()
        self.set_state(start_config)
        for _ in range(5):
            p.stepSimulation()
            time.sleep(0.01)
        return torch.unsqueeze(torch.tensor(start_config), 0)

    def seed(self, s):
        pass

