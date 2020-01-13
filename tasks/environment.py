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


class Environment():
    def __init__(self, experiment_dict):
        self.nsamples_per_update = experiment_dict['nsamples_per_update']
        self.asm_enabled = experiment_dict['enable_asm']
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
        if(not linking):
            self.predict_mask = self.state.positions
        else:
            self.predict_mask = self.state.positions+self.state.links

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space_size,))
        self.actor_critic = opt_cuda(Policy([self.config_size], self.action_space, base_kwargs={'recurrent': False}))


    def take_action(self, action):
        # Get the macroaction that is being executed
        action = action[0].detach().cpu().numpy()
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

    # Reachability for the robotic arm. 
    # TODO: Generalize for arbitrary robotic systems
    def reachable_pos(self, z=None):
        reachability = {
            0.0:(0.45,0.78),
            0.1:(0.45,0.78),
            0.2:(0.43,0.78),
            0.3:(0.38,0.76),
            0.4:(0.30,0.72),
            0.5:(0.32,0.67),
            0.6:(0.32,0.59)
        }
        if(z == None):
            z = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        radius = random.uniform(reachability[int(z*10)/10.0][0], reachability[int(z*10)/10.0][1])
        angle = random.uniform(-math.pi, math.pi)
        return [math.sin(angle) * radius, math.cos(angle) * radius, z]

    def is_stable(self, old_conf, conf, count):
        return (dist(old_conf, conf)<5e-4, count>500)

    def collect_samples(self, graph):
        features = []
        commands = []
        states = torch.zeros([self.nsamples_per_update, self.config_size]).type(torch.FloatTensor)
        prestates = torch.zeros([self.nsamples_per_update, self.config_size]).type(torch.FloatTensor)
        parents = torch.zeros([self.nsamples_per_update, 1]).type(torch.FloatTensor)
        feasibles = torch.zeros([self.nsamples_per_update, 1]).type(torch.FloatTensor)
        actions = torch.zeros([self.nsamples_per_update, self.action_space_size]).type(torch.FloatTensor)
        action_log_probs = torch.zeros([self.nsamples_per_update, 1]).type(torch.FloatTensor)
        values = torch.zeros([self.nsamples_per_update, 1]).type(torch.FloatTensor)
        goal_config = None
        goal_prestate = None
        goal_parent = None
        goal_action = None
        goal_command = None

        # starting_nodes = graph.expand_node(self.nsamples_per_update)
        for i in range(self.nsamples_per_update):
            while True:
                parent = graph.expand_node(1)[0]
                self.set_state(parent.config)
                embedding = torch.unsqueeze(torch.tensor(np.random.uniform(low=-1, high=1, size=self.action_space_size)),0)
                (action, action_log_prob, value, feasible, command) = self.take_action(embedding)
                if action is not None:
                    break

            config = self.get_current_config()
            preconfig = config
            p.stepSimulation(physicsClientId=0)

            count = 0
            while (True):
                for _ in range(10):
                    p.stepSimulation(physicsClientId=0)
                    time.sleep(0.00001)

                config_t = self.get_current_config()
                stable, timeout = self.is_stable(config_t, config, count)
                if(stable or (timeout and self.break_on_timeout)):
                    break

                if(timeout and not self.break_on_timeout):
                    print("Timeout Reset")
                    while True:
                        parent = graph.expand_node(1)[0]
                        self.set_state(parent.config)
                        embedding = torch.unsqueeze(torch.tensor(np.random.uniform(low=-1, high=1, size=self.action_space_size)),0)
                        (action, action_log_prob, value, feasible, command) = self.take_action(embedding)
                        if action is not None:
                            break

                    config = self.get_current_config()
                    count = 0

                #config_t = self.set_legal_config(config_t)
                config = config_t
                count+=1

                
            # Capture perspectives
            if(self.image_based):
                img_arr = torch.unsqueeze(torch.cat(
                    [torch.tensor(take_picture(yaw, pit, 0)).type(torch.FloatTensor).permute(2, 0, 1)
                     for yaw, pit in self.perspectives]), dim=0)
            else:
                img_arr = torch.zeros((1, 3, 84, 84))

            features.append(img_arr)
            commands.append(command)
            parents[i, :] = torch.tensor(parent.node_key)
            actions[i, :] = torch.tensor(action)
            action_log_probs[i, :] = action_log_prob.detach()
            values[i, :] = value
            feasibles[i, :] = int(feasible)
            states[i, :] = torch.tensor(config)
            prestates[i, :] = torch.tensor(preconfig)

            if(self.check_goal_state(config)):
                goal_config = config
                goal_parent = parent.node_key
                goal_action = action
                goal_prestate = preconfig
                goal_command = command
                break

        # TODO: Messy
        return features, states[:i + 1, :], prestates[:i + 1, :],\
               actions[:i + 1, :], action_log_probs[:i + 1, :], values[:i + 1, :], feasibles[:i + 1, :], parents[:i + 1, :], \
               goal_config, goal_prestate, goal_parent, goal_action, goal_command, commands

    # Special Gym wrappers
    def step(self, action, terminate_unreachable=False, state_estimation=False):
        _, feasible, command = self.take_action(action)
        reward = -0.2
        done = False
        pre_stable_state = self.get_current_config()
        self.run_until_stable(dt=0)
        time.sleep(0.01)
        post_stable_state = self.get_current_config()
        if(terminate_unreachable and any([self.macroaction.object_unreachable(obj) for obj in self.objects])):
            print("Block is out of reach. Planning will never complete.")
            sys.exit(1)
        if(self.check_goal_state(post_stable_state)):
            reward = 1.0
            done=True
            
        if(state_estimation):
            inputs = torch.unsqueeze(torch.cat([torch.tensor(take_picture(yaw, pit, 0)).type(torch.FloatTensor).permute(2, 0, 1) for yaw, pit in self.perspectives]), dim=0)
        else:
            inputs = opt_cuda(torch.tensor([0]))

       
        return opt_cuda(torch.unsqueeze(torch.tensor(self.get_current_config()), 0)), opt_cuda(torch.unsqueeze(torch.tensor(reward), 0)), [done], [{"episode": {"r": reward}}], inputs, opt_cuda(torch.unsqueeze(torch.tensor(pre_stable_state), 0)), feasible, command 

    def reset(self):
        start_config = self.get_start_state()
        self.set_state(start_config)
        for _ in range(5):
            p.stepSimulation()
            time.sleep(0.01)
        return opt_cuda(torch.unsqueeze(torch.tensor(start_config), 0))
