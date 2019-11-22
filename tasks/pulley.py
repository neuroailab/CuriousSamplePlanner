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
import pickle
from planning_pybullet.pybullet_tools.kuka_primitives import BodyPose, BodyConf, Command, get_grasp_gen, \
    get_ik_fn, get_free_motion_gen, get_holding_motion_gen
from planning_pybullet.pybullet_tools.utils import WorldSaver, enable_gravity, connect, dump_world, set_pose, \
    Pose, Point, Euler, set_default_camera, stable_z, \
    BLOCK_URDF, load_model, wait_for_user, disconnect, DRAKE_IIWA_URDF, user_input, update_state, disable_real_time, \
    inverse_kinematics, end_effector_from_body, approach_from_grasp, get_joints, get_joint_positions

from planning_pybullet.pybullet_tools.utils import get_pose, set_pose, get_movable_joints, \
    set_joint_positions, add_fixed_constraint, enable_real_time, disable_real_time, joint_controller, \
    enable_gravity, get_refine_fn, user_input, wait_for_duration, link_from_name, get_body_name, sample_placement, \
    end_effector_from_body, approach_from_grasp, plan_joint_motion, GraspInfo, Pose, INF, Point, \
    inverse_kinematics, pairwise_collision, remove_fixed_constraint, Attachment, get_sample_fn, \
    step_simulation, refine_path, plan_direct_joint_motion, get_joint_positions, dump_world, get_link_pose, \
    control_joints

from planning_pybullet.pybullet_tools.kuka_primitives import BodyPath, Attach, Detach
import pickle
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import collections
from planning_pybullet.motion.motion_planners.discrete import astar
import sys
from tasks.macroactions import PickPlace, AddLink, MacroAction
from rl_ppo_rnd.a2c_ppo_acktr.model import Policy
from tasks.environment import Environment
from scripts.utils import *
from gym import spaces


class PulleySeesaw(Environment):
    def is_stable(self, old_conf, conf, count):
        timeout = count > 50
        stable = dist(old_conf, conf) < 6e-5
        return stable, timeout

    def __init__(self, *args):
        super(PulleySeesaw, self).__init__(*args)
        # connect(use_gui=not torch.cuda.is_available())
        connect(use_gui=True)

        if self.detailed_gmp:
            self.robot = p.loadURDF(DRAKE_IIWA_URDF, useFixedBase=True,
                                    globalScaling=1)  # KUKA_IIWA_URDF | DRAKE_IIWA_URDF
        else:
            self.robot = None

        self.KNOT = 0.1
        self.NOKNOT = 0
        self.break_on_timeout = True
        self.config_size = 20
        self.ARM_LEN = 1.6
        self.predict_mask = list(range(self.config_size - 5))
        x_scene_offset = 0.9
        num_small_blue = 5

        self.training = False
        self.perspectives = [(0, -90)]

        self.pulley = p.loadSDF("models/pulley/newsdf.sdf", globalScaling=2.5)
        for pid in self.pulley:
            set_pose(pid,
                     Pose(Point(x=x_scene_offset - 0.19, y=-0.15, z=1.5), Euler(roll=math.pi / 2.0, pitch=0, yaw=1.9)))

        self.floor = p.loadURDF('models/short_floor.urdf', useFixedBase=True)

        self.seesaw = p.loadURDF("models/seesaw.urdf", useFixedBase=True)
        self.seesaw_joint = 1
        set_pose(self.seesaw, Pose(Point(x=x_scene_offset + 0.8, y=0, z=0)))

        self.block_pos = [x_scene_offset - 1, -0.8, 0.05]
        self.black_block = p.loadURDF("models/box_heavy.urdf", useFixedBase=True)
        set_pose(self.black_block, Pose(Point(x=self.block_pos[0], y=self.block_pos[1], z=self.block_pos[2])))

        self.objects = []
        small_boxes = ["small_blue_heavy", "small_purple_heavy", "small_green_heavy", "small_red_heavy",
                       "small_yellow_heavy"]
        for i in range(num_small_blue):
            pos = self.reachable_pos()
            self.objects.append(p.loadURDF("models/" + str(small_boxes[i]) + ".urdf", useFixedBase=False))
            set_pose(self.objects[i], Pose(Point(x=pos[0], y=pos[1], z=0.05)))

        self.red_block = p.loadURDF("models/box_red.urdf", useFixedBase=False)
        set_pose(self.red_block, Pose(Point(x=x_scene_offset + 1.7, y=0, z=0.5)))

        self.useMaximalCoordinates = True
        sphereRadius = 0.01
        self.mass = 1
        self.basePosition = [x_scene_offset - 0.2, -1.7, 1.6]
        self.baseOrientation = [0, 0, 0, 1]

        if self.training:
            p.setGravity(0, 0, -10)
            self.cup = p.loadURDF("models/cup/cup_small.urdf", useFixedBase=True)
            set_pose(self.cup, Pose(Point(x=x_scene_offset - 0.2, y=0.1, z=1)))
            self.knot = self.KNOT

        else:
            self.cupCollideId = p.createCollisionShape(p.GEOM_MESH, fileName="models/cup/Cup/cup_vhacd.obj",
                                                       meshScale=[6, 6, 1.5],
                                                       collisionFrameOrientation=p.getQuaternionFromEuler(
                                                           [math.pi / 2.0, 0, 0]),
                                                       collisionFramePosition=[0.07, 0.3, 0])
            self.cupShapeId = p.createVisualShape(p.GEOM_MESH, fileName="models/cup/Cup/textured-0008192.obj",
                                                  meshScale=[6, 6, 1.5], rgbaColor=[1, 0.886, 0.552, 1],
                                                  visualFrameOrientation=p.getQuaternionFromEuler(
                                                      [math.pi / 2.0, 0, 0]), visualFramePosition=[0.07, 0.3, 0])

            self.colBoxId = p.createCollisionShape(p.GEOM_CYLINDER, radius=sphereRadius, height=0.03,
                                                   halfExtents=[sphereRadius, sphereRadius, sphereRadius],
                                                   collisionFrameOrientation=p.getQuaternionFromEuler(
                                                       [math.pi / 2.0, 0, 0]))
            self.visualcolBoxId = p.createVisualShape(p.GEOM_CYLINDER, radius=sphereRadius, length=0.03,
                                                      halfExtents=[sphereRadius, sphereRadius, sphereRadius],
                                                      rgbaColor=[1, 0.886, 0.552, 1],
                                                      visualFrameOrientation=p.getQuaternionFromEuler(
                                                          [math.pi / 2.0, 0, 0]))

            # visualShapeId = -1

            self.link_Masses = []
            self.linkCollisionShapeIndices = []
            self.linkVisualShapeIndices = []
            self.linkPositions = []
            self.linkOrientations = []
            self.linkInertialFramePositions = []
            self.linkInertialFrameOrientations = []
            self.indices = []
            self.jointTypes = []
            self.axis = []

            numel = 70
            for i in range(numel):
                self.link_Masses.append(0.3)
                self.linkCollisionShapeIndices.append(self.colBoxId)
                self.linkVisualShapeIndices.append(self.visualcolBoxId)
                self.linkPositions.append([0, sphereRadius * 2.0 + 0.01, 0])
                self.linkOrientations.append([0, 0, 0, 1])
                self.linkInertialFramePositions.append([0, 0, 0])
                self.linkInertialFrameOrientations.append([0, 0, 0, 1])
                self.indices.append(i)
                self.jointTypes.append(p.JOINT_FIXED)
                self.axis.append([0, 0, 1])

            self.link_Masses.append(30)
            self.linkCollisionShapeIndices.append(self.cupCollideId)
            self.linkVisualShapeIndices.append(self.cupShapeId)
            self.linkPositions.append([0, 0, 0])
            self.linkOrientations.append([0, 0, 0, 1])
            self.linkInertialFramePositions.append([0, 0, 0])
            self.linkInertialFrameOrientations.append([0, 0, 0, 1])
            self.indices.append(numel)
            self.jointTypes.append(p.JOINT_FIXED)
            self.axis.append([0, 0, 1])

            self.sphereUid = p.createMultiBody(self.mass,
                                               -1,
                                               -1,
                                               self.basePosition,
                                               self.baseOrientation,
                                               linkMasses=self.link_Masses,
                                               linkCollisionShapeIndices=self.linkCollisionShapeIndices,
                                               linkVisualShapeIndices=self.linkVisualShapeIndices,
                                               linkPositions=self.linkPositions,
                                               linkOrientations=self.linkOrientations,
                                               linkInertialFramePositions=self.linkInertialFramePositions,
                                               linkInertialFrameOrientations=self.linkInertialFrameOrientations,
                                               linkParentIndices=self.indices,
                                               linkJointTypes=self.jointTypes,
                                               linkJointAxis=self.axis,
                                               useMaximalCoordinates=self.useMaximalCoordinates)

            p.setRealTimeSimulation(0)

            self.anistropicFriction = [0.00, 0.00, 0.00]
            p.changeDynamics(self.sphereUid, -1, lateralFriction=0, anisotropicFriction=self.anistropicFriction)
            self.keystone = p.loadURDF("models/tiny_green.urdf")
            set_pose(self.keystone, Pose(Point(x=0.7, y=0, z=0.9)))
            p.setGravity(0, 0, -10)
            saved_world = p.restoreState(fileName="./temp/pulley_start_state.bullet")
            # for j in range(100):
            #     p.stepSimulation(physicsClientId=0)

            self.knot = p.createConstraint(self.black_block, -1, self.sphereUid, -1, p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                           parentFramePosition=self.block_pos, childFramePosition=self.block_pos)
            # for j in range(3000):
            #     print(j)
            #     p.stepSimulation(physicsClientId=0)


            # self.keystone = p.loadURDF("models/tiny_green.urdf")
            set_pose(self.keystone, Pose(Point(x=0.7, y=0, z=0.9)))

        # for j in range(1000):
        #     print(j)
        #     p.stepSimulation(physicsClientId=0)

        # saved_world = p.saveState()
        # p.saveBullet("./pulley_start_state.bullet")

        print("setting up action sapace")
        self.macroaction = MacroAction([
            PickPlace(objects=self.objects, robot=self.robot, fixed=self.fixed, gmp=self.detailed_gmp),
        ])
        self.action_space_size = self.macroaction.action_space_size
        self.config_size = 6 * 6 + len(self.macroaction.link_status)  # (4 for links)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space_size,))
        self.actor_critic = opt_cuda(Policy([self.config_size], self.action_space, base_kwargs={'recurrent': False}))
        self.predict_mask = [0, 1, 2] + [6, 7, 8] + [12, 13, 14] + [18, 19, 20]

    def check_goal_state(self, config):
        if self.training:
            keystone, _ = p.getBasePositionAndOrientation(self.cup)
        else:
            keystone, _ = p.getBasePositionAndOrientation(self.keystone)
        tddist = lambda a, b, c: math.sqrt(
            (a - keystone[0]) ** 2 + (b - keystone[1]) ** 2 + (c - keystone[2]) ** 2) < 0.3
        in_bucket = int(tddist(config[0], config[1], config[2])) + int(tddist(config[3], config[4], config[5])) + int(
            tddist(config[6], config[7], config[8])) + tddist(config[9], config[10], config[11]) + int(
            tddist(config[12], config[13], config[14]))
        if in_bucket > 0:
            print(str(in_bucket) + " in bucket")
        if (tddist(config[0], config[1], config[2]) and tddist(config[3], config[4], config[5]) and tddist(config[6],
                                                                                                           config[7],
                                                                                                           config[
                                                                                                               8]) and tddist(
                config[9], config[10], config[11]) and tddist(config[12], config[13], config[14])):
            print(config[-2])
            return config[-1] == self.NOKNOT
        return False

    def get_current_config(self):
        config = []
        for obj in self.objects:
            pos, quat = p.getBasePositionAndOrientation(obj)
            euler = p.getEulerFromQuaternion(quat)
            config.append(pos)
            config.append(euler)
        rpos, rquat = p.getBasePositionAndOrientation(self.red_block)
        reuler = p.getEulerFromQuaternion(rquat)
        config.append(rpos)
        config.append(reuler)

        # joint_state = p.getJointState(self.seesaw, self.seesaw_joint)
        # if(self.knot is not None):
        # 	knot_state = self.KNOT
        # else:
        # 	knot_state = self.NOKNOT

        # return np.concatenate(small_blocks+[rpos]+[[joint_state[0]]]+[[knot_state]])
        return np.concatenate(config)

    def get_available_block(self):
        reachable = False
        break_condition = 0
        while not reachable:
            if break_condition == 100:
                return None
            block_to_move = random.choice([0, 1, 2, 3, 4])
            if self.training:
                keystone, _ = p.getBasePositionAndOrientation(self.cup)
            else:
                keystone, _ = p.getBasePositionAndOrientation(self.keystone)

            blockpos, _ = p.getBasePositionAndOrientation(self.objects[block_to_move])
            tddist = lambda a, b, c: math.sqrt(
                (a - keystone[0]) ** 2 + (b - keystone[1]) ** 2 + (c - keystone[2]) ** 2) < 0.3
            reachable = not tddist(blockpos[0], blockpos[1], blockpos[2])
            break_condition += 1

        return block_to_move

    # def take_action(self, embedding):
    #     joint_state = p.getJointState(self.seesaw, self.seesaw_joint)
    #     collision = True
    #     objects = self.small_blue+[self.red_block]
    #     block_to_move = self.get_available_block()
    #     if(random.uniform(0, 1)>0.95 or block_to_move is None):
    #         block_to_move = 5
    #         if(self.knot is not None):
    #             self.deconstrain()
    #         self.knot=None
    #     else:
    #         while(collision):
    #             # mx, my, _ = self.reachable_pos()
    #             mx = random.uniform(0.7, 0.8)
    #             my = random.uniform(-0.1, 0.1)
    #             # Need to check collisions
    #             # set_pose(objects[block_to_move], Pose(Point(x=mx, y=my, z=random.uniform(0.1, self.ARM_LEN)), Euler(roll = random.uniform(-math.pi, math.pi), pitch = random.uniform(-math.pi, math.pi), yaw = random.uniform(-math.pi, math.pi)) ))
    #             set_pose(objects[block_to_move], Pose(Point(x=mx, y=my, z=random.uniform(0.1, self.ARM_LEN)), Euler(roll = 0, pitch = 0, yaw = 0) ))
    #             collision = self.check_vec_collisions(objects[block_to_move], objects)

    #     block_logit = [0 for _ in range(6)]
    #     block_logit[block_to_move] = 1
    #     return np.concatenate([np.array(block_logit), self.get_current_config()[:15]])



    def constrain(self):
        if not self.training:
            return p.createConstraint(self.black_block, -1, self.sphereUid, -1, p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                      parentFramePosition=self.block_pos, childFramePosition=self.block_pos)
        else:
            return self.KNOT

    def deconstrain(self):
        if not self.training:
            p.removeConstraint(self.knot)

    def set_state(self, conf):
        if not self.training:
            saved_world = p.restoreState(fileName="./temp/pulley_start_state.bullet")
        if conf[-1] > 0:
            if self.knot is not None:
                self.deconstrain()
            self.knot = self.constrain()
        for i in range(5):
            set_pose(self.objects[i], Pose(Point(x=conf[i * 6], y=conf[i * 6 + 1], z=conf[i * 6 + 2]),
                                           Euler(roll=conf[i * 6 + 3], pitch=conf[i * 6 + 4], yaw=conf[i * 6 + 5])))
        set_pose(self.red_block,
                 Pose(Point(x=conf[-6], y=conf[-5], z=conf[-4]), Euler(roll=conf[-3], pitch=conf[-2], yaw=conf[-1])))

    # p.resetJointState(self.seesaw, self.seesaw_joint, conf[-2])
    # if(conf[-1] == self.KNOT):
    # 	if(self.knot is None):
    # 		self.knot = self.constrain()
    # elif(conf[-1] == self.NOKNOT):
    # 	if(self.knot is not None):
    # 		self.deconstrain()
    # 	self.knot = None


    def get_start_state(self):
        pos = []
        z = 0.05
        for i in range(5):
            pos.append([j for j in self.reachable_pos()])

        pos.append([2.6277769963206423, -0.038717869031510414, 0.2586803615589658])

        eu = [0, 0, 0]
        conf = np.array(
            [pos[0][0], pos[0][1], z] + eu + [pos[1][0], pos[1][1], z] + eu + [pos[2][0], pos[2][1], z] + eu + [
                pos[3][0], pos[3][1], z] + eu + [pos[4][0], pos[4][1], z] + eu + [pos[5][0], pos[5][1], pos[5][2]] + eu)
        self.set_state(conf)
        return conf

    @property
    def fixed(self):
        return [self.floor]
