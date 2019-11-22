#!/usr/bin/env python
from __future__ import print_function

from planning_pybullet.pybullet_tools.utils import connect, Euler, set_default_camera, stable_z, \
    DRAKE_IIWA_URDF
from planning_pybullet.pybullet_tools.utils import set_pose, Pose, Point
from scripts.utils import *
from tasks.environment import Environment
from tasks.macroactions import PickPlace, MacroAction
from tasks.state import State


class ThreeBlocks(Environment):
    def __init__(self, *args):
        super(ThreeBlocks, self).__init__(*args)
        connect(use_gui=False)

        if self.detailed_gmp:
            self.robot = p.loadURDF(DRAKE_IIWA_URDF, useFixedBase=True,
                                    globalScaling=1.2)  # KUKA_IIWA_URDF | DRAKE_IIWA_URDF
        else:
            self.robot = None

        # Construct camera
        set_default_camera()

        # Set up objects
        self.floor = p.loadURDF('models/short_floor.urdf', useFixedBase=True)
        self.green_block = p.loadURDF("models/box_green.urdf", useFixedBase=False)
        self.red_block = p.loadURDF("models/box_red.urdf", useFixedBase=False)
        self.blue_block = p.loadURDF("models/box_blue.urdf", useFixedBase=False)

        self.objects = [self.green_block, self.red_block, self.blue_block]
        self.static_objects = []

        # Only used for some curiosity types
        self.perspectives = [(0, -90)]

        # Set up the state space and action space for this task
        self.break_on_timeout = True
        self.macroaction = MacroAction([
            PickPlace(objects=self.objects, robot=self.robot, fixed=self.fixed, gmp=self.detailed_gmp),
            # AddLink(objects = self.objects, robot=self.robot, fixed=self.fixed, gmp=self.detailed_gmp),
        ])

        # Config state attributes
        self.config_state_attrs()

        # Start the simulation
        p.setGravity(0, 0, -10)
        p.stepSimulation(physicsClientId=0)

    @property
    def fixed(self):
        return [self.floor]

    def set_state(self, conf):
        i = 0
        for block in self.objects:
            set_pose(block, Pose(Point(x=conf[i], y=conf[i + 1], z=conf[i + 2]),
                                 Euler(roll=conf[i + 3], pitch=conf[i + 4], yaw=conf[i + 5])))
            i += 6

    def check_goal_state(self, config):
        # collect the y values
        vals = [config[2], config[8], config[14]]
        vals.sort()

        # Two stack
        # if( (vals[0] > 0.06) or (vals[1] > 0.06) or (vals[2] > 0.06)):
        # 	return True

        # Three stack
        if vals[0] < 0.06 and (vals[1] < 0.16 and vals[1] > 0.06) and (vals[2] > 0.16):
            return True
        return False

    def get_current_config(self):
        # Get the object states
        gpos, gquat = p.getBasePositionAndOrientation(self.green_block, physicsClientId=0)
        rpos, rquat = p.getBasePositionAndOrientation(self.red_block, physicsClientId=0)
        ypos, yquat = p.getBasePositionAndOrientation(self.blue_block, physicsClientId=0)

        # Convert quat to euler
        geuler = p.getEulerFromQuaternion(gquat)
        reuler = p.getEulerFromQuaternion(rquat)
        yeuler = p.getEulerFromQuaternion(yquat)

        # Format into a config vector
        return np.concatenate([gpos, geuler, rpos, reuler, ypos, yeuler] + [self.macroaction.link_status])

    def get_start_state(self):
        collision = True
        z = stable_z(self.green_block, self.floor)
        while collision:
            pos1, pos2, pos3 = self.reachable_pos(z=0), self.reachable_pos(z=0), self.reachable_pos(z=0)
            state = State(len(self.objects), len(self.static_objects), len(self.macroaction.link_status))
            state.set_position(0, pos1[0], pos1[1], z)
            state.set_position(1, pos2[0], pos2[1], z)
            state.set_position(2, pos3[0], pos3[1], z)
            self.set_state(state.config)
            collision = check_pairwise_collisions(self.objects)
        return state.config
