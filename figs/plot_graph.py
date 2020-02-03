#!/usr/bin/env python
from __future__ import print_function
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
import networkx as nx
from planning_pybullet.pybullet_tools.kuka_primitives import BodyPose, BodyConf, Command, get_grasp_gen, \
    get_ik_fn, get_free_motion_gen, get_holding_motion_gen
from planning_pybullet.pybullet_tools.utils import WorldSaver, enable_gravity, connect, dump_world, set_pose, \
    Pose, Point, Euler, set_default_camera, stable_z, \
    BLOCK_URDF, load_model, wait_for_user, disconnect, DRAKE_IIWA_URDF, user_input, update_state, disable_real_time,inverse_kinematics,end_effector_from_body,approach_from_grasp 

from planning_pybullet.pybullet_tools.kuka_primitives import BodyPath, Attach, Detach
import pickle
import torch
from torchvision import models, transforms
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import collections
from motion_planners.discrete import astar
import sys

from ComplexMotionPlanning.trainers.plan_graph import PlanGraph, GraphNode



if __name__ == '__main__':
    plan_graph_path = str(sys.argv[1])
    plan_path = str(sys.argv[2])
    plan_graph_filehandler = open(plan_graph_path, 'rb')
    plan_filehandler = open(plan_path, 'rb') 
    plan_graph = pickle.load(plan_graph_filehandler)
    plan = pickle.load(plan_filehandler)

    print(plan_graph)
    print(plan)

    G=nx.Graph()

    colormap = []

    for key, values in plan_graph.plan_graph.items():
        G.add_node(key)

    for key, values in plan_graph.plan_graph.items():
        for val in values:
            G.add_edge(key, val)

    for node in G:
        print(node.node_key)
        print(plan_graph.goal_node.node_key)
        if node.node_key in [p.node_key for p in plan]:
            if node.node_key == plan_graph.goal_node.node_key or node.node_key == plan_graph.start_node.node_key:
                colormap.append("#0000FF")
            else:     
                colormap.append("#00FF00")       
        else:
            colormap.append("red")    

    nx.draw(G, node_color = colormap, node_size=20)
    plt.show()


