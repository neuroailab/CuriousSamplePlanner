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
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import collections
from CuriousSamplePlanner.planning_pybullet.motion.motion_planners.discrete import astar
import sys

def dist(pos1, pos2):
	return np.sqrt(np.sum(np.power(pos1-pos2, 2)))


def reparameterize(X, min_rv, max_rv):
	"""
		Take a random variable X that has support between -1 and 1
		Transform into a random variable that has support between min and max
	"""

	
	return (((X+1)/2.0)% 1)*(max_rv-min_rv)+min_rv

def opt_cuda(t):
	if(len(sys.argv)>2):
		cuda="cuda:"+str(sys.argv[2])
	else:
		cuda="cuda:0"
	if(torch.cuda.is_available()):
		#cuda = "cuda:4"
		return t.cuda(cuda)
	else:
		return t

def check_state_collision(body1, body2):
	return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=0.001, physicsClientId=0)) != 0
	
def check_pairwise_collisions(bodies):
	for i1, body1 in enumerate(bodies):
		for i2, body2 in enumerate(bodies):
			if (body1 != body2 and check_state_collision(body1, body2)):
				print(i1, i2)
				return True
	return False

def check_vec_collisions(body1, bodies):
	for body2 in bodies:
		if (body1 != body2 and check_state_collision(body1, body2)):
			return True
	return False
	
def take_picture(yaw, pitch, roll, size=84):
	camTargetPos = [0, 0, 0]
	nearPlane = 0.01
	farPlane = 100
	fov = 60
	camDistance = 3
	upAxisIndex = 2
	viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex)
	projectionMatrix = p.computeProjectionMatrixFOV(fov, 1, nearPlane, farPlane)
	img_arr = p.getCameraImage(size,
														 size,
														 viewMatrix,
														 projectionMatrix,
														 shadow=1,
														 lightDirection=[1, 1, 1],
														 renderer=p.ER_TINY_RENDERER)

	return img_arr[2][:, :, :3]
