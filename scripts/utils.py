from __future__ import print_function
import pybullet as p
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def dist(pos1, pos2):
	return np.sqrt(np.sum(np.power(pos1-pos2, 2)))


def reparameterize(X, min_rv, max_rv):
	"""
		Take a random variable X that has support between -1 and 1
		Transform into a random variable that has support between min and max
	"""
	return (((X+1)/2.0)% 1)*(max_rv-min_rv)+min_rv


def opt_cuda_str():
	cuda="cuda:0"
	for argv in sys.argv:
		if(argv.isdigit()):
			cuda="cuda:"+str(sys.argv[2])
	return cuda

def opt_cuda(t):
	t = t.type(torch.FloatTensor)
	if(torch.cuda.is_available()):
		cuda="cuda:0"
		for argv in sys.argv:
			if(argv.isdigit()):
				cuda="cuda:"+str(sys.argv[2])
		if(str(sys.argv[2]) == "-1"):
			return t
		else:
			return t.cuda(cuda)
	else:
		return t


def plot_2d_value(value_func):
	values = []
	for i in range(-10, 10):
		value_i = []
		for j in range(-10, 10):
			vf_input = torch.tensor(np.array([i*0.1, j*0.1])).type(torch.FloatTensor)
			print(vf_input)
			vf = value_func(vf_input)
			print(vf)
			value_i.append(vf.item())
		values.append(value_i)
	plt.imshow(np.array(values))
	plt.show()


def check_state_collision(body1, body2):
	return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=0.001, physicsClientId=0)) != 0
	
def check_pairwise_collisions(bodies):
	for i1, body1 in enumerate(bodies):
		for i2, body2 in enumerate(bodies):
			if (body1 != body2 and check_state_collision(body1, body2)):
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


	return np.array(img_arr[2])[:, :, :3], viewMatrix, projectionMatrix
