import os
import torch
import numpy as np

# Constants
pathname = "./data_collection/"
exp_name = "first_attempt"
trajectory_length = 8

# First, get all of the files
from os import listdir
from os.path import isfile, join
path = pathname+exp_name
data_files = [f for f in listdir(path) if isfile(join(path, f))]

total_length = 0
traj_states = []
traj_actions = []
traj_rewards = []
traj_lens = []

g_states = []
g_actions = []
g_rewards = []
g_lens = []

for file in data_files:
	data = torch.load(pathname+exp_name+"/"+file)

	g_states.append(data['states'])
	g_actions.append(data['actions'])
	g_rewards.append(data['rewards'])
	g_lens.append(data['lengths'])
	total_length+=data['lengths']

	if(total_length>trajectory_length):
		# Now we have gotten enough samples to cut it off

		traj_states.append(np.concatenate(g_states, axis=1)[:, :trajectory_length, :])
		traj_actions.append(np.concatenate(g_actions, axis=1)[:, :trajectory_length, :])
		traj_rewards.append(np.concatenate(np.array(g_rewards)))
		traj_lens.append(np.array(g_lens))
		g_states = []
		g_actions = []
		g_rewards = []
		g_lens = []
		total_length = 0

# Now we have all of the trajectories, time to turn it into the data file

total_states = np.concatenate(traj_states, axis=0)
total_actions = np.concatenate(traj_actions, axis=0)
total_rewards = np.concatenate(traj_rewards, axis=0)
total_lens = np.concatenate(traj_lens, axis=0)

data = {
	'states': torch.from_numpy(total_states).float(),
	'actions': torch.from_numpy(total_actions).float(),
	'rewards': torch.from_numpy(total_rewards).float(),
	'lengths': torch.from_numpy(total_lens).float()
}

torch.save(data, "./rl_utils/gail_experts/"+exp_name+".pt")



