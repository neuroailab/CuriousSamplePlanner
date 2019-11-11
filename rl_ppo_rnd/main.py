import copy
import glob
import os
import time
from collections import deque
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from CuriousSamplePlanner.rl_ppo_rnd.a2c_ppo_acktr import algo, utils
from CuriousSamplePlanner.rl_ppo_rnd.a2c_ppo_acktr.model import Policy
from CuriousSamplePlanner.rl_ppo_rnd.a2c_ppo_acktr.storage import RolloutStorage
from gym import spaces
from CuriousSamplePlanner.scripts.utils import *
from CuriousSamplePlanner.trainers.architectures import WorldModel

import pickle
import shutil
from CuriousSamplePlanner.tasks.three_block_stack import ThreeBlocks
from CuriousSamplePlanner.tasks.five_block_stack import FiveBlocks
from CuriousSamplePlanner.tasks.ball_ramp import BallRamp
from CuriousSamplePlanner.tasks.bookshelf import BookShelf
from CuriousSamplePlanner.tasks.pulley import PulleySeesaw

from CuriousSamplePlanner.trainers.dataset import ExperienceReplayBuffer

import sys

def main():
	exp_id = str(sys.argv[1])
	experiment_dict = {
		"world_model_losses": [],
		"num_sampled_nodes": 0,
		"exp_id": exp_id,
		"task": "FiveBlocks",
		"num_training_epochs": 5,
		'batch_size': 128,
		"algo": "ppo",
		'lr': 7e-4,
		'eps': 1e-5,
		'alpha': 0.99,
		'gamma': 0,
		'use_gae': False,
		'entropy_coef': 0,
		'value_loss_coef': 0.5,
		'max_grad_norm': 0.5,
		'recurrent_policy': True,
		'enable_asm': False,
		'detailed_gmp': False,
		'seed': 1,
		'cuda_deterministic':False,
		'num_processes':1,
		'num_steps': 128,
		"learning_rate": 5e-5,
		'ppo_epoch': 4,
		'num_mini_batch': 32,
		'clip_param': 0.2, 
		'log_interval': 10,
		'save_interval': 100,
		'eval_interval': None,
		'num_env_steps': 10e6,
		'log_dir': '/tmp/gym/',
		'nsamples_per_update': 1024,
		'mode': 'EffectPredictionPlanner',
		'training': True, 
		'save_dir': './trained_models/',
		'store_true':False,
		'use_proper_time_limits': False,
		'reset_frequency': 2e-4,
		'recurrent_policy': False,
		'use_linear_lr_decay': False
	}
	experiment_dict['exp_path'] = "/mnt/fs0/arc11_2/pybullet_examples/" + experiment_dict["exp_id"]

	if (os.path.isdir(experiment_dict['exp_path'])):
		shutil.rmtree(experiment_dict['exp_path'])
	os.mkdir(experiment_dict['exp_path'])

	torch.manual_seed(experiment_dict['seed'])
	torch.cuda.manual_seed_all(experiment_dict['seed'])

	if torch.cuda.is_available() and experiment_dict["cuda_deterministic"]:
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True



	torch.set_num_threads(1)
	# device = torch.device("cuda:"+str(int(0)))

	PC = getattr(sys.modules[__name__], experiment_dict['task'])
	print("hello?")
	env = PC(experiment_dict)

	action_low = -1
	action_high = 1




	## Set up the world model for RND operations
	worldModelTarget = opt_cuda(WorldModel(config_size=env.config_size))

	def weights_init_uniform(m):
		classname = m.__class__.__name__            
		# apply a uniform distribution to the weights and a bias=0
		if classname.find('Linear') != -1:
			m.weight.data.uniform_(0.0, 0.02)
			m.bias.data.fill_(0)

	worldModelTarget.apply(weights_init_uniform)
	for param in worldModelTarget.parameters():
		param.requires_grad = False
	worldModel = opt_cuda(WorldModel(config_size=env.config_size))
	criterion = nn.MSELoss()
	optimizer = optim.Adam(worldModel.parameters(), lr=experiment_dict["learning_rate"])


	actor_critic = Policy([env.config_size], env.action_space, base_kwargs={'recurrent': experiment_dict["recurrent_policy"]})



	# actor_critic.to(device)
	if experiment_dict["algo"] == 'a2c':
		agent = algo.A2C_ACKTR(
			actor_critic,
			experiment_dict["value_loss_coef"],
			experiment_dict["entropy_coef"],
			lr=experiment_dict["lr"],
			eps=experiment_dict["eps"],
			alpha=experiment_dict["alpha"],
			max_grad_norm=experiment_dict["max_grad_norm"])
	elif experiment_dict['algo'] == 'ppo':
		agent = algo.PPO(
			actor_critic,
			experiment_dict["clip_param"],
			experiment_dict["ppo_epoch"],
			experiment_dict["num_mini_batch"],
			experiment_dict["value_loss_coef"],
			experiment_dict["entropy_coef"],
			lr=experiment_dict["lr"],
			eps=experiment_dict["eps"],
			max_grad_norm=experiment_dict["max_grad_norm"])
	elif experiment_dict['algo'] == 'acktr':
		agent = algo.A2C_ACKTR(
			actor_critic, experiment_dict["value_loss_coef"], experiment_dict["entropy_coef"], acktr=True)


	rollouts = RolloutStorage(experiment_dict["num_steps"], 1,
							  [env.config_size], env.action_space,
							  actor_critic.recurrent_hidden_state_size)

	obs = env.reset()
	reset_obs = obs
	rollouts.obs[0].copy_(obs)
	# rollouts.to(device)

	episode_rewards = deque(maxlen=1000)
	# Create the replay buffer for training world models
	experience_replay = ExperienceReplayBuffer()

	start = time.time()
	num_updates = int(
		experiment_dict["num_env_steps"]) // experiment_dict["num_steps"] // experiment_dict["num_processes"]
	for j in range(num_updates):

		if experiment_dict['use_linear_lr_decay']:
			# decrease learning rate linearly
			utils.update_linear_schedule(
				agent.optimizer, j, num_updates,
				agent.optimizer.lr if experiment_dict["algo"] == "acktr" else experiment_dict["lr"])

		for step in range(experiment_dict['num_steps']):
			if(random.uniform(0,1) < experiment_dict['reset_frequency']):
				print("RESET")
				env.set_state(reset_obs[0])
			experiment_dict['num_sampled_nodes']+=1
			# Sample actions
			with torch.no_grad():
				value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
					rollouts.obs[step], rollouts.recurrent_hidden_states[step],
					rollouts.masks[step])

			# action = torch.clamp(action, action_low, action_high)
			obs, reward, done, infos = env.step(action)

			obs_tensor = opt_cuda(obs.type(torch.FloatTensor))

			reward = torch.nn.functional.mse_loss(worldModelTarget(obs_tensor), worldModel(obs_tensor)).detach()
			# Insert into buffer for world model training
			targets = opt_cuda(obs)
			filler = opt_cuda(torch.tensor([0]))
			experience_replay.bufferadd_single(filler, torch.squeeze(targets), filler, filler, filler, filler, filler)

			# Inserted for single-task experimentation
			if(done[0]):
				stats_filehandler = open(experiment_dict['exp_path'] + "/stats.pkl", 'wb')
				pickle.dump(experiment_dict, stats_filehandler)
				time.sleep(5)
				exit(1)

			for info in infos:
				if 'episode' in info.keys():
					episode_rewards.append(reward.cpu().numpy())

			# If done then clean the history of observations.
			masks = torch.FloatTensor(
				[[0.0] if done_ else [1.0] for done_ in done])
			bad_masks = torch.FloatTensor(
				[[0.0] if 'bad_transition' in info.keys() else [1.0]
				 for info in infos])
			rollouts.insert(obs, recurrent_hidden_states, action,
							action_log_prob, value, reward, masks, bad_masks)



		with torch.no_grad():
			next_value = actor_critic.get_value(
				rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
				rollouts.masks[-1]).detach()

		rollouts.compute_returns(next_value, experiment_dict["use_gae"], experiment_dict["gamma"],
								 0, experiment_dict["use_proper_time_limits"])

		value_loss, action_loss, dist_entropy = agent.update(rollouts)

		rollouts.after_update()



		# Update world model
		for epoch in range(experiment_dict['num_training_epochs']):
			for next_loaded in enumerate(DataLoader(experience_replay, batch_size=experiment_dict['batch_size'], shuffle=True, num_workers=0)):
				optimizer.zero_grad()
				_, batch = next_loaded
				inputs, labels, prestates, _, _, _, _, index = batch
				outputs = opt_cuda(worldModel(opt_cuda(labels.type(torch.FloatTensor))))
				#outputs_target = opt_cuda(self.worldModelTarget(labels).type(torch.FloatTensor))
				outputs_target=opt_cuda(worldModelTarget(opt_cuda(labels.type(torch.FloatTensor))))

				loss = criterion(outputs[:, env.predict_mask],
									 outputs_target[:, env.predict_mask])
				loss.backward()
				experiment_dict["world_model_losses"].append(loss.item())
				optimizer.step()


		print(j)
		del experience_replay
		experience_replay = ExperienceReplayBuffer()
		# print(episode_rewards)
		if j % experiment_dict['log_interval'] == 0 and len(episode_rewards) > 1:
			total_num_steps = (j + 1) * experiment_dict['num_processes'] * experiment_dict['num_steps']
			end = time.time()
			print(
				"Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.3f}/{:.3f}, min/max reward {:.1f}/{:.1f}\n"
				.format(j, total_num_steps,
						int(total_num_steps / (end - start)),
						len(episode_rewards), np.mean(episode_rewards),
						np.median(episode_rewards), np.min(episode_rewards),
						np.max(episode_rewards), dist_entropy, value_loss,
						action_loss))

		# if (args.eval_interval is not None and len(episode_rewards) > 1
		#         and j % args.eval_interval == 0):
		#     ob_rms = utils.get_vec_normalize(envs).ob_rms
		#     evaluate(actor_critic, ob_rms, args.env_name, args.seed,
		#              args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
	main()
