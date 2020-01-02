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
from CuriousSamplePlanner.trainers.architectures import DynamicsModel

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
		"task": "ThreeBlocks",
		'batch_size': 128,
		"algo": "a2c",
		'lr': 7e-4,
		'eps': 1e-5,
		'alpha': 0.99,
		'gamma': 0.9,
		'use_gae': False,
		'entropy_coef': 0,
		'value_loss_coef': 0.5,
		'max_grad_norm': 0.5,
		'recurrent_policy': True,
		'enable_asm': False,
		'detailed_gmp': False,
		'seed': time.time(),
		'cuda_deterministic':False,
		'num_processes': 1,
		'num_steps': 5,
		'ppo_epoch': 4,
		'num_mini_batch': 5,
		'clip_param': 0.2, 
		'log_interval': 10,
		'save_interval': 100,
		'eval_interval': None,
		'num_env_steps': 1e7,
		'use_splitter': True,
		'split_ratio': 0.2,
		'terminate_unreachable': False,
		'log_dir': '/tmp/gym/',
		'nsamples_per_update': 1024,
		'mode': 'RandomStateEmbeddingPlanner',
		'training': True, 
		'save_dir': './trained_models/',
		'store_true':False,
		'use_proper_time_limits': False,
		'reset_frequency': 0.01,
		'recurrent_policy': False,
		'use_linear_lr_decay': False,
		'rewards': []
	}
	experiment_dict['exp_path'] = "solution_data/" + experiment_dict["exp_id"]

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
	env = PC(experiment_dict)

	action_low = -1
	action_high = 1

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
							  actor_critic.recurrent_hidden_state_size,
							  use_splitter = experiment_dict["use_splitter"],
							  split_ratio = experiment_dict["split_ratio"])

	obs = env.reset()
	transform = list(env.predict_mask)
	random.shuffle(transform)
	reset_obs = obs
	rollouts.obs[0].copy_(obs)
	# rollouts.to(device)

	episode_rewards = deque(maxlen=1000)
	# Create the replay buffer for training world models

	start = time.time()
	no_reward_frame_count = 0
	MAX_NRFC = 100
	num_updates = int(
		experiment_dict["num_env_steps"]) // experiment_dict["num_steps"] // experiment_dict["num_processes"]
	for j in range(num_updates):
		class_index = 0
		for step in range(experiment_dict['num_steps']):
			experiment_dict['num_sampled_nodes']+=1
			# Sample actions
			with torch.no_grad():
				value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
					rollouts.obs[step], rollouts.recurrent_hidden_states[step],
					rollouts.masks[step])

			# action = torch.clamp(action, action_low, action_high)
	
			# print(action)
			# new_action = torch.tensor(np.ones(action.shape))
			# new_action[0][random.randint(0,2)] = 2
			# print(new_action)
			obs, reward, done, infos, inputs, prestate = env.step(action/4.0, terminate_unreachable=experiment_dict['terminate_unreachable'], \
																		  state_estimation=(experiment_dict["mode"]=="StateEstimationPlanner"))

			
			# reward = torch.tensor(obs[0][0].item()+obs[0][1].item()+obs[0][6].item()+obs[0][7].item()+obs[0][12].item()+obs[0][13].item())
			experiment_dict['rewards'].append(reward.item())

			if(reward.item() == 1 or no_reward_frame_count >= MAX_NRFC):
				if(reward.item() == 1):
					class_index = 1
					reward = reward*10
				no_reward_frame_count = 0
				obs = env.reset()

			
			no_reward_frame_count += 1 

			obs_tensor = opt_cuda(obs.type(torch.FloatTensor))
			inputs = opt_cuda(inputs.type(torch.FloatTensor))
			preobs_tensor = opt_cuda(prestate.type(torch.FloatTensor))

			# Insert into buffer for world model training
			filler = opt_cuda(torch.tensor([0]))

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

		rollouts.before_update(class_index)
		value_loss, action_loss, dist_entropy = agent.update(rollouts)
		rollouts.after_update()


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
		if j % experiment_dict['save_interval'] == 0 and len(episode_rewards) > 1:
			with open(experiment_dict['exp_path'] + "/exp_dict.pkl", "wb") as fa:
				pickle.dump(experiment_dict, fa)

		# if (args.eval_interval is not None and len(episode_rewards) > 1
		#         and j % args.eval_interval == 0):
		#     ob_rms = utils.get_vec_normalize(envs).ob_rms
		#     evaluate(actor_critic, ob_rms, args.env_name, args.seed,
		#              args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
	main()
