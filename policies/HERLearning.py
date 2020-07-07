from CuriousSamplePlanner.policies.policy import EnvPolicy
import torch 
import numpy as np
# DDPG imports 
from CuriousSamplePlanner.her.mpi_utils.mpi_utils import sync_networks, sync_grads
from CuriousSamplePlanner.her.rl_modules.replay_buffer import replay_buffer
from CuriousSamplePlanner.her.rl_modules.models import actor, critic
from CuriousSamplePlanner.her.mpi_utils.normalizer import normalizer
from CuriousSamplePlanner.her.her_modules.her import her_sampler
from CuriousSamplePlanner.scripts.utils import *


class HERLearningPolicy(EnvPolicy):
	def __init__(self, *args):


		super(HERLearningPolicy, self).__init__(*args)

		obs = torch.squeeze(self.environment.reset())
		self.ACM = 1
		# close the environment
		env_params = {
			'obs': obs.shape[0],
			'goal': obs.shape[0],
			'action': self.environment.action_space.shape[0],
			'action_max': self.environment.action_space.high[0]*self.ACM,
			'max_timesteps': self.environment.max_timesteps
		}
	
		self.env_params = env_params

		# create the network
		self.actor_network = actor(env_params)
		self.critic_network = critic(env_params)

		# sync the networks across the cpus
		sync_networks(self.actor_network)
		sync_networks(self.critic_network)

		# build up the target network
		self.actor_target_network = actor(env_params)
		self.critic_target_network = critic(env_params)

		# load the weights into the target networks
		self.actor_target_network.load_state_dict(self.actor_network.state_dict())
		self.critic_target_network.load_state_dict(self.critic_network.state_dict())

		# create the optimizer
		self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.experiment_dict['actor_lr'])
		self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.experiment_dict['critic_lr'])

		# her sampler
		self.her_module = her_sampler(self.experiment_dict['replay_strategy'], self.experiment_dict['replay_k'], self.environment.compute_reward)

		# create the replay buffer
		self.buffer = replay_buffer(self.env_params, self.experiment_dict['buffer_size'], self.her_module.sample_her_transitions)

		# create the normalizer
		self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.experiment_dict['clip_range'])
		self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.experiment_dict['clip_range'])

		self.num_steps = 0
		


	def got_reward(self):
		raise NotImplementedError

	def select_action(self, obs):
		obs = torch.squeeze(obs)
		input_tensor = self._preproc_inputs(obs, np.zeros(obs.shape))

		pi = self.actor_network(input_tensor)
		action = self._select_actions(pi)
		return opt_cuda(torch.tensor(action)), {}

	def store_results(self, next_state, action, reward, action_infos, prev_state, done, goal):

		# Squeeze Dimensions
		next_state = torch.squeeze(next_state)
		prev_state = torch.squeeze(prev_state)
		action = torch.squeeze(action)
		goal = torch.squeeze(goal)

		if(self.num_steps == 0):
			# reset the rollouts
			self.mb_obs, self.mb_ag, self.mb_g, self.mb_actions = [], [], [], []
			self.ep_obs, self.ep_ag, self.ep_g, self.ep_actions = [], [], [], []
			self.obs = prev_state
			self.ag = prev_state
			self.g = np.zeros(prev_state.shape) #Desired goal is unknown, just set to zero for now

		# append rollouts
		submit = prev_state.detach().numpy()

		self.ep_obs.append(submit.copy())
		self.ep_ag.append(submit.copy())
		self.ep_g.append(self.g.copy())
		self.ep_actions.append(action.detach().numpy().copy())

		self.num_steps+=1


		# Slight modification of the HER algorithm to allow for environments that terminate after the goal is reached
		
		if(done):
			if(goal != None):
				ns = goal
			else:
				ns = next_state
			# Need to buffer with the last move
			while(len(self.ep_obs) < self.env_params['max_timesteps']):
				self.ep_obs.append(ns.detach().numpy().copy())
				self.ep_ag.append(ns.detach().numpy().copy())
				self.ep_g.append(self.g.copy())
				self.ep_actions.append(action.detach().numpy().copy())

			self.ep_obs.append(ns.numpy().copy())
			self.ep_ag.append(ns.numpy().copy())
			
			self.mb_obs.append(self.ep_obs)
			self.mb_ag.append(self.ep_ag)
			self.mb_g.append(self.ep_g)
			self.mb_actions.append(self.ep_actions)

			#Convert the lists to arrays
			self.mb_obs = np.array(self.mb_obs)
			self.mb_ag = np.array(self.mb_ag)
			self.mb_g = np.array(self.mb_g)
			self.mb_actions = np.array(self.mb_actions)

			self.buffer.store_episode([self.mb_obs, self.mb_ag, self.mb_g, self.mb_actions])
			self._update_normalizer([self.mb_obs, self.mb_ag, self.mb_g, self.mb_actions])

			# reset the rollouts
			self.mb_obs, self.mb_ag, self.mb_g, self.mb_actions = [], [], [], []
			self.ep_obs, self.ep_ag, self.ep_g, self.ep_actions = [], [], [], []
			self.obs = prev_state
			self.ag = prev_state
			self.g = np.zeros(prev_state.shape) #Desired goal is unknown, just set to zero for now

	# pre_process the inputs
	def _preproc_inputs(self, obs, g):
		obs_norm = self.o_norm.normalize(obs)
		g_norm = self.g_norm.normalize(g)
		# concatenate the stuffs
		inputs = np.concatenate([obs_norm, g_norm])
		inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
	  
		return inputs
	
	# this function will choose action for the agent and do the exploration
	def _select_actions(self, pi):
		action = pi.detach().cpu().numpy().squeeze()
		# add the gaussian
		action += self.experiment_dict['noise_eps'] * self.env_params['action_max'] * np.random.randn(*action.shape)
		action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
		# random actions...
		random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
											size=self.env_params['action'])
		# choose if use the random actions
		action += np.random.binomial(1, self.experiment_dict['random_eps'], 1)[0] * (random_actions - action)
		return action

	# update the normalizer
	def _update_normalizer(self, episode_batch):
		mb_obs, mb_ag, mb_g, mb_actions = episode_batch
		mb_obs_next = mb_obs[:, 1:, :]
		mb_ag_next = mb_ag[:, 1:, :]
		# get the number of normalization transitions
		num_transitions = mb_actions.shape[1]
		# create the new buffer to store them
		buffer_temp = {'obs': mb_obs, 
					   'ag': mb_ag,
					   'g': mb_g, 
					   'actions': mb_actions, 
					   'obs_next': mb_obs_next,
					   'ag_next': mb_ag_next,
					   }
		transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
		obs, g = transitions['obs'], transitions['g']
		# pre process the obs and g
		transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
		# update
		self.o_norm.update(transitions['obs'])
		self.g_norm.update(transitions['g'])
		# recompute the stats
		self.o_norm.recompute_stats()
		self.g_norm.recompute_stats()

	def _preproc_og(self, o, g):
		o = np.clip(o, -self.experiment_dict['clip_obs'], self.experiment_dict['clip_obs'])
		g = np.clip(g, -self.experiment_dict['clip_obs'], self.experiment_dict['clip_obs'])
		return o, g

	# soft update
	def _soft_update_target_network(self, target, source):
		for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_((1 - self.experiment_dict['polyak']) * param.data + self.experiment_dict['polyak'] * target_param.data)

	# update the network
	def _update_network(self):
		# sample the episodes
		transitions = self.buffer.sample(self.experiment_dict['batch_size'])
		# pre-process the observation and goal
		o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
		transitions['obs'], transitions['g'] = self._preproc_og(o, g)
		transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
		# start to do the update
		obs_norm = self.o_norm.normalize(transitions['obs'])
		g_norm = self.g_norm.normalize(transitions['g'])
		inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
		obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
		g_next_norm = self.g_norm.normalize(transitions['g_next'])
		inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
		# transfer them into the tensor
		inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
		inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
		actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
		r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) 
	
		# calculate the target Q value function
		with torch.no_grad():
			# do the normalization
			# concatenate the stuffs
			actions_next = self.actor_target_network(inputs_next_norm_tensor)
			q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
			q_next_value = q_next_value.detach()
			target_q_value = r_tensor + self.experiment_dict['gamma'] * q_next_value
			target_q_value = target_q_value.detach()
			# clip the q value
			clip_return = 1 / (1 - self.experiment_dict['gamma'])
			target_q_value = torch.clamp(target_q_value, -clip_return, 0)

		# the q loss
		real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)

		# print(torch.max(target_q_value))
		# print(torch.max(real_q_value))
		critic_loss = (target_q_value - real_q_value).pow(2).mean()
		# print(torch.max(critic_loss))

		# the actor loss
		actions_real = self.actor_network(inputs_norm_tensor)
		actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
		actor_loss += self.experiment_dict['action_l2'] * (actions_real / self.env_params['action_max']).pow(2).mean()

		# start to update the network
		self.actor_optim.zero_grad()
		actor_loss.backward()
		sync_grads(self.actor_network)
		self.actor_optim.step()
		# update the critic_network
		self.critic_optim.zero_grad()
		critic_loss.backward()
		sync_grads(self.critic_network)
		self.critic_optim.step()

		return critic_loss.item(), actor_loss.item()

	def update(self, total_numsteps):
		c_losses = []
		a_losses = []
		for _ in range(self.experiment_dict['n_batches']):
			# train the network
			c_loss, a_loss = self._update_network()
			c_losses.append(c_loss)
			a_losses.append(a_loss)
			# soft update
			self._soft_update_target_network(self.actor_target_network, self.actor_network)
			self._soft_update_target_network(self.critic_target_network, self.critic_network)


		return np.mean(np.array(c_losses)), np.mean(np.array(a_losses)), None
		
