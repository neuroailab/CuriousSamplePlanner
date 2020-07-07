#!/usr/bin/env python
from __future__ import print_function
import time
import pickle
import torch
import sys
from CuriousSamplePlanner.scripts.utils import *
from CuriousSamplePlanner.trainers.plan_graph import PlanGraph
from CuriousSamplePlanner.trainers.dataset import ExperienceReplayBuffer
from CuriousSamplePlanner.trainers.planner import Planner

class RRTPlanner(Planner):
	def __init__(self, experiment_dict):
		super(RRTPlanner, self).__init__(experiment_dict)

	def save_params(self):
		
		with open(self.experiment_dict['exp_path'] + "/exp_dict.pkl", "wb") as fa:
			pickle.dump(self.experiment_dict, fa)
			fa.close()

		graph_filehandler = open(self.experiment_dict['exp_path'] + "/found_graph.pkl", 'wb')
		pickle.dump(self.graph, graph_filehandler)

	def plan(self):
		# Add starting node to the graph
		self.policy.reset()
		start_state = torch.tensor(self.environment.get_start_state())
		print(start_state.shape)
		start_node = self.graph.add_node(start_state, None, None, None, run_index=0)
		run_index = 1

		# Begin RRT loop
		nodes_added = 0
		while (True):
			print("Samples: "+str(run_index*self.experiment_dict["growth_factor"]))
			start_time = time.time()
			# Select a random point within the configuration space for the objects
			sample_config = self.environment.get_random_config()

			# Find the node that is closest to the sample location
			nearest_node = self.graph.nn(sample_config)

			# Sample a bunch of actions from that node
			results = []
			state_action_dict = {}
			# for _ in range(int(self.experiment_dict["batch_size"]/self.experiment_dict["growth_factor"])):
			for _ in range(1):
				self.environment.set_state(nearest_node.config)
				action, ainfo = self.policy.select_action(sample_config)
				result = self.policy.step(torch.squeeze(action))
				state_action_dict[result[0]] = action
				results.append(result)
			if(self.experiment_dict['debug_timing']):
					self.experiment_dict['action_selection_timings'].append(time.time()-start_time)

			# Sort the results based on proximity to the sample
			sorted(results, key=lambda result: self.graph.l2dist(result[0], sample_config), reverse=True)

			# Select the actions that takes you closest to the selected point
			for result in results:
				(next_state, reward, done, infos) = result
				action = state_action_dict[result[0]]
				ntarget = torch.squeeze(next_state.detach().cpu()).numpy()
				naction = torch.squeeze(action.detach().cpu()).numpy()

				added_node = self.graph.add_node(ntarget, ntarget, naction, nearest_node.node_key, run_index = run_index)
				if(done):
					return self.graph, self.graph.get_optimal_plan(start_node, added_node), self.experiment_dict

				nodes_added+=1
				break

			if(nodes_added>=self.experiment_dict["growth_factor"]):
				nodes_added=0
				self.save_params()
				run_index+=1





