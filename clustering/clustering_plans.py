import os
import torch
import numpy as np
# First, get all of the files
from os import listdir
from os.path import isfile, join

import sklearn
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import rankdata
import numpy as np
from sklearn.linear_model import LinearRegression
from collections import defaultdict
import math
import networkx as nx
import matplotlib as mpl


# Planners
from CuriousSamplePlanner.trainers.state_estimation_planner import StateEstimationPlanner
from CuriousSamplePlanner.trainers.random_search_planner import RandomSearchPlanner
from CuriousSamplePlanner.trainers.effect_prediction_planner import EffectPredictionPlanner
from CuriousSamplePlanner.trainers.random_state_embedding_planner import RandomStateEmbeddingPlanner
from CuriousSamplePlanner.scripts.utils import *
from CuriousSamplePlanner.agent.planning_agent import PlanningAgent

from CuriousSamplePlanner.policies.policy import EnvPolicy
import torch 
import numpy as np
from CuriousSamplePlanner.scripts.utils import *


class RegressionPolicy(EnvPolicy):
	def __init__(self,  *args):
		super(RegressionPolicy, self).__init__(*args)

	def adjust(self, X, min_rv, max_rv, starting_rv=-1, ending_rv=1):
		"""
			Take a random variable X that has support between -1 and 1
			Transform into a random variable that has support between min and max
		"""
		return (((X-starting_rv)/(ending_rv-starting_rv))% 1)*(max_rv-min_rv)+min_rv

	def unparameterize(self, arr):
		for i in [2]:
			x = arr[:, i]
			y = arr[:,i+1]
			unscaled_r = np.sqrt(np.power(x, 2)+np.power(y, 2))
			unscaled_theta = np.arctan2(y,x)
			r = self.adjust(unscaled_r, -1, 1, starting_rv = self.min_reach_horiz, ending_rv=self.max_reach_horiz)
			theta = self.adjust(unscaled_theta, -1, 1, starting_rv = -math.pi, ending_rv = math.pi)
			arr[:, i] = theta
			arr[:, i+1] = r
		return arr


	def select_action(self, obs):
		results = self.regression_model.predict(obs)
		return opt_cuda(torch.tensor(self.unparameterize(results)))


def rank(truth):
	return rankdata(truth, method='dense')

def clusters_from_states(truth):
	total = np.zeros(truth.shape[0])
	for i in range(2, truth.shape[1], 6):
		total*=100
		total+=10*np.array((truth[:, i]>0.16), dtype=(np.int))+np.array((truth[:, i]>0.06), dtype=(np.int))
	return rank(total)


def homogeneity(truth, clustering):
	return sklearn.metrics.homogeneity_score(truth, clustering)

# def cluster_purity(truth, clustering):
# 	"""
# 		Input is two one dimensional numpy arrays
# 	"""
# 	ratios = []

# 	clusters = np.unique(truth)
# 	print(clusters)
# 	for cluster_index in clusters:
# 		found_clusters,  = np.where(truth==cluster_index)
# 		cluster_mode = stats.mode(clustering[found_clusters])
# 		print(cluster_mode)
# 		mode_clusters, = np.where(clustering[found_clusters] == cluster_mode[0])

# 		ratios.append(len(mode_clusters)/len(found_clusters))
# 	return sum(ratios)/float(len(ratios))


# Constants
pathname = "./data_collection/solution_data/"
exp_name = "trajs_threeblocks"

path = pathname+exp_name
data_files = [f for f in listdir(path) if isfile(join(path, f))]

total_length = 0
traj_states = []
traj_actions = []
traj_rewards = []
traj_lens = []

g_states = []
g_actions = []
g_aux = []
g_colors = []


print("Num Files: "+str(len(data_files)))
for file in data_files:
	data = torch.load(pathname+exp_name+"/"+file)
	state = data['states']
	# Add stepping states/actions/aux
	g_states.append(torch.squeeze(torch.tensor(data['states']), dim=0))	
	g_actions.append(torch.squeeze(torch.tensor(data['actions']), dim=0))	
	aux = torch.zeros((data['actions'].shape[1]))
	aux[0] = -1 # Start of the trajectory
	g_aux.append(aux)

	# Add goal states/actions/aux
	g_states.append(torch.squeeze(torch.tensor(data['goals']), dim=0))	
	g_actions.append(torch.squeeze(torch.zeros(data['actions'].shape), dim=0))	
	aux = torch.ones((1)) # End of the trajectory
	g_aux.append(aux)


catted_states = torch.cat(g_states,0)
catted_actions = torch.cat(g_actions,0)
catted_aux = torch.cat(g_aux, 0)
colors = clusters_from_states(catted_states)

print(catted_aux)

# EXPLAINED VARIANCE
# catted_actions = torch.cat(g_actions,0)
# catted_rep = torch.cat([catted_states, catted_actions], dim=1)

# print("Shapes")
# print(catted_states.shape)
# print(catted_actions.shape)
# print(catted_rep.shape)

# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(catted_states)

# pca = PCA(n_components=12)
# pca.fit(scaled_data)
# plt.plot(pca.explained_variance_ratio_)
# plt.ylabel("Explained Variance")
# plt.xlabel("# Components")
# plt.title("2-Stack Explained Variance")
# plt.show()


# CLUSTER SCORE CURVES
exps = []
pca_nums = [0, 2, 3, 5, 10]
for pca_num in pca_nums:
	if(pca_num != 0):	
		pca_pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=pca_num))])

		results = pca_pipeline.fit_transform(catted_states)
	else:
		results = catted_states
	runs = []
	for run in range(5):
		homogeneity_scores = []
		for i in range(1, 20):
			scale_pipeline = Pipeline([('scaling', StandardScaler())])
			scaled_catted_states = scale_pipeline.fit_transform(results)
			kmeans_results = KMeans(n_clusters=i).fit_predict(scaled_catted_states)
			
			homo = homogeneity(colors, kmeans_results)
			homogeneity_scores.append(homo)
			print("Cluster Purity for "+str(i)+" clusters: "+str(homo))
		runs.append(homogeneity_scores)
	exps.append(np.expand_dims(np.mean(np.array(runs), axis=0),axis=0) )

# exps = np.transpose(np.concatenate(exps, axis=0))
plt.figure()
for exp in exps:
	print(exp)
	plt.plot(range(1, len(exp[0])+1), exp[0])
plt.title("Cluster Homogeneity Three Blocks")
plt.ylabel("Homogeneity")
plt.xlabel("Num Clusters")
pca_nums[0] = "No PCA"
plt.legend(pca_nums)
plt.show()




# 3D visualization
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# pca_pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=3))])
# results = pca_pipeline.fit_transform(catted_states)
# plt.title("Three Stack No Goal (Actions)")
# ax.scatter(results[:,0], results[:,1], results[:,2], c=colors)
# plt.show()


# REGRESSION Analysis

# # First, select out 2 real clusters of action and state data
# final_cluster_index, = np.where(colors == 2)
# final_states = catted_states[list(final_cluster_index-1), :]
# final_actions = catted_actions[list(final_cluster_index-1), :]

# # print(final_states.shape)
# # print(final_actions.shape)
# def adjust(X, min_rv, max_rv, starting_rv=-1, ending_rv=1):
# 	"""
# 		Take a random variable X that has support between -1 and 1
# 		Transform into a random variable that has support between min and max
# 	"""
# 	return (((X-starting_rv)/(ending_rv-starting_rv))% 1)*(max_rv-min_rv)+min_rv

# max_reach_horiz = 0.45
# min_reach_horiz = 0.4
# def reparameterize(arr):
# 	for i in [2]:
# 		r = adjust(arr[:,i+1], min_reach_horiz, max_reach_horiz)
# 		theta = adjust(arr[:, i], -math.pi, math.pi)
# 		x = r*np.cos(theta) 
# 		y = r*np.sin(theta)
# 		arr[:, i] = x
# 		arr[:, i+1] = y
# 	return arr



# final_actions = reparameterize(final_actions)

# regressor = LinearRegression(normalize=True)  
# regressor.fit(final_states, final_actions) #training the algorithm

# # Second, plot every variable in the action as a function of every variable in the state
# x = np.linspace(-1,1,100)
# for i in [0]:
# 	for j in range(final_actions.shape[1]):
# 		print(i, j)
# 		plt.figure()
# 		plt.scatter(final_states[:, i], final_actions[:, j])
# 		plt.plot(x, regressor.coef_[j][i]*x+regressor.intercept_[j], '-r')
# 		plt.title("Combo: "+str(i)+", "+str(j))
# plt.show()


# Build a Cluster Graph

# class ClusterNode():
# 	def __init__(self):
# 		pass

# 	@property
# 	def reward(self):
# 		print(self.aux)
# 		return torch.sum(self.aux)/float(self.aux.shape[0])
	
# 	@property
# 	def num_samples(self):
# 		return self.states.shape[0]
	
# 	def add_points(self, states, actions, aux):
# 		self.states = states
# 		self.actions = actions
# 		self.aux = aux


# class ClusterGraph():
# 	def __init__(self, states, actions, aux, manual=True):
# 		self.nodes = defaultdict(lambda: ClusterNode())
# 		self.edges = defaultdict(lambda: defaultdict(lambda: 0))
# 		self.states = states
# 		self.actions = actions
# 		self.aux = aux
# 		if(manual):
# 			self.cluster_assignments = self.manual_clustering()
# 		else:
# 			self.cluster_assignments = self.auto_clustering()
# 		self.create_nodes()
# 		self.create_edges()

# 	def rank(self, truth):
# 		return rankdata(truth, method='dense')

# 	def manual_clustering(self):
# 		total = np.zeros(self.states.shape[0])
# 		for i in range(2, self.states.shape[1], 6):
# 			total*=100
# 			total+=10*np.array((self.states[:, i]>0.16), dtype=(np.int))+np.array((self.states[:, i]>0.06), dtype=(np.int))
# 		print(self.rank(total))
# 		return self.rank(total)

# 	def auto_clustering(self, obs):
# 		scale_pipeline = Pipeline([('scaling', StandardScaler())])
# 		scaled_catted_states = scale_pipeline.fit_transform(self.states)
# 		self.kmeans = KMeans(n_clusters=i)
# 		kmeans_results = self.kmeans.fit_predict(scaled_catted_states)
# 		return kmeans_results

# 	def create_nodes(self):
# 		for cluster_index in np.unique(self.cluster_assignments):
# 			final_cluster_index, = np.where(self.cluster_assignments == cluster_index)
# 			cluster_states = self.states[list(final_cluster_index), :]
# 			cluster_actions = self.actions[list(final_cluster_index), :]
# 			cluster_aux = self.aux[list(final_cluster_index)]
# 			self.nodes[cluster_index].add_points(cluster_states, cluster_actions, cluster_aux)

# 		print("Finished Creating Nodes")

# 	def create_edges(self):
# 		# Need to loop through each point to count
# 		for sample_index in range(self.cluster_assignments.shape[0]):
# 			if(self.aux[sample_index] == -1):
# 				# Start (Only outgoing)
# 				current_n = self.nodes[self.cluster_assignments[sample_index]]
# 				after_n = self.nodes[self.cluster_assignments[sample_index+1]]
# 				self.edges[current_n][after_n]+=1

# 			elif(self.aux[sample_index] == 1):
# 				# Goal (Only Incoming)
# 				before_n = self.nodes[self.cluster_assignments[sample_index-1]]
# 				current_n = self.nodes[self.cluster_assignments[sample_index]]
# 				self.edges[before_n][current_n]+=1
# 			else:
# 				# Intermediate (Both incoming and outgoing)
# 				current_n = self.nodes[self.cluster_assignments[sample_index]]
# 				after_n = self.nodes[self.cluster_assignments[sample_index+1]]
# 				before_n = self.nodes[self.cluster_assignments[sample_index-1]]
# 				self.edges[current_n][after_n]+=1
# 				self.edges[before_n][current_n]+=1


# 	def visualize_graph(self):
# 		"""
# 			Visualize the simplified graph
# 		"""
# 		# Spawn the graph
# 		G=nx.DiGraph()

# 		# Add nodes
# 		print("Num nodes: "+str(len(self.nodes.values())))
# 		node_sizes = []
# 		node_colors  = []
# 		for key in self.nodes.values():
# 			node_colors.append(key.reward)
# 			G.add_node(key)
# 			node_sizes.append(key.num_samples)

# 		edge_alphas = []
# 		for source, dest_dict in self.edges.items():
# 			for dest, count in dest_dict.items():
# 				G.add_edge(source, dest)
# 				edge_alphas.append(count)
# 		edge_alphas = [3*float(a)/max(edge_alphas) for a in edge_alphas]
# 		node_sizes = [300*float(n)/max(node_sizes) for n in node_sizes]
# 		N = len(node_sizes)
# 		M = len(edge_alphas)
# 		k = 3/math.sqrt(float(N))
# 		print(node_colors)

# 		# node_sizes = [3 + 10 * i for i in range(len(G))]
# 		# edge_colors = range(2, M + 2)
# 		# edge_alphas = [(5 + i) / (M + 4) for i in range(M)]

# 		# nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='blue')
# 		# edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->',
# 		#                                arrowsize=10, edge_color=edge_colors,
# 		#                                edge_cmap=plt.cm.Blues, width=2)
# 		pos = nx.layout.spring_layout(G, k=k, iterations=100)
# 		nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes,  node_color = node_colors,  cmap=plt.cm.winter)
# 		edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->', arrowsize=10, width=edge_alphas)

# 		# set alpha value for each edge
# 		# for i in range(M):
# 		#     edges[i].set_alpha(edge_alphas[i])

# 		# pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
# 		# pc.set_array(edge_colors)

# 		ax = plt.gca()
# 		ax.set_axis_off()
# 		plt.show()
		

# cluster_graph = ClusterGraph(catted_states, catted_actions, catted_aux)
# cluster_graph.visualize_graph()


# Third, test in an actual environment
# load = "found_pah.pkl"
# exp_id = "testing_regression"
# load_id = "None"

# experiment_dict = {
#     # Hyps
#     "task": "TwoBlocks",
#     "policy": "RandomPolicy",
#     "policy_path": "/mnt/fs0/arc11_2/policy_data_new/normalize_returns_4_update=1/",
#     "return_on_solution": True,
#     "learning_rate": 5e-5,
#     "wm_learning_rate": 7e-4,
#     "sample_cap": 1e7, 
#     "batch_size": 128,
#     'actor_lr': 1e-4,
#     'critic_lr': 1e-3,
#     "node_sampling": "uniform",
#     "mode": "RandomStateEmbeddingPlanner",
#     "feasible_training": True,
#     "nsamples_per_update": 1024,
#     "training": False,
#     'exploration_end': 100, 
#     "exp_id": exp_id,
#     "load_id": load_id,
#     'noise_scale': 0.3,
#     'final_noise_scale': 0.05,
#     'update_interval' : 1,
#     "enable_asm": False, 
#     "growth_factor": 10,
#     "detailed_gmp": False, 
#     "adaptive_batch": True,
#     "num_training_epochs": 30,
#     "infeasible_penalty" : 0,
#     'tau': 0.001,
#     'reward_size': 100,
#     'hidden_size': 64,
#     'use_splitter': True, # Can't use splitter on ppo or a2c because they are on-policy algorithms
#     'split': 0.5,
#     'gamma': 0.5,
#     'ou_noise': True,
#     'param_noise': False,
#     'updates_per_step': 1,
#     'replay_size': 100000,
#     # DRL-Specific
#     'recurrent_policy': False,
#     'algo': 'ppo',
#     'value_loss_coef': 0.5,
#     'reward_alpha': 0,
#     'eps': 5e-5,
#     'entropy_coef': 0,
#     'alpha': 0.99,
#     'max_grad_norm': 0.5,
#     'num_steps': 128,
#     'num_env_steps': 5e5,
#     'use_linear_lr_decay': True,
#     'reset_frequency': 1e-4,
#     'terminate_unreachable': False,
#     'use_gae': False,
#     'use_proper_time_limits': False,
#     'log_interval': 1,
#     'save_interval': 10,
#     'clip_param': 0.2, 
#     'ppo_epoch': 5,
#     'num_mini_batch': 8,
#     # stats
#     "world_model_losses": [],
#     "feasibility":[],
#     "rewards": [],
#     "num_sampled_nodes": 0,
#     "num_graph_nodes": 0,
# }

# experiment_dict['exp_path'] = "./solution_data/" + experiment_dict["exp_id"]
# if (not os.path.isdir("./solution_data")):
#     os.mkdir("./solution_data")
    
# if (os.path.isdir(experiment_dict['exp_path'])):
#     # shutil.rmtree(experiment_dict['exp_path'])
#     pass
# else:
#     os.mkdir(experiment_dict['exp_path'])

# experiment_dict['exp_path'] = "./solution_data/" + experiment_dict["exp_id"]
# experiment_dict['load_path'] = "./solution_data/" + experiment_dict["load_id"]
# if (not os.path.isdir("./solution_data")):
#     os.mkdir("./solution_data")
# #experiment_dict['exp_path'] = "example_images/" + experiment_dict["exp_id"]
# #experiment_dict['load_path'] = 'example_images/' + experiment_dict["load_id"]
# adaptive_batch_lr = {
#     "StateEstimationPlanner": 0.003,
#     "RandomStateEmbeddingPlanner": 0.0005,
#     "EffectPredictionPlanner": 0.001,
#     "RandomSearchPlanner": 0 
# }
# experiment_dict["loss_threshold"] = adaptive_batch_lr[experiment_dict["mode"]]
# PC = getattr(sys.modules[__name__], experiment_dict['mode'])
# planner = PC(experiment_dict)

# ## Add in custom policy
# rp = RegressionPolicy(experiment_dict, planner.environment)
# rp.regression_model=regressor
# rp.max_reach_horiz=max_reach_horiz
# rp.min_reach_horiz=min_reach_horiz 
# planner.policy = rp

# graph, plan, experiment_dict = planner.plan()






