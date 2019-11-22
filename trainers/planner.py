#!/usr/bin/env python
from __future__ import print_function

from scripts.utils import *
from trainers.dataset import ExperienceReplayBuffer
from trainers.plan_graph import PlanGraph


class Planner:
    def __init__(self, experiment_dict):
        self.experiment_dict = experiment_dict

        # Transfer dict properties to class properties
        self.loss_threshold = self.experiment_dict["loss_threshold"]
        self.growth_factor = self.experiment_dict["growth_factor"]
        self.num_training_epochs = self.experiment_dict["num_training_epochs"]
        self.batch_size = self.experiment_dict["batch_size"]
        self.mode = self.experiment_dict['mode']
        self.task = self.experiment_dict['task']
        self.exp_path = self.experiment_dict['exp_path']
        self.load_path = self.experiment_dict['load_path']
        self.sample_cap = self.experiment_dict['sample_cap']
        self.node_sampling = self.experiment_dict['node_sampling']

        # Create the replay buffer for training world models
        self.experience_replay = ExperienceReplayBuffer()

        EC = getattr(sys.modules[__name__], self.experiment_dict["task"])
        self.environment = EC(experiment_dict)

        # Init the plan graph
        self.graph = PlanGraph(node_sampling=self.node_sampling)
        super(Planner, self).__init__()

    def print_exp_dict(self, verbose=False):
        print("Sampled: " + str(self.experiment_dict['num_sampled_nodes']) + "\t Graph Size: " + str(
            self.experiment_dict['num_graph_nodes']) + "\t WM Loss: " + str(
            self.experiment_dict['world_model_losses'][-1]) + "\t Feasibility: " + str(
            self.experiment_dict['feasibility'][-1]))

    # print("Sampled: "+str(self.experiment_dict['num_sampled_nodes'])+"\t Graph Size: "+str(self.experiment_dict['num_graph_nodes'])+"\t WM Loss: "+str(self.experiment_dict['world_model_losses'][-1]))

    def expand_graph(self, run_index):
        raise NotImplementedError

    def plan(self):
        # Set up the starting position
        start_state = self.environment.get_start_state()
        start_node = self.graph.add_node(start_state, None, None, None)
        run_index = 0

        while True:
            features, states, prestates, actions, action_log_probs, \
            values, feasible, parents, goal, goal_prestate, goal_parent, \
            goal_action, goal_command, commands = self.environment.collect_samples(self.graph)
            if goal is not None:
                # self.environment.set_state(goal)
                # for perspective in self.environment.perspectives:
                #     imageio.imwrite(self.exp_path
                #                     + '/GOAL'
                #                     + ',run_index=' + str(run_index)
                #                     + ',parent_index=' + str(goal_parent)
                #                     + ',node_key=' + str(self.graph.node_key) + '.jpg',
                #                     take_picture(perspective[0], perspective[1], 0, size=512))

                goal_node = self.graph.add_node(goal, goal_prestate, goal_action, goal_parent, command=goal_command)
                plan = self.graph.get_optimal_plan(start_node, goal_node)
                break

            targets = opt_cuda(states)
            pretargets = opt_cuda(prestates)
            actions = opt_cuda(actions)
            action_log_probs = opt_cuda(action_log_probs)
            values = opt_cuda(values)
            parent_nodes = opt_cuda(parents)
            feasible = opt_cuda(feasible)
            combined_perspectives = opt_cuda(torch.cat(features))

            self.experience_replay.bufferadd(combined_perspectives, targets, pretargets, \
                                             actions, action_log_probs, values, feasible, parent_nodes, commands)
            self.expand_graph(run_index)
            self.experiment_dict["num_sampled_nodes"] += self.experiment_dict["nsamples_per_update"]
            if self.experiment_dict["num_sampled_nodes"] > self.sample_cap:
                return None, None, self.experiment_dict
            run_index += 1

        return self.graph, plan, self.experiment_dict
