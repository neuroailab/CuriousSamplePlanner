#!/usr/bin/env python
from __future__ import print_function
import pybullet as p
import time
import random
import math
import os
import shutil
import pickle
import collections
from CuriousSamplePlanner.planning_pybullet.motion.motion_planners.discrete import astar
import sys

# Planners
from CuriousSamplePlanner.trainers.random_embedding_planner import RandomEmbeddingPlanner
from CuriousSamplePlanner.trainers.state_estimation_planner import StateEstimationPlanner
from CuriousSamplePlanner.trainers.random_search_planner import RandomSearchPlanner
from CuriousSamplePlanner.trainers.effect_prediction_planner import EffectPredictionPlanner
from CuriousSamplePlanner.trainers.random_state_embedding_planner import RandomStateEmbeddingPlanner
from CuriousSamplePlanner.trainers.recycle_ac_planner import  RecycleACPlanner
from CuriousSamplePlanner.scripts.utils import *
from CuriousSamplePlanner.agent.planning_agent import PlanningAgent


def main(exp_id="no_expid", load_id="no_loadid"):  # control | execute | step

    # Set up the hyperparameters
    experiment_dict = {
        "world_model_losses": [],
        "num_sampled_nodes": 0,
        "num_graph_nodes": 0,
        "num_training_epochs": 30, 
        "learning_rate": 5e-5,
        "sample_cap": 1e7, 
        "batch_size": 128,
        "node_sampling": "uniform",
        "mode": "RecycleACPlanner",
        "feasible_training": True,
        "nsamples_per_update": 1024,
        "training": True, 
        "exp_id": exp_id,
        "load_id": load_id,
        "enable_asm": False, 
        "recycle": False, 
        "growth_factor": 10, 
        "detailed_gmp": False, 
        "task": "BookShelf" 
    }

    lt_dict = {
        "StateEstimationPlanner": 0.003,
        "RandomStateEmbeddingPlanner": 0.0003,
        "EffectPredictionPlanner": 0.001,
        "RandomSearchPlanner": 0,
        "RecycleACPlanner": 0.0003
    }

    experiment_dict["loss_threshold"] = lt_dict[experiment_dict["mode"]]
    experiment_dict['exp_path'] = "./solution_data/" + experiment_dict["exp_id"]
    experiment_dict['load_path'] = "./solution_data/" + experiment_dict["load_id"]
    if (not os.path.isdir("./solution_data")):
        os.mkdir("./solution_data")
    #experiment_dict['exp_path'] = "example_images/" + experiment_dict["exp_id"]
    #experiment_dict['load_path'] = 'example_images/' + experiment_dict["load_id"]

    PC = getattr(sys.modules[__name__], experiment_dict['mode'])
    planner = PC(experiment_dict)


    load = None
    # load = "found_path.pkl"

    if (load == None):
        if (os.path.isdir(experiment_dict['exp_path'])):
            shutil.rmtree(experiment_dict['exp_path'])

        os.mkdir(experiment_dict['exp_path'])
        
        graph, plan, experiment_dict = planner.plan()

        # Save the graph so we can load it back in later
        if(graph is not None):
            graph_filehandler = open(experiment_dict['exp_path'] + "/found_graph.pkl", 'wb')
            filehandler = open(experiment_dict['exp_path'] + "/found_path.pkl", 'wb')

            pickle.dump(graph, graph_filehandler)
            pickle.dump(plan, filehandler)

        stats_filehandler = open(experiment_dict['exp_path'] + "/stats.pkl", 'wb')
        pickle.dump(experiment_dict, stats_filehandler)

    else:
        filehandler = open(experiment_dict['exp_path'] + '/' + load, 'rb')
        plan = pickle.load(filehandler)

        agent = PlanningAgent(planner.environment)
        agent.multistep_plan(plan)


if __name__ == '__main__':
    exp_id = str(sys.argv[1])
    if(len(sys.argv)>3):
        load_id = str(sys.argv[3])
    else:
        load_id = ""

    main(exp_id=exp_id, load_id=load_id)
