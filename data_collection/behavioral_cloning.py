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
from CuriousSamplePlanner.planning_pybullet.pybullet_tools.utils import WorldSaver, enable_gravity, connect, dump_world, set_pose, \
    Pose, Point, Euler, set_default_camera, stable_z, \
    BLOCK_URDF, load_model, wait_for_user, disconnect, DRAKE_IIWA_URDF, user_input, update_state, disable_real_time,inverse_kinematics,end_effector_from_body,approach_from_grasp, get_joints, get_joint_positions


# Planners
from CuriousSamplePlanner.trainers.state_estimation_planner import StateEstimationPlanner
from CuriousSamplePlanner.trainers.random_search_planner import RandomSearchPlanner
from CuriousSamplePlanner.trainers.effect_prediction_planner import EffectPredictionPlanner
from CuriousSamplePlanner.trainers.random_state_embedding_planner import RandomStateEmbeddingPlanner
from CuriousSamplePlanner.trainers.ACPlanner import ACPlanner
from CuriousSamplePlanner.scripts.utils import *
from CuriousSamplePlanner.agent.planning_agent import PlanningAgent


def main(exp_id="no_expid", load_id="no_loadid"):  # control | execute | step

    # load = "found_pah.pkl"
    load = None

    
    # Set up the hyperparameters
    experiment_dict = {
        # Hyps
        "task": "ThreeBlocks",
        "learning_rate": 5e-5,  
        "sample_cap": 1e7, 
        "batch_size": 128,
        "node_sampling": "softmax",
        "mode": "RandomStateEmbeddingPlanner",
        "feasible_training": True,
        "nsamples_per_update": 1024,
        "training": True, 
        "exp_id": exp_id,
        "load_id": load_id,
        "enable_asm": False, 
        "growth_factor": 10,
        "detailed_gmp": False, 
        "adaptive_batch": True,
        "num_training_epochs": 30,
        "infeasible_penalty" : 0,
        # Stats
        "world_model_losses": [],
        "feasibility":[],
        "num_sampled_nodes": 0,
        "num_graph_nodes": 0,
    }

    experiment_dict['exp_path'] = "./solution_data/" + experiment_dict["exp_id"]
    if (not os.path.isdir("./solution_data")):
        os.mkdir("./solution_data")
        
    if (os.path.isdir(experiment_dict['exp_path'])):
        # shutil.rmtree(experiment_dict['exp_path'])
        pass
    else:
        os.mkdir(experiment_dict['exp_path'])
    


    for solution_index in range(2865, 10000):

        experiment_dict['exp_path'] = "./solution_data/" + experiment_dict["exp_id"]
        experiment_dict['load_path'] = "./solution_data/" + experiment_dict["load_id"]
        if (not os.path.isdir("./solution_data")):
            os.mkdir("./solution_data")
        #experiment_dict['exp_path'] = "example_images/" + experiment_dict["exp_id"]
        #experiment_dict['load_path'] = 'example_images/' + experiment_dict["load_id"]
        adaptive_batch_lr = {
            "StateEstimationPlanner": 0.003,
            "RandomStateEmbeddingPlanner": 0.00005,
            "EffectPredictionPlanner": 0.001,
            "RandomSearchPlanner": 0 
        }
        experiment_dict["loss_threshold"] = adaptive_batch_lr[experiment_dict["mode"]]
        PC = getattr(sys.modules[__name__], experiment_dict['mode'])
        planner = PC(experiment_dict)

        graph, plan, experiment_dict = planner.plan()

        # Save the graph so we can load it back in later
        if(graph is not None):
            graph_filehandler = open(experiment_dict['exp_path'] + "/found_graph_"+str(solution_index)+".pkl", 'wb')
            filehandler = open(experiment_dict['exp_path'] + "/found_path_"+str(solution_index)+".pkl", 'wb')
            pickle.dump(graph, graph_filehandler)
            pickle.dump(plan, filehandler)
        
        disconnect()


if __name__ == '__main__':
    exp_id = str(sys.argv[1])
    if(len(sys.argv)>3):
        load_id = str(sys.argv[3])
    else:
        load_id = ""

    main(exp_id=exp_id, load_id=load_id)