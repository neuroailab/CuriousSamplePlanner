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
from planning_pybullet.motion.motion_planners.discrete import astar
import sys

# Planners
from trainers.architectures import DynamicsModel
from trainers.state_estimation_planner import StateEstimationPlanner
from trainers.random_search_planner import RandomSearchPlanner
from trainers.effect_prediction_planner import EffectPredictionPlanner
from trainers.random_state_embedding_planner import RandomStateEmbeddingPlanner
from trainers.ACPlanner import ACPlanner
from scripts.utils import *
from agent.planning_agent import PlanningAgent

from planning_pybullet.pybullet_tools.utils import WorldSaver, enable_gravity, connect, dump_world, set_pose, \
    Pose, Point, Euler, set_default_camera, stable_z, \
    BLOCK_URDF, load_model, wait_for_user, disconnect, DRAKE_IIWA_URDF, user_input, update_state, disable_real_time,inverse_kinematics,end_effector_from_body,approach_from_grasp, get_joints, get_joint_positions


from planning_pybullet.pybullet_tools.utils import multiply, invert, get_pose, set_pose, get_movable_joints, \
    set_joint_positions, add_fixed_constraint_2, enable_real_time, disable_real_time, joint_controller, \
    enable_gravity, get_refine_fn, user_input, wait_for_duration, link_from_name, get_body_name, sample_placement, \
    end_effector_from_body, approach_from_grasp, plan_joint_motion, GraspInfo, Pose, INF, Point, \
    inverse_kinematics, pairwise_collision, remove_fixed_constraint, Attachment, get_sample_fn, \
    step_simulation, refine_path, plan_direct_joint_motion, get_joint_positions, dump_world, get_link_pose,control_joints



def run_csp(experiment_dict, iteration, exp_id="no_expid", load_id="no_loadid"):
    # Set up the hyperparameters

    experiment_dict['exp_path'] = "./solution_data/{}".format(iteration) + experiment_dict["exp_id"]
    experiment_dict['load_path'] = "./solution_data/{}".format(iteration) + experiment_dict["load_id"]
    if not os.path.isdir("./solution_data/{}".format(iteration)):
        os.mkdir("./solution_data/{}".format(iteration))
    # experiment_dict['exp_path'] = "example_images/" + experiment_dict["exp_id"]
    # experiment_dict['load_path'] = 'example_images/' + experiment_dict["load_id"]
    adaptive_batch_lr = {
        "StateEstimationPlanner": 0.003,
        "RandomStateEmbeddingPlanner": 0.00005,
        "EffectPredictionPlanner": 0.001,
        "RandomSearchPlanner": 0
    }
    experiment_dict["loss_threshold"] = adaptive_batch_lr[experiment_dict["mode"]]
    PC = getattr(sys.modules[__name__], experiment_dict['mode'])
    planner = PC(experiment_dict)

    if os.path.isdir(experiment_dict['exp_path']):
        shutil.rmtree(experiment_dict['exp_path'])

    os.mkdir(experiment_dict['exp_path'])

    graph, plan, experiment_dict = planner.plan()

    # Save the graph so we can load it back in later
    if graph is not None:
        # graph_filehandler = open(experiment_dict['exp_path'] + "/found_graph.pkl", 'wb')
        graph_filehandler = open(experiment_dict['exp_path'] + "/found_graph.npy", 'wb')
        filehandler = open(experiment_dict['exp_path'] + "/found_path.pkl", 'wb')

        # pickle.dump(graph, graph_filehandler)
        graph.save(graph_filehandler)
        pickle.dump(plan, filehandler)

    stats_filehandler = open(experiment_dict['exp_path'] + "/stats.pkl", 'wb')
    pickle.dump(experiment_dict, stats_filehandler)

    paths = graph.get_all_paths()
    print('Iteration {}: {}'.format(iteration, paths))
    return paths


def hook(self):
    for (link_object, link, link_transform, _) in self.links:
        if link_object.sphere != None:
            lpos, lquat = p.getBasePositionAndOrientation(link)
            le = p.getEulerFromQuaternion(lquat)
            lpose = Pose(Point(*lpos), Euler(*le))
            set_pose(link_object.sphere, multiply(lpose, link_transform))


def update_dynamics_model(experiment_dict, csp_paths):
    PC = getattr(sys.modules[__name__], experiment_dict['mode'])
    planner = PC(experiment_dict)
    env = planner.environment
    dynamics = DynamicsModel(config_size=env.config_size, action_size=env.action_space_size)
    dynamics = opt_cuda(dynamics)
    dynamics_opt = optim.Adam(dynamics.parameters(), lr=5e-4)  # Only want to finetune

    preds, targets = [], []
    for path in csp_paths:
        env.set_state(path[0].config)
        # final_config = opt_cuda(torch.tensor(path[-1].config).float().unsqueeze(0))
        current = opt_cuda(torch.tensor(path[0].config).float().unsqueeze(0))
        for node in path:
            macroaction = env.macroaction
            feasible, command = macroaction.execute(config=env.get_current_config(), embedding=node.action, sim=True)
            if command is not None:
                command.refine(num_steps=10).execute(time_step=0.001, hook=hook)
                env.run_until_stable(hook=hook, dt=0.01)

            action = opt_cuda(torch.tensor(node.action).unsqueeze(0))
            current = dynamics.forward(current, action)
        preds.append(current)
        targets.append(env.get_current_config())

    preds, targets = opt_cuda(torch.stack(preds)), opt_cuda(torch.stack(targets))
    loss = (preds - targets).pow(2).sum(1).mean()
    dynamics_opt.zero_grad()
    loss.backward()
    dynamics_opt.step()
    torch.save(dynamics.state_dict(), experiment_dict['dynamics_path'])


def cache_iteration(experiment_dict, iteration, exp_id="no_expid", load_id="no_loadid"):
    load = "found_path.pkl"

    # Find the plan and execute it
    experiment_dict['exp_path'] = "./solution_data/{}".format(iteration) + experiment_dict["exp_id"]

    PC = getattr(sys.modules[__name__], experiment_dict['mode'])
    planner = PC(experiment_dict)

    filehandler = open(experiment_dict['exp_path'] + '/' + load, 'rb')
    out_path = experiment_dict['exp_path']
    plan = pickle.load(filehandler)

    agent = PlanningAgent(planner.environment, out_path)
    agent.multistep_plan(plan)


def run_interactive(experiment_dict, exp_id="no_expid", load_id="no_loadid"):  # control | execute | step
    for i in range(experiment_dict['num_iters']):
        csp_paths = run_csp(experiment_dict, i, exp_id, load_id)
        update_dynamics_model(experiment_dict, csp_paths)


def main(exp_id="no_expid", load_id="no_loadid"):
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
        "enable_asm": True,
        "growth_factor": 10,
        "detailed_gmp": False,
        "adaptive_batch": True,
        "num_training_epochs": 30,
        # Stats
        "world_model_losses": [],
        "feasibility": [],
        "num_sampled_nodes": 0,
        "num_graph_nodes": 0,
        "dynamics_path": "out/ThreeBlocks_dynamics.pt",
        "num_iters": 3,
    }

    run_interactive(experiment_dict, exp_id, load_id)
    # cache_iteration(experiment_dict, experiment_dict['num_iters']-1, exp_id, load_id)

if __name__ == '__main__':
    exp_id = str(sys.argv[1])
    if len(sys.argv) > 3:
        load_id = str(sys.argv[3])
    else:
        load_id = ""

    main(exp_id=exp_id, load_id=load_id)
