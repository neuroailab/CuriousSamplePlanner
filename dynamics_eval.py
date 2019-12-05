import sys
import random

import imageio
import numpy as np
import os
import os.path as osp

import shutil
import torch as th
import argparse

from cherry import envs

from dynamics_train import vanilla_mpc, cross_entropy_method, model_predictive_path_integral
from scripts.utils import take_picture, opt_cuda
from tasks.three_block_stack import ThreeBlocks
from trainers.architectures import DynamicsModel, FactoredDynamicsModel, DynamicsCuriosityModel, RNDCuriosityModel
from trainers.plan_graph import PlanGraph

import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-dynamics-path", type=str, required=True, help="Path to dynamics model weights")
    parser.add_argument("-curiosity-path", type=str, required=True, help="Path to curiosity model weights")
    parser.add_argument("-out-path", type=str, required=True, help="Path where output eval data can be stored")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed (default: 1234)")
    parser.add_argument("--planning-mode", type=int, default=1, help="Planning algorithm (0:MPC 1:CEM 2:MPPI)")
    parser.add_argument("--curiosity-mode", type=int, default=1, help="Curiosity metric (0:FD 1:RND)")
    parser.add_argument("--mpc-samples", type=int, default=2000,
                        help="Number of random plan samples for MPC (default: 1000)")
    parser.add_argument("--mpc-horizon", type=int, default=3,
                        help="Planning horizon for MPC (default: 5)")
    args = parser.parse_args()
    return args


def save_env_image(env, path, episode, step):
    for perspective in env.perspectives:
        imageio.imwrite(osp.join(path, '{}_{}.jpg'.format(episode, step)), take_picture(perspective[0], perspective[1], 0, size=512))


def save_curious_path_images(args):
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
        "exp_id": 0,
        "load_id": 0,
        "enable_asm": False,
        "growth_factor": 10,
        "detailed_gmp": True,
        "adaptive_batch": True,
        "num_training_epochs": 30,
        # Stats
        "world_model_losses": [],
        "feasibility": [],
        "num_sampled_nodes": 0,
        "num_graph_nodes": 0,
        "dynamics_path": "",
    }

    env_name = 'ThreeBlocks'
    env = ThreeBlocks(experiment_dict)

    env = envs.Torch(env)

    # obj_factored = True
    obj_factored = False
    if not obj_factored:
        print('Using standard dynamics model')
        dynamics = DynamicsModel(config_size=env.config_size, action_size=env.action_space_size)
    else:
        print('Using factored object dynamics model')
        dynamics = FactoredDynamicsModel(config_size=env.config_size, action_size=env.action_space_size)

    if args.curiosity_mode == 0:
        curiosity = DynamicsCuriosityModel(env)
    elif args.curiosity_mode == 1:
        curiosity = RNDCuriosityModel(env)
    else:
        print("Unknown curiosity metric specified: {}".format(args.curiosity_mode))
        sys.exit(0)
    dynamics = opt_cuda(dynamics)
    curiosity = opt_cuda(curiosity)

    dynamics.load_state_dict(th.load(args.dynamics_path, map_location='cpu'))
    curiosity.load_state_dict(th.load(args.curiosity_path, map_location='cpu'))

    dynamics.eval()
    curiosity.eval()

    if args.planning_mode == 0:
        print('Running vanilla MPC...')
        get_action = lambda state: vanilla_mpc(args, opt_cuda(state), env.action_space, dynamics, curiosity).cpu()
    elif args.planning_mode == 1:
        print('Running cross-entropy method...')
        get_action = lambda state: cross_entropy_method(args, opt_cuda(state), env.action_space, dynamics, curiosity).cpu()
    elif args.planning_mode == 2:
        print('Running model-predictive path integral control...')
        get_action = lambda state: model_predictive_path_integral(args, opt_cuda(state), env.action_space, dynamics,
                                                                  curiosity).cpu()
    else:
        print("Unknown planning mode: {}".format(args.planning_mode))
        sys.exit(0)

    if osp.isdir(args.out_path):
        shutil.rmtree(args.out_path)
    os.mkdir(args.out_path)


    num_episodes = 1
    max_steps = 20
    num_objs = 3
    obj_size = 6
    errors = np.zeros((num_episodes, max_steps))
    os_errors = np.zeros((num_episodes, max_steps))
    obj_os_errors = np.zeros((num_episodes, num_objs, max_steps))
    obj_errors = np.zeros((num_episodes, num_objs, max_steps))

    for e in range(num_episodes):
        num_steps = 0
        obs = env.reset()
        pred_obs = obs.clone()
        done = False
        print('Running evaluation episode {}'.format(e + 1))
        while not done:
            save_env_image(env, args.out_path, e, num_steps)
            action = get_action(obs)
            with th.no_grad():
                os_pred_obs = dynamics.forward(obs, action.unsqueeze(0))

            obs, reward, done, _ = env.step(action)
            with th.no_grad():
                pred_obs = dynamics.forward(pred_obs, action.unsqueeze(0))

            error = (pred_obs - obs).pow(2).mean()
            errors[e, num_steps] = error.item()
            obj_error = list(th.split((pred_obs - obs).pow(2), obj_size, dim=1))
            for i, oe in enumerate(obj_error):
                obj_errors[e, i, num_steps] = oe.mean().item()

            os_error = (os_pred_obs - obs).pow(2).mean()
            os_errors[e, num_steps] = os_error.item()
            obj_os_error = list(th.split((os_pred_obs - obs).pow(2), obj_size, dim=1))
            for i, oe in enumerate(obj_os_error):
                obj_os_errors[e, i, num_steps] = oe.mean().item()

            num_steps += 1

            if num_steps >= max_steps:
                done = True

    data = [errors, os_errors]
    labels = ['Global error', 'One-step error']
    for e, l in zip(data, labels):
        means, stds = np.mean(e, axis=0), np.std(e, axis=0)

        plt.plot(range(max_steps), means, label=l)
        top = means + 1.96 * (stds / num_episodes)
        bot = means - 1.96 * (stds / num_episodes)
        plt.fill_between(range(max_steps), top, bot, alpha=0.5)

    order = ['green', 'red', 'blue']
    for o in range(num_objs):
        # obj_data = [obj_errors[:, o, :], obj_os_errors[:, o, :]]
        obj_data = [obj_os_errors[:, o, :]]
        # obj_labels = ['Global {} error'.format(order[o]), 'One-step {} error'.format(order[o])]
        obj_labels = ['One-step {} error'.format(order[o])]
        for e, l in zip(obj_data, obj_labels):
            means, stds = np.mean(e, axis=0), np.std(e, axis=0)

            plt.plot(range(max_steps), means, label=l)
            top = means + 1.96 * (stds / num_episodes)
            bot = means - 1.96 * (stds / num_episodes)
            plt.fill_between(range(max_steps), top, bot, alpha=0.5)

    plt.grid()
    plt.xlabel('Steps')
    plt.ylabel('MSE')
    plt.title('Error Propagation across {} Episodes'.format(num_episodes))
    plt.legend()
    plt.show()

def csp_graph_errors():
    graph_path = ''
    plan_graph = PlanGraph(plan_graph_path=graph_path)
    paths = plan_graph.get_all_paths()


def main(args):
    save_curious_path_images(args)


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    main(args)