#!/usr/bin/env python3
import pdb
import sys
import gym
import random
import argparse
import numpy as np
import pybullet
import pybullet_envs

import torch as th
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.uniform import Uniform
from torch.distributions.multivariate_normal import MultivariateNormal

import cherry as ch
from cherry import envs

from scripts.utils import opt_cuda
from tasks.three_block_stack import ThreeBlocks
from trainers.architectures import DynamicsModel, DynamicsCuriosityModel
from trainers.architectures import RNDCuriosityModel


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--seed", type=int, default=1234, help="Random seed (default: 1234)")
    parser.add_argument("--planning-mode", type=int, default=0, help="Planning algorithm (0:MPC 1:CEM 2:MPPI)")
    parser.add_argument("--curiosity-mode", type=int, default=0, help="Curiosity metric (0:FD 1:RND)")
    parser.add_argument("--total-steps", type=int, default=10000, help="Total number of agent steps (default: 1e5)")
    parser.add_argument("--dynamics-lr", type=float, default=1e-3, help="Dynamics model learning rate (default: 1e-3)")
    parser.add_argument("--curiosity-lr", type=float, default=1e-3,
                        help="Curiosity model learning rate (default: 1e-3)")
    parser.add_argument("--update-steps", type=int, default=32, help="Number of steps between updates (default: 32")
    parser.add_argument("--dynamics-epochs", type=int, default=10, help="Number of dynamics model epochs (default: 10)")
    parser.add_argument("--curiosity-epochs", type=int, default=10,
                        help="Number of curiosity model epochs (default: 10)")
    parser.add_argument("--dynamics-bsz", type=int, default=64, help="Dynamics model batch size (default: 64)")
    parser.add_argument("--curiosity-bsz", type=int, default=64, help="Curiosity model batch size (default: 64)")
    parser.add_argument("--dynamics-num-batches", type=int, default=1,
                        help="Number of dynamics model minibatch updates per-step (default: 32)")
    parser.add_argument("--curiosity-num-batches", type=int, default=1,
                        help="Number of curiosity model minibatch updates per-step (default: 32)")

    parser.add_argument("--mpc-samples", type=int, default=1000,
                        help="Number of random plan samples for MPC (default: 1000)")
    parser.add_argument("--mpc-horizon", type=int, default=10,
                        help="Planning horizon for MPC (default: 10)")
    parser.add_argument("--render", action="store_true", default=False,
                        help="Render the environment (default: False)")
    parser.add_argument("--record", action="store_true", default=False,
                        help="Record GIFs of the environment during training(default: False)")
    parser.add_argument("--tb", action="store_true", default=False,
                        help="Cache training metadata to Tensorboard (default: False)")
    args = parser.parse_args()
    return args


def dynamics_update(args, replay, optimizer, dynamics, env, writer, iteration):
    # Logging
    dynamics_losses = []
    mean = lambda a: sum(a) / len(a)

    # Perform some optimization steps
    for step in range(args.dynamics_epochs * args.dynamics_num_batches):
        batch = replay.sample(args.dynamics_bsz)

        pred_next_states = dynamics.forward(batch.state(), batch.action())

        loss = (pred_next_states - batch.next_state()).pow(2).mean()

        # Take optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dynamics_losses.append(loss)

    print('Mean TLoss: {}'.format(mean(dynamics_losses).item()))

    # Log metrics
    env.log('dynamics loss', mean(dynamics_losses).item())
    if writer is not None:
        writer.add_scalar('Dynamics/loss', mean(dynamics_losses).item(), iteration)


def curiosity_update(args, replay, optimizer, dynamics, curiosity, env, writer, iteration):
    # Logging
    curiosity_losses = []
    mean = lambda a: sum(a) / len(a)

    # Perform some optimization steps
    for step in range(args.curiosity_epochs * args.curiosity_num_batches):
        batch = replay.sample(args.curiosity_bsz)

        if args.curiosity_mode == 0:
            loss = curiosity.compute_loss(batch.state(), batch.action(), batch.next_state(), dynamics)
        elif args.curiosity_mode == 1:
            loss = curiosity.compute_loss(batch.state(), batch.action())
        else:
            print("Unknown curiosity metric specified: {}".format(args.curiosity_mode))
            sys.exit(0)

        # Take optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        curiosity_losses.append(loss)

    print('Mean CLoss: {}'.format(mean(curiosity_losses).item()))
    # Log metrics
    env.log('curiosity loss', mean(curiosity_losses).item())
    if writer is not None:
        writer.add_scalar('Curiosity/loss', mean(curiosity_losses).item(), iteration)


def vanilla_mpc(state, action_space, dynamics, curiosity):
    action_dim = action_space.shape[0]
    num_samples = 1000
    horizon = 10
    distro = Uniform(low=-1., high=1.)
    random_plans = opt_cuda(distro.sample(th.Size([num_samples, horizon, action_dim])))
    plan_scores = opt_cuda(th.zeros((num_samples,)))
    state_repeat = state.repeat(num_samples, 1)
    with th.no_grad():
        for h in range(horizon):
            actions = random_plans[:, h, :]
            scores = curiosity.forward(state_repeat, actions).squeeze(-1)
            plan_scores += scores
            state_repeat = dynamics.forward(state_repeat, actions)

        _, indices = th.topk(plan_scores, 1)
        return random_plans[indices[0], 0, :]


def cross_entropy_method(state, action_space, dynamics, curiosity):
    action_dim = action_space.shape[0]
    num_samples = 1000
    horizon = 10
    num_iters = 5
    alpha = 0.9
    num_best = int(0.75 * num_samples)
    means = opt_cuda(th.zeros(horizon, action_dim))
    covars = opt_cuda(th.ones_like(means))
    with th.no_grad():
        for m in range(num_iters):
            action_distro = MultivariateNormal(means, th.diag_embed(covars, offset=0))
            state_repeat = state.repeat(num_samples, 1)
            random_plans = opt_cuda(action_distro.sample(th.Size([num_samples])))
            random_plans = th.clamp(random_plans, -1., 1.)
            plan_scores = opt_cuda(th.zeros((num_samples,)))

            for h in range(horizon):
                actions = random_plans[:, h, :]
                scores = curiosity.forward(state_repeat, actions).squeeze(-1)
                plan_scores += scores
                state_repeat = dynamics.forward(state_repeat, actions)

            _, indices = th.topk(plan_scores, num_best)
            best_plans = random_plans[indices, :, :]
            best_mean_plan = best_plans.mean(0)
            best_var_plan = best_plans.std(0).pow(2)
            means = alpha * best_mean_plan + (1. - alpha) * means
            covars = alpha * best_var_plan + (1. - alpha) * covars
        return means[0]


def model_predictive_path_integral(state, action_space, dynamics, curiosity):
    action_dim = action_space.shape[0]
    num_samples = 2000
    horizon = 10
    num_iters = 5
    alpha = 0.9
    reward_weight = 10.
    means = opt_cuda(th.zeros(horizon, action_dim))
    action_distro = MultivariateNormal(th.zeros_like(means), th.eye(action_dim))

    with th.no_grad():
        for m in range(num_iters):
            state_repeat = state.repeat(num_samples, 1)
            base_random_plans = opt_cuda(action_distro.sample(th.Size([num_samples])))
            base_random_plans = th.clamp(base_random_plans, -1., 1.)
            plan_scores = opt_cuda(th.zeros((num_samples,)))

            prev_noise = opt_cuda(th.zeros((num_samples, action_dim)))
            for h in range(horizon):
                noise = alpha * base_random_plans[:, h, :] + (1. - alpha) * prev_noise
                actions = noise + means[h, :].unsqueeze(0).repeat(num_samples, 1)
                actions = th.clamp(actions, -1., 1.)
                scores = curiosity.forward(state_repeat, actions).squeeze()
                plan_weights = th.nn.functional.softmax(reward_weight * scores, dim=0).unsqueeze(-1)
                plan_scores += scores
                means[h, :] = (plan_weights * base_random_plans[:, h, :]).sum(0)
                state_repeat = dynamics.forward(state_repeat, actions)
                prev_noise = noise
        return means[0]


def main(args):
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
        "detailed_gmp": False,
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

    env = envs.Logger(env, interval=args.update_steps)
    # env = envs.Normalizer(env, states=True, rewards=True)
    env = envs.Torch(env)
    if args.record:
        env = envs.Recorder(env, 'out/videos/', format='gif')
    env = envs.Runner(env)

    writer = SummaryWriter(log_dir='out/tensorboard/{}'.format(env_name)) if args.tb else None

    dynamics = DynamicsModel(config_size=env.config_size, action_size=env.action_space_size)
    if args.curiosity_mode == 0:
        curiosity = DynamicsCuriosityModel(env)
    elif args.curiosity_mode == 1:
        curiosity = RNDCuriosityModel(env)
    else:
        print("Unknown curiosity metric specified: {}".format(args.curiosity_mode))
        sys.exit(0)
    dynamics = opt_cuda(dynamics)
    curiosity = opt_cuda(curiosity)

    dynamics_dataset = ch.ExperienceReplay()
    dynamics_opt = optim.Adam(dynamics.parameters(), lr=args.dynamics_lr)
    # dynamics_opt = optim.Adam(dynamics.parameters(), lr=args.dynamics_lr, weight_decay=1e-2)
    curiosity_opt = optim.Adam(curiosity.parameters(), lr=args.curiosity_lr)
    num_updates = args.total_steps // args.update_steps + 1
    if args.planning_mode == 0:
        print('Running vanilla MPC...')
        get_action = lambda state: vanilla_mpc(opt_cuda(state), env.action_space, dynamics, curiosity).cpu()
    elif args.planning_mode == 1:
        print('Running cross-entropy method...')
        get_action = lambda state: cross_entropy_method(opt_cuda(state), env.action_space, dynamics, curiosity).cpu()
    elif args.planning_mode == 2:
        print('Running model-predictive path integral control...')
        get_action = lambda state: model_predictive_path_integral(opt_cuda(state), env.action_space, dynamics,
                                                                  curiosity).cpu()
    else:
        print("Unknown planning mode: {}".format(args.planning_mode))
        sys.exit(0)

    for epoch in range(num_updates):
        # Collect ground-truth environment data
        # replay = env.run(get_action, steps=args.update_steps, render=args.render)
        replay = env.run(get_action, steps=args.update_steps, render=False)
        # dynamics_dataset += replay
        dynamics_dataset = replay
        if len(dynamics_dataset) > int(1e6):
            dynamics_dataset = dynamics_dataset[-int(1e6):]

        # Update forward dynamics model
        if th.cuda.is_available():
            dynamics_dataset = dynamics_dataset.cuda(device='0')
        dynamics_update(args, dynamics_dataset, dynamics_opt, dynamics, env, writer, epoch)
        if epoch % 10 == 0:
            th.save(dynamics.state_dict(), 'out/{}_dynamics.pt'.format(env_name))

        # Update curiosity model
        curiosity_update(args, dynamics_dataset, curiosity_opt, dynamics, curiosity, env, writer, epoch)

    th.save(dynamics.state_dict(), 'out/{}_dynamics.pt'.format(env_name))
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    main(args)
