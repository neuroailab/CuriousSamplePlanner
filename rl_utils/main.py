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

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
import sys


def main():

    # Set up the hyperparameters
    experiment_dict = {
        # Hyps
        "task": "ThreeBlocks",
        "learning_rate": 5e-5,  
        "sample_cap": 1e7, 
        "batch_size": 128,
        'actor_lr': 1e-4,
        'critic_lr': 1e-3,
        "node_sampling": "uniform",
        "mode": "RandomStateEmbeddingPlanner",
        "feasible_training": True,
        "nsamples_per_update": 1024,
        "training": True,
        'exploration_end': 100, 
        "exp_id": "test",
        "load_id": None,
        'noise_scale': 0.3,
        'final_noise_scale': 0.05,
        'update_interval' : 1,
        "enable_asm": False, 
        "growth_factor": 10,
        "detailed_gmp": False, 
        "adaptive_batch": True,
        "num_training_epochs": 30,
        "infeasible_penalty" : 0,
        'tau': 0.001,
        'reward_size': 100,
        'hidden_size': 64,
        'use_splitter': False, # Can't use splitter on ppo or a2c because they are on-policy algorithms
        'split': 0.2,
        'gamma': 0.9,
        'ou_noise': True,
        'param_noise': False,
        'updates_per_step': 1,
        'replay_size': 100000,
        # Stats
        "world_model_losses": [],
        "feasibility":[],
        "rewards": [],
        "num_sampled_nodes": 0,
        "num_graph_nodes": 0,
    }

    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    EC =  getattr(sys.modules[__name__], experiment_dict["task"])
    env = EC(experiment_dict)

    actor_critic = Policy(
        env.observation_space.shape,
        env.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(env.observation_space.shape) == 1
        discr = gail.Discriminator(
            env.observation_space.shape[0] + env.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))
        
        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4)
        drop_last = len(expert_dataset) > args.gail_batch_size
        print(drop_last)
        print("Gail batch size: "+str(args.gail_batch_size))
        print(len(expert_dataset))
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)
        print("Loader 1")
        print(gail_train_loader)
        for g in gail_train_loader:
            print("Len g:"+str(len(g[0])))

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              env.observation_space.shape, env.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = env.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs

            obs, reward, done, infos = env.step(action)
            print(reward)
            episode_rewards.append(reward)

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

        if args.gail:
            if j >= 10:
                env.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # torch.save([
            #     actor_critic,
            #     getattr(utils.get_vec_normalize(env), 'ob_rms', None)
            # ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        # if (args.eval_interval is not None and len(episode_rewards) > 1
        #         and j % args.eval_interval == 0):
        #     ob_rms = utils.get_vec_normalize(env).ob_rms
        #     evaluate(actor_critic, ob_rms, args.env_name, args.seed,
        #              args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
