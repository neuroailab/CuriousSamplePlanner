import copy
import glob
import os
import time
from collections import deque
import shutil
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
import pickle
from CuriousSamplePlanner.scripts.utils import *
from CuriousSamplePlanner.tasks.three_block_stack import ThreeBlocks
from CuriousSamplePlanner.tasks.ball_ramp import BallRamp
from CuriousSamplePlanner.tasks.pulley import PulleySeesaw
from CuriousSamplePlanner.tasks.bookshelf import BookShelf
from CuriousSamplePlanner.tasks.five_block_stack import FiveBlocks

import sys


def main(exp_id="no_expid", load_id="no_loadid"):

    # Set up the hyperparameters
    experiment_dict = {

        "algo": "ppo",
        "gail": True, 
        "gail_experts_dir":'./rl_utils/gail_experts',
        "gail_batch_size": 128,
        "gail_epoch": 5,
        "lr": 3e-4,
        "eps": 1e-5,
        "alpha": 0.99,
        "gamma": 0.9,
        "use_gae": True,
        "gae_lambda": 0.95,
        "entropy_coef": 0,
        "value_loss_coef": 0.5,
        "max_grad_norm": 0.5,
        "seed": 1,
        "cuda_deterministic": False,
        "num_processes": 1,
        "num_steps": 128,
        "ppo_epoch": 10,
        "num_mini_batch": 32,
        "clip_param": 0.2,
        "log_interval": 1,
        "save_interval": 100,
        "eval_interval": None,
        "num_env_steps": 10e6,
        "env_name": "HalfCheetah-v2",
        "log_dir": "/tmp/gym/",
        "save_dir": "./trained_models/",
        "use_proper_time_limits": True,
        "recurrent_policy": False,
        "use_linear_lr_decay": True,
        "custom_environment": False,
        # Hyps
        "learning_rate": 5e-5,  
        "sample_cap": 1e7, 
        "batch_size": 128,
        "node_sampling": "uniform",
        "mode": "RandomStateEmbeddingPlanner",
        "feasible_training": True,
        "nsamples_per_update": 1024,
        "training": True,
        'exploration_end': 100, 
        "exp_id": exp_id,
        "load_id": load_id,
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
        'ou_noise': True,
        'param_noise': False,
        'updates_per_step': 1,
        'replay_size': 100000,
        # Stats
        "world_model_losses": [],
        "losses": [],
        "rewards": [],
        "num_sampled_nodes": 0,
        "num_graph_nodes": 0,
    }

    if(torch.cuda.is_available()):
        prefix = "/mnt/fs0/arc11_2/solution_data/"
    else:
        prefix = "./solution_data/"

    experiment_dict['exp_path'] = prefix + experiment_dict["exp_id"]
    experiment_dict['load_path'] = prefix + experiment_dict["load_id"]

    if (os.path.isdir(experiment_dict['exp_path'])):
        shutil.rmtree(experiment_dict['exp_path'])

    os.mkdir(experiment_dict['exp_path'])


    torch.manual_seed(experiment_dict['seed'])
    torch.cuda.manual_seed_all(experiment_dict['seed'])

    if torch.cuda.is_available() and experiment_dict['cuda_deterministic']:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(experiment_dict['log_dir'])
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if(experiment_dict['custom_environment']):
        EC =  getattr(sys.modules[__name__], experiment_dict["task"])
        env = EC(experiment_dict)
    else:
        env = gym.make(experiment_dict['env_name'])




    actor_critic = Policy(
        env.observation_space.shape,
        env.action_space,
        base_kwargs={'recurrent': experiment_dict['recurrent_policy']})
    actor_critic.to(device)

    if experiment_dict['algo'] == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            experiment_dict['value_loss_coef'],
            experiment_dict['entropy_coef'],
            lr=experiment_dict['lr'],
            eps=experiment_dict['eps'],
            alpha=experiment_dict['alpha'],
            max_grad_norm=experiment_dict['max_grad_norm'])
    elif experiment_dict['algo'] == 'ppo':
        agent = algo.PPO(
            actor_critic,
            experiment_dict['clip_param'],
            experiment_dict['ppo_epoch'],
            experiment_dict['num_mini_batch'],
            experiment_dict['value_loss_coef'],
            experiment_dict['entropy_coef'],
            lr=experiment_dict['lr'],
            eps=experiment_dict['eps'],
            max_grad_norm=experiment_dict['max_grad_norm'])
    elif experiment_dict['algo'] == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, experiment_dict['value_loss_coef'], experiment_dict['entropy_coef'], acktr=True)

    if experiment_dict['gail']:
        assert len(env.observation_space.shape) == 1
        discr = gail.Discriminator(
            env.observation_space.shape[0] + env.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            experiment_dict['gail_experts_dir'], "trajs_{}.pt".format(
                experiment_dict['env_name'].split('-')[0].lower()))
        
        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4)
        drop_last = len(expert_dataset) > experiment_dict['gail_batch_size']
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=experiment_dict['gail_batch_size'],
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(experiment_dict['num_steps'], experiment_dict['num_processes'],
                              env.observation_space.shape, env.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = env.reset()
    rollouts.obs[0].copy_(torch.tensor(obs))
    rollouts.to(device)

    episode_rewards = deque(maxlen=1000)

    start = time.time()
    num_updates = int(experiment_dict['num_env_steps']) // experiment_dict['num_steps'] // experiment_dict['num_processes']
    for j in range(num_updates):
        if experiment_dict['use_linear_lr_decay']:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if experiment_dict['algo'] == "acktr" else experiment_dict['lr'])

        for step in range(experiment_dict['num_steps']):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = env.step(torch.squeeze(action))
            obs = opt_cuda(torch.tensor(obs))
            experiment_dict['rewards'].append(reward.item())
            episode_rewards.append(reward)

            if(reward == 1 and experiment_dict['custom_environment']):
                obs = env.reset()


            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in [done]])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in [infos]])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)


        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if experiment_dict['gail']:
            # if j >= 10:
            #     env.venv.eval()

            gail_epoch = experiment_dict['gail_epoch']
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts)

            for step in range(experiment_dict['num_steps']):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], experiment_dict['gamma'],
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, experiment_dict['use_gae'], experiment_dict['gamma'],
                                 experiment_dict['gae_lambda'], experiment_dict['use_proper_time_limits'])

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % experiment_dict['save_interval'] == 0
                or j == num_updates - 1) and experiment_dict['save_dir'] != "":
            save_path = os.path.join(experiment_dict['save_dir'], experiment_dict['algo'])
            try:
                os.makedirs(save_path)
            except OSError:
                pass

        if j % experiment_dict['log_interval'] == 0 and len(episode_rewards) > 1:

            # Save the data so far
            with open(experiment_dict['exp_path'] + "/exp_dict.pkl", "wb") as fa:
                pickle.dump(experiment_dict, fa)
                fa.close()

            total_num_steps = (j + 1) * experiment_dict['num_processes'] * experiment_dict['num_steps']
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.2f}/{:.2f}, min/max reward {:.2f}/{:.2f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))


if __name__ == '__main__':
    exp_id = str(sys.argv[1])
    if(len(sys.argv)>3):
        load_id = str(sys.argv[3])
    else:
        load_id = ""

    main(exp_id=exp_id, load_id=load_id)
