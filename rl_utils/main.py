import copy
import glob
import os
import time
from collections import deque
import pickle
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import shutil
import sys
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from CuriousSamplePlanner.scripts.utils import *



class Args:
    def __init__(self, **entries):
        self.__dict__.update(entries)



def main(exp_id="no_expid", load_id="no_loadid"):
    experiment_dict = {
            "algo": "ppo",
            "gail": True, 
            "gail_experts_dir":'./rl_utils/gail_experts',
            "gail_batch_size": 128,
            "gail_epoch": 5,
            "lr": 3e-4,
            "gail_lr": 1e-3,
            "eps": 1e-5,
            "alpha": 0.99,
            "gamma": 0.5,
            "use_gae": True,
            "gae_lambda": 0.95,
            "entropy_coef": 0,
            "value_loss_coef": 0.5,
            "max_grad_norm": 0.5,
            "seed": 1,
            "cuda_deterministic": False,
            "num_processes": 1,
            "num_steps": 2048,
            "ppo_epoch": 10,
            "num_mini_batch": 32,
            "clip_param": 0.2,
            "log_interval": 1,
            "save_interval": 100,
            "eval_interval": None,
            "num_env_steps": 10e6,
            "expert_examples": 2048,
            # "env_name": "HalfCheetah-v2",
            "env_name": "ThreeBlocks",
            "log_dir": "/tmp/gym/",
            "save_dir": "./trained_models/",
            "use_proper_time_limits": True,
            "recurrent_policy": False,
            "use_linear_lr_decay": True,
            "custom_environment": False,

            # Hyps 
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
            "cuda": torch.cuda.is_available(),
            # Stats
            "world_model_losses": [],
            "losses": [],
            "gail_loss": [],
            "rewards": [],
            "num_sampled_nodes": 0,
            "num_graph_nodes": 0,
    }

    exp_type = exp_id.split("::")[1]
    change_k, change_v = exp_type.split("=")
    experiment_dict[change_k] = float(change_v)

    if(torch.cuda.is_available()):
        prefix = "/mnt/fs0/arc11_2/solution_data_new/"
    else:
        prefix = "./solution_data/"

    experiment_dict['exp_path'] = prefix + experiment_dict["exp_id"]
    experiment_dict['load_path'] = prefix + experiment_dict["load_id"]

    if (os.path.isdir(experiment_dict['exp_path'])):
        shutil.rmtree(experiment_dict['exp_path'])

    os.mkdir(experiment_dict['exp_path'])

    args = Args(**experiment_dict)
    # args = get_args()

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

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False, experiment_dict = experiment_dict)

    actor_critic = opt_cuda(Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy}))


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
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device, gail_lr=args.gail_lr)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))
        
        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=args.expert_examples, subsample_frequency=1)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.cuda()

    episode_rewards = deque(maxlen=1000)
    episode_gail_losses = deque(maxlen=1000)

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
            obs, reward, done, infos = envs.step(action)

            episode_rewards.append(reward.item())
            
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
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                gl = discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)
                episode_gail_losses.append(gl)

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

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:

            experiment_dict['rewards'].append(np.mean(episode_rewards))
            experiment_dict['gail_loss'].append(np.mean(episode_gail_losses))

            # Save the data so far
            with open(experiment_dict['exp_path'] + "/exp_dict.pkl", "wb") as fa:
                pickle.dump(experiment_dict, fa)
                fa.close()


            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.3f}/{:.3f}, min/max reward {:.3f}/{:.3f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)

if __name__ == '__main__':
    exp_id = str(sys.argv[1])
    if(len(sys.argv)>3):
        load_id = str(sys.argv[3])
    else:
        load_id = ""

    main(exp_id=exp_id, load_id=load_id)