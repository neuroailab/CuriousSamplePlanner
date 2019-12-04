import sys

import time
from itertools import accumulate

import torch
import pickle
import random
import argparse
import numpy as np
import os.path as osp

import torch.nn as nn
from cherry import envs
from cherry import models
from torch.distributions import Normal

from scripts.utils import opt_cuda
from tasks.three_block_stack import ThreeBlocks


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-path", type=str, required=True, help="Path to expert dataset")
    args = parser.parse_args()
    return args


class PolicyNet(nn.Module):
    def __init__(self, obs_size, action_size):
        super(PolicyNet, self).__init__()
        self.actor = models.robotics.RoboticsActor(obs_size, action_size, layer_sizes=[64, 64])

        self.logstd = nn.Parameter(torch.zeros(action_size))

    def forward(self, x):
        action_scores = self.actor(x)
        # action_density = Normal(loc=action_scores, scale=self.logstd.exp())
        # return action_density.rsample()
        return action_scores


def train(args):
    states, actions, _ = pickle.load(open(args.path, 'rb'))
    converged = False
    epsilon = 1e-3
    prev_val_loss = None
    policy = opt_cuda(PolicyNet(states.shape[-1], actions.shape[-1]))
    policy.train()
    num_samples = states.shape[0]
    train_val_split = 0.9
    policy_opt = torch.optim.Adam(policy.parameters(), lr=1e-3, weight_decay=1e-2)
    # policy_opt = torch.optim.Adam(policy.parameters(), lr=1e-3, weight_decay=0.0)
    num_epochs = 0
    train_idxs = random.sample(range(num_samples), int(train_val_split * num_samples))
    val_idxs = list(set(range(num_samples)) - set(train_idxs))

    train_states = opt_cuda(torch.tensor(states[train_idxs, :])).float()
    train_actions = opt_cuda(torch.tensor(actions[train_idxs, :])).float()
    val_states = opt_cuda(torch.tensor(states[val_idxs, :])).float()
    val_actions = opt_cuda(torch.tensor(actions[val_idxs, :])).float()
    # bsz = 8
    bsz = 64
    num_batches = int(len(train_idxs) / bsz)
    min_epochs = 2000
    while not converged:
        batch_idxs = [list(range(len(train_idxs)))[x - y:x] for x, y in zip(accumulate([bsz] * num_batches), [bsz] * num_batches)]
        random.shuffle(batch_idxs)
        for batch in batch_idxs:
            batch_states, batch_actions = train_states[batch, :], train_actions[batch, :]
            policy_actions = policy.forward(batch_states)
            loss = (policy_actions - batch_actions).pow(2).sum(1).mean(0)
            policy_opt.zero_grad()
            loss.backward()
            policy_opt.step()
        num_epochs += 1

        with torch.no_grad():
            policy_actions = policy.forward(val_states)
            val_loss = (policy_actions - val_actions).pow(2).sum(1).mean(0).item()

        print('Epoch {} | Train loss: {} | Val loss: {}'.format(num_epochs, loss.item(), val_loss))
        if prev_val_loss is not None:
            if abs(val_loss - prev_val_loss) <= epsilon and num_epochs >= min_epochs:
                converged = True

        prev_val_loss = val_loss
    torch.save(policy.state_dict(), osp.join(osp.dirname(args.path), 'bc_imitation_policy.pt'))


def eval(args):
    states, actions, _ = pickle.load(open(args.path, 'rb'))
    policy = opt_cuda(PolicyNet(states.shape[-1], actions.shape[-1]))
    policy.eval()
    policy_path = osp.join(osp.dirname(args.path), 'bc_imitation_policy.pt')
    policy.load_state_dict(torch.load(policy_path, map_location='cpu' if not torch.cuda.is_available() else "cuda:0"))

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
        # "detailed_gmp": False,
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

    # env = envs.Logger(env, interval=args.update_steps)
    # env = envs.Normalizer(env, states=True, rewards=True)
    env = envs.Torch(env)
    obs = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            action = policy.forward(obs)
            print(action)
        time.sleep(1)
        obs, reward, done, _ = env.step(action.squeeze(0).data.numpy())


def main(args):
    # retrain = True
    retrain = False

    if retrain:
        train(args)
    eval(args)

if __name__ == '__main__':
    args = parse_args()
    main(args)
