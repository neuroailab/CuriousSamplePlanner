#!/usr/bin/env python
from __future__ import print_function

import torch
from torch import nn

from cherry import models
import cherry.distributions as dist


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvWorldModel(nn.Module):
    def __init__(self, config_size=9, num_perspectives=1):
        super(ConvWorldModel, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(3 * num_perspectives, 32, 8, stride=4), nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                                  nn.Conv2d(64, 32, 3, stride=1), nn.ReLU(), Flatten(),
                                  nn.Linear(1568 * num_perspectives, 128), nn.ReLU(), nn.Linear(128, config_size))

    def forward(self, inputs):
        return self.conv(inputs)


class SkinnyWorldModel(nn.Module):
    def __init__(self, config_size=9):
        super(SkinnyWorldModel, self).__init__()
        hidden = 128
        self.mlp = nn.Sequential(nn.Linear(config_size, config_size))

    def forward(self, config):
        l = self.mlp(config)
        return l


class WorldModel(nn.Module):
    def __init__(self, config_size=9):
        super(WorldModel, self).__init__()
        hidden = 128
        self.mlp = nn.Sequential(nn.Linear(config_size, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, config_size))

    def forward(self, config):
        l = self.mlp(config)
        return l


class DynamicsModel(nn.Module):
    def __init__(self, config_size=0, action_size=0):
        super(DynamicsModel, self).__init__()
        hidden = 128
        self.mlp = nn.Sequential(nn.Linear(config_size + action_size, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                 nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, config_size))

    def forward(self, config, action):
        state_action = torch.cat([config, action], dim=1)
        l = self.mlp(state_action)
        return l


class RNDCuriosityModel(nn.Module):
    def __init__(self, env):
        super(RNDCuriosityModel, self).__init__()
        self.hsz = 128
        self.osz = 64
        self.teacher = models.robotics.RoboticsMLP(env.state_size + env.action_size, self.osz,
                                                     layer_sizes=[self.hsz, self.hsz])
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.student = models.robotics.RoboticsMLP(env.state_size + env.action_size, self.osz,
                                                     layer_sizes=[self.hsz, self.hsz])

    def forward(self, state, action):
        state_action = torch.cat((state, action), dim=-1)
        teacher_pred = self.teacher(state_action)
        student_pred = self.student(state_action)
        return (teacher_pred - student_pred).pow(2).mean(-1)

    def compute_loss(self, state, action):
        return self.forward(state, action).mean()

    def parameters(self, recurse=True):
        return self.student.parameters(recurse=recurse)
