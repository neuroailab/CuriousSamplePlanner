#!/usr/bin/env python
from __future__ import print_function

from collections import defaultdict

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
        self.mlp = models.robotics.RoboticsMLP(config_size + action_size, config_size, layer_sizes=[hidden, hidden])

    def forward(self, config, action):
        state_action = torch.cat([config, action], dim=1)
        l = self.mlp(state_action)
        return l + config


class FactoredDynamicsModel(nn.Module):
    def __init__(self, config_size, action_size, obj_size=6):
        super(FactoredDynamicsModel, self).__init__()
        hidden = 128
        self.obj_size = obj_size
        self.unary = models.robotics.RoboticsMLP(self.obj_size + action_size, self.obj_size,
                                                 layer_sizes=[hidden, hidden])
        self.binary = models.robotics.RoboticsMLP(2*self.obj_size + action_size, self.obj_size,
                                                  layer_sizes=[hidden, hidden])

    def forward(self, config, action):
        objs = list(torch.split(config, self.obj_size, dim=1))
        unary_inputs = [torch.cat((obj, action), dim=1) for obj in objs]
        unary_outputs = [self.unary(ui) for ui in unary_inputs]
        obj_binary_outputs = []
        for i in range(len(objs)):
            binary_inputs = []
            for j in range(len(objs)):
                if i == j:
                    continue
                binary_inputs.append(torch.cat((objs[i], objs[j], action), dim=1))
            binary_outputs = [self.binary(bi) for bi in binary_inputs]
            obj_binary_outputs.append(torch.stack(binary_outputs, dim=0).sum(0))

        assert len(objs) == len(unary_outputs) == len(obj_binary_outputs)
        obj_outputs = [o + u + b for o, u, b in zip(objs, unary_outputs, obj_binary_outputs)]
        ret = torch.cat(tuple(obj_outputs), dim=1)
        assert ret.size() == config.size()
        return ret


class CuriosityModel(nn.Module):
    def __init__(self, env):
        super(CuriosityModel, self).__init__()
        hidden = 128
        self.curiosity = nn.Sequential(
            nn.Linear(env.state_size + env.action_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, state, action):
        state_action = torch.cat((state, action), dim=-1)
        return self.curiosity(state_action)


class DynamicsCuriosityModel(CuriosityModel):
    def __init__(self, env):
        super(DynamicsCuriosityModel, self).__init__(env)

    def compute_loss(self, state, action, next_state, dynamics):
        pred_next_states = dynamics.forward(state, action)

        dynamics_loss = (pred_next_states - next_state).pow(2).sum(-1).detach()
        pred_dynamics_loss = self.forward(state, action)
        return (pred_dynamics_loss - dynamics_loss).pow(2).mean()


class RNDCuriosityModel(nn.Module):
    def __init__(self, env):
        super(RNDCuriosityModel, self).__init__()
        hidden = 128
        self.osz = 64
        self.teacher = nn.Sequential(
            nn.Linear(env.state_size + env.action_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, self.osz))
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.student = nn.Sequential(
            nn.Linear(env.state_size + env.action_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, self.osz))

    def forward(self, state, action):
        state_action = torch.cat((state, action), dim=-1)
        teacher_pred = self.teacher(state_action)
        student_pred = self.student(state_action)
        return (teacher_pred - student_pred).pow(2).mean(-1)

    def compute_loss(self, state, action):
        return self.forward(state, action).mean()

    def parameters(self, recurse=True):
        return self.student.parameters(recurse=recurse)
