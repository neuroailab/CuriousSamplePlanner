import sys
import pickle
import argparse
import numpy as np
import os.path as osp

import trainers


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-path", type=str, required=True, help="Path to expert trajectories")
    args = parser.parse_args()
    return args


def main(args):
    # num_trajs = 25
    # num_trajs = 100
    num_trajs = 1000
    form = 'expert{}/found_path.pkl'

    states, actions, next_states = [], [], []

    for i in range(num_trajs):
        plan_path = osp.join(args.path, form.format(i))
        plan = pickle.load(open(plan_path, 'rb'))
        print(plan_path, len(plan))
        for j in range(1, len(plan)):
            graph_node = plan[j]
            state, action, next_state = graph_node.preconfig, graph_node.action, graph_node.config
            states.append(state)
            actions.append(action)
            next_states.append(next_state)

    states, actions, next_states = np.array(states), np.array(actions), np.array(next_states)
    print(states.shape, actions.shape, next_states.shape)
    pickle.dump((states, actions, next_states), open(osp.join(args.path, 'expert_dataset.pkl'), 'wb'))

if __name__ == '__main__':
    args = parse_args()
    main(args)
