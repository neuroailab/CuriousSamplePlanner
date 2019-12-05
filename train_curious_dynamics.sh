#!/bin/bash

time python dynamics_train.py --planning-mode 1 --curiosity-mode 1 --total-steps 1000000 --mpc-horizon 3 --update-steps 100 --dynamics-num-batches 4 --curiosity-num-batches 4 --tb
