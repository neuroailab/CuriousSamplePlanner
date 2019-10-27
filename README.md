# Curious Sample Planner
Multi-Step Motion Planning in a Pybullet Environment


# How to run an experiment
```
git clone https://github.com/neuroailab/CuriousSamplePlanner.git
cd CuriousSamplePlanner
python3 -m pip install -r requirements.txt
python3 main.py <exp_id> <gpu_num>
```


# Understanding CSP hyperparameters

### world_model_losses
Keeps track of the losses from the world model. 

### num_sampled_nodes
Keeps track of the number of macro-actions tested

### num_graph_nodes
Keeps track of the number of nodes currently in the graph

### num_training_epochs
Number of epochs to train the world model until updating the graph

### learning_rate
Learning rate of the world model

### actor_learning_rate
Learning rate of the actor in actor-critic

### critic_learning_rate
Learning rate of the critic in actor-critic

### sample_cap
Maximum number of samples before terminating.

### batch_size
Batch size for the world model

### node_sampling
Batch size for the world model

### node_sampling
Node sampling modes:
- uniform select nodes uniformly from each state
- softmax select nodes from softmax of probabilities on each graph state

### mode
RecycleACPlanner

### nsamples_per_update
TOOD

### exp_id
ID for the experiment you're currently running

### load_id
ID for experiment you've run before if you want to visualize the results

### enable_asm
Do you want to enable adversarial action selection from each node or uniform action selection

### recycle
Hyperparameter for 
NOTE: Recycle has not been tested for a while.

### growth_factor
How much to grow the graph by at every planning iteration

### detailed_gmp
There are two modes for geometric motion planning.
Rough Mode -- Much quicker but sometimes gives invalid plans. 
Detailed Mode -- Slower but always gives a correct plan.

### task
The task you want to run
Available Tasks
- ThreeBlocks
- FiveBlocks
- BallRamp
- Bookshelf
- PulleySeesaw